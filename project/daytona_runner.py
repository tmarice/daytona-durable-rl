import asyncio
import json
import logging
import shlex
import tarfile
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from daytona import AsyncDaytona
from daytona.v0 import (
    AsyncSandbox,
    CreateSandboxFromSnapshotParams,
    CreateSecretParams,
    CreateSnapshotParams,
    Image,
    Resources,
    SandboxState,
    SessionExecuteRequest,
    UpdateSecretParams,
    VolumeMount,
)
from daytona.v0.common.errors import DaytonaError, DaytonaNotFoundError, DaytonaSpotEvictedError
from daytona.v0.common.snapshot import Snapshot
from daytona.v0.common.volume import Volume

logger = logging.getLogger(__name__)


# TODO Allow only valid state transitions
class DaytonaJobStatus(StrEnum):
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


@dataclass
class DaytonaJobResult:
    job_id: str
    status: DaytonaJobStatus
    exit_code: int | None = None
    logs: str | None = None
    # TODO Add output artifacts


class DaytonaJob:
    def __init__(
        self,
        daytona: AsyncDaytona,
        snapshot: Snapshot,
        volume: Volume,
        source_package: bytes,
        entrypoint: str,
        config: dict,
        secrets: dict[str, str] | None = None,
    ):
        # TODO Stable job IDs would allow resuming a job after runner restart
        self.id = uuid.uuid4().hex
        self.status = DaytonaJobStatus.PENDING
        self._code_dir = f"/run/{self.id}/code"
        self._data_dir = f"/data/jobs/{self.id}"
        self._checkpoint_dir = f"{self._data_dir}/checkpoints"
        self.attempt_id: str | None = None
        self._entrypoint = entrypoint
        self._config = dict(config)

        self._daytona = daytona
        self._snapshot = snapshot
        self._source_package = source_package
        self._volume = volume
        self._secrets = secrets or {}

    async def run_attempt(self) -> DaytonaJobResult:
        """Runs the job in a new sandbox, which is deleted before this returns or raises."""
        self.attempt_id = uuid.uuid4().hex
        self.status = DaytonaJobStatus.PREPARING
        sandbox_name = f"job-{self.id}-{self.attempt_id[:8]}"
        sandbox = None
        try:
            sandbox = await self._create_sandbox(sandbox_name)
            await self._upload_source_package(sandbox)
            await sandbox.process.create_session(self.id)
            response = await sandbox.process.execute_session_command(
                self.id, SessionExecuteRequest(command=self._build_command(), run_async=True)
            )
            self.status = DaytonaJobStatus.RUNNING
            result = await self._wait_for_command(sandbox, response.cmd_id)
            self.status = result.status
            return result
        except DaytonaSpotEvictedError:
            self.status = DaytonaJobStatus.PREEMPTED
            raise
        except asyncio.CancelledError:
            self.status = DaytonaJobStatus.CANCELED
            raise
        except Exception:
            self.status = DaytonaJobStatus.FAILED
            raise
        finally:
            await self._delete_sandbox(sandbox, sandbox_name)

    async def _wait_for_command(self, sandbox: AsyncSandbox, command_id: str) -> DaytonaJobResult:
        while True:
            await self._raise_if_preempted(sandbox)

            try:
                command = await sandbox.process.get_session_command(self.id, command_id)
                if command.exit_code is not None:
                    logs = await sandbox.process.get_session_command_logs(self.id, command_id)
            except DaytonaError:
                await self._raise_if_preempted(sandbox)
                raise

            if command.exit_code is not None:
                return DaytonaJobResult(
                    job_id=self.id,
                    status=(
                        DaytonaJobStatus.COMPLETED
                        if command.exit_code == 0
                        else DaytonaJobStatus.FAILED
                    ),
                    exit_code=command.exit_code,
                    logs=logs.output,
                )

            await asyncio.sleep(5)  # TODO Make this configurable

    async def _raise_if_preempted(self, sandbox: AsyncSandbox) -> None:
        try:
            await sandbox.refresh_data()
        except DaytonaNotFoundError:
            reason = "sandbox no longer exists"
        else:
            if sandbox.spot_evicted_at:
                reason = "spot instance eviction"
            elif sandbox.state in (
                SandboxState.STOPPING,
                SandboxState.STOPPED,
                SandboxState.DESTROYING,
                SandboxState.DESTROYED,
            ):
                reason = f"sandbox is {sandbox.state}"
            else:
                return

        raise DaytonaSpotEvictedError(f"Job {self.id} was preempted: {reason}.")

    async def _delete_sandbox(self, sandbox: AsyncSandbox | None, name: str) -> None:
        """Best effort, so that a failed deletion cannot replace the attempt's outcome."""
        try:
            if sandbox is None:
                sandbox = await self._daytona.v0.get(name)
            await sandbox.delete(wait=False)
        except DaytonaNotFoundError:
            pass
        except Exception:
            logger.exception(
                "Failed to delete sandbox %s for job %s; its TTL will remove it", name, self.id
            )

    async def _create_sandbox(self, name: str) -> AsyncSandbox:
        try:
            # TODO Handle not enough capacity failures -- retry after x seconds
            sandbox = await self._daytona.v0.create(
                CreateSandboxFromSnapshotParams(
                    name=name,
                    snapshot=self._snapshot.name,
                    volumes=[VolumeMount(volume_id=self._volume.id, mount_path="/data")],
                    secrets=self._secrets,
                    auto_stop_interval=0,
                    ttl_minutes=65,
                    ephemeral=True,
                    spot=True,
                )
            )
            return sandbox
        except DaytonaError:
            logger.exception("Failed to create sandbox for job %s", self.id)
            raise

    async def _upload_source_package(self, sandbox: AsyncSandbox):
        # TODO Consider using ephemeral sandbox for uploading source code to the volume, and then reuse source
        archive_path = f"/tmp/{self.id}-source.tar.gz"
        # TODO Handle upload failures
        await sandbox.fs.upload_file(self._source_package, archive_path)
        result = await sandbox.process.exec(
            f"mkdir -p {self._code_dir} {self._data_dir} {shlex.quote(self._checkpoint_dir)} && "
            f"tar --no-same-owner -xzf {archive_path} -C {self._code_dir}",
            timeout=60,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to prepare job {self.id} sandbox: {result.result}")

    def _build_command(self) -> str:
        module, function = self._entrypoint.split(":", 1)
        code = (
            "import importlib; "
            f"getattr(importlib.import_module({module!r}), {function!r})("
            f"{self._config!r}, experiment_id={self.id!r}, attempt_id={self.attempt_id!r}, "
            f"checkpoint_dir={self._checkpoint_dir!r})"
        )
        command = f"cd {shlex.quote(self._code_dir)} && " + shlex.join(["python", "-uc", code])

        return command


class DaytonaRunner:
    def __init__(
        self,
        work_dir: str,
        entrypoint: str,
        resources: dict,
        volume_name: str,
        secrets: dict[str, str] | None = None,  # {"WANDB_API_KEY": "your_api_key"}
        max_concurrent_jobs: int = 1,
        verbose: bool = False,
    ):
        if max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be at least 1")
        self._workdir = Path(work_dir)
        self._entrypoint = entrypoint
        self._resources = resources
        self._volume_name = volume_name
        self._max_concurrent_jobs = max_concurrent_jobs
        self._verbose = verbose

        self._jobs = {}
        self._source_package = self._package_source()
        self._daytona = AsyncDaytona()
        self._secrets = secrets or {}
        self._secret_names = None

    async def close(self) -> None:
        """Close the SDK client after all job tasks have finished cleaning up."""
        await self._daytona.close()

    async def create_job(
        self,
        config: dict,
    ) -> DaytonaJob:
        snapshot = await self._get_or_create_snapshot()
        volume = await self._get_or_create_volume()
        if self._secret_names is None:
            self._secret_names = await self._create_or_update_secrets(self._secrets)

        job = DaytonaJob(
            daytona=self._daytona,
            snapshot=snapshot,
            volume=volume,
            source_package=self._source_package,
            entrypoint=self._entrypoint,
            config=config,
            secrets=self._secret_names,
        )
        self._jobs[job.id] = job

        return job

    async def run(self, jobs: Iterable[DaytonaJob]) -> AsyncIterator[DaytonaJobResult]:
        """Yield job results, then raise any job exceptions together after the batch finishes."""
        slots = asyncio.Semaphore(self._max_concurrent_jobs)
        tasks = [asyncio.create_task(self._run_job(job, slots)) for job in jobs]
        errors = []

        try:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                except Exception as exc:  # noqa
                    errors.append(exc)
                else:
                    yield result
        finally:
            for task in tasks:
                task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

        if errors:
            raise ExceptionGroup("Some jobs failed", errors)

    async def _run_job(self, job: DaytonaJob, slots: asyncio.Semaphore) -> DaytonaJobResult:
        # TODO Handle timeouts
        while True:
            async with slots:
                try:
                    return await job.run_attempt()
                except DaytonaSpotEvictedError:
                    logger.warning("Job %s was preempted, re-running...", job.id)

    async def _get_or_create_snapshot(self) -> Snapshot:
        uv_lock_path = self._workdir / "uv.lock"
        pyproject_toml_path = self._workdir / "pyproject.toml"

        # TODO libgl1 is hardcoded for now -- allow extra libs to be specified in config
        image = (
            Image.base("python:3.14-slim-bookworm")
            .run_commands("apt-get update && apt-get install -y --no-install-recommends libgl1")
            .pip_install_from_pyproject(str(pyproject_toml_path), optional_dependencies=[])
        )
        key = sha256(
            b"\0".join(
                [
                    uv_lock_path.read_bytes(),
                    pyproject_toml_path.read_bytes(),
                    image.dockerfile().encode(),
                    json.dumps(self._resources, sort_keys=True).encode(),
                ]
            )
        ).hexdigest()
        try:
            snapshot = await self._daytona.v0.snapshot.get(key)
        except DaytonaNotFoundError:
            async with asyncio.timeout(1200):
                snapshot = await self._daytona.v0.snapshot.create(
                    CreateSnapshotParams(
                        name=key,
                        image=image,
                        resources=Resources(**self._resources),
                    ),
                    # TODO Use verbose flag to control logging
                    on_logs=lambda chunk: print(chunk, end="", flush=True),
                )

        # TODO Check snapshot state
        return snapshot

    async def _get_or_create_volume(self) -> Volume:
        volume = await self._daytona.v0.volume.get(self._volume_name, create=True)

        async with asyncio.timeout(60):
            while volume.state != "ready":
                if volume.state == "error":
                    raise RuntimeError(f"Volume {self._volume_name} is in error state.")

                await asyncio.sleep(1)
                volume = await self._daytona.v0.volume.get(self._volume_name)

        return volume

    async def _create_or_update_secrets(self, secrets: dict) -> dict:
        existing_ids = {}
        secret_names = {}
        if not secrets:
            return secret_names

        cursor = None
        while True:
            page = await self._daytona.v0.secret.list(cursor=cursor)
            existing_ids |= {s.name: s.id for s in page.items}
            cursor = page.next_cursor
            if cursor is None:
                break

        for env_name, value in secrets.items():
            name = f"daytona-runner-{env_name}"
            if name in existing_ids:
                # Stored values cannot be read back to compare, so always overwrite.
                await self._daytona.v0.secret.update(
                    existing_ids[name], UpdateSecretParams(value=value)
                )
            else:
                await self._daytona.v0.secret.create(CreateSecretParams(name=name, value=value))

            secret_names[env_name] = name

        return secret_names

    # TODO Perhaps parse .gitignore
    SOURCE_EXCLUDES = (
        ".git",
        ".venv",
        ".cache",
        ".pytest_cache",
        ".ruff_cache",
        ".aws",
        ".ssh",
        ".env*",
        ".DS_Store",
        "__pycache__",
        "*.py[cod]",
        "runs",
        "wandb",
    )

    def _package_source(self) -> bytes:
        if not self._workdir.is_dir():
            raise NotADirectoryError(self._workdir)

        with BytesIO() as buffer:
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                archive.add(
                    self._workdir,
                    arcname=".",
                    filter=lambda member: (
                        None
                        if any(
                            fnmatchcase(Path(member.name).name, pattern)
                            for pattern in self.SOURCE_EXCLUDES
                        )
                        else member
                    ),
                )
            return buffer.getvalue()
