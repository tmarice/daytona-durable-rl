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
    CreateSnapshotParams,
    Image,
    Resources,
    SessionExecuteRequest,
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


class DaytonaJob:
    def __init__(
        self,
        daytona: AsyncDaytona,
        snapshot: Snapshot,
        volume: Volume,
        source_package: bytes,
        entrypoint: str,
        config: dict,
        env_vars: dict | None = None,
    ):
        self.id = uuid.uuid4().hex
        self.status = DaytonaJobStatus.PENDING
        self._code_dir = f"/run/{self.id}/code"
        self._data_dir = f"/data/jobs/{self.id}"
        self._checkpoint_dir = f"{self._data_dir}/checkpoints"
        self._sandbox = None
        self._command_id = None

        config = dict(config)
        config["checkpoint_dir"] = self._checkpoint_dir
        self._command = self._build_command(entrypoint, config)

        self._daytona = daytona
        self._snapshot = snapshot
        self._source_package = source_package
        self._volume = volume
        self._env_vars = env_vars or {}

    async def start(self):
        """Starts the Daytona sandbox and runs the specified job."""
        if self.status in (DaytonaJobStatus.PREEMPTED, DaytonaJobStatus.PENDING):
            self.status = DaytonaJobStatus.PREPARING
            self._sandbox = await self._create_sandbox()
            await self._upload_source_package(self._sandbox)

        else:
            raise RuntimeError(f"Cannot start job {self.id} in status {self.status}")

        await self._sandbox.process.create_session(self.id)
        response = await self._sandbox.process.execute_session_command(
            self.id, SessionExecuteRequest(command=self._command, run_async=True)
        )

        self._command_id = response.cmd_id
        self.status = DaytonaJobStatus.RUNNING

    async def wait(self) -> DaytonaJobResult:
        while True:
            await self._sandbox.refresh_data()

            # TODO Handle manual deletion as preemption to be able to simulate failure
            if self._sandbox.spot_evicted_at:
                self.status = DaytonaJobStatus.PREEMPTED
                raise DaytonaSpotEvictedError(
                    f"Job {self.id} was preempted by spot instance eviction."
                )

            command = await self._sandbox.process.get_session_command(self.id, self._command_id)
            if command.exit_code is not None:
                if command.exit_code == 0:
                    self.status = DaytonaJobStatus.COMPLETED
                else:
                    self.status = DaytonaJobStatus.FAILED

                logs = await self._sandbox.process.get_session_command_logs(
                    self.id, self._command_id
                )
                return DaytonaJobResult(
                    job_id=self.id,
                    status=self.status,
                    exit_code=command.exit_code,
                    logs=logs.output,
                )

            await asyncio.sleep(5)  # TODO Make this configurable

    async def cancel(self):
        await self.cleanup()
        self.status = DaytonaJobStatus.CANCELED

    async def cleanup(self, wait: bool = False) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.delete(wait=wait)
            except DaytonaNotFoundError:
                pass
            except Exception:
                logger.exception("Failed to delete sandbox for job %s", self.id)
                raise
            self._sandbox = None
            self._command_id = None

    async def _create_sandbox(self) -> AsyncSandbox:
        try:
            # TODO Handle not enough capacity failures -- retry after x seconds
            sandbox = await self._daytona.v0.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=self._snapshot.name,
                    volumes=[VolumeMount(volume_id=self._volume.id, mount_path="/data")],
                    env_vars=self._env_vars,
                    auto_stop_interval=0,
                    ttl_minutes=65,
                    ephemeral=True,
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

    def _build_command(self, entrypoint: str, config: dict) -> str:
        module, function = entrypoint.split(":", 1)
        code = (
            "import importlib; "
            f"getattr(importlib.import_module({module!r}), {function!r})({config!r})"
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
        env_vars: dict | None = None,
        max_concurrent_jobs: int = 1,
        verbose: bool = False,
    ):
        if max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be at least 1")
        self._workdir = Path(work_dir)
        self._entrypoint = entrypoint
        self._resources = resources
        self._volume_name = volume_name
        # TODO Use daytona secrets API instead
        self._env_vars = env_vars or {}
        self._max_concurrent_jobs = max_concurrent_jobs
        self._verbose = verbose

        self._jobs = {}
        self._source_package = self._package_source()
        self._daytona = AsyncDaytona()

    async def close(self) -> None:
        """Close the SDK client after all job tasks have finished cleaning up."""
        await self._daytona.close()

    async def create_job(
        self,
        config: dict,
    ) -> DaytonaJob:
        snapshot = await self._get_or_create_snapshot()
        volume = await self._get_or_create_volume()

        job = DaytonaJob(
            daytona=self._daytona,
            snapshot=snapshot,
            volume=volume,
            source_package=self._source_package,
            entrypoint=self._entrypoint,
            config=config,
            env_vars=self._env_vars,
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
                    await job.start()
                    return await job.wait()
                except DaytonaSpotEvictedError:
                    logger.warning("Job %s was preempted, re-running...", job.id)
                finally:
                    await job.cleanup(wait=False)

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


class ExperimentSession:
    def __init__(
        self,
        experiment_id: str,
        attempt_id: str,
        checkpoint_dir: str,
    ):
        self._checkpoint_dir = Path(checkpoint_dir)

    def save_checkpoint(self, checkpoint_data: dict):
        # TODO Save checkpoint data to new file in checkpoint_dir with unique name, and replace latest_checkpoint with symlink to it
        pass

    def restore_checkpoint(self) -> dict:
        # TODO Read latest checkpoint file from checkpoint_dir and return its contents as dict, or None if no checkpoint exists
        pass
