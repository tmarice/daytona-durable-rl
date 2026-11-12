import asyncio
import json
import logging
import shlex
import tarfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from aiohttp import ClientConnectionError, ClientPayloadError
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
from daytona.v0.common.errors import (
    DaytonaConnectionError,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaSpotEvictedError,
)
from daytona.v0.common.process import (
    ExecuteResponse,
    SessionCommandLogsResponse,
    SessionExecuteResponse,
)
from daytona.v0.common.secret import ListSecretsResponse, Secret
from daytona.v0.common.snapshot import Snapshot
from daytona.v0.common.volume import Volume
from daytona_api_client_async.exceptions import ApiException
from daytona_toolbox_api_client_async import Command

logger = logging.getLogger(__name__)


class DaytonaAPI:
    def __init__(self, max_retries: int) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._max_retries = max_retries
        self._client = AsyncDaytona()

    async def _request[T, **P](
        self,
        name: str,
        operation: Callable[P, Awaitable[T]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        retries = 0
        delay = 5
        while True:
            try:
                return await operation(*args, **kwargs)
            except (
                DaytonaError,
                ApiException,
                ClientConnectionError,
                ClientPayloadError,
                TimeoutError,
            ) as error:
                status = getattr(error, "status_code", None) or getattr(error, "status", None)
                transient = isinstance(
                    error,
                    (
                        DaytonaConnectionError,
                        ClientConnectionError,
                        ClientPayloadError,
                        TimeoutError,
                    ),
                ) or status in {408, 429, 500, 502, 503, 504}
                if (
                    isinstance(error, DaytonaSpotEvictedError)
                    or not transient
                    or retries >= self._max_retries
                ):
                    raise
                retries += 1
                logger.warning(
                    "Daytona %s failed (%s, HTTP %s); retry %d/%d in %ds",
                    name,
                    type(error).__name__,
                    status,
                    retries,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def close(self) -> None:
        await self._client.close()

    async def create_sandbox(self, params: CreateSandboxFromSnapshotParams) -> AsyncSandbox:
        async def create_once() -> AsyncSandbox:
            creation = asyncio.create_task(self._client.v0.create(params))
            try:
                return await asyncio.shield(creation)
            except asyncio.CancelledError:
                # Let this one request settle before DaytonaJob's finally block
                # looks up the sandbox by its existing name and deletes it.
                try:
                    await creation
                except Exception:
                    logger.exception("Sandbox creation failed during cancellation")
                raise

        return await self._request("sandbox.create", create_once)

    async def get_sandbox(self, name: str) -> AsyncSandbox:
        return await self._request("sandbox.get", self._client.v0.get, name)

    async def refresh_sandbox(self, sandbox: AsyncSandbox) -> None:
        await self._request("sandbox.refresh", sandbox.refresh_data)

    async def delete_sandbox(self, sandbox: AsyncSandbox) -> None:
        await self._request("sandbox.delete", sandbox.delete, wait=False)

    async def get_snapshot(self, name: str) -> Snapshot:
        return await self._request("snapshot.get", self._client.v0.snapshot.get, name)

    async def create_snapshot(
        self,
        params: CreateSnapshotParams,
        *,
        on_logs: Callable[[str], None] | None = None,
    ) -> Snapshot:
        return await self._request(
            "snapshot.create",
            self._client.v0.snapshot.create,
            params,
            on_logs=on_logs,
        )

    async def get_volume(self, name: str, *, create: bool = False) -> Volume:
        return await self._request("volume.get", self._client.v0.volume.get, name, create=create)

    async def list_secrets(self, *, cursor: str | None = None) -> ListSecretsResponse:
        return await self._request("secret.list", self._client.v0.secret.list, cursor=cursor)

    async def create_secret(self, params: CreateSecretParams) -> Secret:
        return await self._request("secret.create", self._client.v0.secret.create, params)

    async def update_secret(self, secret_id: str, params: UpdateSecretParams) -> Secret:
        return await self._request(
            "secret.update",
            self._client.v0.secret.update,
            secret_id,
            params,
        )

    async def upload_file(self, sandbox: AsyncSandbox, data: bytes, path: str) -> None:
        await self._request("fs.upload_file", sandbox.fs.upload_file, data, path)

    async def exec(
        self,
        sandbox: AsyncSandbox,
        command: str,
        *,
        timeout: int,
    ) -> ExecuteResponse:
        return await self._request("process.exec", sandbox.process.exec, command, timeout=timeout)

    async def create_session(self, sandbox: AsyncSandbox, session_id: str) -> None:
        await self._request("process.create_session", sandbox.process.create_session, session_id)

    async def execute_session_command(
        self,
        sandbox: AsyncSandbox,
        session_id: str,
        request: SessionExecuteRequest,
    ) -> SessionExecuteResponse:
        return await self._request(
            "process.execute_session_command",
            sandbox.process.execute_session_command,
            session_id,
            request,
        )

    async def get_session_command(
        self,
        sandbox: AsyncSandbox,
        session_id: str,
        command_id: str,
    ) -> Command:
        return await self._request(
            "process.get_session_command",
            sandbox.process.get_session_command,
            session_id,
            command_id,
        )

    async def get_session_command_logs(
        self,
        sandbox: AsyncSandbox,
        session_id: str,
        command_id: str,
    ) -> SessionCommandLogsResponse:
        return await self._request(
            "process.get_session_command_logs",
            sandbox.process.get_session_command_logs,
            session_id,
            command_id,
        )


type JobCallback = Callable[[DaytonaJob, AsyncSandbox], Awaitable[None]]


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
        api: DaytonaAPI,
        snapshot: Snapshot,
        volume: Volume,
        source_package: bytes,
        entrypoint: str,
        config: dict,
        secrets: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
        on_poll: JobCallback | None = None,
        on_attempt_end: JobCallback | None = None,
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

        self._api = api
        self._snapshot = snapshot
        self._source_package = source_package
        self._volume = volume
        self._secrets = secrets or {}
        self._env_vars = env_vars or {}
        self._on_poll = on_poll
        self._on_attempt_end = on_attempt_end

    async def run_attempt(self) -> DaytonaJobResult:
        """Runs the job in a new sandbox, which is deleted before this returns or raises."""
        self.attempt_id = uuid.uuid4().hex
        self.status = DaytonaJobStatus.PREPARING
        sandbox_name = f"job-{self.id}-{self.attempt_id[:8]}"
        sandbox = None
        logger.info("Job %s | preparing attempt %s", self.id[:8], self.attempt_id[:8])
        try:
            sandbox = await self._create_sandbox(sandbox_name)
            logger.info("Job %s | sandbox allocated; staging source", self.id[:8])
            await self._upload_source_package(sandbox)
            await self._api.create_session(sandbox, self.id)
            response = await self._api.execute_session_command(
                sandbox,
                self.id,
                SessionExecuteRequest(command=self._build_command(), run_async=True),
            )
            self.status = DaytonaJobStatus.RUNNING
            logger.info("Job %s | worker started", self.id[:8])
            result = await self._wait_for_command(sandbox, response.cmd_id)
            self.status = result.status
            logger.log(
                logging.INFO if result.exit_code == 0 else logging.ERROR,
                "Job %s | worker exited: %s (code %s)",
                self.id[:8],
                str(self.status).lower(),
                result.exit_code,
            )
            return result
        except DaytonaSpotEvictedError:
            self.status = DaytonaJobStatus.PREEMPTED
            raise
        except asyncio.CancelledError:
            self.status = DaytonaJobStatus.CANCELED
            logger.info("Job %s | canceled; cleaning up", self.id[:8])
            raise
        except Exception:
            self.status = DaytonaJobStatus.FAILED
            logger.exception("Job %s | attempt failed", self.id[:8])
            raise
        finally:
            try:
                if sandbox is not None:
                    logger.info(
                        "Job %s | collecting attempt results (%s)",
                        self.id[:8],
                        str(self.status).lower(),
                    )
                    await self._notify(self._on_attempt_end, sandbox)
            finally:
                await self._delete_sandbox(sandbox, sandbox_name)

    async def _notify(self, callback: JobCallback | None, sandbox: AsyncSandbox) -> None:
        if callback is not None:
            try:
                await callback(self, sandbox)
            except Exception:
                logger.exception(
                    "Reporting failed for job %s, attempt %s", self.id, self.attempt_id
                )

    async def _wait_for_command(self, sandbox: AsyncSandbox, command_id: str) -> DaytonaJobResult:
        while True:
            await self._raise_if_preempted(sandbox)

            try:
                command = await self._api.get_session_command(sandbox, self.id, command_id)
                if command.exit_code is not None:
                    logs = await self._api.get_session_command_logs(sandbox, self.id, command_id)
            except DaytonaError:
                await self._raise_if_preempted(sandbox)
                raise

            await self._notify(self._on_poll, sandbox)
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
            await self._api.refresh_sandbox(sandbox)
        except DaytonaNotFoundError:
            reason = "sandbox no longer exists"
        else:
            if sandbox.spot_evicted_at:
                reason = "spot instance eviction"
            # TODO Maybe remove this is preemption is common enough
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
                sandbox = await self._api.get_sandbox(name)
            await self._api.delete_sandbox(sandbox)
            logger.info("Job %s | sandbox deletion requested", self.id[:8])
        except DaytonaNotFoundError:
            pass
        except Exception:
            logger.exception(
                "Failed to delete sandbox %s for job %s; its TTL will remove it", name, self.id
            )

    async def _create_sandbox(self, name: str) -> AsyncSandbox:
        try:
            sandbox = await self._api.create_sandbox(
                CreateSandboxFromSnapshotParams(
                    name=name,
                    snapshot=self._snapshot.name,
                    volumes=[VolumeMount(volume_id=self._volume.id, mount_path="/data")],
                    secrets=self._secrets,
                    env_vars=self._env_vars,
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
        await self._api.upload_file(sandbox, self._source_package, archive_path)
        result = await self._api.exec(
            sandbox,
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
        env_vars: dict[str, str] | None = None,
        max_retries: int = 5,
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
        self._api = DaytonaAPI(max_retries)
        self._secrets = secrets or {}
        self._secret_names = None
        self._env_vars = env_vars or {}

    async def close(self) -> None:
        """Close the SDK client after all job tasks have finished cleaning up."""
        await self._api.close()

    async def create_job(
        self,
        config: dict,
        *,
        on_poll: JobCallback | None = None,
        on_attempt_end: JobCallback | None = None,
    ) -> DaytonaJob:
        snapshot = await self._get_or_create_snapshot()
        volume = await self._get_or_create_volume()
        if self._secret_names is None:
            self._secret_names = await self._create_or_update_secrets(self._secrets)

        job = DaytonaJob(
            api=self._api,
            snapshot=snapshot,
            volume=volume,
            source_package=self._source_package,
            entrypoint=self._entrypoint,
            config=config,
            secrets=self._secret_names,
            env_vars=self._env_vars,
            on_poll=on_poll,
            on_attempt_end=on_attempt_end,
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
        async with slots:
            while True:
                try:
                    return await job.run_attempt()
                except DaytonaSpotEvictedError:
                    logger.warning(
                        "Job %s | preempted; retrying with its existing slot", job.id[:8]
                    )

    async def _get_or_create_snapshot(self) -> Snapshot:
        uv_lock_path = self._workdir / "uv.lock"
        pyproject_toml_path = self._workdir / "pyproject.toml"

        # TODO libgl1 is hardcoded for now -- allow extra libs to be specified in config
        image = (
            Image.base("python:3.14-slim-bookworm")
            .run_commands("apt-get update && apt-get install -y --no-install-recommends libgl1")
            .pip_install_from_requirements(str(self._workdir / "requirements.txt"))
            .run_commands("apt-get update && apt-get install -y --no-install-recommends ffmpeg")
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
            snapshot = await self._api.get_snapshot(key)
        except DaytonaNotFoundError:
            logger.info("Building snapshot %s", key[:12])
            async with asyncio.timeout(1200):
                snapshot = await self._api.create_snapshot(
                    CreateSnapshotParams(
                        name=key,
                        image=image,
                        resources=Resources(**self._resources),
                    ),
                    on_logs=(
                        (lambda chunk: print(chunk, end="", flush=True)) if self._verbose else None
                    ),
                )

        if not self._jobs:
            logger.info("Using snapshot %s", key[:12])
        # TODO Check snapshot state
        return snapshot

    async def _get_or_create_volume(self) -> Volume:
        volume = await self._api.get_volume(self._volume_name, create=True)

        async with asyncio.timeout(60):
            while volume.state != "ready":
                if volume.state == "error":
                    raise RuntimeError(f"Volume {self._volume_name} is in error state.")

                await asyncio.sleep(1)
                volume = await self._api.get_volume(self._volume_name)

        return volume

    async def _create_or_update_secrets(self, secrets: dict) -> dict:
        existing_ids = {}
        secret_names = {}
        if not secrets:
            return secret_names

        cursor = None
        while True:
            page = await self._api.list_secrets(cursor=cursor)
            existing_ids |= {s.name: s.id for s in page.items}
            cursor = page.next_cursor
            if cursor is None:
                break

        for env_name, value in secrets.items():
            name = f"daytona-runner-{env_name}"
            if name in existing_ids:
                # Stored values cannot be read back to compare, so always overwrite.
                await self._api.update_secret(existing_ids[name], UpdateSecretParams(value=value))
            else:
                await self._api.create_secret(CreateSecretParams(name=name, value=value))

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
