import asyncio
import logging
import shlex
import tarfile
import uuid
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

# TODO Is this good?
logger = logging.getLogger(__name__)


# TODO State machine
class DaytonaJobStatus(StrEnum):
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class DaytonaJob:
    def __init__(
        self,
        daytona: AsyncDaytona,
        snapshot: Snapshot,
        volume: Volume,
        source_package: bytes,
        entrypoint: str,
        config: dict,
    ):
        self.id = uuid.uuid4().hex
        self.status = DaytonaJobStatus.PENDING
        self.sandbox = None
        self._code_dir = f"/run/{self.id}/code"
        self._data_dir = f"/data/jobs/{self.id}"
        self._command_id = None
        config = dict(config)
        config.setdefault("checkpoint_path", f"{self._data_dir}/checkpoints")
        self._checkpoint_dir = config["checkpoint_path"]
        self._command = self._build_command(entrypoint, config)

        self._daytona = daytona
        self._snapshot = snapshot
        self._source_package = source_package
        self._volume = volume

    async def start(self):
        """Starts the Daytona sandbox and runs the specified job."""
        if self.status == DaytonaJobStatus.PREEMPTED:
            logger.warning("Job %s was preempted, restarting...", self.id)
            self.status = DaytonaJobStatus.PREPARING
            if self.sandbox is not None and self.sandbox.recoverable:
                await self.sandbox.recover()
            else:
                self.sandbox = await self._create_sandbox()

        elif self.status == DaytonaJobStatus.PENDING:
            self.status = DaytonaJobStatus.PREPARING
            self.sandbox = await self._create_sandbox()

        await self.sandbox.process.create_session(self.id)
        response = await self.sandbox.process.execute_session_command(
            self.id, SessionExecuteRequest(command=self._command, run_async=True)
        )

        self._command_id = response.cmd_id
        self.status = DaytonaJobStatus.RUNNING

    async def wait(self):
        while True:
            await self.sandbox.refresh_data()

            # TODO Handle manual deletion as preemption to be able to simulate failure
            if self.sandbox.spot_evicted_at:
                raise DaytonaSpotEvictedError(
                    f"Job {self.id} was preempted by spot instance eviction."
                )

            command = await self.sandbox.process.get_session_command(self.id, self._command_id)
            if command.exit_code is not None:
                if command.exit_code == 0:
                    self.status = DaytonaJobStatus.COMPLETED
                else:
                    self.status = DaytonaJobStatus.FAILED

                return

            await asyncio.sleep(5)  # TODO Make this configurable

    async def cancel(self, wait: bool = True):
        await self.sandbox.delete(wait=wait)
        self.status = DaytonaJobStatus.CANCELED

    async def _create_sandbox(self) -> AsyncSandbox:
        try:
            # TODO Handle not enough capacity failures -- retry after x seconds
            sandbox = await self._daytona.v0.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=self._snapshot.name,
                    volumes=[VolumeMount(volume_id=self._volume.id, mount_path="/data")],
                    auto_stop_interval=0,
                    ttl_minutes=65,
                    ephemeral=True,
                )
            )
        except DaytonaError:
            logger.exception("Failed to create sandbox for job %s", self.id)
            raise
        else:
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

        return sandbox

    def _build_command(self, entrypoint: str, config: dict) -> str:
        module, function = entrypoint.split(":", 1)
        code = (
            "import importlib; "
            f"getattr(importlib.import_module({module!r}), {function!r})({config!r})"
        )
        command = f"cd {shlex.quote(self._code_dir)} && " + shlex.join(["python", "-uc", code])

        return command


class DaytonaRunner:
    # TODO Handle sigterm and sigint

    def __init__(
        self,
        work_dir: str,
        entrypoint: str,
        resources: dict,
        volume_name: str,
        secrets: list[str],
        max_concurrent_jobs: int = 1,
        verbose: bool = False,
    ):
        self._workdir = Path(work_dir)
        self._entrypoint = entrypoint
        self._resources = resources
        self._volume_name = volume_name
        self._secrets = secrets
        self._max_concurrent_jobs = max_concurrent_jobs
        self._verbose = verbose

        self._daytona = AsyncDaytona()
        self._jobs = {}
        self._source_package = self._package_source()

    async def create_job(
        self,
        config: dict,  # {"checkpoint_path": "/path/to/folder/checkpoints"}
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
        )
        self._jobs[job.id] = job

        return job

    async def run(self, jobs):
        slots = asyncio.Semaphore(self._max_concurrent_jobs)
        tasks = [asyncio.create_task(self._run_job(job, slots)) for job in jobs]

        try:
            for task in asyncio.as_completed(tasks):
                yield await task
        finally:
            # TODO Do I want to cancel all tasks if one fails?
            for task in tasks:
                task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_job(self, job: DaytonaJob, slots: asyncio.Semaphore):
        async with slots:
            try:
                await job.start()
                await job.wait()
            except DaytonaSpotEvictedError:
                # TODO Unify logging - do not scatter across all classes
                logger.warning("Job %s was preempted, re-running...", job.id)
                await job.start()
                return await self._run_job(job, slots)
            except Exception:
                logger.exception("Job %s failed", job.id)

    async def _get_or_create_snapshot(self) -> Snapshot:
        uv_lock_path = self._workdir / "uv.lock"
        pyproject_toml_path = self._workdir / "pyproject.toml"

        # TODO Is uv_lock enough for cache key?
        key = sha256(uv_lock_path.read_bytes()).hexdigest()
        try:
            snapshot = await self._daytona.v0.snapshot.get(key)
        except DaytonaNotFoundError:
            # TODO Determine python version dynamically from pyproject.toml
            # TODO Install from uv.lock and include required system packages.
            image = Image.base("python:3.14-slim-bookworm").pip_install_from_pyproject(
                str(pyproject_toml_path), optional_dependencies=[]
            )
            snapshot = await self._daytona.v0.snapshot.create(
                CreateSnapshotParams(
                    name=key,
                    image=image,
                    resources=Resources(**self._resources),
                ),
                timeout=1200,
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
        checkpoint_path: str,
    ):
        self._checkpoint_path = Path(checkpoint_path)

    def save_checkpoint(self, checkpoint_data: dict):
        # TODO Save checkpoint data to new file in checkpoint_path with unique name, and replace latest_checkpoint with symlink to it
        pass

    def restore_checkpoint(self) -> dict:
        # TODO Read latest checkpoint file from checkpoint_path and return its contents as dict, or None if no checkpoint exists
        pass
