import logging
import uuid
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from daytona import AsyncDaytona
from daytona.v0 import CreateSandboxFromImageParams, Image, Resources
from daytona.v0.common.errors import DaytonaNotFoundError

# TODO Is this good?
logger = logging.getLogger(__name__)


class DaytonaJobStatus(StrEnum):
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class DaytonaJob:
    def __init__(self, snapshot):
        self.id = uuid.uuid4().hex
        self.status = DaytonaJobStatus.PENDING

    async def start(self):
        """Starts the Daytona sandbox and runs the specified job."""

    async def wait(self):
        pass

    async def cancel(self):
        pass


class DaytonaJobResult:
    pass


class DaytonaRunner:
    # TODO Handle sigterm and sigint
    # TODO Make async so it can handle multiple jobs at once

    def __init__(self, max_concurrent_jobs: int = 1):
        self._max_concurrent_jobs = max_concurrent_jobs
        self._daytona = AsyncDaytona()
        self._jobs = {}

    def create_job(
        self,
        entrypoint: str,  # physical_ai.main:run  -- expects run(checkpoint_path: str) -> None
        config: dict,  # {"work_dir": /path/to/folder", "checkpoint_path": "/path/to/folder/checkpoints" , "resources": {"cpu": 4, "memory": 8, "disk": 10}, "pyproject_toml_path": "/path/to/folder/pyproject.toml"}
    ) -> DaytonaJob:
        """Prepares a new job for execution in a Daytona sandbox.

        1. get or create snapshot
        2. create job with specified snapshot
        """

    def _get_or_create_snapshot(self, resources: dict, pyproject_toml_path: Path):
        daytona = Daytona()

        key = sha256(pyproject_toml_path.read_bytes()).hexdigest()
        try:
            snapshot = daytona.v0.snapshot.get(key)
        except DaytonaNotFoundError:
            image = Image.base("python:3.14-slim-bookworm").pip_install_from_requirements(
                pyproject_toml_path  # TODO check if Path is OK instead of str
            )
            snapshot = Daytona().v0.create_snapshot(
                CreateSandboxFromImageParams(
                    image=image,
                    resources=Resources(**resources),
                    auto_stop_interval=0,
                    ttl_minutes=65,
                    ephemeral=True,
                ),
                timeout=1200,
                on_snapshot_create_logs=lambda chunk: print(chunk, end="", flush=True),
            )

        return snapshot

    async def run(self, jobs):
        """Monitors the execution of multiple jobs and re-run them if we detect spot instance preemption."""


class ExperimentSession:
    def __init__(
        self,
        experiment_id: str,
        attempt_id: str,
        checkpoint_path: str,
    ):
        pass

    def save_checkpoint(self, checkpoint_data: dict):
        pass

    def restore_checkpoint(self) -> dict:
        pass
