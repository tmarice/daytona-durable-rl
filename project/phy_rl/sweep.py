"""Workload adapter for training and checking durable experiment sessions."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__:
    from .checkpointing import checkpoint_from_data
    from .daytona_experiment_session import ExperimentSession
    from .train import run as train
else:
    from checkpointing import checkpoint_from_data
    from daytona_experiment_session import ExperimentSession
    from train import run as train

REPORT_PREFIX = "DAYTONA_REPORT "


def inspect_checkpoint(session: ExperimentSession) -> dict:
    data = session.restore_checkpoint()
    if data is None:
        raise RuntimeError(f"No committed checkpoint for {session.experiment_id}")
    assert session.latest_checkpoint is not None
    checkpoint = checkpoint_from_data(session.latest_checkpoint, data)
    return {
        "experiment_id": session.experiment_id,
        "reader_attempt_id": session.attempt_id,
        "writer_attempt_id": checkpoint.manifest["metadata"]["attempt_id"],
        "env_steps": checkpoint.manifest["env_steps"],
        "completed_updates": checkpoint.manifest["completed_updates"],
        "path": str(checkpoint.path),
    }


def run(config: dict, *, experiment_id: str, attempt_id: str, checkpoint_dir: str) -> None:
    if "verify_experiments" in config:
        # A new CPU sandbox mounts the same volume after training sandboxes were
        # deleted. Use the original experiment identities and a new reader attempt.
        jobs_directory = Path(checkpoint_dir).parent.parent
        report = {
            "checkpoints": [
                inspect_checkpoint(
                    ExperimentSession(
                        job_id,
                        attempt_id,
                        jobs_directory / job_id / "checkpoints",
                    )
                )
                for job_id in config["verify_experiments"]
            ]
        }
    else:
        # Metrics and plotting need a regular filesystem (append/rename). Only
        # immutable checkpoints are written directly to Daytona's FUSE volume.
        with TemporaryDirectory(prefix="phy-rl-attempt-") as temporary:
            local = Path(temporary) / attempt_id
            destination = Path(checkpoint_dir).parent / "attempts" / attempt_id
            try:
                directory = train(
                    {**config, "output": temporary},
                    experiment_id=experiment_id,
                    attempt_id=attempt_id,
                    checkpoint_dir=checkpoint_dir,
                )
                report = {
                    "summary": json.loads((directory / "summary.json").read_text()),
                    "resolved": json.loads((directory / "resolved_config.json").read_text()),
                    "checkpoint": inspect_checkpoint(
                        ExperimentSession(
                            experiment_id,
                            attempt_id,
                            checkpoint_dir,
                        )
                    ),
                    "artifacts": str(destination),
                }
            finally:
                # Copy bytes only: object-backed mounts need not support chmod,
                # timestamp changes, or any of copytree's other metadata writes.
                for source in local.rglob("*"):
                    if source.is_file():
                        target = destination / source.relative_to(local)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
    print(REPORT_PREFIX + json.dumps(report, allow_nan=False), flush=True)
