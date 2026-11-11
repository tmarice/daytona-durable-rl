"""Experiment identity and JSON checkpoint storage, without SDK dependencies."""

import json
import uuid
from pathlib import Path


def latest_checkpoint_path(directory: Path) -> Path | None:
    """Only completion markers participate in checkpoint selection."""
    try:
        marker = max(
            (path for path in directory.iterdir() if path.suffix == ".complete"),
            default=None,
        )
    except FileNotFoundError:
        return None
    return marker.with_suffix("") if marker is not None else None


def read_checkpoint_document(path: Path) -> tuple[Path, dict]:
    """Shared reader for session restore and inspecting configuration before a run."""
    path = path.resolve()
    if path.is_dir():
        selected = latest_checkpoint_path(path)
        if selected is None:
            raise ValueError(f"No committed checkpoint in {path}")
        path = selected
    if not Path(f"{path}.complete").is_file():
        raise ValueError(f"Missing completion marker for {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


class ExperimentSession:
    """Experiment identity and JSON checkpoints; one active writer per directory."""

    def __init__(self, experiment_id: str, attempt_id: str, checkpoint_dir: str | Path) -> None:
        self.experiment_id = experiment_id
        self.attempt_id = attempt_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.latest_checkpoint: Path | None = None

    def save_checkpoint(self, data: dict) -> Path:
        """Write immutable JSON, then mark it committed after the file closes."""
        encoded = json.dumps(
            {"experiment_id": self.experiment_id, "attempt_id": self.attempt_id, "data": data},
            allow_nan=False,
        )
        previous = latest_checkpoint_path(self.checkpoint_dir)
        sequence = int(previous.name.split("-")[1]) + 1 if previous is not None else 1
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint-{sequence:020d}-{uuid.uuid4().hex}.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        # No rename, symlink, or mutable latest pointer is required on the mount.
        # A process killed before this marker leaves an ignored, uncommitted file.
        with Path(f"{path}.complete").open("x"):
            pass
        self.latest_checkpoint = path.resolve()
        return self.latest_checkpoint

    def restore_checkpoint(self, path: Path | None = None) -> dict | None:
        """Load an explicit checkpoint, or the newest committed checkpoint."""
        if path is None:
            path = latest_checkpoint_path(self.checkpoint_dir)
            if path is None:
                return None
        path, document = read_checkpoint_document(path)
        if document["experiment_id"] != self.experiment_id:
            raise ValueError(f"Checkpoint at {path} belongs to a different experiment")
        self.latest_checkpoint = path
        return document["data"]
