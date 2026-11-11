"""Synchronous, committed local checkpoints with validated full-state restore."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

FORMAT_VERSION = 1


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_synced(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit_checkpoint(
    root: Path, payload: bytes, *, env_steps: int, completed_updates: int, metadata: dict
) -> Path:
    """Publish immutable payload/manifest, then atomically replace the latest pointer."""
    root.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    name = f"update-{completed_updates:08d}-{token}"
    temporary = root / f".pending-{token}"
    destination = root / name
    pointer = root / f".latest-{token}.tmp"
    temporary.mkdir()
    try:
        manifest = _json_bytes(
            {
                "format_version": FORMAT_VERSION,
                "env_steps": env_steps,
                "completed_updates": completed_updates,
                "payload_size": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "metadata": metadata,
            }
        )
        _write_synced(temporary / "state.msgpack", payload)
        _write_synced(temporary / "manifest.json", manifest)
        _sync_directory(temporary)
        os.replace(temporary, destination)
        _sync_directory(root)
        _write_synced(
            pointer,
            _json_bytes(
                {"checkpoint": name, "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
            ),
        )
        os.replace(pointer, root / "latest.json")
        _sync_directory(root)
    finally:
        # SIGKILL can leave these behind; readers never consider temporary files.
        if temporary.exists():
            shutil.rmtree(temporary)
        pointer.unlink(missing_ok=True)
    return destination / "manifest.json"


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    manifest: dict
    payload: bytes

    def restore(self, template):
        """Reconstruct typed Flax/Optax state and reject incompatible array layouts."""
        import jax
        import numpy as np
        from flax import serialization

        restored = serialization.from_bytes(template, self.payload)
        expected, expected_tree = jax.tree_util.tree_flatten_with_path(template)
        actual, actual_tree = jax.tree_util.tree_flatten(restored)
        if actual_tree != expected_tree:
            raise ValueError("Checkpoint state structure is incompatible")
        for (key, before), after in zip(expected, actual, strict=True):
            before, after = np.asarray(before), np.asarray(after)
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError(
                    f"Checkpoint array {jax.tree_util.keystr(key)} is incompatible: "
                    f"expected {before.shape}/{before.dtype}, got {after.shape}/{after.dtype}"
                )
            if not np.isfinite(after).all():
                raise ValueError("Checkpoint contains non-finite learner state")
        if int(restored["learner"].env_steps) != self.manifest["env_steps"]:
            raise ValueError("Checkpoint payload and manifest progress disagree")
        return restored


def read_checkpoint(path: Path) -> Checkpoint:
    """Read an explicit manifest, a checkpoint directory, or a latest pointer."""
    path = path.resolve()
    if path.is_dir():
        path /= "latest.json" if (path / "latest.json").exists() else "manifest.json"
    try:
        data = path.read_bytes()
        document = json.loads(data)
        if "checkpoint" in document:
            name = document["checkpoint"]
            if not isinstance(name, str) or Path(name).name != name or name.startswith("."):
                raise ValueError("Invalid checkpoint pointer")
            path = path.parent / name / "manifest.json"
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != document["manifest_sha256"]:
                raise ValueError("Checkpoint manifest checksum mismatch")
            document = json.loads(data)
        if document["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format version")
        for field in ("env_steps", "completed_updates", "payload_size"):
            if type(document[field]) is not int or document[field] <= 0:
                raise ValueError(f"Invalid checkpoint {field}")
        payload = (path.parent / "state.msgpack").read_bytes()
        if len(payload) != document["payload_size"]:
            raise ValueError("Checkpoint payload size mismatch")
        if hashlib.sha256(payload).hexdigest() != document["payload_sha256"]:
            raise ValueError("Checkpoint payload checksum mismatch")
        if not isinstance(document["metadata"], dict):
            raise TypeError("Invalid checkpoint metadata")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid or incomplete checkpoint at {path}: {error}") from error
    return Checkpoint(path, document, payload)


def save_state(root: Path, state, *, completed_updates: int, metadata: dict) -> Path:
    import jax
    from flax import serialization

    # Finish device work and serialize before the next update donates its buffers.
    state = jax.device_get(state)
    return commit_checkpoint(
        root,
        serialization.to_bytes(state),
        env_steps=int(state["learner"].env_steps),
        completed_updates=completed_updates,
        metadata=metadata,
    )
