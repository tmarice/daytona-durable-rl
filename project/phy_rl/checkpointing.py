"""Encode and validate full RL state; ExperimentSession handles persistence."""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .daytona_experiment_session import ExperimentSession, read_checkpoint_document
else:
    from daytona_experiment_session import ExperimentSession, read_checkpoint_document

FORMAT_VERSION = 2


def commit_checkpoint(
    session: ExperimentSession,
    payload: bytes,
    *,
    env_steps: int,
    completed_updates: int,
    metadata: dict,
) -> Path:
    """Encode binary learner state inside a JSON-compatible checkpoint dictionary."""
    return session.save_checkpoint(
        {
            "format_version": FORMAT_VERSION,
            "env_steps": env_steps,
            "completed_updates": completed_updates,
            "payload_size": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_base64": base64.b64encode(payload).decode("ascii"),
            "metadata": {
                **metadata,
                "experiment_id": session.experiment_id,
                "attempt_id": session.attempt_id,
            },
        }
    )


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    manifest: dict
    payload: bytes

    def restore(self, template: Any) -> Any:
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


def checkpoint_from_data(path: Path, document: dict) -> Checkpoint:
    """Validate the workload-specific contents returned by a session."""
    try:
        if document["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format version")
        for field in ("env_steps", "completed_updates", "payload_size"):
            if type(document[field]) is not int or document[field] <= 0:
                raise ValueError(f"Invalid checkpoint {field}")
        payload = base64.b64decode(document["payload_base64"], validate=True)
        if len(payload) != document["payload_size"]:
            raise ValueError("Checkpoint payload size mismatch")
        if hashlib.sha256(payload).hexdigest() != document["payload_sha256"]:
            raise ValueError("Checkpoint payload checksum mismatch")
        if not isinstance(document["metadata"], dict):
            raise TypeError("Invalid checkpoint metadata")
    except (KeyError, TypeError, ValueError, binascii.Error) as error:
        raise ValueError(f"Invalid checkpoint at {path}: {error}") from error
    return Checkpoint(path, document, payload)


def read_checkpoint(path: Path) -> Checkpoint:
    """Inspect a new-format checkpoint file or directory, including CLI configuration."""
    path, document = read_checkpoint_document(path)
    return checkpoint_from_data(path, document["data"])


def save_state(
    session: ExperimentSession, state: dict, *, completed_updates: int, metadata: dict
) -> Path:
    import jax
    from flax import serialization

    # Finish device work and serialize before the next update donates its buffers.
    state = jax.device_get(state)
    return commit_checkpoint(
        session,
        serialization.to_bytes(state),
        env_steps=int(state["learner"].env_steps),
        completed_updates=completed_updates,
        metadata=metadata,
    )
