"""Publication and CLI contracts, without importing training dependencies."""

import contextlib
import dataclasses
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from daytona_runner.checkpointing import commit_checkpoint, read_checkpoint
from daytona_runner.train import Config, config_values, parse_args


class CheckpointTests(unittest.TestCase):
    def commit(self, root, payload=b"complete state", update=1):
        return commit_checkpoint(
            root,
            payload,
            env_steps=update * 64,
            completed_updates=update,
            metadata={"requested": config_values(dataclasses.replace(Config(), seed=22))},
        )

    def test_manifest_and_latest_validate_payload_and_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.commit(root)
            second = self.commit(root, b"next state", update=2)
            self.assertEqual(read_checkpoint(first).payload, b"complete state")
            for path in (root, root / "latest.json", second.parent, second):
                with self.subTest(path=path):
                    checkpoint = read_checkpoint(path)
                    self.assertEqual(checkpoint.payload, b"next state")
                    self.assertEqual(checkpoint.manifest["env_steps"], 128)

    def test_interruption_at_each_publish_stage_preserves_previous_checkpoint(self):
        for fail_on in ("payload", "manifest", "directory", "pointer_write", "pointer_publish"):
            with self.subTest(stage=fail_on), tempfile.TemporaryDirectory() as temporary:
                from daytona_runner import checkpointing

                root = Path(temporary)
                previous = self.commit(root)
                original_write = checkpointing._write_synced
                original_replace = os.replace

                def write(path, data, fail_on=fail_on, original_write=original_write):
                    if (
                        fail_on == "payload"
                        and path.name == "state.msgpack"
                        or fail_on == "manifest"
                        and path.name == "manifest.json"
                        or fail_on == "pointer_write"
                        and path.name.endswith(".tmp")
                    ):
                        path.write_bytes(data[:3])
                        raise OSError("interrupted write")
                    original_write(path, data)

                def replace(
                    source, destination, fail_on=fail_on, original_replace=original_replace
                ):
                    if (
                        fail_on == "directory"
                        and source.is_dir()
                        or fail_on == "pointer_publish"
                        and destination.name == "latest.json"
                    ):
                        raise OSError("interrupted publication")
                    original_replace(source, destination)

                with (
                    mock.patch.object(checkpointing, "_write_synced", side_effect=write),
                    mock.patch.object(checkpointing.os, "replace", side_effect=replace),
                    self.assertRaises(OSError),
                ):
                    self.commit(root, b"new state", update=2)
                self.assertEqual(read_checkpoint(root).path, previous.resolve())
                self.assertEqual(read_checkpoint(root).payload, b"complete state")

    def test_corruption_and_uncommitted_directories_are_rejected(self):
        for mode in ("truncated", "checksum", "manifest", "missing", "version"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self.commit(root)
                payload = manifest.parent / "state.msgpack"
                if mode == "truncated":
                    payload.write_bytes(b"short")
                elif mode == "checksum":
                    payload.write_bytes(b"x" * payload.stat().st_size)
                elif mode == "manifest":
                    manifest.write_text("{}")
                elif mode == "version":
                    document = json.loads(manifest.read_text())
                    document["format_version"] = 999
                    manifest.write_text(json.dumps(document))
                else:
                    payload.unlink()
                with self.assertRaises(ValueError):
                    read_checkpoint(root)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            read_checkpoint(Path(temporary))

    def test_resume_inherits_configuration_and_allows_only_runtime_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.commit(root)
            config = parse_args(["--resume", str(root), "--checkpoint-interval-steps", "7"])
            self.assertEqual(config.seed, 22)
            self.assertEqual(config.total_steps, Config().total_steps)
            self.assertEqual(config.checkpoint_interval_steps, 7)
            self.assertEqual(parse_args(["--resume", str(root)]).backend, "cpu")
            self.assertEqual(parse_args(["--resume", str(root), "--gpu"]).backend, "gpu")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--resume", str(root), "--seed", "11"])


if __name__ == "__main__":
    unittest.main()
