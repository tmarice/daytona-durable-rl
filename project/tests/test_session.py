"""Generic session persistence, without any training or Daytona dependencies."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phy_rl.daytona_experiment_session import ExperimentSession


class SessionTests(unittest.TestCase):
    def test_session_import_does_not_require_sdk_or_training_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "from phy_rl.daytona_experiment_session import ExperimentSession",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_round_trip_across_attempts_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as root:
            session = ExperimentSession("experiment", "first", root)
            self.assertIsNone(session.restore_checkpoint())
            first = session.save_checkpoint({"step": 1, "nested": {"values": [1, None, True]}})
            replacement = ExperimentSession("experiment", "second", root)
            self.assertEqual(replacement.restore_checkpoint()["step"], 1)
            second = replacement.save_checkpoint({"step": 2})
            self.assertEqual(replacement.latest_checkpoint, second)
            self.assertEqual(session.restore_checkpoint(), {"step": 2})
            self.assertEqual(session.restore_checkpoint(first)["step"], 1)
            self.assertEqual(session.latest_checkpoint, first)
            self.assertEqual(session.restore_checkpoint(Path(root)), {"step": 2})
            self.assertEqual(json.loads(second.read_text())["attempt_id"], "second")
            with self.assertRaisesRegex(ValueError, "different experiment"):
                ExperimentSession("other", "third", root).restore_checkpoint()

    def test_interrupted_payload_and_marker_writes_preserve_previous_checkpoint(self):
        for stage in ("payload", "marker"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                session = ExperimentSession("experiment", "attempt", root)
                previous = session.save_checkpoint({"step": 1})
                original_open = Path.open

                def interrupted_open(
                    path, mode="r", *args, stage=stage, original_open=original_open, **kwargs
                ):
                    if mode == "x" and path.suffix == ".complete" and stage == "marker":
                        raise OSError("marker write failed")
                    stream = original_open(path, mode, *args, **kwargs)
                    if mode == "x" and path.suffix == ".json" and stage == "payload":

                        def partial_write(data):
                            stream.write(data[:5])
                            raise OSError("payload write failed")

                        wrapper = mock.MagicMock(wraps=stream)
                        wrapper.__enter__.return_value = wrapper
                        wrapper.__exit__.side_effect = lambda *args: stream.close()
                        wrapper.write.side_effect = partial_write
                        return wrapper
                    return stream

                with mock.patch.object(Path, "open", interrupted_open), self.assertRaises(OSError):
                    session.save_checkpoint({"step": 2})
                self.assertEqual(session.latest_checkpoint, previous)
                self.assertEqual(session.restore_checkpoint(), {"step": 1})
                next_path = session.save_checkpoint({"step": 3})
                self.assertEqual(session.restore_checkpoint(), {"step": 3})
                self.assertEqual(int(next_path.name.split("-")[1]), 2)

    def test_corrupt_selected_checkpoint_fails_without_falling_back(self):
        for corruption in ("invalid_json", "missing_file"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as root:
                session = ExperimentSession("experiment", "attempt", root)
                previous = session.save_checkpoint({"step": 1})
                latest = session.save_checkpoint({"step": 2})
                if corruption == "missing_file":
                    latest.unlink()
                else:
                    latest.write_text("{")
                with self.assertRaises((json.JSONDecodeError, FileNotFoundError)):
                    session.restore_checkpoint()
                self.assertEqual(session.restore_checkpoint(previous), {"step": 1})

    def test_unserializable_data_does_not_create_files_and_explicit_missing_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "checkpoints"
            session = ExperimentSession("experiment", "attempt", directory)
            for data in ({"nan": float("nan")}, {"bytes": b"binary"}):
                with self.assertRaises((TypeError, ValueError)):
                    session.save_checkpoint(data)
            self.assertFalse(directory.exists())
            with self.assertRaises(ValueError):
                session.restore_checkpoint(directory / "missing.json")
            pending = Path(root) / "checkpoint.json"
            pending.write_text('{"data": {}}')
            with self.assertRaisesRegex(ValueError, "completion marker"):
                session.restore_checkpoint(pending)

    def test_storage_errors_are_not_treated_as_an_empty_checkpoint_directory(self):
        session = ExperimentSession("experiment", "attempt", "/volume/checkpoints")
        with mock.patch.object(Path, "iterdir", side_effect=PermissionError("unreadable")):
            with self.assertRaises(PermissionError):
                session.restore_checkpoint()
            with self.assertRaises(PermissionError):
                session.save_checkpoint({"step": 1})


if __name__ == "__main__":
    unittest.main()
