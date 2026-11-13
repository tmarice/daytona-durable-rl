"""Publication and CLI contracts, without importing training dependencies."""

import contextlib
import dataclasses
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from phy_rl.checkpointing import commit_checkpoint, load_baseline, read_checkpoint, save_baseline
from phy_rl.daytona_experiment_session import ExperimentSession
from phy_rl.train import Config, config_values, parse_args


class CheckpointTests(unittest.TestCase):
    def commit(self, root, payload=b"complete state", update=1):
        return commit_checkpoint(
            ExperimentSession("experiment", "attempt", root),
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
            for path in (root, second):
                with self.subTest(path=path):
                    checkpoint = read_checkpoint(path)
                    self.assertEqual(checkpoint.payload, b"next state")
                    self.assertEqual(checkpoint.manifest["env_steps"], 128)

    def test_corruption_and_uncommitted_directories_are_rejected(self):
        for mode in ("truncated", "checksum", "manifest", "missing", "version"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = self.commit(root)
                envelope = json.loads(path.read_text())
                document = envelope["data"]
                if mode == "truncated":
                    document["payload_base64"] = "c2hvcnQ="
                elif mode == "checksum":
                    document["payload_sha256"] = "wrong"
                elif mode == "manifest":
                    envelope["data"] = {}
                elif mode == "version":
                    document["format_version"] = 999
                else:
                    del document["payload_base64"]
                path.write_text(json.dumps(envelope))
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


@unittest.skipUnless(importlib.util.find_spec("flax"), "Requires training dependencies")
class BaselineTests(unittest.TestCase):
    def test_checkpoint_baseline_preserves_evaluation_and_trajectory_arrays(self) -> None:
        import numpy as np

        baseline = {
            "evaluation": {"training_step": 0, "episode_returns": [1.5], "mean_return": 1.5},
            "rollouts": {"qpos": np.arange(12, dtype=np.float32).reshape(1, 4, 3)},
            "policy_params": {"weights": np.array([1.0, 2.0], dtype=np.float32)},
        }
        with tempfile.TemporaryDirectory() as root:
            session = ExperimentSession("experiment", "first", root)
            reference = save_baseline(session, baseline)
            path = commit_checkpoint(
                session,
                b"state",
                env_steps=64,
                completed_updates=1,
                metadata={"baseline": reference},
            )
            restored = load_baseline(read_checkpoint(path))
            self.assertEqual(restored["evaluation"], baseline["evaluation"])
            np.testing.assert_array_equal(
                restored["rollouts"]["qpos"], baseline["rollouts"]["qpos"]
            )
            np.testing.assert_array_equal(
                restored["policy_params"]["weights"], baseline["policy_params"]["weights"]
            )
            (Path(root) / reference["file"]).write_bytes(b"interrupted or corrupt baseline")
            with self.assertRaisesRegex(ValueError, "Baseline checksum mismatch"):
                load_baseline(read_checkpoint(path))

    def test_legacy_checkpoint_has_no_original_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = commit_checkpoint(
                ExperimentSession("experiment", "first", root),
                b"state",
                env_steps=64,
                completed_updates=1,
                metadata={},
            )
            self.assertIsNone(load_baseline(read_checkpoint(path)))


if __name__ == "__main__":
    unittest.main()
