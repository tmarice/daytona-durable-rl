"""Opt-in real MuJoCo/MJX/PPO execution; requires installed dependencies."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from phy_rl.checkpointing import read_checkpoint
from phy_rl.evaluation import evaluate_policy
from phy_rl.train import Config, parse_args, run


@unittest.skipUnless(
    os.environ.get("RUN_MUJOCO_SMOKE") == "1", "Set RUN_MUJOCO_SMOKE=1 to execute real training"
)
class RealTrainingTests(unittest.TestCase):
    def test_training_without_checkpoints_never_serializes_state(self):
        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch("phy_rl.train.save_state", side_effect=AssertionError("Saved state")),
        ):
            directory = run(
                Config(
                    total_steps=128,
                    num_envs=4,
                    episode_length=32,
                    unroll_length=8,
                    batch_size=4,
                    num_minibatches=2,
                    updates_per_batch=2,
                    num_evals=2,
                    eval_episodes=2,
                    output=Path(root),
                    checkpoints=False,
                )
            )
            summary = json.loads((directory / "summary.json").read_text())
            resolved = json.loads((directory / "resolved_config.json").read_text())
            self.assertEqual(summary["actual_steps"], 128)
            self.assertGreater(summary["policy_parameter_change_l2"], 0)
            self.assertFalse(summary["checkpoints_enabled"])
            self.assertFalse(resolved["checkpoints_enabled"])
            self.assertIsNone(summary["latest_checkpoint"])
            self.assertFalse((directory / "checkpoints").exists())
            self.assertFalse((directory / "checkpoint_status.json").exists())

    def test_wandb_receives_real_training_video_and_resumed_attempt(self):
        import wandb

        calls = []

        def capture_run(**kwargs):
            remote = SimpleNamespace(
                id=kwargs["id"],
                project=kwargs["project"],
                entity=kwargs["entity"],
                url="https://example.com/test-run",
                summary={},
                define_metric=mock.Mock(),
                log=mock.Mock(),
                save=mock.Mock(),
                finish=mock.Mock(),
            )
            calls.append((kwargs, remote))
            return remote

        # The simulator, trainer, checkpoints, SDK media objects, and video encoder
        # are real. Only the remote run transport is replaced: this test never uploads.
        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch.object(wandb, "init", side_effect=capture_run),
            mock.patch("dotenv.load_dotenv"),
        ):
            first = run(
                Config(
                    total_steps=128,
                    num_envs=4,
                    episode_length=32,
                    unroll_length=8,
                    batch_size=4,
                    num_minibatches=2,
                    updates_per_batch=2,
                    num_evals=2,
                    eval_episodes=2,
                    output=Path(root),
                    video=True,
                    wandb=True,
                    wandb_mode="online",
                    wandb_entity="test-team",
                    wandb_project="test-project",
                )
            )
            checkpoint = next((first / "checkpoints").glob("update-00000001-*/manifest.json"))
            second = run(parse_args(["--resume", str(checkpoint), "--output", root]))
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0]["id"], calls[1][0]["id"])
            self.assertEqual(calls[1][0]["resume"], "allow")
            for directory, (_, remote) in zip((first, second), calls, strict=True):
                records = [call.args[0] for call in remote.log.call_args_list]
                self.assertTrue(any("eval/episode_reward" in record for record in records))
                self.assertTrue(
                    any("heldout/after/balance_success_rate" in record for record in records)
                )
                self.assertTrue(
                    any(
                        isinstance(record.get("policy/comparison"), wandb.Video)
                        for record in records
                    )
                )
                self.assertTrue(all(record["attempt_id"] == directory.name for record in records))
                self.assertTrue(all(not call.kwargs for call in remote.log.call_args_list))
                self.assertEqual(remote.summary["attempt_status"], "completed")
                remote.finish.assert_called_once_with(exit_code=0)
                identity = read_checkpoint(directory / "checkpoints").manifest["metadata"]["wandb"]
                self.assertEqual(identity["id"], remote.id)
            self.assertEqual(calls[1][1].log.call_args_list[0].args[0]["env_steps"], 64)
            self.assertEqual(calls[1][1].summary["steps_this_attempt"], 64)

    def test_balance_requires_every_step_in_bounds_and_the_full_horizon(self):
        import numpy as np

        qpos = np.zeros((4, 3, 2))
        qpos[0, :, 1] = 2 * np.pi + 0.01  # Upright after wrapping the angle.
        qpos[1, 1, 1] = np.deg2rad(12.1)  # Fell outside the limit, then recovered.
        qpos[2, 1, 0] = 1.01  # Cart left the permitted range, then returned.
        rollouts = {
            "qpos": qpos,
            "returns": np.full((4, 3), 100.0),  # High reward alone cannot pass.
            "actions": np.zeros((4, 3, 1)),
            "length": np.array([2, 2, 2, 1]),  # Last episode stopped early.
        }
        with mock.patch("phy_rl.evaluation.capture_rollouts", return_value=rollouts):
            result, filmed = evaluate_policy(
                SimpleNamespace(dt=0.01), None, None, Config(episode_length=2, eval_episodes=4)
            )
        self.assertEqual(result["balance"]["episode_success"], [True, False, False, False])
        self.assertEqual(result["balance"]["success_count"], 1)
        self.assertEqual(result["balance"]["episode_balanced_fraction"][-1], 0.5)
        self.assertIs(filmed, rollouts)
        self.assertEqual(result["episode_returns"], filmed["returns"][:, -1].tolist())

    def test_cpu_policy_updates_and_checkpoints_every_update_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            directory = run(
                Config(
                    total_steps=1024,
                    num_envs=4,
                    episode_length=32,
                    unroll_length=8,
                    batch_size=4,
                    num_minibatches=2,
                    updates_per_batch=2,
                    num_evals=2,
                    eval_episodes=2,
                    output=Path(root),
                )
            )
            summary = json.loads((directory / "summary.json").read_text())
            resolved = json.loads((directory / "resolved_config.json").read_text())
            self.assertEqual(resolved["backend"], "cpu")
            self.assertGreaterEqual(summary["actual_steps"], 1024)
            self.assertGreater(summary["policy_parameter_change_l2"], 0)
            self.assertEqual(len(summary["evaluation_after"]["episode_returns"]), 2)
            self.assertEqual(summary["evaluation_after"]["balance"]["episode_count"], 2)
            self.assertGreater((directory / "learning_curves.png").stat().st_size, 1000)
            checkpoints = sorted((directory / "checkpoints").glob("update-*/manifest.json"))
            self.assertEqual(
                [read_checkpoint(path).manifest["completed_updates"] for path in checkpoints],
                list(range(1, 17)),
            )
            self.assertEqual(read_checkpoint(directory / "checkpoints").manifest["env_steps"], 1024)

    def test_sigkill_then_fresh_process_restores_full_state_and_original_target(self):
        import jax
        import numpy as np
        from flax import serialization

        def execute(code, args):
            return subprocess.run(
                [sys.executable, "-c", code, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # This wraps only the real durable save. SIGKILL bypasses all worker
            # exception handling/finalization, after the first periodic commit.
            interrupted = execute(
                """
import os, signal
from unittest.mock import patch
from phy_rl import train
save = train.save_state
def save_then_kill(*args, **kwargs):
    save(*args, **kwargs)
    os.kill(os.getpid(), signal.SIGKILL)
with patch.object(train, 'save_state', save_then_kill):
    raise SystemExit(train.main())
""",
                [
                    "--total-steps",
                    "512",
                    "--num-envs",
                    "4",
                    "--episode-length",
                    "16",
                    "--unroll-length",
                    "8",
                    "--batch-size",
                    "4",
                    "--num-minibatches",
                    "2",
                    "--updates-per-batch",
                    "2",
                    "--num-evals",
                    "3",
                    "--eval-episodes",
                    "2",
                    "--checkpoint-interval-steps",
                    "3",
                    "--output",
                    str(root),
                ],
            )
            self.assertEqual(
                interrupted.returncode, -signal.SIGKILL, interrupted.stdout + interrupted.stderr
            )
            (first,) = root.iterdir()
            saved = read_checkpoint(first / "checkpoints")
            self.assertEqual(saved.manifest["env_steps"], 192)
            self.assertEqual(saved.manifest["completed_updates"], 3)
            self.assertFalse((first / "summary.json").exists())

            # Capture the actual reconstructed typed state before any updates.
            # Comparing every leaf catches accidental optimizer/RNG reinitialization.
            audit = root / "restored.msgpack"
            resumed = execute(
                """
import os
from pathlib import Path
from flax import serialization
from unittest.mock import patch
from phy_rl.checkpointing import Checkpoint
from phy_rl import train
restore = Checkpoint.restore
def audited_restore(self, template):
    state = restore(self, template)
    Path(os.environ['RESTORE_AUDIT']).write_bytes(serialization.to_bytes(state))
    return state
with patch.object(Checkpoint, 'restore', audited_restore):
    raise SystemExit(train.main())
""".replace("os.environ['RESTORE_AUDIT']", repr(str(audit))),
                ["--resume", str(saved.path), "--checkpoint-interval-steps", "5"],
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            expected = serialization.msgpack_restore(saved.payload)
            restored = serialization.msgpack_restore(audit.read_bytes())
            self.assertEqual(
                jax.tree_util.tree_structure(expected), jax.tree_util.tree_structure(restored)
            )
            for before, after in zip(
                jax.tree_util.tree_leaves(expected),
                jax.tree_util.tree_leaves(restored),
                strict=True,
            ):
                np.testing.assert_array_equal(before, after)

            (second,) = [path for path in root.iterdir() if path.is_dir() and path != first]
            summary = json.loads((second / "summary.json").read_text())
            self.assertEqual(summary["starting_step"], 192)
            self.assertEqual(summary["actual_steps"], 512)
            self.assertEqual(summary["steps_this_attempt"], 320)
            self.assertEqual(summary["evaluation_before"]["policy_origin"], "checkpoint")
            rows = [json.loads(row) for row in (second / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["step"], 192)
            self.assertEqual([row["step"] for row in rows], sorted(row["step"] for row in rows))
            checkpoints = sorted((second / "checkpoints").glob("update-*/manifest.json"))
            self.assertEqual(
                [read_checkpoint(path).manifest["completed_updates"] for path in checkpoints],
                [5, 8],
            )
            final = read_checkpoint(second / "checkpoints")
            final_state = serialization.msgpack_restore(final.payload)
            # Adam's counter must include the passes before AND after the kill.
            adam = final_state["learner"]["optimizer_state"]["1"]["0"]
            self.assertEqual(int(adam["count"]), 8 * 2 * 2)

            completed = execute(
                "from phy_rl.train import main; raise SystemExit(main())",
                ["--resume", str(final.path)],
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            (third,) = [
                path for path in root.iterdir() if path.is_dir() and path not in (first, second)
            ]
            summary = json.loads((third / "summary.json").read_text())
            self.assertEqual(summary["steps_this_attempt"], 0)
            self.assertEqual(summary["policy_parameter_change_l2"], 0)
            self.assertFalse((third / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
