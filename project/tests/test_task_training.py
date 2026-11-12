"""Real robot training, evaluation, video, and resume; opt in with RUN_MUJOCO_TASKS=1."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phy_rl.evaluation import evaluate_policy
from phy_rl.train import Config


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Requires training dependencies")
class EvaluationMetricTests(unittest.TestCase):
    def test_forward_speed_uses_named_joint_and_metrics_average_active_steps(self) -> None:
        import numpy as np

        for task, index in (("CheetahRun", 0), ("WalkerWalk", 1)):
            with self.subTest(task=task):
                qpos = np.zeros((1, 3, 2))
                qpos[0, :, index] = [0, 0.5, 1]
                qpos[0, :, 1 - index] = [0, 100, 200]
                env = SimpleNamespace(
                    dt=0.25,
                    mj_model=SimpleNamespace(
                        joint=lambda name, index=index: SimpleNamespace(qposadr=[index])
                    ),
                )
                with patch(
                    "phy_rl.evaluation.capture_rollouts",
                    return_value={
                        "qpos": qpos,
                        "length": np.array([2]),
                        "returns": np.array([[0, 1, 2]]),
                        "metric/reward/standing": np.array([[0.2, 0.6]]),
                    },
                ):
                    result, _ = evaluate_policy(env, None, None, Config(task=task))
                self.assertEqual(result["task_metrics"]["forward_speed_m_per_s"], 2)
                self.assertAlmostEqual(result["task_metrics"]["reward/standing"], 0.4)


@unittest.skipUnless(os.environ.get("RUN_MUJOCO_TASKS") == "1", "Set RUN_MUJOCO_TASKS=1")
class TaskTrainingTests(unittest.TestCase):
    def exercise(self, task: str) -> None:
        # Separate processes release compiled programs and simulator memory between tasks.
        script = """
import dataclasses, json, sys
from pathlib import Path
from phy_rl.train import run, parse_args
root = Path(sys.argv[2])
first = run({
    "task": sys.argv[1], "backend": "cpu", "output": str(root),
    "total_steps": 16, "num_envs": 2, "episode_length": 8,
    "batch_size": 2, "num_minibatches": 1, "unroll_length": 4,
    "updates_per_batch": 1, "num_evals": 2, "eval_episodes": 1,
    "checkpoint_interval_steps": 1, "video": True,
})
checkpoint = sorted((first / "checkpoints").glob("checkpoint-*.json"))[0]
second = run(dataclasses.replace(parse_args(["--resume", str(checkpoint)]), video=False))
(root / "attempts.json").write_text(json.dumps([str(first), str(second)]))
"""
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run(
                [sys.executable, "-c", script, task, root],
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            first, second = map(Path, json.loads((Path(root) / "attempts.json").read_text()))
            initial = json.loads((first / "summary.json").read_text())
            resumed = json.loads((second / "summary.json").read_text())
            resolved = json.loads((first / "resolved_config.json").read_text())
            self.assertEqual(resolved["requested"]["task"], task)
            self.assertEqual(resolved["backend"], "cpu")
            self.assertEqual(initial["actual_steps"], 16)
            self.assertGreater(initial["policy_parameter_change_l2"], 0)
            self.assertTrue(initial["evaluation_after"]["task_metrics"])
            self.assertNotIn("balance", initial["evaluation_after"])
            self.assertEqual(resumed["starting_step"], 8)
            self.assertEqual(resumed["steps_this_attempt"], 8)
            self.assertEqual(resumed["experiment_id"], initial["experiment_id"])
            self.assertGreater(resumed["policy_parameter_change_l2"], 0)
            for name in ("policy_comparison.mp4", "policy_comparison.png", "learning_curves.png"):
                self.assertGreater((first / name).stat().st_size, 1000)
            video = json.loads((first / "video.json").read_text())
            self.assertEqual(video["renderer"], "mujoco_native")
            if task == "Go1JoystickFlatTerrain":
                self.assertEqual(resolved["network"]["value_obs_key"], "privileged_state")

    def test_cheetah(self) -> None:
        self.exercise("CheetahRun")

    def test_walker(self) -> None:
        self.exercise("WalkerWalk")

    def test_go1(self) -> None:
        self.exercise("Go1JoystickFlatTerrain")

    def test_panda(self) -> None:
        self.exercise("PandaPickCube")


if __name__ == "__main__":
    unittest.main()
