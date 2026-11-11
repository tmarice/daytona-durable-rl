"""Task presets and robot asset staging without training or network access."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phy_rl.checkpointing import commit_checkpoint
from phy_rl.daytona_experiment_session import ExperimentSession
from phy_rl.tasks import TASK_DEFAULTS, network_settings, prepare_robot_assets
from phy_rl.train import config_from_values, config_values, parse_args


class TaskTests(unittest.TestCase):
    def test_cli_and_runner_dicts_select_presets_and_allow_overrides(self) -> None:
        for task in TASK_DEFAULTS:
            with self.subTest(task=task):
                cli = parse_args(["--task", task, "--gpu"])
                self.assertEqual(cli, config_from_values({"task": task, "backend": "gpu"}))
                cli.validate()
                overridden = config_from_values({"task": task, "learning_rate": 0.0007})
                self.assertEqual(overridden.learning_rate, 0.0007)
                self.assertEqual(cli.batch_size * cli.num_minibatches % cli.num_envs, 0)
                if not task.startswith("Cartpole"):
                    self.assertEqual(
                        cli.total_steps
                        % (cli.batch_size * cli.num_minibatches * cli.unroll_length),
                        0,
                    )
        self.assertEqual(config_from_values({"task": "PandaPickCube"}).episode_length, 150)
        self.assertEqual(config_from_values({"task": "Go1JoystickFlatTerrain"}).reward_scaling, 1)
        self.assertEqual(
            network_settings("Go1JoystickFlatTerrain")["value_obs_key"], "privileged_state"
        )

    def test_resume_uses_saved_values_instead_of_current_task_defaults(self) -> None:
        original = config_from_values({"task": "WalkerWalk", "total_steps": 123_456})
        with tempfile.TemporaryDirectory() as root:
            checkpoint = commit_checkpoint(
                ExperimentSession("experiment", "first", Path(root)),
                b"unused test payload",
                env_steps=8,
                completed_updates=1,
                metadata={"requested": config_values(original)},
            )
            resumed = parse_args(["--resume", str(checkpoint)])
            self.assertEqual(resumed.task, "WalkerWalk")
            self.assertEqual(resumed.total_steps, 123_456)
            self.assertEqual(resumed.num_envs, original.num_envs)
            with self.assertRaisesRegex(ValueError, "task.*conflicts"):
                config_from_values({"resume": checkpoint, "task": "PandaPickCube"})

    def test_assets_download_only_selected_robot_at_pinned_revision(self) -> None:
        mjx_env = SimpleNamespace(MENAGERIE_COMMIT_SHA="pinned-revision", MENAGERIE_PATH="old")
        tree = {
            "tree": [
                {"type": "tree", "path": "unitree_go1/assets"},
                {"type": "blob", "path": "unitree_go1/assets/trunk.stl"},
                {"type": "blob", "path": "unitree_go1/LICENSE"},
                {"type": "blob", "path": "unrelated_robot/mesh.stl"},
            ]
        }

        def fetch(url: str, timeout: int) -> io.BytesIO:
            self.assertIn("pinned-revision", url)
            return io.BytesIO(json.dumps(tree).encode() if "api.github.com" in url else b"asset")

        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(sys.modules, {"mujoco_playground._src": SimpleNamespace(mjx_env=mjx_env)}),
            patch("phy_rl.tasks.urlopen", side_effect=fetch) as download,
        ):
            prepare_robot_assets("CheetahRun", Path(root))
            download.assert_not_called()
            prepare_robot_assets("Go1JoystickFlatTerrain", Path(root))
            self.assertEqual(download.call_count, 3)
            self.assertEqual(
                (mjx_env.MENAGERIE_PATH / "unitree_go1/assets/trunk.stl").read_bytes(), b"asset"
            )
            self.assertFalse((mjx_env.MENAGERIE_PATH / "unrelated_robot").exists())


if __name__ == "__main__":
    unittest.main()
