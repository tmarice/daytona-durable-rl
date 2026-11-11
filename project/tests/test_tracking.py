"""W&B lifecycle and resume contracts without requiring credentials or the SDK."""

import dataclasses
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from phy_rl.checkpointing import commit_checkpoint
from phy_rl.daytona_experiment_session import ExperimentSession
from phy_rl.tracking import WandbTracker
from phy_rl.train import Config, config_values, parse_args, run, training_config


class TrackingTests(unittest.TestCase):
    def test_task_evaluation_metrics_are_logged_with_attempt_and_step(self):
        with tempfile.TemporaryDirectory() as root:
            tracker = WandbTracker(Path(root), attempt_id="attempt-2")
            tracker.run = SimpleNamespace(log=mock.Mock())
            tracker.evaluation(
                128,
                "after",
                {
                    "mean_return": 5.0,
                    "std_return": 1.0,
                    "mean_length": 100,
                    "task_metrics": {"reward/tracking_lin_vel": 0.8, "forward_speed_m_per_s": 1.2},
                },
            )
            logged = tracker.run.log.call_args.args[0]
            self.assertEqual(logged["heldout/after/task/reward/tracking_lin_vel"], 0.8)
            self.assertEqual(logged["heldout/after/task/forward_speed_m_per_s"], 1.2)
            self.assertEqual(logged["env_steps"], 128)
            self.assertEqual(logged["attempt_id"], "attempt-2")

    def test_tracking_options_do_not_change_training_compatibility_and_resume_inherits_them(self):
        config = dataclasses.replace(Config(), wandb=True, wandb_project="robotics")
        self.assertEqual(training_config(config), training_config(Config()))
        with tempfile.TemporaryDirectory() as root:
            commit_checkpoint(
                ExperimentSession("experiment", "attempt", root),
                b"state",
                env_steps=64,
                completed_updates=1,
                metadata={"requested": config_values(config)},
            )
            self.assertTrue(parse_args(["--resume", root]).wandb)
            self.assertEqual(parse_args(["--resume", root]).wandb_project, "robotics")
            self.assertFalse(parse_args(["--resume", root, "--no-wandb"]).wandb)

    def test_resume_uses_saved_destination_and_preserves_replayed_steps(self):
        remote = SimpleNamespace(
            id="persistent-id",
            project="robotics",
            entity="test-team",
            url="https://example.com/run",
            define_metric=mock.Mock(),
            log=mock.Mock(),
            summary={},
        )
        sdk = SimpleNamespace(init=mock.Mock(return_value=remote), Settings=mock.Mock())
        previous = {"id": remote.id, "project": remote.project, "entity": remote.entity}
        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch.dict(
                sys.modules, {"wandb": sdk, "dotenv": SimpleNamespace(load_dotenv=mock.Mock())}
            ),
            mock.patch.dict(
                os.environ, {"WANDB_PROJECT": "unrelated", "WANDB_ENTITY": "another-team"}
            ),
        ):
            tracker = WandbTracker(Path(root), attempt_id="attempt-2")
            tracker.start(
                Config(wandb=True, wandb_mode="online"), "experiment", {"seed": 11}, previous
            )
            self.assertEqual(sdk.init.call_args.kwargs["resume"], "allow")
            self.assertEqual(sdk.init.call_args.kwargs["id"], "persistent-id")
            self.assertEqual(sdk.init.call_args.kwargs["entity"], "test-team")
            self.assertEqual(sdk.init.call_args.kwargs["project"], "robotics")
            tracker.log(128, {"eval/reward": 1.0})
            tracker.log(64, {"eval/reward": 0.5})
            calls = remote.log.call_args_list
            self.assertEqual([call.args[0]["env_steps"] for call in calls], [128, 64])
            self.assertTrue(all(not call.kwargs for call in calls))
            self.assertTrue(all(call.args[0]["attempt_id"] == "attempt-2" for call in calls))
            self.assertEqual(
                json.loads((Path(root) / "wandb.json").read_text())["id"], "persistent-id"
            )

    def test_finalization_uploads_all_result_files_and_media(self):
        remote = SimpleNamespace(summary={}, save=mock.Mock(), log=mock.Mock(), finish=mock.Mock())
        sdk = SimpleNamespace(
            Video=mock.Mock(return_value="video-object"),
            Image=mock.Mock(side_effect=lambda path: f"image:{Path(path).name}"),
        )
        result_files = {
            "requested_config.json",
            "resolved_config.json",
            "attempt.json",
            "wandb.json",
            "metrics.jsonl",
            "metrics.csv",
            "learning_curves.png",
            "evaluation_before.json",
            "evaluation_after.json",
            "policy_comparison.png",
            "policy_comparison.mp4",
            "summary.json",
            "status.json",
            "checkpoint_status.json",
            "video.json",
        }
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"wandb": sdk}):
            path = Path(root)
            for name in result_files | {".env", "state.msgpack"}:
                (path / name).write_text("placeholder")
            tracker = WandbTracker(path, attempt_id="attempt-2")
            tracker.run = remote
            tracker.finish("completed", {"actual_steps": 128})
            self.assertEqual(remote.log.call_args.args[0]["policy/comparison"], "video-object")
            self.assertEqual(
                remote.log.call_args.args[0]["plots/learning_curves"], "image:learning_curves.png"
            )
            self.assertEqual(
                remote.log.call_args.args[0]["plots/policy_comparison"],
                "image:policy_comparison.png",
            )
            self.assertEqual(
                {Path(c.args[0]).name for c in remote.save.call_args_list}, result_files
            )
            for call in remote.save.call_args_list:
                self.assertEqual(
                    call.kwargs, {"base_path": str(path.resolve().parent), "policy": "now"}
                )
            remote.finish.assert_called_once_with(exit_code=0)
            self.assertEqual(remote.summary["attempt_status"], "completed")
            self.assertEqual(remote.summary["actual_steps"], 128)

    def test_failed_attempt_uploads_available_metrics_plots_and_error(self):
        remote = SimpleNamespace(summary={}, save=mock.Mock(), log=mock.Mock(), finish=mock.Mock())
        sdk = SimpleNamespace(Image=mock.Mock(return_value="plot-object"))
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"wandb": sdk}):
            path = Path(root)
            files = {
                "metrics.jsonl",
                "metrics.csv",
                "learning_curves.png",
                "status.json",
                "error.txt",
            }
            for name in files:
                (path / name).write_text("placeholder")
            tracker = WandbTracker(path, attempt_id="failed-attempt")
            tracker.run = remote
            tracker.log(64, {"training/loss": 0.25})
            tracker.finish("failed")
            self.assertEqual({Path(c.args[0]).name for c in remote.save.call_args_list}, files)
            self.assertEqual(remote.log.call_args.args[0]["env_steps"], 64)
            self.assertEqual(remote.log.call_args.args[0]["plots/learning_curves"], "plot-object")
            remote.finish.assert_called_once_with(exit_code=1)

    def test_failed_training_finishes_tracking_without_masking_original_error(self):
        for error, status in (
            (RuntimeError("training failed"), "failed"),
            (KeyboardInterrupt(), "interrupted"),
        ):
            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as root,
                mock.patch("phy_rl.train.WandbTracker") as tracker,
                mock.patch("phy_rl.train.train", side_effect=error),
            ):
                tracker.return_value.finish.side_effect = RuntimeError("upload failed")
                with self.assertRaises(type(error)) as caught:
                    run(Config(wandb=True, output=Path(root)))
                self.assertIs(caught.exception, error)
                tracker.return_value.finish.assert_called_once_with(status, None)


if __name__ == "__main__":
    unittest.main()
