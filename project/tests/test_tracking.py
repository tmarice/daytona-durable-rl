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
from phy_rl.tracking import WandbTracker
from phy_rl.train import Config, config_values, parse_args, run, training_config


class TrackingTests(unittest.TestCase):
    def test_tracking_options_do_not_change_training_compatibility_and_resume_inherits_them(self):
        config = dataclasses.replace(Config(), wandb=True, wandb_project="robotics")
        self.assertEqual(training_config(config), training_config(Config()))
        with tempfile.TemporaryDirectory() as root:
            commit_checkpoint(
                Path(root),
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
            tracker = WandbTracker(Path(root))
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
            self.assertTrue(all(call.args[0]["attempt_id"] == Path(root).name for call in calls))
            self.assertEqual(
                json.loads((Path(root) / "wandb.json").read_text())["id"], "persistent-id"
            )

    def test_finalization_uploads_video_and_only_explicit_result_files(self):
        remote = SimpleNamespace(summary={}, save=mock.Mock(), log=mock.Mock(), finish=mock.Mock())
        sdk = SimpleNamespace(Video=mock.Mock(return_value="video-object"), Image=mock.Mock())
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"wandb": sdk}):
            path = Path(root)
            for name in ("summary.json", "policy_comparison.mp4", ".env", "state.msgpack"):
                (path / name).write_text("placeholder")
            tracker = WandbTracker(path)
            tracker.run = remote
            tracker.finish("completed", {"actual_steps": 128})
            self.assertEqual(remote.log.call_args.args[0]["policy/comparison"], "video-object")
            self.assertEqual(
                [Path(c.args[0]).name for c in remote.save.call_args_list], ["summary.json"]
            )
            remote.finish.assert_called_once_with(exit_code=0)
            self.assertEqual(remote.summary["attempt_status"], "completed")

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
