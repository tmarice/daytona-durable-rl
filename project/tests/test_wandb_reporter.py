"""Controller reporting with in-memory Daytona files and a fake W&B transport."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from daytona.v0.common.errors import DaytonaNotFoundError

from daytona_runner import DaytonaJobStatus
from wandb_reporter import WandbReporter


def metric(step: int, reward: float = 1.0) -> bytes:
    return (
        json.dumps(
            {
                "step": step,
                "elapsed_seconds": 2.0,
                "metrics": {"eval/episode_reward": reward},
            }
        )
        + "\n"
    ).encode()


class ReporterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.run = SimpleNamespace(
            id="wandb-id",
            project="project",
            entity="team",
            url="https://example.com/run",
            define_metric=Mock(),
            log=Mock(),
            summary={},
            config=SimpleNamespace(update=Mock()),
            save=Mock(),
            finish=Mock(),
        )
        self.sdk = SimpleNamespace(
            init=Mock(return_value=self.run),
            Settings=Mock(),
            Image=Mock(),
            Video=Mock(),
        )
        self.enterContext(patch.dict(sys.modules, {"wandb": self.sdk}))
        self.reporter = WandbReporter(
            {"wandb": False, "total_steps": 128}, output=self.root, project="project"
        )
        self.job = SimpleNamespace(id="job", attempt_id="first", status=DaytonaJobStatus.RUNNING)
        self.files = {}

        async def download(path: str, timeout: int) -> bytes:
            if path not in self.files:
                raise DaytonaNotFoundError("file not found")
            return self.files[path]

        self.sandbox = SimpleNamespace(
            fs=SimpleNamespace(download_file=AsyncMock(side_effect=download))
        )
        await self.reporter.start(self.job)

    def put(self, name: str, data: bytes) -> None:
        self.files[f"/run/job/code/runs/{self.job.attempt_id}/{name}"] = data

    async def test_start_uses_an_independent_run_without_controller_system_metrics(self) -> None:
        options = self.sdk.init.call_args.kwargs
        self.assertEqual(options["reinit"], "create_new")
        self.assertEqual(options["mode"], "online")
        self.assertEqual(options["config"]["experiment_id"], "job")
        self.assertEqual(
            options["config"], {"wandb": False, "total_steps": 128, "experiment_id": "job"}
        )
        self.assertTrue(self.sdk.Settings.call_args.kwargs["x_disable_stats"])
        self.run.define_metric.assert_any_call("*", step_metric="env_steps")
        self.assertEqual(
            json.loads((self.root / "job/wandb.json").read_text())["url"], self.run.url
        )

    async def test_poll_waits_for_files_and_complete_lines_without_dropping_repeated_steps(
        self,
    ) -> None:
        await self.reporter.poll(self.job, self.sandbox)
        self.run.log.assert_not_called()
        first, second = metric(64), metric(64, 2.0)
        self.put("metrics.jsonl", first + second[:20])
        await self.reporter.poll(self.job, self.sandbox)
        await self.reporter.poll(self.job, self.sandbox)
        self.assertEqual(self.run.log.call_count, 1)
        self.put("metrics.jsonl", first + second)
        await self.reporter.poll(self.job, self.sandbox)
        self.assertEqual(
            [c.args[0]["eval/episode_reward"] for c in self.run.log.call_args_list], [1.0, 2.0]
        )
        self.assertEqual((self.root / "job/first/metrics.jsonl").read_bytes(), first + second)

    async def test_progress_is_throttled_and_repeated_polls_preserve_metric_age(self) -> None:
        clock = Mock(return_value=100.0)
        with (
            patch("wandb_reporter.time", SimpleNamespace(monotonic=clock)),
            self.assertLogs("wandb_reporter", level="INFO") as captured,
        ):
            await self.reporter.poll(self.job, self.sandbox)
            clock.return_value = 105.0
            self.put("metrics.jsonl", metric(64, 3.5))
            await self.reporter.poll(self.job, self.sandbox)
            clock.return_value = 110.0
            await self.reporter.poll(self.job, self.sandbox)
            clock.return_value = 130.0
            await self.reporter.poll(self.job, self.sandbox)

        messages = [record.getMessage() for record in captured.records]
        self.assertEqual(
            messages,
            [
                "Job job | worker running; awaiting first metric",
                "Job job | evaluation reward=3.50 at step=64",
                "Job job | training steps=64/128 (50.0%) | last new metric 25s ago",
            ],
        )
        self.run.log.assert_called_once()

    async def test_replacement_attempt_drops_stale_progress_and_reports_immediately(self) -> None:
        clock = Mock(return_value=100.0)
        self.put("metrics.jsonl", metric(128))
        with patch("wandb_reporter.time", SimpleNamespace(monotonic=clock)):
            await self.reporter.poll(self.job, self.sandbox)
            self.job.attempt_id = "replacement"
            clock.return_value = 105.0
            with self.assertLogs("wandb_reporter", level="INFO") as captured:
                await self.reporter.poll(self.job, self.sandbox)
                self.put("metrics.jsonl", metric(64))
                clock.return_value = 135.0
                await self.reporter.poll(self.job, self.sandbox)

        messages = [record.getMessage() for record in captured.records]
        self.assertEqual(messages[0], "Job job | worker running; awaiting first metric")
        self.assertIn("training steps=64/128 (50.0%)", messages[-1])
        self.assertNotIn("100.0%", "\n".join(messages))
        self.sdk.init.assert_called_once()

    async def test_retry_resets_the_cursor_but_keeps_the_run_and_labels_replayed_steps(
        self,
    ) -> None:
        self.put("metrics.jsonl", metric(128))
        await self.reporter.poll(self.job, self.sandbox)
        self.job.status = DaytonaJobStatus.PREEMPTED
        await self.reporter.collect(self.job, self.sandbox)
        self.run.finish.assert_not_called()
        self.job.attempt_id = "replacement"
        self.job.status = DaytonaJobStatus.RUNNING
        self.put("metrics.jsonl", metric(64))
        await self.reporter.poll(self.job, self.sandbox)
        self.assertEqual(
            [
                (c.args[0]["attempt_id"], c.args[0]["env_steps"])
                for c in self.run.log.call_args_list
            ],
            [("first", 128), ("replacement", 64)],
        )
        self.sdk.init.assert_called_once()
        self.assertTrue((self.root / "job/first/metrics.jsonl").is_file())
        self.assertTrue((self.root / "job/replacement/metrics.jsonl").is_file())

    async def test_evaluations_are_logged_once_at_their_training_step(self) -> None:
        self.put("evaluation_before.json", b'{"training_step":')
        await self.reporter.poll(self.job, self.sandbox)
        for phase, step in (("before", 0), ("after", 128)):
            self.put(
                f"evaluation_{phase}.json",
                json.dumps(
                    {
                        "training_step": step,
                        "mean_return": 2,
                        "std_return": 1,
                        "mean_length": 32,
                        "task_metrics": {"forward_speed_m_per_s": 0.5},
                        "balance": {"success_count": 1, "episode_count": 2},
                    }
                ).encode(),
            )
        await self.reporter.poll(self.job, self.sandbox)
        await self.reporter.poll(self.job, self.sandbox)
        self.assertEqual([c.args[0]["env_steps"] for c in self.run.log.call_args_list], [0, 128])
        self.assertIn("heldout/after/mean_return", self.run.log.call_args.args[0])
        self.assertEqual(
            self.run.log.call_args.args[0]["heldout/after/task/forward_speed_m_per_s"], 0.5
        )
        self.assertEqual(self.run.log.call_args.args[0]["heldout/after/balance_success_rate"], 0.5)

    async def test_collection_uploads_media_at_last_known_step_without_a_summary(self) -> None:
        self.put("metrics.jsonl", metric(64))
        self.put("policy_comparison.mp4", b"video")
        self.put("policy_comparison.png", b"image")
        await self.reporter.collect(self.job, self.sandbox)
        directory = self.root / "job/first"
        self.sdk.Video.assert_called_once_with(
            str(directory / "policy_comparison.mp4"), format="mp4"
        )
        self.sdk.Image.assert_called_once_with(str(directory / "policy_comparison.png"))
        media = self.run.log.call_args.args[0]
        self.assertEqual(media["env_steps"], 64)
        self.assertEqual(media["attempt_id"], "first")
        self.assertIs(media["policy/comparison"], self.sdk.Video.return_value)
        self.assertIs(media["plots/policy_comparison"], self.sdk.Image.return_value)

    async def test_final_collection_drains_metrics_and_downloads_only_result_files(self) -> None:
        self.put("metrics.jsonl", metric(64))
        await self.reporter.poll(self.job, self.sandbox)
        self.put("metrics.jsonl", metric(64) + metric(128))
        self.put("summary.json", b'{"actual_steps":128}')
        self.put("resolved_config.json", b'{"backend":"gpu"}')
        self.put("error.txt", b"diagnostic")
        self.put(".env", b"secret")
        self.put("state.msgpack", b"checkpoint payload")
        self.job.status = DaytonaJobStatus.COMPLETED
        await self.reporter.collect(self.job, self.sandbox)
        self.assertEqual(self.run.log.call_count, 2)
        saved = {Path(c.args[0]).name for c in self.run.save.call_args_list}
        self.assertEqual(
            saved, {"metrics.jsonl", "summary.json", "resolved_config.json", "error.txt"}
        )
        self.assertEqual(self.run.summary["actual_steps"], 128)
        self.run.config.update.assert_called_once_with(
            {"worker": {"backend": "gpu"}}, allow_val_change=True
        )
        self.run.finish.assert_not_called()
        await self.reporter.finish(self.job.status)
        await self.reporter.finish(self.job.status)
        self.run.finish.assert_called_once_with(exit_code=0)
        self.assertEqual(self.run.summary["job_status"], "completed")

    async def test_upload_failure_retains_files_and_finishes_the_run(self) -> None:
        self.put("metrics.jsonl", metric(64))
        self.run.save.side_effect = RuntimeError("W&B unavailable")
        with self.assertLogs("wandb_reporter", level="WARNING"):
            await self.reporter.collect(self.job, self.sandbox)
        await self.reporter.finish(DaytonaJobStatus.FAILED)
        self.run.finish.assert_called_once_with(exit_code=1)
        self.assertTrue((self.root / "job/first/metrics.jsonl").is_file())
        status = json.loads((self.root / "job/reporting_status.json").read_text())
        self.assertIn("W&B unavailable", status["errors"][0])

    async def test_evicted_sandbox_read_failure_keeps_previous_metrics(self) -> None:
        self.put("metrics.jsonl", metric(64))
        await self.reporter.poll(self.job, self.sandbox)
        self.sandbox.fs.download_file.side_effect = TimeoutError("sandbox gone")
        self.job.status = DaytonaJobStatus.PREEMPTED
        with self.assertLogs("wandb_reporter", level="WARNING"):
            await self.reporter.collect(self.job, self.sandbox)
        self.assertEqual((self.root / "job/first/metrics.jsonl").read_bytes(), metric(64))
        self.run.finish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
