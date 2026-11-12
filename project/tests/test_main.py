"""Library usage example and Ctrl+C cleanup without remote sandboxes."""

import asyncio
import io
import json
import os
import subprocess
import sys
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import main as example
from daytona_runner import DaytonaJobResult, DaytonaJobStatus, DaytonaRunner
from phy_rl.tasks import TASK_DEFAULTS
from phy_rl.train import config_from_values


class TrainingGridTests(unittest.TestCase):
    def test_every_allowed_task_has_valid_configs_with_equal_training_budgets(self) -> None:
        self.assertEqual({task.value for task in example.Task}, set(TASK_DEFAULTS))
        for task in example.Task:
            with self.subTest(task=task):
                grid = example.build_training_grid(task)
                self.assertEqual(len(grid), 12)
                self.assertEqual([config["seed"] for config in grid[:2]], [11, 22])
                self.assertEqual(
                    {config["total_steps"] for config in grid}, {grid[0]["total_steps"]}
                )
                for values in grid:
                    # The runner embeds config reprs in a remote Python command.
                    self.assertIs(type(values["task"]), str)
                    self.assertEqual(values, json.loads(json.dumps(values)))
                    config = config_from_values(values)
                    self.assertEqual(config.task, task.value)
                    self.assertEqual(config.backend, "gpu")
                    self.assertTrue(config.video)
                    self.assertFalse(config.wandb)
                    self.assertEqual(
                        config.batch_size * config.num_minibatches % config.num_envs, 0
                    )
                    steps_per_update = (
                        config.batch_size * config.num_minibatches * config.unroll_length
                    )
                    self.assertEqual(
                        config.total_steps % (steps_per_update * (config.num_evals - 1)), 0
                    )

    def test_robot_baselines_keep_task_presets_and_candidates_change_one_setting(self) -> None:
        for task in example.Task:
            with self.subTest(task=task):
                grid = example.build_training_grid(task)
                baseline = grid[0]
                for key, value in TASK_DEFAULTS[task.value].items():
                    self.assertEqual(baseline[key], value)
                for seed in (11, 22):
                    candidates = [config for config in grid if config["seed"] == seed]
                    self.assertEqual(len(candidates), 6)
                    self.assertEqual(
                        len({json.dumps(config, sort_keys=True) for config in candidates}), 6
                    )
                    for config in candidates[1:]:
                        changed = {key for key in config if config[key] != candidates[0][key]}
                        self.assertEqual(len(changed), 1)
                        self.assertLessEqual(
                            changed,
                            {"learning_rate", "entropy_cost", "updates_per_batch", "batch_size"},
                        )


class ExampleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(example, "load_dotenv"))
        self.enterContext(patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}))
        os.environ.pop("WANDB_SILENT", None)
        self.enterContext(patch.object(example.logging, "basicConfig"))
        for name in (example.__name__, "daytona_runner", "wandb_reporter"):
            logger = example.logging.getLogger(name)
            self.addCleanup(logger.setLevel, logger.level)
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.reporters = []

        def make_reporter(*args, **kwargs):
            reporter = SimpleNamespace(
                start=AsyncMock(),
                poll=AsyncMock(),
                collect=AsyncMock(),
                finish=AsyncMock(),
                directory=self.root / str(len(self.reporters)),
            )
            reporter.directory.mkdir()
            self.reporters.append(reporter)
            return reporter

        self.enterContext(patch.object(example, "WandbReporter", side_effect=make_reporter))

    async def test_results_close_the_stream_before_the_client(self) -> None:
        events = []

        async def results(jobs):
            try:
                for job in jobs:
                    yield DaytonaJobResult(
                        job.id,
                        DaytonaJobStatus.COMPLETED,
                        exit_code=0,
                        logs=f"worker output for {job.id}\n",
                    )
            finally:
                events.append("stream closed")

        runner = SimpleNamespace(
            create_job=AsyncMock(
                side_effect=[
                    SimpleNamespace(id=str(i), status=DaytonaJobStatus.COMPLETED) for i in range(12)
                ]
            ),
            run=results,
            close=AsyncMock(side_effect=lambda: events.append("client closed")),
        )
        with (
            patch.object(example, "DaytonaRunner", return_value=runner) as constructor,
            redirect_stdout(io.StringIO()) as output,
        ):
            await example.main()
        self.assertEqual(events, ["stream closed", "client closed"])
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(os.environ["WANDB_SILENT"], "true")
        for index, reporter in enumerate(self.reporters):
            self.assertEqual(
                (reporter.directory / "worker.log").read_text(), f"worker output for {index}\n"
            )
        configs = [
            config_from_values(call.kwargs["config"]) for call in runner.create_job.call_args_list
        ]
        self.assertEqual({config.learning_rate for config in configs}, {0.0003, 0.001})
        self.assertEqual(len(configs), 12)
        self.assertEqual({config.task for config in configs}, {"CartpoleSwingup"})
        self.assertEqual(
            {
                (
                    config.learning_rate,
                    config.entropy_cost,
                    config.updates_per_batch,
                    config.batch_size,
                    config.seed,
                )
                for config in configs
            },
            {
                (*candidate, seed)
                for candidate in (
                    (0.001, 0.01, 16, 1024),
                    (0.0003, 0.01, 16, 1024),
                    (0.001, 0.003, 16, 1024),
                    (0.001, 0.03, 16, 1024),
                    (0.001, 0.01, 8, 1024),
                    (0.001, 0.01, 16, 512),
                )
                for seed in (11, 22)
            },
        )
        for config in configs:
            self.assertEqual(config.backend, "gpu")
            self.assertFalse(config.wandb)
            self.assertEqual(config.total_steps, 62_914_560)
            self.assertEqual(config.batch_size * config.num_minibatches % config.num_envs, 0)
            steps_per_update = config.batch_size * config.num_minibatches * config.unroll_length
            self.assertEqual(config.total_steps % (steps_per_update * (config.num_evals - 1)), 0)
        self.assertEqual(constructor.call_args.kwargs["work_dir"], "phy_rl")
        self.assertEqual(constructor.call_args.kwargs["entrypoint"], "train:run")
        self.assertEqual(constructor.call_args.kwargs["resources"]["gpu"], 1)
        self.assertEqual(constructor.call_args.kwargs["max_concurrent_jobs"], 2)
        self.assertNotIn("env_vars", constructor.call_args.kwargs)
        self.assertNotIn("secrets", constructor.call_args.kwargs)
        for call, reporter in zip(runner.create_job.call_args_list, self.reporters, strict=True):
            self.assertIs(call.kwargs["on_poll"], reporter.poll)
            self.assertIs(call.kwargs["on_attempt_end"], reporter.collect)
            reporter.start.assert_awaited_once()
            reporter.finish.assert_any_await(DaytonaJobStatus.COMPLETED)

    async def test_preparation_failure_still_closes_the_client(self) -> None:
        error = RuntimeError("snapshot failed")
        runner = SimpleNamespace(create_job=AsyncMock(side_effect=error), close=AsyncMock())
        with (
            patch.object(example, "DaytonaRunner", return_value=runner),
            self.assertRaises(RuntimeError) as caught,
        ):
            await example.main()
        self.assertIs(caught.exception, error)
        runner.close.assert_awaited_once()

    async def test_result_processing_failure_cleans_up_other_jobs_before_client_close(self) -> None:
        events = []
        running = asyncio.Event()
        result = DaytonaJobResult("first", DaytonaJobStatus.COMPLETED, exit_code=0)

        async def finish_first() -> DaytonaJobResult:
            await running.wait()
            return result

        async def run_until_cancelled() -> None:
            running.set()
            try:
                await asyncio.Future()
            finally:
                # A cancelled attempt still awaits the deletion of its sandbox.
                await asyncio.sleep(0)
                events.append("other cleaned")

        jobs = []

        def create_job(*, config: dict, **callbacks) -> SimpleNamespace:
            job = SimpleNamespace(
                id="first" if not jobs else str(len(jobs)),
                status=DaytonaJobStatus.RUNNING,
                run_attempt=AsyncMock(side_effect=run_until_cancelled if jobs else finish_first),
            )
            jobs.append(job)
            return job

        runner = object.__new__(DaytonaRunner)
        runner._max_concurrent_jobs = 2
        runner.create_job = AsyncMock(side_effect=create_job)
        runner._api = SimpleNamespace(
            close=AsyncMock(side_effect=lambda: events.append("client closed"))
        )

        def fail_result_logging(message: str, *args) -> None:
            if message.startswith("Sweep progress"):
                raise RuntimeError("output failed")

        with (
            patch.object(example, "DaytonaRunner", return_value=runner),
            patch.object(example.logger, "info", side_effect=fail_result_logging),
            self.assertRaisesRegex(RuntimeError, "output failed"),
        ):
            async with asyncio.timeout(5):
                await example.main()
        started_others = sum(job.run_attempt.await_count for job in jobs[1:])
        self.assertGreaterEqual(started_others, 1)
        self.assertEqual(events, [*["other cleaned"] * started_others, "client closed"])

    async def test_tracking_start_failure_finishes_prepared_runs_without_launching_jobs(
        self,
    ) -> None:
        runner = SimpleNamespace(
            create_job=AsyncMock(
                return_value=SimpleNamespace(
                    id="job",
                    status=DaytonaJobStatus.PENDING,
                )
            ),
            run=Mock(),
            close=AsyncMock(),
        )
        reporter = SimpleNamespace(
            start=AsyncMock(side_effect=RuntimeError("W&B unavailable")),
            poll=AsyncMock(),
            collect=AsyncMock(),
            finish=AsyncMock(),
        )
        with (
            patch.object(example, "DaytonaRunner", return_value=runner),
            patch.object(example, "WandbReporter", return_value=reporter),
            self.assertRaisesRegex(RuntimeError, "W&B unavailable"),
        ):
            await example.main()
        runner.run.assert_not_called()
        reporter.finish.assert_awaited_once_with(DaytonaJobStatus.CANCELED)
        runner.close.assert_awaited_once()


@unittest.skipUnless(os.name == "posix", "SIGINT probe uses POSIX signals")
class SignalTests(unittest.TestCase):
    def test_sigint_awaits_sandbox_cleanup_before_closing_client(self) -> None:
        script = textwrap.dedent("""
            import asyncio
            import os
            import runpy
            import signal
            from types import SimpleNamespace
            from unittest.mock import AsyncMock, patch
            from daytona_runner import DaytonaAPI, DaytonaJob, DaytonaRunner

            async def interrupt():
                os.kill(os.getpid(), signal.SIGINT)
                await asyncio.Future()

            async def delete(*, wait):
                print("delete started", flush=True)
                await asyncio.sleep(0.01)
                print("delete finished", flush=True)

            async def close():
                print("client closed", flush=True)

            sandbox = SimpleNamespace(
                refresh_data=interrupt,
                delete=delete,
                process=SimpleNamespace(
                    create_session=AsyncMock(),
                    execute_session_command=AsyncMock(
                        return_value=SimpleNamespace(cmd_id="command")
                    ),
                ),
            )
            api = object.__new__(DaytonaAPI)
            api._max_retries = 0
            api._client = SimpleNamespace(close=close)
            job = DaytonaJob(api, None, None, b"", "physical_ai.main:train", {})
            job._create_sandbox = AsyncMock(return_value=sandbox)
            job._upload_source_package = AsyncMock()
            runner = object.__new__(DaytonaRunner)
            runner._max_concurrent_jobs = 1
            runner.create_job = AsyncMock(return_value=job)
            runner._api = api
            with (
                patch("daytona_runner.DaytonaRunner", return_value=runner),
                patch("wandb_reporter.WandbReporter", return_value=SimpleNamespace(
                    start=AsyncMock(), poll=AsyncMock(), collect=AsyncMock(), finish=AsyncMock(),
                )),
                patch("dotenv.load_dotenv"),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
            ):
                runpy.run_path("main.py", run_name="__main__")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(), ["delete started", "delete finished", "client closed"]
        )
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
