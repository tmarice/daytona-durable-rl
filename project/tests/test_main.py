"""Library usage example and Ctrl+C cleanup without remote sandboxes."""

import asyncio
import io
import os
import subprocess
import sys
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main as example
from daytona_runner import DaytonaJobResult, DaytonaJobStatus, DaytonaRunner
from phy_rl.train import config_from_values


class ExampleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(example, "load_dotenv"))
        self.enterContext(patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}))

    async def test_results_close_the_stream_before_the_client(self) -> None:
        events = []

        async def results(jobs):
            try:
                for job in jobs:
                    yield DaytonaJobResult(job, DaytonaJobStatus.COMPLETED, exit_code=0)
            finally:
                events.append("stream closed")

        runner = SimpleNamespace(
            create_job=AsyncMock(return_value="job"),
            run=results,
            close=AsyncMock(side_effect=lambda: events.append("client closed")),
        )
        with (
            patch.object(example, "DaytonaRunner", return_value=runner) as constructor,
            redirect_stdout(io.StringIO()),
        ):
            await example.main()
        self.assertEqual(events, ["stream closed", "client closed"])
        configs = [
            config_from_values(call.kwargs["config"]) for call in runner.create_job.call_args_list
        ]
        self.assertEqual({config.learning_rate for config in configs}, {0.0003, 0.001})
        self.assertEqual(len(configs), 12)
        self.assertEqual({config.task for config in configs}, {"CartpoleSwingup"})
        self.assertEqual(
            {(config.learning_rate, config.entropy_cost, config.seed) for config in configs},
            {
                (lr, entropy_cost, seed)
                for lr in (0.0003, 0.001)
                for entropy_cost in (0.001, 0.01)
                for seed in (11, 22, 33)
            },
        )
        for config in configs:
            self.assertEqual(config.backend, "gpu")
            self.assertTrue(config.wandb)
            self.assertGreaterEqual(config.total_steps, 7_864_320)
            self.assertEqual(config.batch_size * config.num_minibatches % config.num_envs, 0)
        self.assertEqual(constructor.call_args.kwargs["work_dir"], "phy_rl")
        self.assertEqual(constructor.call_args.kwargs["entrypoint"], "train:run")
        self.assertEqual(constructor.call_args.kwargs["resources"]["gpu"], 1)
        self.assertEqual(constructor.call_args.kwargs["max_concurrent_jobs"], 2)

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

        def create_job(*, config: dict) -> SimpleNamespace:
            job = SimpleNamespace(
                run_attempt=AsyncMock(side_effect=run_until_cancelled if jobs else finish_first)
            )
            jobs.append(job)
            return job

        runner = object.__new__(DaytonaRunner)
        runner._max_concurrent_jobs = 2
        runner.create_job = AsyncMock(side_effect=create_job)
        runner._daytona = SimpleNamespace(
            close=AsyncMock(side_effect=lambda: events.append("client closed"))
        )
        with (
            patch.object(example, "DaytonaRunner", return_value=runner),
            patch.object(example, "print", side_effect=RuntimeError("output failed")),
            self.assertRaisesRegex(RuntimeError, "output failed"),
        ):
            async with asyncio.timeout(5):
                await example.main()
        started_others = sum(job.run_attempt.await_count for job in jobs[1:])
        self.assertGreaterEqual(started_others, 1)
        self.assertEqual(events, [*["other cleaned"] * started_others, "client closed"])


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
            from daytona_runner import DaytonaJob, DaytonaRunner

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
            job = DaytonaJob(None, None, None, b"", "physical_ai.main:train", {})
            job._create_sandbox = AsyncMock(return_value=sandbox)
            job._upload_source_package = AsyncMock()
            runner = object.__new__(DaytonaRunner)
            runner._max_concurrent_jobs = 1
            runner.create_job = AsyncMock(return_value=job)
            runner._daytona = SimpleNamespace(close=close)
            with (
                patch("daytona_runner.DaytonaRunner", return_value=runner),
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
