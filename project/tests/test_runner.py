"""Batch failure isolation and cancellation without creating remote sandboxes."""

import asyncio
import importlib.util
import sys
import unittest
from contextlib import aclosing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

# The orchestration module currently shares its name with the training package.
spec = importlib.util.spec_from_file_location(
    "orchestration_runner", Path(__file__).resolve().parents[1] / "daytona_runner.py"
)
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.runner = object.__new__(runner_module.DaytonaRunner)
        self.runner._max_concurrent_jobs = 1

    async def test_job_exceptions_are_grouped_after_other_jobs_finish(self) -> None:
        preparation_error = RuntimeError("upload failed")
        monitoring_error = TimeoutError("command lookup timed out")
        expected = runner_module.DaytonaJobResult(
            job_id="successful", status=runner_module.DaytonaJobStatus.COMPLETED, exit_code=0
        )
        jobs = [
            SimpleNamespace(start=AsyncMock(side_effect=preparation_error), wait=AsyncMock()),
            SimpleNamespace(start=AsyncMock(), wait=AsyncMock(side_effect=monitoring_error)),
            SimpleNamespace(start=AsyncMock(), wait=AsyncMock(return_value=expected)),
        ]
        results = []

        with self.assertRaises(ExceptionGroup) as caught:
            async for result in self.runner.run(jobs):
                results.append(result)

        self.assertEqual(results, [expected])
        self.assertCountEqual(caught.exception.exceptions, [preparation_error, monitoring_error])
        for job in jobs:
            job.start.assert_awaited_once()
        jobs[0].wait.assert_not_awaited()
        jobs[1].wait.assert_awaited_once()
        jobs[2].wait.assert_awaited_once()

    async def test_results_stream_before_the_batch_finishes(self) -> None:
        self.runner._max_concurrent_jobs = 2
        release = asyncio.Event()
        completed = runner_module.DaytonaJobResult(
            job_id="completed", status=runner_module.DaytonaJobStatus.COMPLETED, exit_code=0
        )
        failed = runner_module.DaytonaJobResult(
            job_id="failed", status=runner_module.DaytonaJobStatus.FAILED, exit_code=1
        )

        async def wait_for_release() -> runner_module.DaytonaJobResult:
            await release.wait()
            return failed

        jobs = [
            SimpleNamespace(start=AsyncMock(), wait=AsyncMock(return_value=completed)),
            SimpleNamespace(start=AsyncMock(), wait=AsyncMock(side_effect=wait_for_release)),
        ]

        async with asyncio.timeout(5), aclosing(self.runner.run(jobs)) as results:
            self.assertIs(await anext(results), completed)
            release.set()
            self.assertIs(await anext(results), failed)
            with self.assertRaises(StopAsyncIteration):
                await anext(results)

    async def test_caller_cancellation_stops_running_and_queued_tasks(self) -> None:
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def wait_until_cancelled() -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        running = SimpleNamespace(
            start=AsyncMock(), wait=AsyncMock(side_effect=wait_until_cancelled)
        )
        queued = SimpleNamespace(start=AsyncMock(), wait=AsyncMock())

        async def consume() -> None:
            async for _ in self.runner.run([running, queued]):
                pass

        async with asyncio.timeout(5):
            consumer = asyncio.create_task(consume())
            try:
                await started.wait()
                consumer.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await consumer
            finally:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)

        self.assertTrue(stopped.is_set())
        queued.start.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
