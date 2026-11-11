"""Batch failure isolation and cancellation without creating remote sandboxes."""

import asyncio
import unittest
from contextlib import aclosing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

import daytona_runner as runner_module


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.runner = object.__new__(runner_module.DaytonaRunner)
        self.runner._max_concurrent_jobs = 1

    async def test_cleanup_logs_deletion_failures_and_tolerates_an_absent_sandbox(self) -> None:
        job = runner_module.DaytonaJob(None, None, None, b"", "main:train", {})
        error = TimeoutError("deletion timed out")
        sandbox = SimpleNamespace(delete=AsyncMock(side_effect=error))
        job._sandbox = sandbox
        with (
            self.assertLogs(runner_module.logger, level="ERROR"),
            self.assertRaises(TimeoutError) as caught,
        ):
            await job.cleanup()
        self.assertIs(caught.exception, error)
        self.assertIs(job._sandbox, sandbox)

        sandbox.delete.side_effect = runner_module.DaytonaNotFoundError("already deleted")
        await job.cleanup()
        self.assertIsNone(job._sandbox)
        await job.cleanup()
        self.assertEqual(sandbox.delete.await_count, 2)

    async def test_snapshot_cache_separates_resources_and_dependency_recipes(self) -> None:
        snapshots = SimpleNamespace(get=AsyncMock(return_value="cached snapshot"))
        self.runner._daytona = SimpleNamespace(v0=SimpleNamespace(snapshot=snapshots))
        with TemporaryDirectory() as root:
            self.runner._workdir = Path(root)
            self.runner._resources = {"cpu": 4, "memory": 8, "disk": 10}
            (Path(root) / "uv.lock").write_text("unchanged lock")
            pyproject = Path(root) / "pyproject.toml"
            pyproject.write_text('[project]\ndependencies = ["example-package==1"]\n')
            await self.runner._get_or_create_snapshot()
            first = snapshots.get.await_args.args[0]
            await self.runner._get_or_create_snapshot()
            self.assertEqual(snapshots.get.await_args.args[0], first)
            self.runner._resources["cpu"] = 8
            await self.runner._get_or_create_snapshot()
            second = snapshots.get.await_args.args[0]
            self.assertNotEqual(second, first)
            pyproject.write_text('[project]\ndependencies = ["example-package==2"]\n')
            await self.runner._get_or_create_snapshot()
            self.assertNotEqual(snapshots.get.await_args.args[0], second)

    async def test_job_exceptions_are_grouped_after_other_jobs_finish(self) -> None:
        preparation_error = RuntimeError("upload failed")
        monitoring_error = TimeoutError("command lookup timed out")
        expected = runner_module.DaytonaJobResult(
            job_id="successful", status=runner_module.DaytonaJobStatus.COMPLETED, exit_code=0
        )
        jobs = [
            SimpleNamespace(
                start=AsyncMock(side_effect=preparation_error),
                wait=AsyncMock(),
                cleanup=AsyncMock(),
            ),
            SimpleNamespace(
                start=AsyncMock(), wait=AsyncMock(side_effect=monitoring_error), cleanup=AsyncMock()
            ),
            SimpleNamespace(
                start=AsyncMock(), wait=AsyncMock(return_value=expected), cleanup=AsyncMock()
            ),
        ]
        results = []

        with self.assertRaises(ExceptionGroup) as caught:
            async for result in self.runner.run(jobs):
                results.append(result)

        self.assertEqual(results, [expected])
        self.assertCountEqual(caught.exception.exceptions, [preparation_error, monitoring_error])
        for job in jobs:
            job.start.assert_awaited_once()
            job.cleanup.assert_awaited_once_with(wait=False)
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
            SimpleNamespace(
                start=AsyncMock(), wait=AsyncMock(return_value=completed), cleanup=AsyncMock()
            ),
            SimpleNamespace(
                start=AsyncMock(), wait=AsyncMock(side_effect=wait_for_release), cleanup=AsyncMock()
            ),
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
            start=AsyncMock(), wait=AsyncMock(side_effect=wait_until_cancelled), cleanup=AsyncMock()
        )
        queued = SimpleNamespace(start=AsyncMock(), wait=AsyncMock(), cleanup=AsyncMock())

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
        running.cleanup.assert_awaited_once_with(wait=False)
        queued.start.assert_not_awaited()
        queued.cleanup.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
