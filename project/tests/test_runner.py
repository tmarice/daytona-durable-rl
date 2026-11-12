"""Batch failure isolation and cancellation without creating remote sandboxes."""

import asyncio
import shlex
import sys
import unittest
from contextlib import aclosing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import daytona_runner as runner_module


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.runner = object.__new__(runner_module.DaytonaRunner)
        self.runner._max_concurrent_jobs = 1
        self.runner._jobs = {}

    @staticmethod
    def make_api(client: object = None, max_retries: int = 0) -> runner_module.DaytonaAPI:
        with patch.object(runner_module, "AsyncDaytona", return_value=client):
            return runner_module.DaytonaAPI(max_retries)

    @staticmethod
    def make_sandbox() -> SimpleNamespace:
        return SimpleNamespace(
            state=runner_module.SandboxState.STARTED,
            spot_evicted_at=None,
            refresh_data=AsyncMock(),
            delete=AsyncMock(),
            process=SimpleNamespace(
                create_session=AsyncMock(),
                execute_session_command=AsyncMock(return_value=SimpleNamespace(cmd_id="command")),
                get_session_command=AsyncMock(return_value=SimpleNamespace(exit_code=0)),
                get_session_command_logs=AsyncMock(return_value=SimpleNamespace(output="done")),
            ),
        )

    @staticmethod
    def make_job(sandbox: SimpleNamespace, config: dict | None = None) -> runner_module.DaytonaJob:
        """A job that receives the given sandbox without calling the SDK."""
        job = runner_module.DaytonaJob(
            RunnerTests.make_api(),
            None,
            None,
            b"",
            "training:run",
            config or {},
        )
        job._create_sandbox = AsyncMock(return_value=sandbox)
        job._upload_source_package = AsyncMock()
        return job

    @staticmethod
    def make_sdk_job(**calls: AsyncMock) -> runner_module.DaytonaJob:
        """A job whose sandbox creation and lookup reach the given SDK calls."""
        job = runner_module.DaytonaJob(
            RunnerTests.make_api(SimpleNamespace(v0=SimpleNamespace(**calls))),
            SimpleNamespace(name="gpu-snapshot"),
            SimpleNamespace(id="checkpoint-volume"),
            b"",
            "training:run",
            {},
        )
        job._upload_source_package = AsyncMock()
        return job

    async def test_spot_jobs_retry_eviction_stops_and_deleted_sandboxes(self) -> None:
        cases = [
            ("spot eviction", runner_module.SandboxState.STARTED),
            ("stopping", runner_module.SandboxState.STOPPING),
            ("stopped", runner_module.SandboxState.STOPPED),
            ("destroying", runner_module.SandboxState.DESTROYING),
            ("destroyed", runner_module.SandboxState.DESTROYED),
            ("missing", runner_module.SandboxState.DESTROYED),
        ]
        for cause, state in cases:
            with self.subTest(cause=cause):
                interrupted, replacement = self.make_sandbox(), self.make_sandbox()
                interrupted.state = state
                if cause == "spot eviction":
                    interrupted.spot_evicted_at = "2026-11-11T12:00:00Z"
                if cause == "missing":
                    interrupted.refresh_data.side_effect = runner_module.DaytonaNotFoundError(
                        "already deleted"
                    )
                create = AsyncMock(side_effect=[interrupted, replacement])
                job = self.make_sdk_job(create=create)

                async def delete_interrupted(
                    wait: bool, job: runner_module.DaytonaJob = job, cause: str = cause
                ) -> None:
                    self.assertEqual(job.status, runner_module.DaytonaJobStatus.PREEMPTED)
                    if cause == "missing":
                        raise runner_module.DaytonaNotFoundError("already deleted")

                interrupted.delete.side_effect = delete_interrupted
                with self.assertLogs(runner_module.logger, level="WARNING"):
                    async with asyncio.timeout(5):
                        results = [result async for result in self.runner.run([job])]

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].job_id, job.id)
                self.assertEqual(results[0].status, runner_module.DaytonaJobStatus.COMPLETED)
                self.assertEqual(results[0].logs, "done")
                self.assertEqual(create.await_count, 2)
                for call in create.await_args_list:
                    params = call.args[0]
                    self.assertIs(params.spot, True)
                    self.assertEqual(params.snapshot, "gpu-snapshot")
                    self.assertEqual(params.volumes[0].volume_id, "checkpoint-volume")
                    self.assertTrue(params.name.startswith(f"job-{job.id}-"))
                self.assertNotEqual(*(call.args[0].name for call in create.await_args_list))
                interrupted.process.get_session_command.assert_not_awaited()
                interrupted.delete.assert_awaited_once_with(wait=False)
                replacement.delete.assert_awaited_once_with(wait=False)
                self.assertEqual(job.status, runner_module.DaytonaJobStatus.COMPLETED)

    async def test_eviction_while_the_sandbox_is_created_is_retried(self) -> None:
        sandbox = self.make_sandbox()
        evicted = runner_module.DaytonaSpotEvictedError("evicted while starting")
        create = AsyncMock(side_effect=[evicted, sandbox])
        get = AsyncMock(side_effect=runner_module.DaytonaNotFoundError("already destroyed"))
        job = self.make_sdk_job(create=create, get=get)

        with self.assertLogs(runner_module.logger, level="WARNING"):
            async with asyncio.timeout(5):
                results = [result async for result in self.runner.run([job])]

        self.assertEqual(
            [result.status for result in results], [runner_module.DaytonaJobStatus.COMPLETED]
        )
        self.assertEqual(create.await_count, 2)
        get.assert_awaited_once_with(create.await_args_list[0].args[0].name)
        sandbox.delete.assert_awaited_once_with(wait=False)

    async def test_eviction_replacement_retries_capacity_failure_in_the_same_new_attempt(
        self,
    ) -> None:
        evicted, replacement = self.make_sandbox(), self.make_sandbox()
        evicted.spot_evicted_at = "2026-11-12T09:37:00Z"
        create = AsyncMock(
            side_effect=[
                evicted,
                runner_module.DaytonaError("no capacity", status_code=503),
                replacement,
            ]
        )
        job = self.make_sdk_job(create=create)
        job._api._max_retries = 2

        with (
            patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep,
            self.assertLogs(runner_module.logger, level="WARNING"),
        ):
            results = [result async for result in self.runner.run([job])]

        self.assertEqual(results[0].status, runner_module.DaytonaJobStatus.COMPLETED)
        names = [call.args[0].name for call in create.await_args_list]
        self.assertNotEqual(names[0], names[1])
        self.assertEqual(names[1], names[2])
        sleep.assert_awaited_once_with(5)
        evicted.delete.assert_awaited_once_with(wait=False)
        replacement.delete.assert_awaited_once_with(wait=False)

    async def test_preparation_and_launch_retries_preserve_the_attempt_and_checkpoint_path(
        self,
    ) -> None:
        sandbox = self.make_sandbox()
        sandbox.fs = SimpleNamespace(upload_file=AsyncMock())
        sandbox.process.exec = AsyncMock(
            side_effect=[
                runner_module.DaytonaError("bad gateway", status_code=502),
                SimpleNamespace(exit_code=0),
            ]
        )
        sandbox.process.execute_session_command.side_effect = [
            runner_module.DaytonaError("bad gateway", status_code=502),
            SimpleNamespace(cmd_id="command"),
        ]
        create = AsyncMock(return_value=sandbox)
        job = self.make_sdk_job(create=create)
        job._api._max_retries = 2
        del job._upload_source_package  # Exercise the real preparation calls.

        with (
            patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep,
            self.assertLogs(runner_module.logger, level="WARNING"),
        ):
            result = await job.run_attempt()

        self.assertEqual(result.status, runner_module.DaytonaJobStatus.COMPLETED)
        create.assert_awaited_once()
        sandbox.fs.upload_file.assert_awaited_once()
        sandbox.process.create_session.assert_awaited_once_with(job.id)
        preparation = sandbox.process.exec.await_args_list
        self.assertEqual(preparation[0], preparation[1])
        launches = sandbox.process.execute_session_command.await_args_list
        self.assertEqual(launches[0], launches[1])
        command = launches[0].args[1].command
        self.assertIn(job.attempt_id, command)
        self.assertIn(job._checkpoint_dir, command)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [5, 5])
        sandbox.delete.assert_awaited_once_with(wait=False)

    async def test_sandbox_is_deleted_by_name_when_creation_times_out(self) -> None:
        orphan = self.make_sandbox()
        create = AsyncMock(side_effect=runner_module.DaytonaError("sandbox start timed out"))
        get = AsyncMock(return_value=orphan)
        job = self.make_sdk_job(create=create, get=get)

        with (
            self.assertLogs(runner_module.logger, level="ERROR"),
            self.assertRaises(runner_module.DaytonaError),
        ):
            await job.run_attempt()

        self.assertEqual(job.status, runner_module.DaytonaJobStatus.FAILED)
        get.assert_awaited_once_with(create.await_args.args[0].name)
        orphan.delete.assert_awaited_once_with(wait=False)

    async def test_cancellation_waits_for_inflight_creation_then_deletes_the_sandbox(self) -> None:
        creating = asyncio.Event()
        release = asyncio.Event()
        orphan = self.make_sandbox()

        async def create_after_release(params: object) -> SimpleNamespace:
            creating.set()
            await release.wait()
            return orphan

        create = AsyncMock(side_effect=create_after_release)
        get = AsyncMock(return_value=orphan)
        job = self.make_sdk_job(create=create, get=get)

        async with asyncio.timeout(5):
            attempt = asyncio.create_task(job.run_attempt())
            await creating.wait()
            attempt.cancel()
            await asyncio.sleep(0)
            self.assertFalse(attempt.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await attempt

        self.assertEqual(job.status, runner_module.DaytonaJobStatus.CANCELED)
        get.assert_awaited_once_with(create.await_args.args[0].name)
        orphan.delete.assert_awaited_once_with(wait=False)

    async def test_cancellation_during_creation_backoff_does_not_create_another_sandbox(
        self,
    ) -> None:
        sleeping = asyncio.Event()

        async def wait_in_backoff(delay: int) -> None:
            sleeping.set()
            await asyncio.Future()

        create = AsyncMock(side_effect=runner_module.DaytonaError("no capacity", status_code=503))
        get = AsyncMock(side_effect=runner_module.DaytonaNotFoundError("nothing created"))
        job = self.make_sdk_job(create=create, get=get)
        job._api._max_retries = 5
        with (
            patch("daytona_runner.asyncio.sleep", side_effect=wait_in_backoff),
            self.assertLogs(runner_module.logger, level="WARNING"),
        ):
            async with asyncio.timeout(5):
                attempt = asyncio.create_task(job.run_attempt())
                await sleeping.wait()
                attempt.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await attempt

        self.assertEqual(job.status, runner_module.DaytonaJobStatus.CANCELED)
        create.assert_awaited_once()
        get.assert_awaited_once_with(create.await_args.args[0].name)
        job._upload_source_package.assert_not_awaited()

    async def test_deletion_failure_is_logged_without_replacing_the_result(self) -> None:
        sandbox = self.make_sandbox()
        sandbox.delete.side_effect = TimeoutError("deletion timed out")
        job = self.make_job(sandbox)

        with self.assertLogs(runner_module.logger, level="ERROR"):
            result = await job.run_attempt()

        self.assertEqual(result.status, runner_module.DaytonaJobStatus.COMPLETED)
        self.assertEqual(job.status, runner_module.DaytonaJobStatus.COMPLETED)

        sandbox.delete.side_effect = runner_module.DaytonaNotFoundError("already deleted")
        with self.assertNoLogs(runner_module.logger, level="ERROR"):
            await job.run_attempt()

    async def test_stop_during_process_requests_is_preemption(self) -> None:
        for method in ("get_session_command", "get_session_command_logs"):
            with self.subTest(method=method):
                sandbox = self.make_sandbox()
                job = self.make_job(sandbox)

                async def stop_during_request(
                    *args: object, sandbox: SimpleNamespace = sandbox
                ) -> None:
                    sandbox.state = runner_module.SandboxState.DESTROYING
                    raise runner_module.DaytonaError("sandbox stopped during request")

                getattr(sandbox.process, method).side_effect = stop_during_request
                with self.assertRaises(runner_module.DaytonaSpotEvictedError):
                    await job.run_attempt()
                self.assertEqual(job.status, runner_module.DaytonaJobStatus.PREEMPTED)
                self.assertEqual(sandbox.refresh_data.await_count, 2)
                sandbox.delete.assert_awaited_once_with(wait=False)

    async def test_process_errors_in_running_sandbox_are_not_preemption(self) -> None:
        for method in ("get_session_command", "get_session_command_logs"):
            with self.subTest(method=method):
                sandbox = self.make_sandbox()
                job = self.make_job(sandbox)
                error = runner_module.DaytonaError("unrelated process error")
                getattr(sandbox.process, method).side_effect = error

                with self.assertRaises(runner_module.DaytonaError) as caught:
                    await job.run_attempt()
                self.assertIs(caught.exception, error)
                self.assertEqual(job.status, runner_module.DaytonaJobStatus.FAILED)
                sandbox.delete.assert_awaited_once_with(wait=False)

    async def test_training_failure_is_not_preemption(self) -> None:
        sandbox = self.make_sandbox()
        sandbox.process.get_session_command.return_value.exit_code = 1
        job = self.make_job(sandbox)
        result = await job.run_attempt()
        self.assertEqual(result.status, runner_module.DaytonaJobStatus.FAILED)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(job.status, runner_module.DaytonaJobStatus.FAILED)

    async def test_reporting_hooks_poll_and_collect_before_deletion_for_each_outcome(self) -> None:
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code):
                sandbox = self.make_sandbox()
                sandbox.process.get_session_command.return_value.exit_code = exit_code
                job = self.make_job(sandbox)
                events = []

                async def poll(current, worker, job=job, sandbox=sandbox, events=events):
                    self.assertIs(current, job)
                    self.assertIs(worker, sandbox)
                    self.assertEqual(current.status, runner_module.DaytonaJobStatus.RUNNING)
                    events.append("poll")

                async def collect(current, worker, events=events, sandbox=sandbox):
                    events.append(str(current.status))
                    sandbox.delete.assert_not_awaited()

                job._on_poll = poll
                job._on_attempt_end = collect
                sandbox.delete.side_effect = lambda events=events, **kwargs: events.append(
                    "deleted"
                )
                result = await job.run_attempt()
                self.assertEqual(events, ["poll", str(result.status), "deleted"])

    async def test_reporting_failure_does_not_replace_training_result_or_skip_deletion(
        self,
    ) -> None:
        sandbox = self.make_sandbox()
        job = self.make_job(sandbox)
        job._on_poll = AsyncMock(side_effect=RuntimeError("poll failed"))
        job._on_attempt_end = AsyncMock(side_effect=RuntimeError("collection failed"))
        with self.assertLogs(runner_module.logger, level="ERROR"):
            result = await job.run_attempt()
        self.assertEqual(result.status, runner_module.DaytonaJobStatus.COMPLETED)
        job._on_attempt_end.assert_awaited_once_with(job, sandbox)
        sandbox.delete.assert_awaited_once()

    async def test_collection_runs_on_exception_eviction_and_cancellation(self) -> None:
        for error, status in (
            (RuntimeError("process failed"), runner_module.DaytonaJobStatus.FAILED),
            (
                runner_module.DaytonaSpotEvictedError("evicted"),
                runner_module.DaytonaJobStatus.PREEMPTED,
            ),
            (asyncio.CancelledError(), runner_module.DaytonaJobStatus.CANCELED),
        ):
            with self.subTest(status=status):
                sandbox = self.make_sandbox()
                sandbox.process.get_session_command.side_effect = error
                job = self.make_job(sandbox)

                async def collect(current, worker, status=status, sandbox=sandbox):
                    self.assertEqual(current.status, status)
                    sandbox.delete.assert_not_awaited()

                job._on_attempt_end = AsyncMock(side_effect=collect)
                with self.assertRaises(type(error)):
                    await job.run_attempt()
                job._on_attempt_end.assert_awaited_once_with(job, sandbox)
                sandbox.delete.assert_awaited_once()

    async def test_launch_contract_keeps_experiment_and_changes_attempt_on_retry(self) -> None:
        config = {"learning_rate": 0.001, "task": "quoted ' value"}
        sandbox = self.make_sandbox()
        job = self.make_job(sandbox, config)
        entrypoint = Mock()
        with patch.dict(sys.modules, {"training": SimpleNamespace(run=entrypoint)}):
            for _ in range(2):
                await job.run_attempt()
                request = sandbox.process.execute_session_command.await_args.args[1]
                exec(shlex.split(request.command)[-1])  # noqa: S102 -- exercise our launch command
        first, second = entrypoint.call_args_list
        self.assertEqual(first.args, (config,))
        self.assertEqual(second.args, (config,))
        self.assertEqual(first.kwargs["experiment_id"], job.id)
        self.assertEqual(second.kwargs["experiment_id"], job.id)
        self.assertNotEqual(first.kwargs["attempt_id"], second.kwargs["attempt_id"])
        self.assertEqual(first.kwargs["checkpoint_dir"], f"/data/jobs/{job.id}/checkpoints")
        self.assertEqual(first.kwargs["checkpoint_dir"], second.kwargs["checkpoint_dir"])
        self.assertNotIn("checkpoint_dir", config)
        self.assertEqual(sandbox.delete.await_count, 2)

    async def test_runtime_environment_reaches_sandbox_without_entering_command(self) -> None:
        sandbox = self.make_sandbox()
        create = AsyncMock(return_value=sandbox)
        client = SimpleNamespace(v0=SimpleNamespace(create=create))
        with (
            TemporaryDirectory() as root,
            patch.object(runner_module, "AsyncDaytona", return_value=client),
        ):
            runner = runner_module.DaytonaRunner(
                work_dir=root,
                entrypoint="training:run",
                resources={"gpu": 1},
                volume_name="test",
                env_vars={"WANDB_API_KEY": "runtime-only-key"},
            )
            runner._get_or_create_snapshot = AsyncMock(return_value=SimpleNamespace(name="cached"))
            runner._get_or_create_volume = AsyncMock(return_value=SimpleNamespace(id="volume"))
            job = await runner.create_job({"wandb": True})
            await job._create_sandbox("test")
        self.assertEqual(create.await_args.args[0].env_vars, {"WANDB_API_KEY": "runtime-only-key"})
        self.assertEqual(create.await_args.args[0].secrets, {})
        self.assertNotIn("runtime-only-key", job._build_command())

    async def test_runner_configures_one_shared_adapter_for_its_jobs(self) -> None:
        client = SimpleNamespace(close=AsyncMock())
        for supplied, expected in (({}, 5), ({"max_retries": 0}, 0), ({"max_retries": 2}, 2)):
            with (
                self.subTest(supplied=supplied),
                TemporaryDirectory() as root,
                patch.object(runner_module, "AsyncDaytona", return_value=client),
            ):
                runner = runner_module.DaytonaRunner(
                    work_dir=root,
                    entrypoint="training:run",
                    resources={},
                    volume_name="test",
                    **supplied,
                )
                runner._get_or_create_snapshot = AsyncMock(
                    return_value=SimpleNamespace(name="cached")
                )
                runner._get_or_create_volume = AsyncMock(return_value=SimpleNamespace(id="volume"))
                first = await runner.create_job({"seed": 11})
                second = await runner.create_job({"seed": 22})
                self.assertIs(first._api, runner._api)
                self.assertIs(second._api, runner._api)
                self.assertEqual(runner._api._max_retries, expected)
                await runner.close()
        self.assertEqual(client.close.await_count, 3)

    async def test_secrets_are_updated_when_they_exist_and_created_otherwise(self) -> None:
        pages = [
            SimpleNamespace(
                items=[SimpleNamespace(name="unrelated", id="unrelated-id")], next_cursor="next"
            ),
            SimpleNamespace(
                items=[SimpleNamespace(name="daytona-runner-WANDB_API_KEY", id="wandb-id")],
                next_cursor=None,
            ),
        ]
        secret = SimpleNamespace(
            list=AsyncMock(side_effect=pages), update=AsyncMock(), create=AsyncMock()
        )
        self.runner._api = self.make_api(SimpleNamespace(v0=SimpleNamespace(secret=secret)))

        names = await self.runner._create_or_update_secrets(
            {"WANDB_API_KEY": "rotated", "HF_TOKEN": "new"}
        )

        self.assertEqual(
            names,
            {
                "WANDB_API_KEY": "daytona-runner-WANDB_API_KEY",
                "HF_TOKEN": "daytona-runner-HF_TOKEN",
            },
        )
        self.assertEqual(
            [call.kwargs["cursor"] for call in secret.list.await_args_list], [None, "next"]
        )
        secret.update.assert_awaited_once()
        secret_id, update = secret.update.await_args.args
        self.assertEqual((secret_id, update.value), ("wandb-id", "rotated"))
        secret.create.assert_awaited_once()
        (create,) = secret.create.await_args.args
        self.assertEqual((create.name, create.value), ("daytona-runner-HF_TOKEN", "new"))

    async def test_snapshot_cache_separates_resources_and_dependency_recipes(self) -> None:
        snapshots = SimpleNamespace(get=AsyncMock(return_value="cached snapshot"))
        self.runner._api = self.make_api(SimpleNamespace(v0=SimpleNamespace(snapshot=snapshots)))
        with TemporaryDirectory() as root:
            self.runner._workdir = Path(root)
            self.runner._resources = {"cpu": 4, "memory": 8, "disk": 10}
            (Path(root) / "uv.lock").write_text("unchanged lock")
            (Path(root) / "requirements.txt").write_text("example-package==1\n")
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
            SimpleNamespace(run_attempt=AsyncMock(side_effect=preparation_error)),
            SimpleNamespace(run_attempt=AsyncMock(side_effect=monitoring_error)),
            SimpleNamespace(run_attempt=AsyncMock(return_value=expected)),
        ]
        results = []

        with self.assertRaises(ExceptionGroup) as caught:
            async for result in self.runner.run(jobs):
                results.append(result)

        self.assertEqual(results, [expected])
        self.assertCountEqual(caught.exception.exceptions, [preparation_error, monitoring_error])
        for job in jobs:
            job.run_attempt.assert_awaited_once()

    async def test_evicted_job_keeps_its_slot_ahead_of_queued_jobs(self) -> None:
        order = []
        completed = runner_module.DaytonaJobResult(
            job_id="recovering", status=runner_module.DaytonaJobStatus.COMPLETED, exit_code=0
        )

        async def recovering_attempt() -> runner_module.DaytonaJobResult:
            order.append("recovering")
            if len(order) == 1:
                # Let the fresh job queue for the occupied slot before eviction.
                await asyncio.sleep(0)
                raise runner_module.DaytonaSpotEvictedError("evicted")
            return completed

        async def fresh_attempt() -> runner_module.DaytonaJobResult:
            order.append("fresh")
            return runner_module.DaytonaJobResult(
                job_id="fresh", status=runner_module.DaytonaJobStatus.COMPLETED, exit_code=0
            )

        jobs = [
            SimpleNamespace(id="recovering", run_attempt=recovering_attempt),
            SimpleNamespace(id="fresh", run_attempt=fresh_attempt),
        ]
        with self.assertLogs("daytona_runner", level="WARNING"):
            results = [result async for result in self.runner.run(jobs)]

        self.assertEqual(order, ["recovering", "recovering", "fresh"])
        self.assertCountEqual([result.job_id for result in results], ["recovering", "fresh"])

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
            SimpleNamespace(run_attempt=AsyncMock(return_value=completed)),
            SimpleNamespace(run_attempt=AsyncMock(side_effect=wait_for_release)),
        ]

        async with asyncio.timeout(5), aclosing(self.runner.run(jobs)) as results:
            self.assertIs(await anext(results), completed)
            release.set()
            self.assertIs(await anext(results), failed)
            with self.assertRaises(StopAsyncIteration):
                await anext(results)

    async def test_caller_cancellation_stops_running_and_queued_tasks(self) -> None:
        started = asyncio.Event()

        async def wait_until_cancelled(*args: object) -> None:
            started.set()
            await asyncio.Future()

        sandbox = self.make_sandbox()
        sandbox.process.get_session_command.side_effect = wait_until_cancelled
        running = self.make_job(sandbox)
        queued = SimpleNamespace(run_attempt=AsyncMock())

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

        self.assertEqual(running.status, runner_module.DaytonaJobStatus.CANCELED)
        sandbox.delete.assert_awaited_once_with(wait=False)
        queued.run_attempt.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
