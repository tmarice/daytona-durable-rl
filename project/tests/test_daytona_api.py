"""API retry policy without credentials, remote requests, or real backoff waits."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientConnectionError, ClientPayloadError
from daytona.v0.common.errors import (
    DaytonaConnectionError,
    DaytonaConnectionTimeoutError,
    DaytonaError,
    DaytonaSpotEvictedError,
    DaytonaTimeoutError,
)
from daytona_api_client_async.exceptions import ApiException

from daytona_runner import DaytonaAPI


class DaytonaAPITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = SimpleNamespace(
            close=AsyncMock(),
            v0=SimpleNamespace(
                get=AsyncMock(),
                secret=SimpleNamespace(list=AsyncMock()),
            ),
        )
        self.constructor = self.enterContext(
            patch("daytona_runner.AsyncDaytona", return_value=self.client)
        )
        self.api = DaytonaAPI(max_retries=5)

    def test_negative_budget_is_rejected_before_creating_a_client(self) -> None:
        self.constructor.reset_mock()
        with self.assertRaisesRegex(ValueError, "max_retries"):
            DaytonaAPI(max_retries=-1)
        self.constructor.assert_not_called()

    async def test_success_preserves_the_return_value_and_does_not_sleep(self) -> None:
        expected = object()
        self.client.v0.get.return_value = expected
        with patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertIs(await self.api.get_sandbox("worker"), expected)
        self.client.v0.get.assert_awaited_once_with("worker")
        sleep.assert_not_awaited()

    async def test_each_retryable_http_status_recovers_with_sdk_and_raw_api_errors(self) -> None:
        for status in (408, 429, 500, 502, 503, 504):
            for error in (
                DaytonaError("unavailable", status_code=status),
                ApiException(status=status),
            ):
                with self.subTest(status=status, error=type(error).__name__):
                    expected = object()
                    request = self.client.v0.secret.list
                    request.reset_mock()
                    request.side_effect = [error, expected]
                    with (
                        patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep,
                        self.assertLogs("daytona_runner", level="WARNING"),
                    ):
                        self.assertIs(await self.api.list_secrets(cursor="page-2"), expected)
                    self.assertEqual(request.await_count, 2)
                    self.assertTrue(
                        all(c.kwargs == {"cursor": "page-2"} for c in request.await_args_list)
                    )
                    sleep.assert_awaited_once_with(5)

    async def test_connection_errors_and_transport_timeouts_are_retried(self) -> None:
        for error in (
            DaytonaConnectionError("connection lost"),
            DaytonaConnectionTimeoutError("read timed out"),
            ClientConnectionError("connection lost"),
            ClientPayloadError("truncated response"),
            TimeoutError("read timed out"),
        ):
            with self.subTest(error=type(error).__name__):
                request = AsyncMock(side_effect=[error, "ok"])
                with (
                    patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep,
                    self.assertLogs("daytona_runner", level="WARNING"),
                ):
                    self.assertEqual(await self.api._request("test", request), "ok")
                sleep.assert_awaited_once_with(5)
                self.assertEqual(request.await_count, 2)

    async def test_backoff_doubles_and_caps_at_sixty_seconds(self) -> None:
        api = DaytonaAPI(max_retries=6)
        request = AsyncMock(side_effect=[DaytonaError("busy", status_code=503)] * 6 + ["ok"])
        with (
            patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep,
            self.assertLogs("daytona_runner", level="WARNING"),
        ):
            self.assertEqual(await api._request("test", request), "ok")
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [5, 10, 20, 40, 60, 60])
        self.assertEqual(request.await_count, 7)

    async def test_exhaustion_and_zero_budget_reraise_the_final_error(self) -> None:
        for budget in (0, 2, 5):
            with self.subTest(budget=budget):
                api = DaytonaAPI(max_retries=budget)
                errors = [DaytonaError(str(i), status_code=503) for i in range(budget + 1)]
                request = AsyncMock(side_effect=errors)
                with (
                    patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep,
                    patch("daytona_runner.logger.warning"),
                    self.assertRaises(DaytonaError) as caught,
                ):
                    await api._request("test", request)
                self.assertIs(caught.exception, errors[-1])
                self.assertEqual(request.await_count, budget + 1)
                self.assertEqual(sleep.await_count, budget)

    async def test_permanent_errors_and_evictions_propagate_immediately(self) -> None:
        errors = [
            error
            for status in (400, 401, 403, 404, 409, 410, 422, 501)
            for error in (
                DaytonaError("permanent", status_code=status),
                ApiException(status=status),
            )
        ] + [
            DaytonaSpotEvictedError("evicted", status_code=503),
            DaytonaTimeoutError("sandbox did not start"),
            DaytonaError("invalid configuration"),
            RuntimeError("programming error"),
        ]
        for error in errors:
            with self.subTest(
                error=type(error).__name__, status=getattr(error, "status_code", None)
            ):
                request = AsyncMock(side_effect=error)
                with (
                    patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep,
                    self.assertRaises(type(error)) as caught,
                ):
                    await self.api._request("test", request)
                self.assertIs(caught.exception, error)
                request.assert_awaited_once()
                sleep.assert_not_awaited()

    async def test_nonzero_command_exit_code_is_not_an_api_failure(self) -> None:
        response = SimpleNamespace(exit_code=7, result="training failed")
        sandbox = SimpleNamespace(process=SimpleNamespace(exec=AsyncMock(return_value=response)))
        with patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertIs(await self.api.exec(sandbox, "command", timeout=60), response)
        sandbox.process.exec.assert_awaited_once_with("command", timeout=60)
        sleep.assert_not_awaited()

    async def test_cancellation_during_backoff_stops_further_requests(self) -> None:
        sleeping = asyncio.Event()

        async def wait_in_backoff(delay: int) -> None:
            sleeping.set()
            await asyncio.Future()

        self.client.v0.get.side_effect = DaytonaError("busy", status_code=503)
        with (
            patch("daytona_runner.asyncio.sleep", side_effect=wait_in_backoff),
            self.assertLogs("daytona_runner", level="WARNING"),
        ):
            async with asyncio.timeout(5):
                request = asyncio.create_task(self.api.get_sandbox("worker"))
                await sleeping.wait()
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
        self.client.v0.get.assert_awaited_once_with("worker")

    async def test_concurrent_requests_have_independent_budgets(self) -> None:
        api = DaytonaAPI(max_retries=2)
        error = DaytonaError("busy", status_code=503)
        first = AsyncMock(side_effect=[error, error, "first"])
        second = AsyncMock(side_effect=[error, "second"])
        real_sleep = asyncio.sleep

        async def yield_in_backoff(delay: int) -> None:
            await real_sleep(0)

        with (
            patch("daytona_runner.asyncio.sleep", side_effect=yield_in_backoff) as sleep,
            self.assertLogs("daytona_runner", level="WARNING"),
        ):
            results = await asyncio.gather(
                api._request("first", first),
                api._request("second", second),
            )
        self.assertEqual(results, ["first", "second"])
        self.assertEqual((first.await_count, second.await_count), (3, 2))
        self.assertCountEqual([c.args[0] for c in sleep.await_args_list], [5, 5, 10])

    async def test_retry_logs_do_not_expose_request_arguments_or_error_bodies(self) -> None:
        secret = "secret-value-do-not-log"
        request = AsyncMock(side_effect=[ApiException(status=503, reason=secret), None])
        with (
            patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock),
            self.assertLogs("daytona_runner", level="WARNING") as logs,
        ):
            await self.api._request("secret.update", request, secret, value=secret)
        output = "\n".join(logs.output)
        self.assertNotIn(secret, output)
        self.assertIn("secret.update", output)
        self.assertIn("HTTP 503", output)
        self.assertIn("retry 1/5 in 5s", output)

    async def test_client_close_is_not_retried(self) -> None:
        self.client.close.side_effect = TimeoutError("close failed")
        with (
            patch("daytona_runner.asyncio.sleep", new_callable=AsyncMock) as sleep,
            self.assertRaises(TimeoutError),
        ):
            await self.api.close()
        self.client.close.assert_awaited_once()
        sleep.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
