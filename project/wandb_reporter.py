"""Report a remote phy_rl job to W&B from the local controller."""

# Reporting is best effort; transport and filesystem errors must not mask training outcomes.
# ruff: noqa: BLE001

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path, PurePosixPath

from daytona.v0 import AsyncSandbox
from daytona.v0.common.errors import DaytonaNotFoundError

from daytona_runner import DaytonaJob, DaytonaJobStatus

logger = logging.getLogger(__name__)

# The worker/controller file contract. No worker modules are imported on the controller.
RESULT_FILES = (
    "requested_config.json",
    "resolved_config.json",
    "attempt.json",
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
    "error.txt",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class WandbReporter:
    def __init__(
        self,
        config: dict,
        *,
        output: Path = Path("runs/daytona"),
        project: str | None = None,
        entity: str | None = None,
    ) -> None:
        self.config = dict(config)
        self.output = output
        self.project = project
        self.entity = entity
        self.run = None
        self.attempt_id = None
        self.attempt_directory = None
        self.last_env_steps = 0
        self.directory = None
        self.processed_lines = 0
        self.evaluations = set()
        self.errors = []
        self.latest_progress = None
        self.last_metric_at = 0.0
        self.last_progress_log = 0.0

    async def start(self, job: DaytonaJob) -> None:
        import wandb

        self.directory = self.output / job.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.run = await asyncio.to_thread(
            wandb.init,
            project=self.project or "daytona-experiment-runner",
            entity=self.entity,
            id=hashlib.sha256(job.id.encode()).hexdigest()[:24],
            name=job.id,
            resume="allow",
            reinit="create_new",
            mode="online",
            dir=str(self.directory.resolve()),
            config={**self.config, "experiment_id": job.id},
            # Automatic system metrics would describe the controller, not the GPU worker.
            settings=wandb.Settings(
                disable_git=True,
                save_code=False,
                console="off",
                x_disable_stats=True,
            ),
        )
        self.run.define_metric("env_steps")
        self.run.define_metric("*", step_metric="env_steps")
        self.run.summary.update({"job_status": "pending", "tracking_location": "controller"})
        write_json(
            self.directory / "wandb.json",
            {
                "id": self.run.id,
                "project": self.run.project,
                "entity": self.run.entity,
                "url": self.run.url,
                "experiment_id": job.id,
            },
        )
        logger.info("Job %s | W&B: %s", job.id[:8], self.run.url)

    def _attempt(self, job: DaytonaJob) -> tuple[Path, str]:
        assert self.directory is not None and self.run is not None
        assert job.attempt_id is not None
        if self.attempt_id != job.attempt_id:
            directory = self.directory / job.attempt_id
            directory.mkdir(parents=True, exist_ok=True)
            self.attempt_id = job.attempt_id
            self.attempt_directory = directory
            self.last_env_steps = 0
            self.processed_lines = 0
            self.latest_progress = None
            self.last_metric_at = 0.0
            self.last_progress_log = 0.0
            self.evaluations = set()
            self.run.summary.update({"attempt_id": job.attempt_id, "job_status": "running"})
        output = PurePosixPath(self.config.get("output", "runs"))
        if not output.is_absolute():
            output = PurePosixPath(f"/run/{job.id}/code") / output
        assert self.attempt_directory is not None
        return self.attempt_directory, str(output / job.attempt_id)

    def _log(self, env_steps: int, values: dict) -> None:
        self.last_env_steps = int(env_steps)
        self.run.log({**values, "env_steps": int(env_steps), "attempt_id": self.attempt_id})

    def _evaluation(self, phase: str, result: dict) -> None:
        values = {
            f"heldout/{phase}/{key}": result[key]
            for key in ("mean_return", "std_return", "mean_length")
        }
        values.update(
            {
                f"heldout/{phase}/task/{name}": value
                for name, value in result.get("task_metrics", {}).items()
            }
        )
        if "balance" in result:
            balance = result["balance"]
            values[f"heldout/{phase}/balance_success_rate"] = (
                balance["success_count"] / balance["episode_count"]
            )
        self._log(result["training_step"], values)

    def _upload_results(self, directory: Path, summary: dict | None) -> None:
        import wandb

        if summary is not None:
            self.run.summary.update(summary)
        media = {}
        for name in ("learning_curves.png", "policy_comparison.png"):
            path = directory / name
            if path.is_file():
                media[f"plots/{path.stem}"] = wandb.Image(str(path))
        video = directory / "policy_comparison.mp4"
        if video.is_file():
            media["policy/comparison"] = wandb.Video(str(video), format="mp4")
        if media:
            step = summary["actual_steps"] if summary is not None else self.last_env_steps
            self._log(step, media)
        for name in RESULT_FILES:
            path = directory / name
            if path.is_file():
                self.run.save(str(path), base_path=str(directory.parent), policy="now")

    async def _download(self, sandbox: AsyncSandbox, path: str) -> bytes | None:
        try:
            # The SDK accepts its timeout as a positional argument.
            return await sandbox.fs.download_file(path, 15)
        except DaytonaNotFoundError:
            return None

    async def poll(self, job: DaytonaJob, sandbox: AsyncSandbox) -> None:
        directory, remote = self._attempt(job)
        content = await self._download(sandbox, f"{remote}/metrics.jsonl")
        if content is not None:
            # A concurrent append can leave the last line incomplete. Read it next time.
            complete, separator, _ = content.rpartition(b"\n")
            lines = complete.splitlines() if separator else []
            (directory / "metrics.jsonl").write_bytes(complete + separator)
            for line in lines[self.processed_lines :]:
                row = json.loads(line)
                self._log(
                    row["step"],
                    {
                        **row["metrics"],
                        "attempt/elapsed_seconds": row["elapsed_seconds"],
                    },
                )
                self.processed_lines += 1
                self.latest_progress = row
                self.last_metric_at = time.monotonic()
                reward = row["metrics"].get("eval/episode_reward")
                if reward is not None:
                    logger.info(
                        "Job %s | evaluation reward=%.2f at step=%s",
                        job.id[:8],
                        reward,
                        f"{row['step']:,}",
                    )

        self._log_progress(job)

        for phase in ("before", "after"):
            if phase in self.evaluations:
                continue
            name = f"evaluation_{phase}.json"
            content = await self._download(sandbox, f"{remote}/{name}")
            if content is None:
                continue
            try:
                evaluation = json.loads(content)
            except json.JSONDecodeError:
                continue  # The worker has not finished writing this file yet.
            (directory / name).write_bytes(content)
            self._evaluation(phase, evaluation)
            self.evaluations.add(phase)
            logger.info(
                "Job %s | held-out %s return=%.2f",
                job.id[:8],
                phase,
                evaluation["mean_return"],
            )

    def _log_progress(self, job: DaytonaJob) -> None:
        now = time.monotonic()
        if job.status != DaytonaJobStatus.RUNNING or now - self.last_progress_log < 30:
            return
        self.last_progress_log = now
        if self.latest_progress is None:
            logger.info("Job %s | worker running; awaiting first metric", job.id[:8])
            return
        step = self.latest_progress["step"]
        logger.info(
            "Job %s | training steps=%s/%s (%.1f%%) | last new metric %.0fs ago",
            job.id[:8],
            f"{step:,}",
            f"{self.config['total_steps']:,}",
            100 * step / self.config["total_steps"],
            now - self.last_metric_at,
        )

    def _error(self, message: str, error: Exception) -> None:
        self.errors.append(f"{message}: {error}")
        logger.warning("%s: %s", message, error)

    async def collect(self, job: DaytonaJob, sandbox: AsyncSandbox) -> None:
        directory, remote = self._attempt(job)
        try:
            await self.poll(job, sandbox)
        except Exception as error:
            self._error("Final metric read failed", error)

        # Download before sandbox deletion; upload only our explicit result files.
        for name in RESULT_FILES:
            try:
                content = await self._download(sandbox, f"{remote}/{name}")
                if content is not None:
                    (directory / name).write_bytes(content)
            except Exception as error:
                self._error(f"Could not download {name}", error)
                if job.status == DaytonaJobStatus.PREEMPTED:
                    break  # An evicted sandbox's local files may already be gone.

        write_json(
            directory / "collection.json",
            {
                "attempt_id": job.attempt_id,
                "status": job.status,
                "errors": list(self.errors),
            },
        )
        try:
            self.run.summary.update(
                {
                    "attempt_id": job.attempt_id,
                    "attempt_status": str(job.status).lower(),
                }
            )
            resolved = directory / "resolved_config.json"
            if resolved.is_file():
                self.run.config.update(
                    {"worker": json.loads(resolved.read_text())},
                    allow_val_change=True,
                )
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
            logger.info("Job %s | submitting results and available media to W&B", job.id[:8])
            await asyncio.to_thread(self._upload_results, directory, summary)
        except Exception as error:
            self._error("W&B result upload failed; local files were retained", error)

    async def finish(self, status: DaytonaJobStatus) -> None:
        if self.run is None:
            return
        run, self.run = self.run, None
        logger.info("Job %s | finalizing W&B sync", self.directory.name[:8])
        try:
            run.summary.update({"job_status": str(status).lower(), "reporting_errors": self.errors})
        except Exception as error:
            self._error("W&B summary update failed", error)
        finally:
            try:
                exit_code = 0 if status == DaytonaJobStatus.COMPLETED else 1
                if status == DaytonaJobStatus.CANCELED:
                    exit_code = 130
                await asyncio.to_thread(run.finish, exit_code=exit_code)
            except Exception as error:
                self._error("W&B finalization failed; local files were retained", error)
            assert self.directory is not None
            write_json(
                self.directory / "reporting_status.json",
                {
                    "status": status,
                    "errors": self.errors,
                },
            )
            logger.info(
                "Job %s | %s; reporting errors=%d; files=%s",
                self.directory.name[:8],
                str(status).lower(),
                len(self.errors),
                self.directory,
            )
