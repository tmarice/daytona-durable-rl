"""Sweep task-specific PPO settings on two concurrent Daytona H100s."""

import asyncio
import logging
import os
from contextlib import aclosing
from datetime import UTC, datetime
from enum import StrEnum

from dotenv import load_dotenv

from daytona_runner import DaytonaJobStatus, DaytonaRunner
from wandb_reporter import WandbReporter

logger = logging.getLogger(__name__)


class Task(StrEnum):
    CARTPOLE_BALANCE = "CartpoleBalance"
    CARTPOLE_SWINGUP = "CartpoleSwingup"
    CHEETAH_RUN = "CheetahRun"
    WALKER_WALK = "WalkerWalk"
    GO1_JOYSTICK_FLAT_TERRAIN = "Go1JoystickFlatTerrain"
    PANDA_PICK_CUBE = "PandaPickCube"


def build_training_grid(task: Task) -> list[dict]:
    """Compare a task's baseline with five single-setting changes across two seeds."""
    # All sweep settings are explicit here; the controller does not import worker code.
    baseline = {
        "task": task.value,
        "total_steps": 62_914_560,
        "num_envs": 2048,
        "episode_length": 1000,
        "learning_rate": 0.001,
        "entropy_cost": 0.01,
        "reward_scaling": 10.0,
        "discounting": 0.995,
        "unroll_length": 30,
        "batch_size": 1024,
        "num_minibatches": 32,
        "updates_per_batch": 16,
        "num_evals": 2,
        "eval_episodes": 8,
        "checkpoint_interval_steps": 4,
        "checkpoints": True,
        "backend": "gpu",
        "video": True,
        "output": "runs",
        "resume": None,
        "wandb": False,
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_mode": None,
    }
    cartpole = {
        "episode_length": 2000,
        "num_evals": 9,
        "eval_episodes": 32,
        "checkpoint_interval_steps": 1,
    }
    presets = {
        Task.CARTPOLE_BALANCE: cartpole,
        Task.CARTPOLE_SWINGUP: cartpole,
        Task.CHEETAH_RUN: {"total_steps": 15_728_640},
        # 64 collection/training cycles, with evaluation every eight cycles.
        Task.WALKER_WALK: {"total_steps": 62_914_560, "num_evals": 9},
        Task.GO1_JOYSTICK_FLAT_TERRAIN: {
            "total_steps": 16_384_000,
            "num_envs": 8192,
            "learning_rate": 0.0003,
            "reward_scaling": 1.0,
            "discounting": 0.97,
            "batch_size": 256,
            "unroll_length": 20,
            "updates_per_batch": 4,
        },
        Task.PANDA_PICK_CUBE: {
            "total_steps": 8_192_000,
            "episode_length": 150,
            "reward_scaling": 1.0,
            "discounting": 0.97,
            "entropy_cost": 0.02,
            "batch_size": 512,
            "unroll_length": 10,
            "updates_per_batch": 8,
        },
    }
    baseline.update(presets[task])

    batch_size = baseline["batch_size"] // 2
    learning_rate = 0.0003
    if task == Task.GO1_JOYSTICK_FLAT_TERRAIN:
        batch_size = baseline["batch_size"] * 2
        learning_rate = 0.0001

    candidates = [
        {},
        {"learning_rate": learning_rate},
        {"entropy_cost": baseline["entropy_cost"] * 0.3},
        {"entropy_cost": baseline["entropy_cost"] * 3},
        {"updates_per_batch": baseline["updates_per_batch"] // 2},
        {"batch_size": batch_size},
    ]
    return [
        {**baseline, **candidate, "seed": seed} for candidate in candidates for seed in (11, 22)
    ]


async def main() -> None:
    load_dotenv(override=False)
    os.environ.setdefault("WANDB_SILENT", "true")
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in (__name__, "daytona_runner", "wandb_reporter"):
        logging.getLogger(name).setLevel(logging.INFO)
    concurrency = 4
    task = Task.PANDA_PICK_CUBE
    training_grid = build_training_grid(task)
    os.environ.setdefault("WANDB_RUN_GROUP", f"{task}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    runner = DaytonaRunner(
        work_dir="phy_rl",
        entrypoint="train:run",
        resources={"cpu": 4, "memory": 16, "disk": 20, "gpu": 1, "gpu_type": "H100"},
        volume_name="experiment_volume",
        max_concurrent_jobs=concurrency,
        env_vars={"MUJOCO_GL": "osmesa"},
    )
    jobs = []
    reporters = {}
    try:
        logger.info(
            "Sweep %s | %d trials | %d concurrent H100s",
            os.environ["WANDB_RUN_GROUP"],
            len(training_grid),
            concurrency,
        )
        for index, config in enumerate(training_grid, start=1):
            reporter = WandbReporter(
                config,
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
            )
            job = await runner.create_job(
                config=config,
                on_poll=reporter.poll,
                on_attempt_end=reporter.collect,
            )
            jobs.append(job)
            reporters[job.id] = reporter
            logger.info(
                "Queued %d/%d | job=%s seed=%d lr=%g entropy=%g batch=%d updates=%d",
                index,
                len(training_grid),
                job.id[:8],
                config["seed"],
                config["learning_rate"],
                config["entropy_cost"],
                config["batch_size"],
                config["updates_per_batch"],
            )
            await reporter.start(job)
        finished = 0
        async with aclosing(runner.run(jobs)) as results:
            async for result in results:
                reporter = reporters[result.job_id]
                if result.logs:
                    try:
                        (reporter.directory / "worker.log").write_text(
                            result.logs, encoding="utf-8"
                        )
                    except OSError:
                        logger.exception("Job %s | could not save worker log", result.job_id[:8])
                await reporter.finish(result.status)
                finished += 1
                logger.info("Sweep progress | %d/%d results received", finished, len(jobs))
    finally:
        try:
            for job in jobs:
                status = job.status
                if status not in (DaytonaJobStatus.COMPLETED, DaytonaJobStatus.FAILED):
                    status = DaytonaJobStatus.CANCELED
                await reporters[job.id].finish(status)
        finally:
            await runner.close()
            logger.info("Controller stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
