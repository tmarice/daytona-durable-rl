"""Sweep Cartpole swing-up PPO settings on two concurrent Daytona H100s."""

import asyncio
import os
from contextlib import aclosing

from dotenv import load_dotenv

from daytona_runner import DaytonaRunner


async def main() -> None:
    load_dotenv(override=False)
    runner = DaytonaRunner(
        work_dir="phy_rl",
        entrypoint="train:run",
        resources={"cpu": 4, "memory": 8, "disk": 20, "gpu": 1, "gpu_type": "H100"},
        volume_name="experiment_volume",
        max_concurrent_jobs=2,
        secrets={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
    )
    try:
        # Task choices: rough duration of the full 12-trial sweep on one H100,
        # running sequentially with the current presets and video disabled:
        #   CartpoleBalance        ~25-65 min
        #   CartpoleSwingup        ~25-65 min
        #   CheetahRun             ~30-70 min
        #   WalkerWalk             ~35-75 min
        #   Go1JoystickFlatTerrain ~30-70 min
        #   PandaPickCube          ~35-75 min
        # Unmeasured planning estimates: published A100 PPO throughput as a proxy,
        # plus an assumed 2-5 min/trial for startup, JIT, evaluation, and result I/O.
        # Cartpole estimates are less reliable because our batching differs more.
        # Excludes initial image building, capacity queues, and preemption retries.
        # Reference: https://arxiv.org/html/2502.08844v1 (Tables IV, VII, IX; Section VI).
        training_grid = [
            {
                "task": "CartpoleSwingup",
                "backend": "gpu",
                "seed": seed,
                "learning_rate": lr,
                "entropy_cost": entropy_cost,
                "wandb": True,
                "wandb_project": os.getenv("WANDB_PROJECT"),
                "wandb_entity": os.getenv("WANDB_ENTITY"),
            }
            for lr in (0.0003, 0.001)
            for entropy_cost in (0.001, 0.01)
            for seed in (11, 22, 33)
        ]

        jobs = [await runner.create_job(config=cfg) for cfg in training_grid]
        async with aclosing(runner.run(jobs)) as results:
            async for result in results:
                print(result)
    finally:
        await runner.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
