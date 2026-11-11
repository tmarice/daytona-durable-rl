"""Example library usage; replace the work directory and training entrypoint as needed."""

import asyncio
import os
from contextlib import aclosing

from dotenv import load_dotenv

from daytona_runner import DaytonaRunner


async def main() -> None:
    load_dotenv(override=False)
    runner = DaytonaRunner(
        work_dir=".",
        entrypoint="physical_ai.main:train",
        resources={"cpu": 4, "memory": 8, "disk": 10},
        volume_name="experiment_volume",
        secrets={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
    )
    try:
        sweep = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [32, 64, 128],
            "num_layers": [2, 3, 4],
        }
        training_grid = [
            {"learning_rate": lr, "batch_size": bs, "num_layers": nl}
            for lr in sweep["learning_rate"]
            for bs in sweep["batch_size"]
            for nl in sweep["num_layers"]
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
