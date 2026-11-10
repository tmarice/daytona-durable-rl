import asyncio

from .daytona import DaytonaRunner


async def main():
    runner = DaytonaRunner(volume="experiment_volume")

    # TODO Adjust for our current problem
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

    jobs = [
        runner.create_job(
            entrypoint="physical_ai.main:train",
            config={
                "work_dir": "/work_dir",
                "resources": {"cpu": 4, "memory": 8, "disk": 10},
                "secrets": ["WANDB_API_KEY"],
            }
            | cfg,
        )
        for cfg in training_grid
    ]

    async for result in runner.run(jobs):
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
