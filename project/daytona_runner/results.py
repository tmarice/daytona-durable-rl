"""Local, inspectable metrics and result artifacts for a single training attempt."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_run_directory(root: Path, task: str, seed: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / f"{task}-seed{seed}-{stamp}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


class Metrics:
    """Flush each progress callback so partial runs retain their observations."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.started = time.monotonic()
        self.rows: list[dict[str, Any]] = []

    def record(self, step: int, values: dict[str, Any]) -> None:
        step = int(step)
        if step < 0 or (self.rows and step < self.rows[-1]["step"]):
            raise ValueError("Training step must be nonnegative and must not move backwards")
        scalars = {key: float(value) for key, value in values.items()}
        invalid = [key for key, value in scalars.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"Non-finite training metrics: {', '.join(invalid)}")
        row = {"step": step, "elapsed_seconds": time.monotonic() - self.started, "metrics": scalars}
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        reward = scalars.get("eval/episode_reward")
        suffix = f", evaluation reward={reward:.3f}" if reward is not None else ""
        print(f"step={step:,}, elapsed={row['elapsed_seconds']:.1f}s{suffix}", flush=True)

    def export_csv(self) -> None:
        names = sorted({key for row in self.rows for key in row["metrics"]})
        with (self.directory / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "elapsed_seconds", *names])
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        "step": row["step"],
                        "elapsed_seconds": row["elapsed_seconds"],
                        **row["metrics"],
                    }
                )

    def plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
        reward_rows = [row for row in self.rows if "eval/episode_reward" in row["metrics"]]
        axes[0].plot(
            [row["step"] for row in reward_rows],
            [row["metrics"]["eval/episode_reward"] for row in reward_rows],
            marker="o",
        )
        axes[0].set(title="Policy evaluation during training", ylabel="Mean episode reward")
        losses = sorted({key for row in self.rows for key in row["metrics"] if "loss" in key})
        for key in losses:
            points = [row for row in self.rows if key in row["metrics"]]
            axes[1].plot(
                [row["step"] for row in points], [row["metrics"][key] for row in points], label=key
            )
        if losses:
            axes[1].legend(fontsize="small")
        else:
            axes[1].text(
                0.5, 0.5, "No loss metrics reported", ha="center", transform=axes[1].transAxes
            )
        axes[1].set(title="PPO losses", ylabel="Loss")
        for axis in axes:
            axis.set_xlabel("Training environment steps")
            axis.grid(alpha=0.2)
        fig.savefig(self.directory / "learning_curves.png", dpi=160)
        plt.close(fig)
