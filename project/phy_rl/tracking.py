"""Optional W&B reporting; local checkpoints remain the recovery source of truth."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

if __package__:
    from .results import write_json
else:
    from results import write_json


class WandbTracker:
    def __init__(self, directory: Path, *, attempt_id: str) -> None:
        self.directory = directory.resolve()
        self.attempt_id = attempt_id
        self.run = None
        self.identity = None
        self._last_env_steps = 0

    def start(self, config, experiment_id: str, parameters: dict, previous: dict | None):
        # Import lazily so local training and the lightweight tests need no W&B.
        import wandb
        from dotenv import load_dotenv

        dotenv_path = Path.cwd() / ".env"
        if not dotenv_path.is_file():
            dotenv_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path, override=False)

        mode = config.wandb_mode or os.environ.get("WANDB_MODE", "online")
        if mode not in {"online", "offline"}:
            raise ValueError("W&B mode must be online or offline; use --no-wandb to disable it")
        if mode == "offline" and config.resume is not None:
            raise ValueError("W&B offline mode cannot resume history; use online or --no-wandb")
        project = config.wandb_project or os.environ.get(
            "WANDB_PROJECT", "daytona-experiment-runner"
        )
        entity = config.wandb_entity or os.environ.get("WANDB_ENTITY")
        run_id = hashlib.sha256(experiment_id.encode()).hexdigest()[:24]
        if previous:
            for field in ("project", "entity"):
                supplied = getattr(config, f"wandb_{field}")
                if supplied is not None and supplied != previous[field]:
                    raise ValueError(f"--wandb-{field} conflicts with the checkpoint's W&B run")
            project, entity, run_id = previous["project"], previous["entity"], previous["id"]

        self.run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            name=experiment_id,
            resume="allow" if mode == "online" else None,
            mode=mode,
            dir=str(self.directory),
            config={**parameters, "experiment_id": experiment_id},
            # Upload only our explicit result files, never repository code or .env.
            settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
        )
        self.identity = {
            "id": self.run.id,
            "project": self.run.project,
            "entity": self.run.entity,
        }
        write_json(
            self.directory / "wandb.json",
            {**self.identity, "url": self.run.url, "mode": mode, "attempt_id": self.attempt_id},
        )
        self.run.define_metric("env_steps")
        self.run.define_metric("*", step_metric="env_steps")
        self.run.summary.update({"attempt_id": self.attempt_id, "attempt_status": "running"})
        print(
            f"W&B: {self.run.url or 'offline logs in ' + str(self.directory / 'wandb')}", flush=True
        )

    def log(self, env_steps: int, values: dict) -> None:
        self._last_env_steps = int(env_steps)
        if self.run is not None:
            # Let W&B maintain its own monotonically increasing history step.
            # Environment steps may repeat after restoring an older checkpoint.
            self.run.log({**values, "env_steps": int(env_steps), "attempt_id": self.attempt_id})

    def evaluation(self, env_steps: int, phase: str, result: dict):
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
        self.log(env_steps, values)

    def finish(self, status: str, summary: dict | None = None) -> None:
        if self.run is None:
            return
        import wandb

        try:
            self.run.summary.update({"attempt_status": status, "attempt_id": self.attempt_id})
            if summary is not None:
                self.run.summary.update(summary)
            media = {}
            for name in ("learning_curves.png", "policy_comparison.png"):
                path = self.directory / name
                if path.is_file():
                    media[f"plots/{path.stem}"] = wandb.Image(str(path))
            if (self.directory / "policy_comparison.mp4").is_file():
                media["policy/comparison"] = wandb.Video(
                    str(self.directory / "policy_comparison.mp4"), format="mp4"
                )
            if media:
                step = summary["actual_steps"] if summary is not None else self._last_env_steps
                self.log(step, media)
            # Attempt subdirectories keep earlier attempts' files intact on resume.
            for name in (
                "requested_config.json",
                "resolved_config.json",
                "attempt.json",
                "wandb.json",
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
            ):
                path = self.directory / name
                if path.is_file():
                    self.run.save(str(path), base_path=str(self.directory.parent), policy="now")
        finally:
            self.run.finish(exit_code={"completed": 0, "interrupted": 130}.get(status, 1))
