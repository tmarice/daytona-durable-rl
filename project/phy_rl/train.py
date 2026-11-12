"""Train, checkpoint, and resume a MuJoCo Playground policy on CPU or NVIDIA GPU."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .checkpointing import checkpoint_from_data, read_checkpoint, save_state
    from .daytona_experiment_session import (
        ExperimentSession,
        latest_checkpoint_path,
        read_checkpoint_document,
    )
    from .evaluation import evaluate_policy
    from .results import Metrics, create_run_directory, write_json
    from .tasks import TASK_DEFAULTS, network_settings, prepare_robot_assets
    from .tracking import WandbTracker
else:
    # Daytona uploads the contents of phy_rl and imports this module as `train`.
    from checkpointing import checkpoint_from_data, read_checkpoint, save_state
    from daytona_experiment_session import (
        ExperimentSession,
        latest_checkpoint_path,
        read_checkpoint_document,
    )
    from evaluation import evaluate_policy
    from results import Metrics, create_run_directory, write_json
    from tasks import TASK_DEFAULTS, network_settings, prepare_robot_assets
    from tracking import WandbTracker


@dataclass(frozen=True)
class Config:
    task: str = "CartpoleBalance"
    seed: int = 11
    total_steps: int = 8_388_608
    num_envs: int = 128
    episode_length: int = 2000
    learning_rate: float = 0.001
    entropy_cost: float = 0.01
    reward_scaling: float = 10.0
    discounting: float = 0.995
    unroll_length: int = 32
    batch_size: int = 128
    num_minibatches: int = 8
    updates_per_batch: int = 8
    num_evals: int = 9
    eval_episodes: int = 32
    video: bool = False
    output: Path = Path("runs")
    checkpoints: bool = True
    checkpoint_interval_steps: int = 1
    resume: Path | None = None
    backend: str = "cpu"
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str | None = None

    def validate(self) -> None:
        if self.wandb_mode not in {None, "online", "offline"}:
            raise ValueError("wandb_mode must be online or offline")
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("backend must be cpu or gpu")
        if self.task not in TASK_DEFAULTS:
            raise ValueError(f"Supported tasks are {', '.join(TASK_DEFAULTS)}")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if type(self.checkpoint_interval_steps) is not int:
            raise ValueError("checkpoint_interval_steps must be a positive integer")
        for name in (
            "total_steps",
            "num_envs",
            "episode_length",
            "unroll_length",
            "batch_size",
            "num_minibatches",
            "updates_per_batch",
            "eval_episodes",
            "checkpoint_interval_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_evals < 2:
            raise ValueError("num_evals must be at least 2 for before/after evaluation")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.entropy_cost) or self.entropy_cost < 0:
            raise ValueError("entropy_cost must be finite and nonnegative")
        if not math.isfinite(self.reward_scaling) or self.reward_scaling <= 0:
            raise ValueError("reward_scaling must be finite and positive")
        if not math.isfinite(self.discounting) or not 0 < self.discounting <= 1:
            raise ValueError("discounting must be in (0, 1]")
        if self.batch_size * self.num_minibatches % self.num_envs:
            raise ValueError("batch_size * num_minibatches must be divisible by num_envs")


RUNTIME_OPTIONS = {
    "output",
    "video",
    "resume",
    "checkpoints",
    "checkpoint_interval_steps",
    "backend",
    "wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_mode",
}


def config_values(config: Config) -> dict:
    return {
        **dataclasses.asdict(config),
        "output": str(config.output),
        "resume": str(config.resume) if config.resume is not None else None,
    }


def training_config(config: Config) -> dict:
    return {
        key: value for key, value in config_values(config).items() if key not in RUNTIME_OPTIONS
    }


def config_from_values(supplied: dict) -> Config:
    """Inherit resume settings while rejecting explicit training changes."""
    supplied = dict(supplied)
    supplied.setdefault("backend", Config().backend)
    values = {}
    if supplied.get("resume") is not None:
        checkpoint = read_checkpoint(Path(supplied["resume"]))
        values = checkpoint.manifest["metadata"]["requested"].copy()
        for key, value in supplied.items():
            if key not in RUNTIME_OPTIONS and values.get(key) != value:
                raise ValueError(f"--{key.replace('_', '-')} conflicts with the checkpoint")
    else:
        values.update(TASK_DEFAULTS.get(supplied.get("task", Config().task), {}))
    values.update(supplied)
    if "output" in values:
        values["output"] = Path(values["output"])
    if values.get("resume") is not None:
        values["resume"] = Path(values["resume"])
    config = Config(**values)
    config.validate()
    return config


def parse_args(argv: list[str] | None = None) -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--cpu",
        dest="backend",
        action="store_const",
        const="cpu",
        help="Train on CPU (default)",
    )
    device.add_argument(
        "--gpu",
        dest="backend",
        action="store_const",
        const="gpu",
        help="Train on an NVIDIA GPU; requires CUDA-enabled JAX",
    )
    parser.add_argument("--task", choices=list(TASK_DEFAULTS), default=defaults.task)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
    parser.add_argument("--episode-length", type=int, default=defaults.episode_length)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--entropy-cost", type=float, default=defaults.entropy_cost)
    parser.add_argument("--reward-scaling", type=float, default=defaults.reward_scaling)
    parser.add_argument("--discounting", type=float, default=defaults.discounting)
    parser.add_argument("--unroll-length", type=int, default=defaults.unroll_length)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-minibatches", type=int, default=defaults.num_minibatches)
    parser.add_argument("--updates-per-batch", type=int, default=defaults.updates_per_batch)
    parser.add_argument("--num-evals", type=int, default=defaults.num_evals)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument(
        "--video", action="store_true", help="Save a before/after policy video (requires ffmpeg)"
    )
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        help="Log to Weights & Biases using credentials from the environment or .env",
    )
    parser.add_argument(
        "--wandb-project", help="W&B project (default: WANDB_PROJECT or daytona-experiment-runner)"
    )
    parser.add_argument(
        "--wandb-entity", help="W&B user/team (default: WANDB_ENTITY or account default)"
    )
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline"], help="Default: WANDB_MODE or online"
    )
    parser.add_argument(
        "--checkpoints",
        action=argparse.BooleanOptionalAction,
        help="Save training checkpoints (default: enabled); --no-checkpoints disables all saves",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=defaults.checkpoint_interval_steps,
        help="Save every N completed PPO updates (default: 1); always save the final update",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a session checkpoint JSON file or checkpoint directory",
    )
    # Preserve which options the caller supplied: omitted settings come from the
    # checkpoint, while conflicting training overrides are explicit errors.
    for action in parser._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
    supplied = vars(parser.parse_args(argv))
    try:
        config = config_from_values(supplied)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return config


def ppo_settings(config: Config) -> dict:
    """Explicit settings, shared between training and the saved run configuration."""
    return {
        "num_timesteps": config.total_steps,
        "seed": config.seed,
        "num_envs": config.num_envs,
        "episode_length": config.episode_length,
        "learning_rate": config.learning_rate,
        "entropy_cost": config.entropy_cost,
        "unroll_length": config.unroll_length,
        "batch_size": config.batch_size,
        "num_minibatches": config.num_minibatches,
        "num_updates_per_batch": config.updates_per_batch,
        "num_evals": config.num_evals,
        "num_eval_envs": config.eval_episodes,
        "action_repeat": 1,
        "normalize_observations": True,
        "reward_scaling": config.reward_scaling,
        "discounting": config.discounting,
        "clipping_epsilon": 0.2,
        "gae_lambda": 0.95,
        "max_grad_norm": 1.0,
        "max_devices_per_host": 1,
        "num_resets_per_eval": 0,
        "deterministic_eval": True,
        "log_training_metrics": True,
        "training_metrics_steps": config.batch_size * config.num_minibatches * config.unroll_length,
        "run_evals": True,
        "checkpoint_interval_steps": config.checkpoint_interval_steps,
    }


def initialize_jax(backend: str):
    """Select the requested platform before any training dependencies initialize JAX."""
    # CUDA remains mandatory and the default; progress callbacks also need a CPU device.
    platform_name = "cuda,cpu" if backend == "gpu" else "cpu"
    os.environ["JAX_PLATFORMS"] = platform_name
    import jax

    # Also cover callers that imported JAX before invoking run(). A backend
    # already initialized elsewhere cannot be changed; the check below rejects it.
    jax.config.update("jax_platforms", platform_name)
    try:
        actual_backend = jax.default_backend()
    except (RuntimeError, AssertionError) as error:
        if backend == "gpu":
            raise RuntimeError(
                "--gpu requires an available NVIDIA GPU and CUDA-enabled JAX. "
                "Use --cpu for CPU training."
            ) from error
        raise
    if actual_backend != backend:
        raise RuntimeError(
            f"Requested {backend} training, but JAX is already using {actual_backend}. "
            f"Start a new Python process with --{backend}."
        )
    return jax


def train(
    config: Config,
    directory: Path,
    metrics: Metrics,
    session: ExperimentSession,
    tracker: WandbTracker | None = None,
) -> dict:
    data = session.restore_checkpoint(config.resume)
    checkpoint = None
    if data is not None:
        assert session.latest_checkpoint is not None
        checkpoint = checkpoint_from_data(session.latest_checkpoint, data)
    jax = initialize_jax(config.backend)
    os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib"))
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry, wrapper

    if __package__:
        from ._vendor import brax_ppo as ppo
    else:
        from _vendor import brax_ppo as ppo

    env_config = registry.get_default_config(config.task)
    env_config.impl = "jax"
    env_config.episode_length = config.episode_length
    env_config.vision = False
    prepare_robot_assets(config.task, directory)
    env = registry.load(config.task, config=env_config)
    eval_env = registry.load(config.task, config=env_config)
    settings = ppo_settings(config)
    architecture = network_settings(config.task)
    versions = {
        package: importlib.metadata.version(package)
        for package in (
            "playground",
            "brax",
            "jax",
            "jaxlib",
            "mujoco",
            "mujoco-mjx",
            "flax",
            "optax",
            "numpy",
            "matplotlib",
        )
    }
    compatibility = {
        "training_config": training_config(config),
        "ppo": {
            key: value for key, value in settings.items() if key != "checkpoint_interval_steps"
        },
        "environment": json.loads(env_config.to_json()),
        "network": architecture,
        "versions": versions,
        "trainer_sha256": hashlib.sha256(Path(ppo.__file__).read_bytes()).hexdigest(),
    }
    if checkpoint is not None and checkpoint.manifest["metadata"]["compatibility"] != compatibility:
        raise ValueError(
            "Checkpoint training configuration, versions, or trainer source is incompatible"
        )
    starting_step = checkpoint.manifest["env_steps"] if checkpoint is not None else 0
    steps_per_update = config.batch_size * config.num_minibatches * config.unroll_length
    if checkpoint is not None and (
        checkpoint.manifest["completed_updates"] * steps_per_update != starting_step
    ):
        raise ValueError("Checkpoint update count and environment-step count disagree")
    metadata = {
        "requested": config_values(config),
        "compatibility": compatibility,
        "experiment_id": session.experiment_id,
        "attempt_id": session.attempt_id,
        "resumed_from": str(checkpoint.path) if checkpoint is not None else None,
    }
    previous_tracking = checkpoint.manifest["metadata"].get("wandb") if checkpoint else None
    if tracker is not None:
        tracker.start(
            config,
            session.experiment_id,
            {**training_config(config), "network": architecture, "versions": versions},
            previous_tracking,
        )
        metadata["wandb"] = tracker.identity
        tracker.log(
            starting_step, {"attempt/starting_step": starting_step, "attempt/status": "running"}
        )
    elif previous_tracking is not None:
        # Disabling tracking for one attempt must not lose the remote run identity.
        metadata["wandb"] = previous_tracking
    write_json(directory / "attempt.json", {**metadata, "starting_step": starting_step})
    write_json(
        directory / "resolved_config.json",
        {
            "requested": config_values(config),
            "environment": json.loads(env_config.to_json()),
            "ppo": settings,
            "network": architecture,
            "versions": versions,
            "brax_source": json.loads(
                importlib.metadata.distribution("brax").read_text("direct_url.json") or "null"
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_kinds": [device.device_kind for device in jax.devices()],
            "checkpoints_enabled": config.checkpoints,
            "checkpoint_interval_unit": "completed_ppo_updates",
            "environment_steps_per_update": steps_per_update,
            "starting_step": starting_step,
        },
    )
    print(
        f"Training {config.task} on {jax.devices()} with {config.num_envs} environments", flush=True
    )
    if config.checkpoints:
        print(
            f"Starting at {starting_step:,} environment steps; saving every "
            f"{config.checkpoint_interval_steps} completed PPO update(s) "
            f"({steps_per_update:,} environment steps per update).",
            flush=True,
        )
    else:
        print(
            f"Starting at {starting_step:,} environment steps; checkpointing disabled.", flush=True
        )
    initial_evaluation = None
    initial_policy_params = None
    initial_rollouts = None

    def inspect_initial_policy(step, make_policy, params):
        nonlocal initial_evaluation, initial_policy_params, initial_rollouts
        if initial_evaluation is None:
            # The trainer donates device buffers. Retain an independent host copy
            # for the final update measurement, never a reference to donated data.
            initial_policy_params = jax.tree_util.tree_map(
                lambda value: jax.device_get(value).copy(), params[1]
            )
            initial_evaluation, initial_rollouts = evaluate_policy(
                eval_env, make_policy, params, config
            )
            initial_evaluation["training_step"] = int(step)
            initial_evaluation["policy_origin"] = (
                "checkpoint" if checkpoint else "random_initialization"
            )
            write_json(directory / "evaluation_before.json", initial_evaluation)
            if tracker is not None:
                tracker.evaluation(step, "before", initial_evaluation)

    latest_checkpoint = str(checkpoint.path) if checkpoint is not None else None

    def checkpoint_completed_update(state, completed_updates):
        nonlocal latest_checkpoint
        started = time.monotonic()
        path = save_state(
            session,
            state,
            completed_updates=completed_updates,
            metadata=metadata,
        )
        latest_checkpoint = str(path.resolve())
        write_json(
            directory / "checkpoint_status.json",
            {
                "latest_checkpoint": latest_checkpoint,
                "completed_updates": completed_updates,
                "env_steps": int(state["learner"].env_steps),
                "save_seconds": time.monotonic() - started,
            },
        )
        print(f"Checkpoint committed: update={completed_updates} {path.resolve()}", flush=True)

    def report_progress(step, values):
        metrics.record(step, values)
        if tracker is not None:
            row = metrics.rows[-1]
            tracker.log(step, {**row["metrics"], "attempt/elapsed_seconds": row["elapsed_seconds"]})

    make_policy, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **architecture),
        progress_fn=report_progress,
        policy_params_fn=inspect_initial_policy,
        **({"checkpoint_fn": checkpoint_completed_update} if config.checkpoints else {}),
        restore_state_fn=checkpoint.restore if checkpoint is not None else None,
        **settings,
    )
    if initial_evaluation is None:
        raise RuntimeError("Trainer did not expose its initial policy for baseline evaluation")
    actual_steps = max((row["step"] for row in metrics.rows), default=starting_step)
    final_evaluation, final_rollouts = evaluate_policy(eval_env, make_policy, params, config)
    final_evaluation["training_step"] = actual_steps
    write_json(directory / "evaluation_after.json", final_evaluation)
    squared_changes = jax.tree_util.tree_map(
        lambda before, after: ((after - before) ** 2).sum(), initial_policy_params, params[1]
    )
    policy_change = math.sqrt(
        sum(float(value) for value in jax.tree_util.tree_leaves(squared_changes))
    )
    if tracker is not None:
        tracker.evaluation(actual_steps, "after", final_evaluation)
    if not math.isfinite(policy_change) or (actual_steps > starting_step and policy_change == 0):
        raise RuntimeError("Training did not produce a finite, nonzero policy parameter update")
    if actual_steps < config.total_steps:
        raise RuntimeError(
            f"Trainer stopped at {actual_steps} steps before target {config.total_steps}"
        )
    summary = {
        "requested_steps": config.total_steps,
        "actual_steps": actual_steps,
        "evaluation_before": initial_evaluation,
        "evaluation_after": final_evaluation,
        "mean_return_change": final_evaluation["mean_return"] - initial_evaluation["mean_return"],
        "policy_parameter_change_l2": policy_change,
        "checkpoints_enabled": config.checkpoints,
        "starting_step": starting_step,
        "steps_this_attempt": actual_steps - starting_step,
        "completed_updates": actual_steps // steps_per_update,
        "experiment_id": session.experiment_id,
        "attempt_id": session.attempt_id,
        "resumed_from": str(checkpoint.path) if checkpoint is not None else None,
        "latest_checkpoint": latest_checkpoint,
    }
    if config.video:
        if __package__:
            from .video import write_comparison
        else:
            from video import write_comparison

        summary["video"] = write_comparison(
            eval_env, initial_rollouts, final_rollouts, config, directory
        )
    return summary


def run(
    config: Config | dict,
    *,
    experiment_id: str | None = None,
    attempt_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> Path:
    if isinstance(config, dict):
        config = config_from_values(config)
    config.validate()
    resume = config.resume
    if resume is None and checkpoint_dir is not None:
        resume = latest_checkpoint_path(Path(checkpoint_dir))
    if resume is not None:
        config = dataclasses.replace(config, resume=Path(resume))
        if experiment_id is None:
            _, document = read_checkpoint_document(config.resume)
            experiment_id = document["experiment_id"]
    if attempt_id is None:
        directory = create_run_directory(config.output, config.task, config.seed)
        attempt_id = directory.name
    else:
        if not attempt_id or Path(attempt_id).name != attempt_id or attempt_id in {".", ".."}:
            raise ValueError("attempt_id must be a single directory name")
        directory = config.output / attempt_id
        directory.mkdir(parents=True, exist_ok=False)
    session = ExperimentSession(
        experiment_id if experiment_id is not None else attempt_id,
        attempt_id,
        checkpoint_dir if checkpoint_dir is not None else directory / "checkpoints",
    )
    metrics = Metrics(directory)
    write_json(
        directory / "requested_config.json",
        config_values(config),
    )
    print(f"Results: {directory.resolve()}", flush=True)
    write_json(directory / "status.json", {"status": "running"})
    started = time.monotonic()
    tracker = WandbTracker(directory, attempt_id=session.attempt_id) if config.wandb else None
    summary = None
    status = "failed"
    try:
        if config.video and shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "--video requires ffmpeg on PATH; install it before running training"
            )
        summary = train(config, directory, metrics, session, tracker)
        metrics.export_csv()
        metrics.plot()
        summary["wall_seconds"] = time.monotonic() - started
        write_json(directory / "summary.json", summary)
        write_json(directory / "status.json", {"status": "completed"})
        status = "completed"
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metrics.export_csv()
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(
            directory / "status.json",
            {
                "status": status,
                "error": str(error),
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        if tracker is not None:
            try:
                tracker.finish(status, summary)
            except Exception as error:  # noqa: BLE001 -- best-effort reporting at shutdown.
                # Never hide a training error or discard successful local results
                # because uploading results failed during shutdown.
                print(
                    f"W&B finalization failed: {error}. Local results were retained.",
                    file=sys.stderr,
                )
    print(
        f"Mean evaluation return: {summary['evaluation_before']['mean_return']:.3f} -> "
        f"{summary['evaluation_after']['mean_return']:.3f}\nResults: {directory.resolve()}",
        flush=True,
    )
    return directory


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        run(config)
    except ModuleNotFoundError as error:
        print(
            f"Missing training dependency: {error.name}. Install the project dependencies first (see README.md).",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print(
            (
                "Training interrupted; metrics and any committed checkpoints were retained. "
                "Use --resume with the session's checkpoint directory or a checkpoint JSON file."
                if config.checkpoints
                else "Training interrupted; metrics were retained. Checkpointing was disabled."
            ),
            file=sys.stderr,
        )
        return 130
    except Exception as error:  # noqa: BLE001 -- CLI boundary; run() retained the traceback.
        print(
            f"Training failed: {error}. See error.txt in the printed results directory.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
