"""Short GPU training presets, based on Playground 0.2.0's Brax PPO settings."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

DM_CONTROL_DEFAULTS = {
    "num_envs": 2048,
    "episode_length": 1000,
    "batch_size": 1024,
    "num_minibatches": 32,
    "unroll_length": 30,
    "updates_per_batch": 16,
    "num_evals": 2,
    "eval_episodes": 8,
    "checkpoint_interval_steps": 4,
}

# These are bounded experiment budgets, not claims of converged policies or runtime.
TASK_DEFAULTS = {
    "CartpoleBalance": {},
    "CartpoleSwingup": {},
    "CheetahRun": {**DM_CONTROL_DEFAULTS, "total_steps": 15_728_640},
    "WalkerWalk": {**DM_CONTROL_DEFAULTS, "total_steps": 7_864_320},
    "Go1JoystickFlatTerrain": {
        "total_steps": 16_384_000,
        "num_envs": 8192,
        "episode_length": 1000,
        "learning_rate": 0.0003,
        "reward_scaling": 1.0,
        "discounting": 0.97,
        "batch_size": 256,
        "num_minibatches": 32,
        "unroll_length": 20,
        "updates_per_batch": 4,
        "num_evals": 2,
        "eval_episodes": 8,
        "checkpoint_interval_steps": 4,
    },
    "PandaPickCube": {
        "total_steps": 8_192_000,
        "num_envs": 2048,
        "episode_length": 150,
        "reward_scaling": 1.0,
        "discounting": 0.97,
        "entropy_cost": 0.02,
        "batch_size": 512,
        "num_minibatches": 32,
        "unroll_length": 10,
        "updates_per_batch": 8,
        "num_evals": 2,
        "eval_episodes": 8,
        "checkpoint_interval_steps": 4,
    },
}


def network_settings(task: str) -> dict:
    if task.startswith("Cartpole"):
        return {"policy_hidden_layer_sizes": [64, 64], "value_hidden_layer_sizes": [64, 64]}
    if task == "Go1JoystickFlatTerrain":
        return {
            "policy_hidden_layer_sizes": [512, 256, 128],
            "value_hidden_layer_sizes": [512, 256, 128],
            "policy_obs_key": "state",
            "value_obs_key": "privileged_state",
        }
    return {
        "policy_hidden_layer_sizes": [32, 32, 32, 32],
        "value_hidden_layer_sizes": [256, 256, 256, 256, 256],
    }


def prepare_robot_assets(task: str, directory: Path) -> None:
    """Download only the selected robot at Playground's pinned Menagerie revision.

    Playground otherwise clones the entire repository with git, which is absent
    from the runner image. Keep assets in the attempt directory, outside uploads.
    """
    robots = {"Go1JoystickFlatTerrain": "unitree_go1", "PandaPickCube": "franka_emika_panda"}
    if task not in robots:
        return
    from mujoco_playground._src import mjx_env

    robot = robots[task]
    revision = mjx_env.MENAGERIE_COMMIT_SHA
    root = directory / ".assets" / revision
    url = f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{revision}?recursive=1"
    print(f"Downloading {robot} assets at {revision}", flush=True)
    with urlopen(url, timeout=60) as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise RuntimeError("Menagerie asset listing was truncated")
    files = [
        item["path"]
        for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith(f"{robot}/")
    ]
    if not files:
        raise RuntimeError(f"No Menagerie assets found for {robot}")

    def download(name: str) -> None:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{revision}/{name}"
        )
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, files))
    mjx_env.MENAGERIE_PATH = root
