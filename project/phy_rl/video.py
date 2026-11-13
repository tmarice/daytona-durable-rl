"""Render recorded evaluation trajectories with MuJoCo's native offscreen renderer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco

if __package__:
    from .results import write_json
else:
    from results import write_json

WIDTH = 640
HEIGHT = 480


def configure_rendering() -> None:
    """Select headless CPU rendering on Linux before MuJoCo is imported."""
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")


def create_renderer(model: mujoco.MjModel) -> mujoco.Renderer:
    import mujoco

    # Some robot models allocate a 4K buffer by default. Match our output size.
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    try:
        return mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    except Exception as error:
        raise RuntimeError(
            "Could not initialize native MuJoCo rendering. On headless Linux, install "
            "libosmesa6 and set MUJOCO_GL=osmesa before starting Python. "
            "MUJOCO_GL=egl requires working EGL graphics support."
        ) from error


def camera_for_task(task: str) -> str | mujoco.MjvCamera:
    import mujoco

    cameras = {
        "CartpoleBalance": "fixed",
        "CartpoleSwingup": "fixed",
        "CheetahRun": "side",
        "WalkerWalk": "side",
        "Go1JoystickFlatTerrain": "track",
    }
    if task in cameras:
        return cameras[task]
    if task != "PandaPickCube":
        raise ValueError(f"No camera configured for {task}")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.45, 0, 0.3)
    camera.distance = 1.6
    camera.azimuth = 135
    camera.elevation = -25
    return camera


def scene_options() -> mujoco.MjvOption:
    import mujoco

    options = mujoco.MjvOption()
    # These models use groups 0-2 for visible geometry and group 3 for collisions.
    options.geomgroup[:] = (1, 1, 1, 0, 0, 0)
    options.sitegroup[:] = 0
    return options


def check_rendering(env, task: str) -> None:
    """Render one frame before starting expensive training."""
    import mujoco

    data = mujoco.MjData(env.mj_model)
    mujoco.mj_forward(env.mj_model, data)
    with create_renderer(env.mj_model) as renderer:
        renderer.update_scene(data, camera=camera_for_task(task), scene_option=scene_options())
        renderer.render()


def write_comparison(
    env, before: dict, after: dict, config, directory: Path, *, baseline_step: int = 0
) -> dict:
    """Stream native before/after frames with labels into an H.264 MP4."""
    import matplotlib

    matplotlib.use("Agg")
    import mujoco
    import numpy as np
    from matplotlib import animation
    from matplotlib import pyplot as plt

    model = env.mj_model
    data = mujoco.MjData(model)
    camera = camera_for_task(config.task)
    options = scene_options()
    stride = max(1, round(0.04 / env.dt))
    frame_steps = list(range(0, config.episode_length + 1, stride))
    if frame_steps[-1] != config.episode_length:
        frame_steps.append(config.episode_length)
    fps = 1 / (env.dt * stride)
    episodes = min(4 if config.task.startswith("Cartpole") else 2, config.eval_episodes)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6), facecolor="#f4f6fa")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.82, wspace=0.02)
    fig.suptitle(f"{config.task} | PPO policy before and after training", fontsize=16, y=0.97)
    heading = fig.text(0.5, 0.91, "", ha="center", fontsize=11)
    fig.text(
        0.5,
        0.025,
        "MuJoCo native rendering | Same evaluation starts | 1x speed",
        ha="center",
        fontsize=10,
    )
    images = []
    titles = (
        "INITIAL POLICY" if baseline_step == 0 else f"BASELINE AT STEP {baseline_step:,}",
        "FINAL POLICY",
    )
    for axis in axes:
        images.append(axis.imshow(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)))
        axis.set_axis_off()

    path = directory / "policy_comparison.mp4"
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        with create_renderer(model) as renderer, writer.saving(fig, str(path), dpi=100):
            for episode in range(episodes):
                for step in frame_steps:
                    heading.set_text(
                        f"Episode {episode + 1}/{episodes} | Simulation time {step * env.dt:.2f} s"
                    )
                    for axis, image, rollout, title in zip(
                        axes, images, (before, after), titles, strict=True
                    ):
                        data.qpos[:] = rollout["qpos"][episode, step]
                        data.mocap_pos[:] = rollout["mocap_pos"][episode, step]
                        data.mocap_quat[:] = rollout["mocap_quat"][episode, step]
                        data.time = step * env.dt
                        # Reconstruct the recorded pose without advancing the simulation.
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera, scene_option=options)
                        image.set_data(renderer.render())
                        ended = step >= int(rollout["length"][episode])
                        axis.set_title(
                            f"{title} | Return {rollout['returns'][episode, step]:.1f}"
                            + (" | End" if ended else ""),
                            fontsize=11,
                        )
                    writer.grab_frame()
                    if episode == 0 and step == frame_steps[len(frame_steps) // 2]:
                        fig.savefig(directory / "policy_comparison.png", dpi=100)
    finally:
        plt.close(fig)

    metadata = {
        "path": path.name,
        "baseline_training_step": baseline_step,
        "renderer": "mujoco_native",
        "camera": camera if isinstance(camera, str) else "fixed_panda",
        "frame_width": WIDTH,
        "frame_height": HEIGHT,
        "fps": fps,
        "playback_speed": 1.0,
        "frame_count": episodes * len(frame_steps),
        "episode_indices": list(range(episodes)),
        "control_dt": env.dt,
        "seed": config.seed,
        "evaluation_seed_fold_in": 1_000_003,
        "episode_returns_before": before["returns"][:episodes, -1].tolist(),
        "episode_returns_after": after["returns"][:episodes, -1].tolist(),
    }
    write_json(directory / "video.json", metadata)
    print(f"Video: {path.resolve()}", flush=True)
    return metadata
