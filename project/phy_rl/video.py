"""Render recorded MuJoCo states without requiring an OpenGL display."""

from __future__ import annotations

import math
from pathlib import Path

if __package__:
    from .results import write_json
else:
    from results import write_json


def write_comparison(env, before: dict, after: dict, config, directory: Path) -> dict:
    """Stream a labeled, real-time comparison into an H.264 MP4."""
    if not config.task.startswith("Cartpole"):
        return write_robot_comparison(env, before, after, config, directory)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import animation
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    # These task geometries are defined in Playground's Cartpole MJCF.
    cart_half_width, _, cart_half_height = env.mj_model.geom("cart").size
    pole_length = float(2 * env.mj_model.geom("pole_1").size[1])
    cart_height = float(env.mj_model.body("cart").pos[2])
    stride = max(1, round(0.04 / env.dt))
    speed = 1.0
    fps = speed / (env.dt * stride)
    frame_steps = list(range(0, config.episode_length + 1, stride))
    if frame_steps[-1] != config.episode_length:
        frame_steps.append(config.episode_length)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), facecolor="#f4f6fa")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.15, top=0.78, wspace=0.12)
    fig.suptitle(f"{config.task}  |  PPO policy before and after training", fontsize=18, y=0.96)
    heading = fig.text(0.5, 0.86, "", ha="center", fontsize=12)
    fig.text(
        0.5,
        0.045,
        "Actual MuJoCo states, 2D view  |  Identical starts, deterministic actions  |  1x speed",
        ha="center",
        fontsize=10,
        color="#445066",
    )
    artists = []
    for axis, title, color in zip(
        axes,
        ("RESTORED POLICY" if config.resume else "INITIAL POLICY", "ATTEMPT END POLICY"),
        ("#bb6538", "#2269a5"),
        strict=True,
    ):
        axis.set(xlim=(-3, 3), ylim=(-0.15, 2.3), aspect="equal")
        axis.set_facecolor("#f4f6fa")
        axis.set_title(title, fontsize=13, color=color, fontweight="bold", pad=28)
        axis.axhline(cart_height, color="#a6afbd", lw=2, zorder=0)
        axis.axvline(0, color="#c3cad5", ls="--", lw=1, zorder=0)
        axis.set_xticks([-2, -1, 0, 1, 2], labels=["-2 m", "-1 m", "0", "1 m", "2 m"])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        cart = Rectangle((0, 0), 2 * cart_half_width, 2 * cart_half_height, color=color)
        axis.add_patch(cart)
        (pole,) = axis.plot([], [], color=color, lw=7, solid_capstyle="round")
        (pivot,) = axis.plot([], [], "o", color="#263348", markersize=5)
        (control,) = axis.plot([], [], color=color, lw=6, solid_capstyle="round")
        axis.plot([-1, 1], [0.1, 0.1], color="#d4dae3", lw=6, zorder=0)
        stats = axis.text(0.5, 1.05, "", ha="center", transform=axis.transAxes, fontsize=10)
        action_text = axis.text(0, -0.13, "", ha="center", fontsize=10)
        artists.append((cart, pole, pivot, control, stats, action_text))

    path = directory / "policy_comparison.mp4"
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    # Keep the demo watchable even when numerical evaluation uses many episodes.
    video_episodes = min(4, config.eval_episodes)
    try:
        with writer.saving(fig, str(path), dpi=100):
            for episode in range(video_episodes):
                for step in frame_steps:
                    heading.set_text(
                        f"Episode {episode + 1}/{video_episodes}  |  "
                        f"Simulation time {step * env.dt:.2f} s  |  "
                        f"{config.total_steps:,} requested training steps"
                    )
                    for rollout, parts in zip((before, after), artists, strict=True):
                        cart, pole, pivot, control, stats, action_text = parts
                        x, angle = rollout["qpos"][episode, step]
                        cart.set_xy((x - cart_half_width, cart_height - cart_half_height))
                        pole.set_data(
                            [x, x + pole_length * math.sin(angle)],
                            [cart_height, cart_height + pole_length * math.cos(angle)],
                        )
                        pivot.set_data([x], [cart_height])
                        action = float(rollout["actions"][episode, step, 0])
                        control.set_data([0, action], [0.1, 0.1])
                        action_text.set_text(f"Control input: {action:+.2f}  (range -1 to +1)")
                        ended = step >= int(rollout["length"][episode])
                        stats.set_text(
                            f"Return: {rollout['returns'][episode, step]:.1f}  |  "
                            f"Cart: {x:+.2f} m  |  "
                            f"Pole: {math.degrees(math.atan2(math.sin(angle), math.cos(angle))):+.1f}°"
                            + ("  |  End" if ended else "")
                        )
                    writer.grab_frame()
                    if episode == 0 and step == frame_steps[len(frame_steps) // 2]:
                        fig.savefig(directory / "policy_comparison.png", dpi=100)
    finally:
        plt.close(fig)
    metadata = {
        "path": path.name,
        "renderer": "matplotlib_2d_from_mujoco_qpos",
        "fps": fps,
        "playback_speed": speed,
        "frame_count": video_episodes * len(frame_steps),
        "episode_indices": list(range(video_episodes)),
        "control_dt": env.dt,
        "seed": config.seed,
        "evaluation_seed_fold_in": 1_000_003,
        "episode_returns_before": before["returns"][:video_episodes, -1].tolist(),
        "episode_returns_after": after["returns"][:video_episodes, -1].tolist(),
    }
    write_json(directory / "video.json", metadata)
    print(f"Video: {path.resolve()}", flush=True)
    return metadata


def model_segments(model, data) -> list:
    """Body links and primitive outlines in world coordinates; no mesh renderer."""
    import itertools

    import mujoco
    import numpy as np

    segments = [
        [data.xpos[parent], data.xpos[body]]
        for body, parent in enumerate(model.body_parentid)
        if parent > 0
    ]
    corners = np.array(list(itertools.product((-1, 1), repeat=3)))
    edges = [
        (a, b) for a in range(8) for b in range(a + 1, 8) if (corners[a] != corners[b]).sum() == 1
    ]
    for index, kind in enumerate(model.geom_type):
        center = data.geom_xpos[index]
        rotation = data.geom_xmat[index].reshape(3, 3)
        size = model.geom_size[index]
        if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            offset = rotation[:, 2] * size[1]
            segments.append([center - offset, center + offset])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            vertices = (corners * size) @ rotation.T + center
            segments.extend([[vertices[a], vertices[b]] for a, b in edges])
    return segments


def write_robot_comparison(env, before: dict, after: dict, config, directory: Path) -> dict:
    """Render a 3D link/geometry view of the exact evaluated robot trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import mujoco
    import numpy as np
    from matplotlib import animation
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    model = env.mj_model
    data = mujoco.MjData(model)
    stride = max(1, round(0.05 / env.dt))
    frame_steps = list(range(0, config.episode_length + 1, stride))
    if frame_steps[-1] != config.episode_length:
        frame_steps.append(config.episode_length)
    fps = 1 / (env.dt * stride)
    episodes = min(2, config.eval_episodes)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), subplot_kw={"projection": "3d"})
    fig.suptitle(f"{config.task} | PPO policy before and after training")
    fig.text(
        0.5, 0.02, "MuJoCo link/geometry view | Same evaluation starts | 1x speed", ha="center"
    )
    artists = []
    for axis, color in zip(axes, ("#bb6538", "#2269a5"), strict=True):
        lines = Line3DCollection([], colors=color, linewidths=3)
        axis.add_collection3d(lines, autolim=False)
        (target,) = axis.plot([], [], [], "*", color="#cc9900", markersize=12)
        axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", zlim=(0, 2))
        axis.view_init(elev=20, azim=-65)
        axis.set_box_aspect((1, 1, 0.8))
        artists.append((lines, target))

    path = directory / "policy_comparison.mp4"
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    titles = ("RESTORED POLICY" if config.resume else "INITIAL POLICY", "ATTEMPT END POLICY")
    try:
        with writer.saving(fig, str(path), dpi=100):
            for episode in range(episodes):
                for step in frame_steps:
                    for axis, (lines, target), rollout, title in zip(
                        axes, artists, (before, after), titles, strict=True
                    ):
                        data.qpos[:] = rollout["qpos"][episode, step]
                        data.mocap_pos[:] = rollout["mocap_pos"][episode, step]
                        data.mocap_quat[:] = rollout["mocap_quat"][episode, step]
                        mujoco.mj_kinematics(model, data)
                        lines.set_segments(model_segments(model, data))
                        target.set_data_3d(data.mocap_pos.T if model.nmocap else np.empty((3, 0)))
                        center = data.xpos[1, :2]
                        axis.set(
                            xlim=(center[0] - 1, center[0] + 1), ylim=(center[1] - 1, center[1] + 1)
                        )
                        axis.set_title(
                            f"{title}\nEpisode {episode + 1} | {step * env.dt:.2f}s | "
                            f"Return {rollout['returns'][episode, step]:.1f}"
                        )
                    writer.grab_frame()
                    if episode == 0 and step == frame_steps[len(frame_steps) // 2]:
                        fig.savefig(directory / "policy_comparison.png", dpi=100)
    finally:
        plt.close(fig)
    metadata = {
        "path": path.name,
        "renderer": "matplotlib_3d_mujoco_kinematics",
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
