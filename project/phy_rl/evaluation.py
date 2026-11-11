"""Score and retain the same held-out trajectories used for policy videos."""

from __future__ import annotations

import math


def capture_rollouts(env, make_policy, params, config) -> dict:
    """Keep positions and controls in memory for the same held-out episode seeds."""
    import jax
    import jax.numpy as jnp

    policy = make_policy(params, deterministic=True)

    def episode(key):
        reset_key, policy_key = jax.random.split(key)
        initial = env.reset(reset_key)

        def advance(carry, _):
            state, key = carry
            active = state.done == 0

            def step(carry):
                state, key = carry
                key, action_key = jax.random.split(key)
                action, _ = policy(state.obs, action_key)
                return env.step(state, action), key, action

            state, key, action = jax.lax.cond(
                active,
                step,
                lambda carry: (*carry, jnp.zeros(env.action_size)),
                (state, key),
            )
            reward = jnp.where(active, state.reward, 0)
            task_metrics = {
                name: jnp.where(active, value, 0) for name, value in state.metrics.items()
            }
            return (state, key), (
                state.data.qpos,
                reward,
                action,
                active,
                state.data.mocap_pos,
                state.data.mocap_quat,
                task_metrics,
            )

        _, (qpos, rewards, actions, active, mocap_pos, mocap_quat, task_metrics) = jax.lax.scan(
            advance, (initial, policy_key), None, length=config.episode_length
        )
        return {
            "qpos": jnp.concatenate((initial.data.qpos[None], qpos)),
            "returns": jnp.concatenate((jnp.zeros(1), jnp.cumsum(rewards))),
            "actions": jnp.concatenate((jnp.zeros((1, env.action_size)), actions)),
            "length": active.sum(),
            "mocap_pos": jnp.concatenate((initial.data.mocap_pos[None], mocap_pos)),
            "mocap_quat": jnp.concatenate((initial.data.mocap_quat[None], mocap_quat)),
            **{f"metric/{name}": values for name, values in task_metrics.items()},
        }

    key = jax.random.fold_in(jax.random.PRNGKey(config.seed), 1_000_003)
    keys = jax.random.split(key, config.eval_episodes)
    return jax.device_get(jax.jit(jax.vmap(episode))(keys))


def evaluate_policy(env, make_policy, params, config) -> tuple[dict, dict]:
    """Return evaluation metrics and their exact trajectories, without saving a policy."""
    import numpy as np

    rollouts = capture_rollouts(env, make_policy, params, config)
    if not all(np.isfinite(value).all() for value in rollouts.values()):
        raise ValueError("Policy evaluation produced non-finite states, actions, or returns")
    returns = rollouts["returns"][:, -1]
    lengths = rollouts["length"]
    result = {
        "episode_returns": returns.tolist(),
        "episode_lengths": lengths.tolist(),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_length": float(lengths.mean()),
        "seed": config.seed,
        "evaluation_seed_fold_in": 1_000_003,
        "deterministic_actions": True,
        "horizon": config.episode_length,
        "horizon_seconds": config.episode_length * env.dt,
    }
    result["task_metrics"] = {
        name.removeprefix("metric/"): float((values.sum(axis=1) / np.maximum(lengths, 1)).mean())
        for name, values in rollouts.items()
        if name.startswith("metric/")
    }
    if config.task in {"CheetahRun", "WalkerWalk"}:
        position_index = env.mj_model.joint("rootx").qposadr[0]
        displacement = (
            rollouts["qpos"][:, -1, position_index] - rollouts["qpos"][:, 0, position_index]
        )
        result["task_metrics"]["forward_speed_m_per_s"] = float(
            (displacement / (np.maximum(lengths, 1) * env.dt)).mean()
        )
    if config.task == "CartpoleBalance":
        qpos = rollouts["qpos"]
        angles = np.abs(np.arctan2(np.sin(qpos[:, :, 1]), np.cos(qpos[:, :, 1])))
        carts = np.abs(qpos[:, :, 0])
        angle_limit = math.radians(12)
        cart_limit = 1.0
        valid = np.arange(config.episode_length)[None, :] < lengths[:, None]
        balanced = (angles[:, 1:] <= angle_limit) & (carts[:, 1:] <= cart_limit) & valid
        max_angles, max_carts = angles.max(axis=1), carts.max(axis=1)
        success = (
            (max_angles <= angle_limit)
            & (max_carts <= cart_limit)
            & (lengths == config.episode_length)
        )
        result["balance"] = {
            "criteria": {"max_abs_pole_angle_deg": 12.0, "max_abs_cart_position_m": cart_limit},
            "episode_max_abs_pole_angle_deg": np.degrees(max_angles).tolist(),
            "episode_max_abs_cart_position_m": max_carts.tolist(),
            "episode_balanced_fraction": balanced.mean(axis=1).tolist(),
            "episode_success": success.tolist(),
            "success_count": int(success.sum()),
            "episode_count": config.eval_episodes,
        }
    return result, rollouts
