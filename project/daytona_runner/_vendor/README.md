# Brax PPO trainer

`brax_ppo.py` is derived from `brax/training/agents/ppo/train.py` at Brax revision
`303d89dfac42919e5fde7c91f24c28a505373b24`, under Apache-2.0 (`LICENSE.brax`).
The rest of Brax is imported from the published `brax==0.14.2` wheel. The vendored
trainer includes the JAX device-placement compatibility fix from that revision;
the published release's upstream PPO trainer still calls the removed APIs.

Local changes expose full learner restore/save callbacks, execute one complete
PPO update per host call, retain the RNG carried between updates, and resume
absolute progress and the original evaluation schedule. Checkpoints include the
learner, training/evaluation random keys, and environment reset keys. Simulation
state is deliberately reset on restore. The PPO loss, rollout, normalization,
and optimizer computations remain upstream code.

This adapter supports this project's single-process, single-device configuration
with `num_resets_per_eval=0`. It rejects other topologies. When upgrading Brax,
verify this adapter against the installed release; do not patch the installed
dependency.
