# Reduced V1: MuJoCo experiments on Daytona

## Approach

Build one training entrypoint and reuse it throughout: manually, through a local experiment runner, and inside Daytona GPU sandboxes.

Use Python 3.14, as selected during local setup, an existing MuJoCo Playground task, MJX, and PPO. Use small CPU settings locally and larger parallel simulation batches on Daytona. Verify dependency and trainer compatibility in the first milestone; justify any need for a different Python version with a concrete dependency limitation.

Keep the existing constraints: Daytona Volumes for cloud artifacts, two concurrent experiments initially, and a planned testing budget of up to $150.

## Mandatory confirmation between steps

Implement one numbered step at a time. Steps 1 and 2 have been authorized and
completed. Step 3 requires separate confirmation.

After completing each step, summarize its implementation, commands, verification results, artifacts, and limitations. Explicitly ask whether to proceed to the next numbered step and wait for the user's confirmation. Do not implement later-step features early or treat silence as approval. Fixes and verification required to finish the current step remain within its scope.

## Milestones

### 1. Manual local training, without checkpoints

Status: complete on 2026-11-06 with Python 3.14.8. The corrected default trains
CartpoleBalance with MJX-JAX and Brax PPO on CPU for 8,388,608 steps. It balances
for 20 seconds in all 32 held-out episodes: worst pole deviation 1.85°, worst cart
displacement 0.181 m, and mean return 1,998.33 / 2,000. The earlier two-second
reward-improvement check was insufficient and has been replaced with an explicit
sustained-balance criterion. The video shows the exact evaluated trajectories.
See [README.md](README.md) for commands and results.

- Run one short MuJoCo training session on this Mac’s CPU.
- Accept explicit task, seed, training target, and hyperparameters.
- Record the resolved configuration and training metrics locally.
- Inspect reward/loss curves and evaluate the resulting policy in the same process.
- Optionally record an MP4 comparison of initial and trained policies with `--video`, using identical held-out starting states and no saved policy.
- No checkpoint saving, scheduler, or cloud infrastructure.

Done when: one documented command trains a policy and produces understandable results.

### 2. Manual local training, with checkpoints

Status: complete on 2026-11-06. `--checkpoint-interval-steps` defaults to 1
completed PPO update and accepts positive integers; the final update is always
saved. Full learner/RNG checkpoints are committed atomically with checksums.
`--resume` inherits the original configuration and target, with separate attempt
artifacts and absolute progress. A real SIGKILL at update 3 / 192 environment
steps was followed by a fresh-process restore and completion at update 8 / 512
steps. All learner/RNG leaves and Adam's cumulative update counter were verified.
The twelve lightweight tests and three real CPU tests passed. See README for
commands, the pinned Brax adaptation, and the simulation-reset limitation.
Await confirmation before step 3.

- Add periodic checkpoint saving and an explicit --resume option.
- Save model, optimizer, normalization, random state, and completed training progress.
- Save only at complete training-update boundaries.
- Stop the process and manually launch a new process from a checkpoint.
- Continue toward the original training target.

Done when: resumed training restores learner state and progresses beyond the saved update.

Simulation episodes may reset on resume; exact trajectory replay is not required.

### 3. Manual Daytona training, without checkpoints

- Package the same training entrypoint and dependencies into a reproducible GPU image.
- Manually create one spot GPU sandbox and launch training.
- Confirm GPU use, inspect logs, and collect metrics if the run completes.
- Provide explicit creation, execution, inspection, and deletion commands.
- Set a sandbox TTL to bound resource lifetime.

Done when: a real GPU training run completes on Daytona and its sandbox is deleted.

If spot termination occurs, record the interruption and manually rerun. No recovery logic yet.

### 4. Manual Daytona training, with checkpoints

- Mount a Daytona Volume under an experiment-specific path.
- Persist checkpoints and metrics independently of sandbox storage.
- Write checkpoint payloads to unique locations and publish a manifest afterward; validate completeness and checksums before loading.
- Manually create a replacement sandbox, mount the same experiment data, and resume.
- Collect results before deleting compute resources; retain the volume.

Done when: a checkpoint survives sandbox deletion and a manually launched replacement continues training.

Termination remains unmanaged. This milestone proves the storage and resume mechanism that automation will later use.

### 5. Simple local experiment control plane

- Expand YAML seeds and hyperparameter combinations into immutable experiment configurations.
- Run each experiment as a local subprocess using the established training entrypoint.
- Enforce a concurrency limit, initially two.
- Give each experiment separate metrics and checkpoint directories.
- Record basic experiment/attempt metadata in local SQLite.
- Display queued, running, succeeded, failed, and cancelled status.
- Support cancellation and subprocess cleanup.

Done when: a four-experiment grid runs locally with two overlapping processes, isolated artifacts, and clear outcomes.

The database records progress, but automatic reconciliation after controller restart is deferred to milestone 8.

### 6. Experiment grids on Daytona spot GPUs

- Add a Daytona execution backend to the same grid runner.
- Allocate one GPU sandbox per experiment.
- Launch up to the concurrency limit and dispatch queued experiments as slots become free.
- Mount experiment-specific volume paths and enable checkpointing.
- Capture sandbox IDs, logs, outcomes, and timing information.
- Delete sandboxes after completion, failure, or cancellation.
- Mark terminated experiments as interrupted; do not reschedule them.

Done when: four experiments run across at most two concurrent GPU sandboxes, with persistent artifacts and verified cleanup.

### 7. Spot termination supervision and automatic resume

- Monitor sandbox and worker status.
- Distinguish confirmed worker loss from temporary API unavailability.
- On confirmed termination, queue a replacement attempt.
- Mount the same experiment data and restore the latest valid checkpoint.
- Preserve experiment identity while assigning a new attempt and sandbox ID.
- Bound retries and runtime; back off when spot capacity is unavailable.
- Avoid retrying invalid configurations or deterministic workload errors.
- Add deliberate sandbox deletion after a nonzero checkpoint for testing.

Done when: an interrupted experiment automatically resumes in a replacement sandbox while other experiments continue.

This milestone assumes the controller remains running.

### 8. Control-plane restart recovery

- Persist provisioning intent, sandbox identity, attempt state, cancellation intent, and cleanup obligations.
- Enforce one active controller per database.
- On restart, reconcile stored state with Daytona before dispatching work.
- Reattach to healthy workers, recognize completed work, and recover interrupted experiments.
- Resolve ambiguous creation or launch outcomes before issuing replacements.
- Retry outstanding cleanup.
- Use a temporary CPU artifact-reader sandbox when persisted results need inspection and no suitable worker remains.

Done when: force-killing and restarting the controller preserves the queue, adopts existing workers, recovers lost workers, and creates no duplicate executions.

Recovery assumes the local database survives; loss of the entire controller machine remains outside V1.

## Shared rules

- One experiment: one resolved task, seed, hyperparameter configuration, and total training target.
- One attempt: one execution process or sandbox; retries remain part of the same experiment.
- Checkpoint recovery: restore validated learner progress, never an incomplete checkpoint.
- Metrics: preserve attempt identity so repeated work after rollback remains visible.
- Cleanup: remove compute resources while retaining experiment artifacts.
- Spot behavior: no automatic fallback to on-demand capacity.
- Manual stages: use direct commands and small scripts; introduce orchestration only at milestone 5.

## V1 acceptance and deferred work

The final demonstration submits four experiments with concurrency two, interrupts one sandbox, automatically resumes its experiment, restarts the controller, and finishes with retained results and no owned compute resources running.

Test checkpoint integrity in milestone 2, volume persistence in milestone 4, scheduling limits in milestone 5, worker recovery in milestone 7, and controller reconciliation in milestone 8.

Defer W&B, ClearML, Isaac Lab, custom dashboards, sophisticated scheduling, and extensive cost benchmarking until these eight milestones work.
