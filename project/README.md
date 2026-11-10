# Daytona Experiment Runner

Step 2 implements manual **CPU MuJoCo training with full-state checkpoints and
explicit resume**. The workload is MuJoCo Playground's `CartpoleBalance`, simulated
using MJX-JAX and trained with a minimally adapted, pinned Brax PPO trainer.
Optional Weights & Biases tracking reports metrics, evaluations, and videos.
`daytona_test.py` runs a single sandbox experiment without checkpoints.
There is no experiment scheduler yet.

**Step 1 baseline:** the default policy balances Cartpole for **20 simulated
seconds in all 32 held-out evaluation episodes**. The worst pole deviation is
1.85° and the worst cart displacement is 0.181 m. Mean return is 1,998.33 / 2,000.
Evaluation measures physical stability throughout each episode, not just reward
improvement. Video frames and evaluation metrics come from the same trajectories.
Step 2 adds checkpoint/recovery verification on short CPU workloads; the balance
measurements below are from the retained step 1 run.

## Install and run

From this directory:

```sh
uv sync --locked --python 3.14
uv run --no-sync python -m daytona_runner.train
```

Python 3.14 is the selected runtime. In this workspace, Nix/devenv sets
`UV_PROJECT_ENVIRONMENT` to the parent `.devenv/state/venv`; the commands above
use that environment. Outside devenv, uv defaults to a project-local `.venv`.
To use a separate environment here, prefix both commands with
`UV_PROJECT_ENVIRONMENT=.venv`. Use `uv run --no-sync` to select the installed
environment even when a previously opened shell still points to Python 3.15.
No cloud credentials are needed for local training.

Select the training device with mutually exclusive flags:

```sh
uv run --no-sync python -m daytona_runner.train --cpu  # Default when omitted
uv run --no-sync python -m daytona_runner.train --gpu  # NVIDIA CUDA
```

`--gpu` requires an available NVIDIA GPU and a compatible CUDA-enabled JAX
installation; it fails if GPU execution is unavailable. Device selection also
applies to resumed runs and defaults to CPU unless `--gpu` is supplied. Changing
devices in an already initialized Python process requires starting a new process.
GPU training has not yet been verified on Daytona. GPU installation requirements:
[JAX documentation](https://docs.jax.dev/en/latest/installation.html).

The Playground and Brax releases are pinned in `pyproject.toml`,
and `uv.lock` records the resolved dependencies. Use `uv sync --locked` to retain
that resolution. The released `playground==0.2.0` exposes the
JAX/MJX API used here; this project does not follow Playground's changing GitHub
main branch. Brax is installed from the published `brax==0.14.2` wheel.
The vendored PPO trainer is derived from upstream revision
`303d89dfac42919e5fde7c91f24c28a505373b24` and includes its fix for removed JAX
device-placement APIs. Training uses this adapter; the published release's
upstream PPO trainer still calls those removed APIs.

The default uses seed 11, 128 parallel simulated environments, 8,388,608 requested
training steps, nine evaluations during training, and 2,000-step (20-second)
episodes. Final evaluation uses 32 fixed held-out starting states. Training defaults
to CPU execution and uses 64-by-64 policy/value networks.

Explicit settings matching the default:

```sh
uv run --no-sync python -m daytona_runner.train \
  --task CartpoleBalance --seed 11 \
  --total-steps 8388608 --num-envs 128 --episode-length 2000 \
  --learning-rate 0.001 --entropy-cost 0.01 \
  --reward-scaling 10 --discounting 0.995 \
  --unroll-length 32 --batch-size 128 --num-minibatches 8 \
  --updates-per-batch 8 --num-evals 9 --eval-episodes 32 \
  --video --output runs
```

`--help` lists the remaining rollout, minibatch, and evaluation controls.
`CartpoleSwingup` is also selectable, but balancing is the initial validation task.
Batch size times minibatch count must be divisible by the number of environments.
Brax may round the requested step count upward to finish complete training batches;
the actual count is recorded separately.

## Single Daytona sandbox

Set `DAYTONA_API_KEY` in `.env` or your environment, then run:

```sh
uv run --no-sync python daytona_test.py        # 4 CPUs, 8 GiB RAM
uv run --no-sync python daytona_test.py --gpu  # One H100; requires GPU access
```

The script builds a Python 3.14 image using `uv.lock`, runs the default
`CartpoleBalance` training with **all checkpoint saves disabled**, downloads
metrics and evaluation results to `runs/daytona-<sandbox-id>.tar.gz`, and deletes
the sandbox in `finally`. Training has a one-hour timeout and the sandbox has a
65-minute lifetime limit. Output is printed when training finishes. Use
`--total-steps 32768` for a shorter run; this does not establish balancing quality.
The GPU image adds CUDA 12 support for the locked JAX version. Cloud execution
and GPU compatibility have not yet been verified.

The local trainer also accepts `--no-checkpoints`; checkpointing remains enabled
by default. No volume or resume workflow is involved in `daytona_test.py`.
The launcher uses the installed [Daytona Python SDK](https://www.daytona.com/docs/python-sdk)
through its `v0` namespace.

## Weights & Biases

Install the updated dependencies with `uv sync --locked --python 3.14`. Set
`WANDB_API_KEY` in `.env`, along with the destination:

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=deep-truck7569-deep-truck-industries
WANDB_PROJECT="Daytona test"
```

Then enable tracking:

```sh
uv run --no-sync python -m daytona_runner.train --wandb --video
```

The script loads `.env` from the working directory, falling back to this project's
`.env`. Existing environment variables take precedence. `.env` is ignored by Git;
the key is never included in run configuration or uploaded result files. Local
training without `--wandb` does not initialize W&B or load `.env`.

`--wandb-project` and `--wandb-entity` override the environment for a new experiment.
Without a project setting, the default is `daytona-experiment-runner`; without an
entity setting, W&B uses the account default. The W&B run URL is printed and saved
in the attempt's `wandb.json`.

Workers log PPO reward/loss metrics directly to W&B, with `env_steps` as the chart
x-axis. Held-out before/after evaluations include mean return and balance success
rate. The run also receives the training configuration, final summary, learning
curve image, explicit JSON/CSV result files, and `policy/comparison` video when
`--video` is enabled. Checkpoint payloads remain local; W&B is not used for recovery.

```sh
# Inherits tracking settings and resumes the same online W&B run.
uv run --no-sync python -m daytona_runner.train \
  --resume runs/<attempt>/checkpoints/latest.json

# Continue locally without tracking this attempt.
uv run --no-sync python -m daytona_runner.train \
  --resume runs/<attempt>/checkpoints/latest.json --no-wandb
```

The checkpoint retains the W&B run ID, project, and entity. Resumed attempts keep
that destination even if `.env` changes; conflicting explicit destination flags
are rejected. Tracking options do not change learner checkpoint compatibility,
so `--wandb` can also be added when resuming an older, untracked checkpoint.
Run only one attempt per experiment at a time.

W&B's internal log counter is separate from `env_steps`. Resuming an older
checkpoint appends observations, including replayed steps, with a new `attempt_id`;
it does not overwrite earlier history. Result files are separated by attempt.
Exceptions and Ctrl-C finalize tracking with a failure/interrupted status.
An upload error during finalization is reported without discarding local results
or masking a training error. Abrupt process termination cannot finalize uploads.

For a fresh run without uploading, use `--wandb --wandb-mode offline`; W&B stores
its logs under the attempt directory. W&B still requires its local background
service in offline mode. Resuming W&B history requires online mode; use
`--no-wandb` to resume training without that service.

## Inspect results

To watch the policy control the cart and pole, record a before/after video during
the training run:

```sh
uv run --no-sync python -m daytona_runner.train --video
```

This produces `policy_comparison.mp4`: a side-by-side 2D view of actual MuJoCo
positions, showing the first four of the 32 held-out episodes with identical
starting states. Episodes are selected in order, not by their scores.
The left panel uses the attempt's starting policy (restored when resuming) and
the right panel uses its final policy. Cumulative reward, cart position, and control input are displayed.
Playback is labeled **1x speed**; the default video lasts about 80 seconds.
The frames use the exact trajectories scored in the evaluation JSON. Pole angle
is displayed alongside the control input, cart position, and cumulative return.

Video encoding requires `ffmpeg` on `PATH` with the `libx264` encoder (already
available in this workspace). Rendering uses Matplotlib on CPU and needs no
OpenGL display. This is a 2D visualization, not MuJoCo's native 3D renderer.
Recording compares the policy at the start and end of this attempt. On resume,
the starting policy is restored from the checkpoint. A completed checkpoint can
also be loaded with `--resume ... --video`; this performs evaluation/video only,
without additional training.

Each run creates a new directory under `runs/` and prints its absolute path. Runs
with the same task and seed never overwrite each other.

| Artifact | Contents |
| --- | --- |
| `requested_config.json` | Effective inputs, including inherited resume settings |
| `attempt.json` | Experiment/attempt identity, starting progress, source checkpoint |
| `wandb.json` | W&B run ID, destination, URL, and mode when tracking is enabled |
| `checkpoints/update-*/state.msgpack` | Full learner state and random keys |
| `checkpoints/update-*/manifest.json` | Progress, configuration, payload size and SHA-256 |
| `checkpoints/latest.json` | Atomically published pointer to the latest committed checkpoint |
| `checkpoint_status.json` | Latest save location, progress, and save duration |
| `resolved_config.json` | Task settings, PPO settings, network architecture, installed versions, and device |
| `metrics.jsonl` | Progress observations, flushed after each callback |
| `metrics.csv` | The same observations in a spreadsheet-friendly format |
| `learning_curves.png` | Evaluation rewards and PPO losses over training steps |
| `evaluation_before.json` | Attempt-start policy evaluation, training step, and policy origin |
| `evaluation_after.json` | Trained policy evaluated on the same seeds and horizon |
| `policy_comparison.mp4` | Before/after policy video when `--video` is supplied |
| `policy_comparison.png` | Preview frame from the video |
| `video.json` | Playback settings, seed, frame count, and depicted episode returns |
| `summary.json` | Evaluation change, actual steps, parameter change, and wall time |
| `status.json` | Running, completed, failed, or interrupted |
| `error.txt` | Failure traceback, when applicable |

Open `learning_curves.png` and inspect `summary.json`. Both baseline and final
evaluation occur in the same process with deterministic actions. The summary
reports return improvement and sustained balance. The balance criterion requires
**every step of the full 20 seconds** to stay within 12° of upright and ±1 m of
centre. It also reports each episode’s maximum angle and cart displacement.
A short PPO run is not guaranteed to improve or satisfy this criterion.
The policy parameter-change measurement verifies that learning actually updated
the network. This is a functional example, not a statistically conclusive benchmark.

### Observed local results

Verified on 2026-11-06 using Python 3.14.8, Playground 0.2.0, JAX/JAXlib 0.12.0.1,
MuJoCo/MJX 3.17.0, and Brax revision `303d89dfac42919e5fde7c91f24c28a505373b24`.
Both policies use the same 32 held-out starts, deterministic actions, and a
20-second horizon.

| Metric | Initial policy | Trained policy |
| --- | --- | --- |
| Episodes balanced for the full 20 seconds | 0 / 32 | 32 / 32 |
| Mean return (maximum 2,000) | 496.01 | 1,998.33 |
| Minimum episode return | 477.41 | 1,996.48 |
| Worst pole deviation from upright | 180.00° | 1.85° |
| Worst absolute cart displacement | 1.834 m | 0.181 m |

The prior 1,048,576-step run used two-second episodes and a much smaller training
batch. Its reward improvement did **not** demonstrate balancing; the video made
that failure clear. The current defaults replace it with longer episodes, larger
batches, more updates, and PPO settings closer to Playground's bundled configuration.

These results cover the task's standard small random initial offsets, with one
training seed and 32 evaluation episodes. They do not establish recovery from
large disturbances or robustness across different training seeds.

The successful run, including the full 32-episode evaluation and the video of the
first four episodes, is retained locally under `runs/` (ignored by Git):

`CartpoleBalance-seed11-20261106T153116784486Z-3a5c2fc5`

The reward plot uses Brax's evaluation episodes. The summary and video use the
separate fixed held-out episodes, so plot and summary reward values differ.

The retained step 1 run has no checkpoint, so its policy cannot be recovered from
metrics. New runs save full learner checkpoints as described below.

## Save and resume

By default, save after **every completed PPO update**. Here a checkpoint “step”
means one rollout batch followed by all its PPO minibatch optimizer passes, not
one simulator step or one Adam update. With the default settings, one PPO update
advances `128 × 8 × 32 = 32,768` environment steps. Evaluation frequency does not
control checkpoint frequency.

```sh
# Save every update (the default).
uv run --no-sync python -m daytona_runner.train

# Save every five completed PPO updates.
uv run --no-sync python -m daytona_runner.train --checkpoint-interval-steps 5

# After interrupting a run, resume its latest committed checkpoint.
uv run --no-sync python -m daytona_runner.train \
  --resume runs/<attempt>/checkpoints/latest.json
```

The interval must be a positive integer. Saves occur at absolute update counts
that are multiples of the interval. The final update is always saved, even when
it is not a multiple. All checkpoints are retained; frequent saves use more disk
space and add synchronous serialization/fsync overhead.

`--resume` also accepts a checkpoint directory or an explicit `manifest.json`.
It inherits the task, seed, hyperparameters, evaluation schedule, and original
training target. Conflicting training overrides are rejected. Runtime options
`--output`, `--video`, `--checkpoint-interval-steps`, and `--cpu`/`--gpu` may change. For example,
resuming at 3,145,728 of 8,388,608 environment steps continues toward 8,388,608;
it does not request another 8,388,608 steps. Existing Brax batch/epoch rounding
still applies. A checkpoint already at the target performs no additional updates.

Every launch creates a new attempt directory and keeps the experiment identity.
Metrics use absolute environment-step counts; earlier attempts remain intact,
including observations of work subsequently lost after the last checkpoint.
`evaluation_before.json` identifies whether the attempt began with a random or
restored policy. Summaries report `starting_step`, `steps_this_attempt`, and the
latest checkpoint. Ctrl-C retains committed checkpoints and marks the attempt
interrupted. SIGKILL cannot update `status.json`; it may still say running, but
its committed checkpoints remain usable for manual resume.

Each checkpoint contains policy/value parameters, Adam moments and counters,
normalization state, completed environment steps, training PRNG streams,
evaluation RNG, and environment reset keys. Save uses Flax MessagePack
serialization, already supplied by the training dependencies. No pickle is used.
The current learning rate is constant; there is no separate scheduler state.
Simulator episodes reset on resume. Exact trajectory replay is not guaranteed.
Compilation and evaluation setup happen again in each process.

Payloads and manifests are written and fsynced in a temporary directory, renamed
to an immutable checkpoint directory, and then published through an atomic
replacement of `latest.json`. Loads verify size, SHA-256, format, configuration,
dependency versions, trainer source, and array shapes/dtypes. Temporary directories
are ignored. A corrupt selected checkpoint fails clearly; an older valid manifest
can be selected explicitly. These guarantees currently cover the local filesystem;
Daytona Volume persistence will be verified separately in step 4.

Brax's upstream checkpoint API saves network/normalizer parameters but omits the
optimizer and progress/RNG state. The small trainer adaptation is documented in
[the vendor notes](daytona_runner/_vendor/README.md). It retains the existing PPO
math and supports one process/device. Checkpoint compatibility is deliberately
strict: different trainer source or dependency versions require a new experiment.

## Verification

The lightweight tests require only Python's standard library:

```sh
uv run --no-sync python -m unittest discover -s tests -v
```

They validate input rejection, metric integrity, output isolation, checkpoint
publication failures, corruption, resume configuration, and failure handling.
They mock training explicitly and do not establish RL compatibility.

After dependencies are installed, run a real, short CPU smoke test:

```sh
RUN_MUJOCO_SMOKE=1 uv run --no-sync \
  python -m unittest tests.test_training_smoke -v
```

Then run the default command above and inspect its retained results. The smoke
tests check genuine policy updates, CPU execution, evaluation/plot artifacts, and
checkpoint cadence. A real subprocess is SIGKILLed after a committed checkpoint;
a fresh subprocess restores every saved learner/RNG leaf and completes the
original target. Adam's update count verifies optimizer continuation. Resuming the
finished checkpoint must perform zero additional training. The tests also verify
that high reward, temporary recovery after exceeding a balance limit, and
premature termination cannot pass the balance criterion.

Verified locally on 2026-11-06: all twelve lightweight tests and all three opt-in
tests passed. The recovery test committed update 3 at 192 environment steps,
terminated with SIGKILL, restored every learner/RNG leaf exactly, and completed
the original target at update 8 / 512 environment steps. Changing the save
interval from 3 to 5 on resume produced checkpoints at updates 5 and 8, including
the final off-cadence save. Adam retained all 32 minibatch optimization steps.
Loading that final checkpoint performed zero further updates. These short runs
verify recovery mechanics, not reward improvement or balancing performance.

W&B integration verification on 2026-11-07: all 19 lightweight tests and all four
opt-in CPU tests passed. The new test exercises real training, checkpoint resume,
video encoding, and W&B media objects while replacing only the remote run
transport. It verifies the same W&B identity across attempts and preserves
replayed environment steps. Live upload verification was blocked by this agent
environment's restriction on the local sockets required by `wandb-core`; these
tests do not establish dashboard connectivity.

## Milestone boundary

Step 2 implements local checkpoint saving and manual resume. **Ask for confirmation
before implementing step 3 (manual Daytona training)**. Follow the explicit
confirmation rule between later steps in [the reduced V1 plan](codex_plan_2.md).

Upstream references:

- [MuJoCo Playground release](https://pypi.org/project/playground/0.2.0/)
- [Brax release](https://pypi.org/project/brax/0.14.2/)
- [Brax compatibility fix included in the vendored trainer](https://github.com/google/brax/commit/303d89dfac42919e5fde7c91f24c28a505373b24)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
