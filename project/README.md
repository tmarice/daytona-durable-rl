# Daytona Experiment Runner

Step 2 implements manual **CPU MuJoCo training with full-state checkpoints and
explicit resume**. The workload is MuJoCo Playground's `CartpoleBalance`, simulated
using MJX-JAX and trained with a minimally adapted, pinned Brax PPO trainer.
Optional Weights & Biases tracking reports metrics, evaluations, and videos.
`daytona_test.py` runs a single sandbox experiment without checkpoints.
`main.py` runs a bounded GPU training sweep through the existing `daytona_runner.py`
library, including Ctrl+C cleanup. The RL implementation lives in `phy_rl`.

**Step 1 baseline:** the default policy balances Cartpole for **20 simulated
seconds in all 32 held-out evaluation episodes**. The worst pole deviation is
1.85° and the worst cart displacement is 0.181 m. Mean return is 1,998.33 / 2,000.
Evaluation measures physical stability throughout each episode, not just reward
improvement. Video frames and evaluation metrics come from the same trajectories.
Step 2 adds checkpoint/recovery verification on short CPU workloads; the balance
measurements below are from the retained step 1 run.

## Install and run

The runner and RL training are independent uv projects, each with its own
`pyproject.toml`, `uv.lock`, and environment:

- The root project installs Daytona and dotenv for the runner and sandbox launchers.
- `phy_rl` installs Playground, Brax, Matplotlib, W&B, and dotenv for training.

Run all commands below from this directory. In Nix/devenv, unset the shared
environment override first so syncing one project does not replace the other's
dependencies:

```sh
unset UV_PROJECT_ENVIRONMENT
uv sync --locked --python 3.14                  # Runner: .venv
uv sync --project phy_rl --locked --python 3.14  # Training: phy_rl/.venv
uv run --project phy_rl --no-sync python -m phy_rl.train
```

Python 3.14 is the selected runtime. Use `--project phy_rl` to select the training
environment while keeping the repository root as the working directory, so module
imports and existing `runs/` paths continue to work. No cloud credentials are
needed for local training. Both projects install dependencies only; source code
runs directly from the checkout.

Select the training device with mutually exclusive flags:

```sh
uv run --project phy_rl --no-sync python -m phy_rl.train --cpu  # Default when omitted
uv run --project phy_rl --no-sync python -m phy_rl.train --gpu  # NVIDIA CUDA
```

`--gpu` requires an available NVIDIA GPU and a compatible CUDA-enabled JAX
installation; it fails if GPU execution is unavailable. Device selection also
applies to resumed runs and defaults to CPU unless `--gpu` is supplied. Changing
devices in an already initialized Python process requires starting a new process.
GPU training has not yet been verified on Daytona. GPU installation requirements:
[JAX documentation](https://docs.jax.dev/en/latest/installation.html).

The Playground and Brax releases are pinned in `phy_rl/pyproject.toml`,
and `phy_rl/uv.lock` records the resolved training dependencies. Use
`uv sync --project phy_rl --locked` to retain that resolution. The released
`playground==0.2.0` exposes the JAX/MJX API used here; this project does not follow Playground's changing GitHub
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
uv run --project phy_rl --no-sync python -m phy_rl.train \
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

The script builds a Python 3.14 image using `phy_rl/pyproject.toml` and
`phy_rl/uv.lock`, runs the default `CartpoleBalance` training with **all checkpoint
saves disabled**, downloads
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

## GPU sweep through the existing runner

Set `DAYTONA_API_KEY` and the W&B settings below in `.env`, install the root
dependencies, and run locally:

```sh
uv run --no-sync python main.py
```

This submits twelve independent CartpoleSwingup PPO training jobs: learning rates 0.0003
and 0.001, entropy costs 0.001 and 0.01, and seeds 11, 22, and 33, in every
combination. All runs use the same task, network
architecture, and training budget. Change the single `task` value in `main.py`
to sweep another supported task. Two spot H100 sandboxes run concurrently, with one
GPU per job. Each Cartpole job requests 8,388,608 training steps, saves a full learner
checkpoint every PPO update and at completion, and evaluates 32 held-out episodes
before and after training.
JAX must report the GPU backend; unavailable CUDA fails instead of falling back.

The configuration uses `work_dir="phy_rl"` and `entrypoint="train:run"`. The runner
uploads that directory and installs dependencies from its `pyproject.toml`.
CUDA is a regular Linux dependency; no extra is required. The runner uses `uv.lock`
for the snapshot cache key, but its pip installation does not enforce the lockfile.

The CLI and runner configuration dictionaries select task-specific defaults from
`phy_rl/tasks.py`, then apply explicit overrides. CartpoleBalance and
CartpoleSwingup remain available with their existing defaults.

| Task | Training steps | Parallel environments | Episode steps | PPO updates |
| --- | ---: | ---: | ---: | ---: |
| CheetahRun | 15,728,640 | 2,048 | 1,000 | 16 |
| WalkerWalk | 7,864,320 | 2,048 | 1,000 | 8 |
| Go1JoystickFlatTerrain | 16,384,000 | 8,192 | 1,000 | 100 |
| PandaPickCube | 8,192,000 | 2,048 | 150 | 50 |

PPO settings and network sizes follow the installed Playground 0.2.0 recipes,
with shorter training budgets and fewer evaluations. Go1's actor uses `state`
observations and its critic uses `privileged_state`. These are simulation
experiments, not policies validated for physical robots. Neither convergence nor
one-minute H100 runtime is established. Compilation, asset downloads, evaluation,
checkpointing, and optional video encoding add to training time.

Run an individual task locally on a CUDA machine:

```sh
uv run --project phy_rl --no-sync python -m phy_rl.train --task CheetahRun --gpu --wandb
uv run --project phy_rl --no-sync python -m phy_rl.train --task PandaPickCube --gpu --video
```

Go1 and Panda download their robot assets from GitHub at Playground's pinned
MuJoCo Menagerie revision. Only the selected robot is downloaded, including its
license, into the attempt's `.assets` directory; git is not required. New attempts
need network access for these downloads. Assets are not uploaded to W&B.

Evaluation records each environment's metric averages over active steps, plus
forward speed in m/s for Cheetah and Walker. These are also logged under
`heldout/before/task/*` and `heldout/after/task/*` in W&B. Cartpole's balance
criterion is applied only to CartpoleBalance.

Checkpoints live at `/data/jobs/<job_id>/checkpoints` on the retained
`experiment_volume`. Attempt metrics, plots, and summaries are written to the
sandbox's local filesystem and uploaded to W&B before training returns. Jobs use
the project and entity from the local environment and the runner forwards the API
key as a secret, overwriting the stored `daytona-runner-WANDB_API_KEY` value on
each run. The local entrypoint prints the runner's results and logs.

Every runner attempt requests `spot=True`. While monitoring a job, spot eviction,
stopping, and deletion all trigger cleanup and a replacement sandbox. The retry
keeps the experiment ID and checkpoint directory, with a new attempt ID. GPU
sandboxes are deleted when stopped, so a missing sandbox also triggers recovery.
Use Ctrl+C to end the local runner; manually stopping a sandbox triggers a retry.
Eviction while a sandbox is still being created is retried as well.

Each attempt owns one sandbox named `job-<job_id>-<attempt prefix>` and deletes it
when the attempt ends. The name lets the runner find and delete a sandbox whose
creation timed out or was cancelled. A failed deletion is logged and left to the
TTL; it does not replace the job's result.

The runner retains its existing 65-minute sandbox TTL, with a separate
1,200-second image-construction timeout. `aclosing()` and `finally`
await sandbox deletion requests and close the SDK client on failure or Ctrl+C.
A second Ctrl+C can interrupt cleanup. Controller-crash recovery and concurrent
writers to the same checkpoint directory are deliberately deferred, as confirmed
for this task. Snapshots and the volume are retained for reuse.

## Weights & Biases

Install the training dependencies with
`uv sync --project phy_rl --locked --python 3.14`. Set
`WANDB_API_KEY` in `.env`, along with the destination:

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=deep-truck7569-deep-truck-industries
WANDB_PROJECT="Daytona test"
```

Then enable tracking:

```sh
uv run --project phy_rl --no-sync python -m phy_rl.train --wandb --video
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

The runner example enables W&B for each job and passes the project/entity from
the local environment. Every generated result file is uploaded under its attempt
directory, including `metrics.jsonl`, `metrics.csv`, both evaluation JSON files,
`summary.json`, the original plot PNGs, and video files when generated. The PNGs
also appear as W&B images. Failed attempts upload available results and `error.txt`.
Finalization waits for W&B to finish syncing before returning to the runner.

```sh
# Inherits tracking settings and resumes the same online W&B run.
uv run --project phy_rl --no-sync python -m phy_rl.train \
  --resume runs/<attempt>/checkpoints

# Continue locally without tracking this attempt.
uv run --project phy_rl --no-sync python -m phy_rl.train \
  --resume runs/<attempt>/checkpoints --no-wandb
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
uv run --project phy_rl --no-sync python -m phy_rl.train --video
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
OpenGL display. Cartpole uses the existing 2D visualization. The four new tasks
use a 3D view of body links and primitive geometry reconstructed from the recorded
MuJoCo positions, including Panda's target. These are schematic views, not native
textured MuJoCo renders. New-task videos show the first two held-out episodes.
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
| `checkpoints/checkpoint-*.json` | Session IDs, progress/configuration, and base64-encoded full learner state with size and SHA-256 |
| `checkpoints/checkpoint-*.json.complete` | Empty marker indicating that the JSON file was fully written and closed |
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
uv run --project phy_rl --no-sync python -m phy_rl.train

# Save every five completed PPO updates.
uv run --project phy_rl --no-sync python -m phy_rl.train --checkpoint-interval-steps 5

# After interrupting a run, resume its latest committed checkpoint.
uv run --project phy_rl --no-sync python -m phy_rl.train \
  --resume runs/<attempt>/checkpoints
```

The interval must be a positive integer. Saves occur at absolute update counts
that are multiples of the interval. The final update is always saved, even when
it is not a multiple. All checkpoints are retained; frequent saves use more disk
space and add synchronous serialization and write overhead.

`--resume` accepts a checkpoint directory or an explicit `checkpoint-*.json` file.
The file must have its companion `.complete` marker. Legacy `manifest.json` /
`latest.json` checkpoints are unsupported; existing files are left untouched.
It inherits the task, seed, hyperparameters, evaluation schedule, and original
training target. Conflicting training overrides are rejected. Runtime options
`--output`, `--video`, `--checkpoint-interval-steps`, and `--cpu`/`--gpu` may change. For example,
resuming at 3,145,728 of 8,388,608 environment steps continues toward 8,388,608;
it does not request another 8,388,608 steps. Existing Brax batch/epoch rounding
still applies. A checkpoint already at the target performs no additional updates.

Every launch creates a new attempt directory; resumed attempts keep the experiment identity.
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
serialization, already supplied by the training dependencies, then base64-encodes
the bytes inside the session JSON dictionary. No pickle is used.
The current learning rate is constant; there is no separate scheduler state.
Simulator episodes reset on resume. Exact trajectory replay is not guaranteed.
Compilation and evaluation setup happen again in each process.

`ExperimentSession` writes a uniquely named, sequentially numbered JSON file,
closes it, then creates its empty completion marker. Restore selects the highest
committed sequence; files without markers are ignored. This uses sequential writes
without symlinks, renames, or a mutable latest pointer. Loads verify size, SHA-256,
format, configuration, dependency versions, trainer source, and array shapes/dtypes.
A corrupt selected checkpoint fails clearly; an older valid JSON file can be
selected explicitly. Mounted-volume persistence across sandbox deletion still
requires separate live verification.

## ExperimentSession integration

`ExperimentSession` lives in `phy_rl/daytona_experiment_session.py` and can be imported without the
Daytona SDK. It accepts JSON-compatible dictionaries; workload-specific serialization
belongs in the training code.

```python
from phy_rl.daytona_experiment_session import ExperimentSession

session = ExperimentSession("experiment-1", "attempt-2", "/data/jobs/experiment-1/checkpoints")
previous = session.restore_checkpoint()  # dict, or None before the first save
path = session.save_checkpoint({"step": 100, "loss": 0.25})
```

Every training invocation constructs a session, including the local CLI. The
runner calls an entry point with the contract:

```python
run(
    config,
    experiment_id="experiment-1",
    attempt_id="attempt-2",
    checkpoint_dir="/data/jobs/experiment-1/checkpoints",
)
```

`phy_rl.train.run` accepts a `Config` or a configuration dictionary and optional
identity/storage keywords. `DaytonaJob` supplies its stable job ID as `experiment_id`,
a fresh `attempt_id` on each launch, and the same `/data/jobs/<job_id>/checkpoints`
directory across retries. The session is passed into training and its IDs are used
in attempt metadata, summaries, checkpoints, and W&B reporting.

An explicit `resume` selects that checkpoint; otherwise, a supplied checkpoint
directory is searched automatically. Missing checkpoints start fresh. Disabling
checkpoint saves still permits restoration. Local calls without identity arguments
generate a new attempt directory and save checkpoints inside it; explicit local
resume inherits the saved experiment ID. Supplied attempt IDs name their output
directories, and reusing an existing attempt directory is rejected.

The session assumes one active writer per experiment. Concurrent writer coordination,
visibility retries, automatic fallback from corrupt checkpoints, and retention cleanup
are intentionally deferred. Checkpoints use the supplied directory; other attempt
artifacts use `output` (default `runs`) and are uploaded when W&B is enabled.

Brax's upstream checkpoint API saves network/normalizer parameters but omits the
optimizer and progress/RNG state. The small trainer adaptation is documented in
[the vendor notes](phy_rl/_vendor/README.md). It retains the existing PPO
math and supports one process/device. Checkpoint compatibility is deliberately
strict: different trainer source or dependency versions require a new experiment.

## Verification

Run the lightweight suite in the root environment (runner tests use the Daytona SDK):

```sh
uv run --no-sync python -m unittest discover -s tests -v
```

They validate input rejection, metric integrity, output isolation, checkpoint
publication failures, corruption, resume configuration, and failure handling.
They mock training explicitly and do not establish RL compatibility.

After dependencies are installed, run a real, short CPU smoke test:

```sh
RUN_MUJOCO_SMOKE=1 uv run --project phy_rl --no-sync \
  python -m unittest tests.test_training_smoke -v
```

Exercise all four new tasks with real, small CPU training runs, video generation,
and a second attempt restored from the first checkpoint (requires ffmpeg and
GitHub access for the robot assets):

```sh
RUN_MUJOCO_TASKS=1 uv run --project phy_rl --no-sync \
  python -m unittest tests.test_task_training -v
```

These tests verify execution and resume, not learned task proficiency or H100 speed.

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

Session integration verified locally on 2026-11-11: 36 lightweight tests and all
six opt-in CPU tests passed. Coverage includes interrupted JSON publication,
retry identities, automatic shared-directory resume with saving disabled,
SIGKILL/full-state recovery, and W&B reporting with its remote transport mocked.
No Daytona sandbox was launched for this verification.

Runner sweep verification on 2026-11-11: all 40 lightweight tests and seven real
CPU tests passed. The new subprocess test trains through the sweep adapter,
checks durable attempt artifacts, and restores the checkpoint from a fresh reader
process. The GPU image built successfully with the pinned CUDA dependencies
(snapshot `716816667aa48395b3654874e062bcb8535dc02e4a351cca8e0f0a7e140c1f07`).
Both H100 sandbox creation requests were rejected with HTTP 400:
`Organization doesn't have GPU credits. Add more by visiting the Wallet page`.
GPU execution is therefore still unverified.

Additional task support verified locally on 2026-11-11: 50 focused unit tests,
the four new task training/video/resume tests, a numerical evaluation-metric test,
and the existing Cartpole/W&B training regression passed. These are small CPU
execution checks; H100 throughput and learned task proficiency remain unmeasured.
Tests for the removed `main.py` staging helpers and CLI/report interface have been
removed. The current grid and source packaging are covered by `test_main` and
`test_workdir`; live Daytona end-to-end coverage needs a new test for the current
entrypoint.

## Milestone boundary

The current task authorizes the bounded runner-based GPU sweep and end-to-end
verification above. Broader controller restart recovery and orchestration from
[the reduced V1 plan](codex_plan_2.md) remain outside this task.

Upstream references:

- [MuJoCo Playground release](https://pypi.org/project/playground/0.2.0/)
- [Brax release](https://pypi.org/project/brax/0.14.2/)
- [Brax compatibility fix included in the vendored trainer](https://github.com/google/brax/commit/303d89dfac42919e5fde7c91f24c28a505373b24)
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
