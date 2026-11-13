# Daytona Experiment Runner

This example runs reinforcement learning jobs in Daytona GPU sandboxes.
The controller runs on your computer. A worker runs inside each sandbox.
Workers save training state in checkpoints on a shared volume.
If a sandbox stops or disappears, the controller starts a replacement.
The replacement worker loads the latest complete checkpoint.

Training uses MuJoCo Playground, MJX, and an
[adapted Brax PPO trainer](phy_rl/_vendor/README.md).
Weights & Biases (W&B) stores metrics, plots, and videos.

## Install

Install `uv`. Use Python 3.14.
Run these commands from the repository root:

```sh
cd project
unset UV_PROJECT_ENVIRONMENT
uv sync --locked --python 3.14
```

Run all remaining commands from `project/`.
The controller and training code use separate Python environments.
The `unset` command prevents a shared uv setting from mixing these environments.

## Train locally

Install the training dependencies:

```sh
uv sync --project phy_rl --locked --python 3.14
```

Start CPU training:

```sh
uv run --project phy_rl --no-sync python -m phy_rl.train --cpu
```

This command trains `CartpoleBalance` for 8,388,608 requested steps.
Local training does not require cloud credentials.

Use `--gpu` for an NVIDIA GPU with CUDA support.
Add `--video` to save a video of the policy before and after training.
Video output requires `ffmpeg` with the `libx264` encoder.
Videos use MuJoCo's native renderer with model meshes, materials, and lighting.
On headless Linux, install `libosmesa6` and set `MUJOCO_GL=osmesa` before starting Python.
The Daytona image includes these dependencies. Training still uses the GPU.
macOS uses its native graphics backend.
Use `MUJOCO_GL=egl` only on Linux systems with working EGL graphics support.
The trainer checks rendering before training starts when `--video` is enabled.
Add `--wandb` to send local results to W&B.
Use `--help` to list all options.

Other tasks are `CartpoleSwingup`, `CheetahRun`, `WalkerWalk`,
`Go1JoystickFlatTerrain`, and `PandaPickCube`.
Go1 and Panda require network access to download robot assets.

## Run on Daytona

This example requires Daytona GPU access, GPU credits, and a W&B account.
The command below uses paid GPU resources.

Create `.env` in `project/`:

```dotenv
DAYTONA_API_KEY=your-daytona-api-key
WANDB_API_KEY=your-wandb-api-key
WANDB_ENTITY=your-wandb-user-or-team
WANDB_PROJECT=daytona-experiment-runner
```

Replace the example values with your credentials and W&B destination.
Review the settings in [main.py](main.py) before you start:

| Setting | Value |
| --- | --- |
| Task | `WalkerWalk` |
| Jobs | Six configurations, each with seeds 11 and 22 |
| Training steps per job | 62,914,560 |
| Maximum active jobs | Two |
| Resources per job | One H100, 4 CPUs, 8 GiB RAM, 20 GiB disk |
| Video output | Enabled |

The sweep compares a baseline with five changes.
Each change affects one setting: learning rate, entropy cost, update count, or batch size.
Select a task with the `Task` enum in `main.py`, for example `Task.WALKER_WALK`.
`build_training_grid(task)` returns twelve configurations with the selected task's settings.
All sweep settings are in `main.py`. The controller does not import the worker package.
Walker uses 62,914,560 steps and nine evaluations. This gives 64 collection and training
cycles, compared with eight cycles in the earlier 7,864,320-step preset.
Other tasks have separate presets in the same function.

Start the sweep:

```sh
uv run --no-sync python main.py
```

The controller sends metrics to W&B as training progresses.
Workers do not need the W&B API key or a connection to W&B.
Before sandbox deletion, the controller tries to download the result files.
Each job keeps one W&B run across replacement attempts.

Press Ctrl+C once to stop the controller and request sandbox deletion.
Wait for cleanup to finish. A second Ctrl+C can interrupt cleanup.
Manually stopping a sandbox causes the controller to start a replacement.
The controller retains snapshots and the shared volume for later use.

The example uses `daytona[v0-sdk]==2.0.0a8` through its `v0` API.
The sandbox image installs packages from `phy_rl/requirements.txt`.
This file is an export of `phy_rl/uv.lock`.
After dependency changes, update the lockfile.
Then regenerate the requirements file:

```sh
uv lock --project phy_rl
uv export --project phy_rl --locked --no-hashes --no-dev --no-emit-project \
  --output-file phy_rl/requirements.txt
```

## Resume training

A checkpoint contains model parameters, optimizer state, normalization state,
random state, and completed training steps.
The trainer saves a checkpoint after each complete PPO update and at completion.
The `.complete` marker identifies a finished checkpoint file.

For local training, replace `ATTEMPT` with the directory name of the previous attempt:

```sh
uv run --project phy_rl --no-sync python -m phy_rl.train \
  --resume runs/ATTEMPT/checkpoints
```

Training continues toward the original step target.
A checkpoint at that target requires no more training updates.
The simulator resets its episodes after restoration.
Training settings and dependency versions must match the checkpoint.

Use `--checkpoint-interval-steps 5` to save every five PPO updates.
This interval counts PPO updates, not simulation steps.
Use `--no-wandb` to resume without W&B uploads.
Add `--gpu` to resume on an NVIDIA GPU. CPU is the default.

Daytona workers resume automatically from the shared checkpoint directory:
`/data/jobs/<job_id>/checkpoints`.
Each replacement gets a new attempt ID and keeps the same job ID.
Use only one worker per checkpoint directory.

## Read the results

Local attempts write to `runs/<attempt>/`.
The controller downloads Daytona results to `runs/daytona/<job_id>/<attempt_id>/`.
Daytona checkpoints stay on the shared volume.
The controller saves each W&B link in `runs/daytona/<job_id>/wandb.json`.

| File | Contents |
| --- | --- |
| `summary.json` | Training steps, elapsed time, and evaluation results |
| `metrics.jsonl`, `metrics.csv` | Training metrics |
| `learning_curves.png` | Reward and loss plots |
| `evaluation_before.json`, `evaluation_after.json` | Policy evaluations on the same starting states |
| `policy_comparison.mp4` | Policy video, when enabled |
| `resolved_config.json` | Settings, dependency versions, and device |
| `attempt.json` | Job identity, attempt identity, and restored progress |
| `status.json`, `error.txt` | Training status and error details |

The controller also saves collection errors and reporting errors locally.
A sudden sandbox loss can remove files before the controller downloads them.
Metrics from earlier successful reads remain on your computer.

Checkpoints reference a sibling `baseline-*.msgpack` file containing the original
evaluation, trajectories, and policy parameters. Keep that file when copying checkpoints.
Retries reuse it for the before/after summary and video, including retries after training
has reached its target. `policy_parameter_change_l2` measures change from this baseline;
`attempt_policy_parameter_change_l2` measures only the current attempt's updates.
Older checkpoints without a baseline remain resumable, but emit a warning and label
their comparison with the restored step instead of presenting it as the original policy.

## Verified behavior and limits

Local `CartpoleBalance` training kept the pole balanced for 20 seconds in all
32 evaluation episodes. This result used one training seed.
The mean return increased from 496.01 to 1,998.33 out of 2,000.

Saved Daytona results include completed H100 training runs and restored checkpoints.
The completed cloud resume checked on 2026-11-12 restored a checkpoint at the final step target.
It performed no further training updates.
Local tests verify that training can continue from an earlier checkpoint after a forced process stop.
The saved results do not yet demonstrate that full sequence on Daytona.

Current limits:

- A controller restart starts a new sweep. It does not recover the previous sweep.
- Each sandbox has a 65-minute lifetime. The sweep has no overall deadline or limit on replacement attempts.
- The runner retries temporary API errors up to five times by default. A lost command response can cause a duplicate launch.
- The snapshot cache key does not include the contents of `requirements.txt`.
- Checkpoints have no automatic retention limit. A corrupt checkpoint stops restoration.
- A worker failure can leave the controller exit code at zero.

## Run tests

Run the unit tests:

```sh
uv run --no-sync python -m unittest discover -s tests -v
```

The unit tests use simulated cloud services.
Install the training dependencies and `ffmpeg`.
Run the native rendering test without training:

```sh
RUN_MUJOCO_RENDER=1 uv run --project phy_rl --no-sync \
  python -m unittest tests.test_video -v
```

Then run the CPU training and recovery tests:

```sh
RUN_MUJOCO_SMOKE=1 uv run --project phy_rl --no-sync \
  python -m unittest tests.test_training_smoke -v
```

Run the additional task tests with network access for robot assets:

```sh
RUN_MUJOCO_TASKS=1 uv run --project phy_rl --no-sync \
  python -m unittest tests.test_task_training -v
```
