# Daytona RL Runner: Distributed, Fault-Tolerant Physical AI Training

## Mission
Build a working Python-first proof of concept that orchestrates multiple reinforcement-learning experiments on Daytona GPU sandboxes, survives worker/sandbox failures, and produces reproducible results and infrastructure metrics.

**Priority:** MuJoCo first (prefer MJX for GPU-accelerated simulation). **Stretch goal:** NVIDIA Isaac Lab compatibility. The engineering deliverable is the scheduler, lifecycle management, persistence, and recovery, not a novel RL algorithm or polished UI.

## Success criteria
1. A documented command submits a small sweep of MuJoCo/MJX training runs with different seeds or hyperparameters.
2. Runs execute with bounded concurrency across Daytona sandboxes, with explicit lifecycle states and reliable cleanup.
3. Run metadata and checkpoints persist outside the sandbox; restarting the local orchestrator does not lose submitted jobs.
4. A deliberate mid-training sandbox termination is detected, and the run resumes from the latest **committed** checkpoint in a new sandbox.
5. A report shows reward/learning curves, completion status, retries, wall time, provisioning/startup overhead, and estimated cost (clearly distinguish estimates from billed charges).
6. A README gives reproducible commands, limitations, observed measurements, and architectural tradeoffs.

## Scope and constraints
- Use Python for the orchestrator, CLI, and tests. Favor a small dependency footprint.
- Use an existing MuJoCo RL task and training implementation. Start with a simple task (e.g., Ant or Hopper) and a modest PPO configuration. Verify task availability in the installed versions.
- Prefer MJX GPU simulation if supported on the selected Daytona GPU; fall back to standard MuJoCo CPU to validate orchestration and label results accordingly.
- Do not assume a GPU sandbox provides the graphics stack required by Isaac Sim.
- Do not assume Daytona snapshots preserve running process state. Recovery is from **application-level training checkpoints** in external durable storage.
- Do not assume exact Daytona SDK method names, GPU availability, prices, billing granularity, or persistence semantics. Check current official docs and installed SDK before implementing.
- Avoid Kubernetes, web dashboards, distributed multi-GPU training, custom robotics environments, or a generic workflow engine.
- Keep resource usage bounded: configurable maximum concurrent GPU sandboxes, runtime budget, and cleanup on cancellation.

## Proposed architecture

```text
CLI (submit/status/cancel/report)
       |
SQLite job store (durable specs, state, attempts, events)
       |
Async scheduler (bounded concurrency, reconciliation, retries)
       |
Daytona provider adapter (create/inspect/exec/stop/delete)
       |
GPU sandbox: prepared environment + MuJoCo/MJX training worker
       |
Checkpoint + metrics uploader -> durable artifact store
       |
Report generator (CSV/JSON/Markdown; plots optional)
```

A single orchestrator process is sufficient for the MVP. SQLite is acceptable, but document the single-scheduler assumption. The sandbox worker should write periodic checkpoints and metrics independently of the orchestrator, so a local crash does not immediately lose training progress.

## Suggested repository structure

```text
rl-runner/
  pyproject.toml
  README.md
  configs/mujoco_sweep.yaml
  rl_runner/
    cli.py
    models.py
    store.py
    scheduler.py
    daytona_provider.py
    artifacts.py
    metrics.py
    reporting.py
    workers/mujoco_train.py
    workers/checkpointing.py
  tests/
    test_scheduler.py
    test_recovery.py
    test_state_machine.py
    test_artifacts.py
  scripts/
    smoke_test.py
    inject_failure.py
```

## Job model and state machine
Each experiment has an immutable specification: task, simulator/backend, algorithm, seed, hyperparameters, GPU requirements, timeout, checkpoint interval, and source/dependency versions.

Track `job_id`, `attempt_id`, `sandbox_id`, heartbeat timestamp, latest committed checkpoint URI, start/end times, errors, and metric/artifact locations.

Suggested states:

```text
QUEUED -> PROVISIONING -> STARTING -> RUNNING -> COMPLETED
              |              |          |
              +--------------+----------+-> RETRY_WAIT -> PROVISIONING
                                             |
                                             +-> FAILED (retry limit)
Any nonterminal state -> CANCELLING -> CANCELLED
```

Enforce valid transitions and persist every transition with timestamp and reason. On orchestrator restart, reconcile nonterminal jobs with Daytona's observed sandbox state. Do not blindly relaunch a job that might still be running.

## Checkpoint and recovery contract
- Training worker periodically writes checkpoint files locally, then uploads to durable storage outside the sandbox.
- Include model parameters, optimizer state, scheduler state, step/iteration, RNG state(s), and relevant normalization statistics. Document any simulator/environment state that cannot be restored exactly.
- Upload to a unique attempt/versioned key, verify completeness (size/checksum or equivalent), then **atomically publish a manifest/pointer** to the latest valid checkpoint.
- Never treat an incomplete upload as resumable.
- On lost sandbox, start a new attempt from the latest committed checkpoint; preserve the job's history.
- Guarantee **at-least-once attempts**, not exactly-once execution. Make result publication idempotent using job/attempt identifiers.
- Define recovery as resuming training progress, **not bit-for-bit deterministic trajectory replay**.
- If a run fails due to invalid training code/configuration, classify as a terminal workload failure rather than retrying indefinitely.

For the MVP, a durable object store is preferred (S3-compatible if available). A local filesystem store is acceptable for initial testing only if it is explicitly outside the Daytona sandbox and survives sandbox deletion. Use an abstraction so it can be replaced.

## Daytona provider responsibilities
- Provision GPU sandbox with pinned image/dependency setup and resource configuration.
- Run worker with environment/configuration and artifact credentials scoped to the task.
- Observe state and collect stdout/stderr, exit codes, and timestamps.
- Apply timeouts, cancellation, retries with exponential backoff and jitter for transient API failures.
- Delete resources in `finally`/reconciliation paths; support a cleanup command for orphaned sandboxes.
- Avoid logging secrets; use least-privilege credentials.
- Distinguish API failures, capacity errors, sandbox loss, worker errors, and timeout errors.

## Implementation phases

### Phase 0: Verify capabilities (time-box)
1. Read current Daytona GPU sandbox docs, Python SDK docs, pricing/billing docs, and lifecycle semantics.
2. Create one GPU sandbox, check GPU visibility, CUDA/JAX compatibility, filesystem behavior, and command execution.
3. Record hardware, driver, CUDA/JAX versions, image details, provisioning latency, and limitations.
4. Confirm the available mechanism for persisting checkpoints outside the sandbox.

**Gate:** If GPU/MJX is blocked, continue orchestration using CPU MuJoCo while recording the blocker. Do not spend the entire project debugging dependencies.

### Phase 1: MuJoCo training worker
1. Run a known working MuJoCo task locally first.
2. Implement a headless training entrypoint with deterministic seed/config inputs and periodic metrics.
3. Implement checkpoint save/load and a test that resumes from a checkpoint.
4. Run the same worker in a Daytona sandbox.

**Gate:** One experiment completes and exports checkpoint + metrics.

### Phase 2: Orchestration
1. Define YAML experiment specs and expand a small parameter/seed sweep.
2. Implement durable job store and bounded asynchronous scheduler.
3. Add Daytona provisioning, execution, monitoring, cleanup, status, and cancellation.
4. Run at least 2 experiments concurrently if quotas permit.

**Gate:** Jobs survive CLI/orchestrator restart and produce distinct artifacts.

### Phase 3: Fault tolerance
1. Add periodic durable checkpoint upload and committed checkpoint manifest.
2. Implement reconciliation, retry classification, and resume on replacement sandbox.
3. Add controlled failure injection: terminate a running sandbox after a committed checkpoint.
4. Demonstrate successful recovery and preserve the complete attempt history.

**Gate:** A killed sandbox results in resumed training, without losing the entire experiment.

### Phase 4: Evaluation and report
1. Compare sequential vs bounded-concurrent execution on identical small workloads.
2. Measure cold provisioning, environment startup, training throughput, checkpoint overhead, recovery time, and cleanup.
3. Compare uninterrupted run vs failure-and-recovery run; explain differences in reward trajectory.
4. Estimate spend from measured billable durations and published rates, explicitly stating assumptions.
5. Generate a concise report with results, caveats, and recommended Daytona improvements.

### Stretch: Isaac Lab
- Verify Isaac Sim requirements (NVIDIA driver, GPU, Vulkan/EGL/rendering capabilities, container setup).
- Attempt a minimal headless Isaac Lab task and a few PPO iterations with RSL-RL.
- Implement an `IsaacLabWorker` conforming to the same job/artifact/checkpoint contract.
- If blocked, provide a reproducible compatibility matrix, logs, and exact failure point. Do not jeopardize the MuJoCo deliverable.

## CLI (illustrative, implement as appropriate)

```bash
rl-runner validate configs/mujoco_sweep.yaml
rl-runner submit configs/mujoco_sweep.yaml
rl-runner status
rl-runner status <job-id>
rl-runner cancel <job-id>
rl-runner report <experiment-id> --output report.md
rl-runner cleanup --dry-run
```

Example config (adapt fields to actual implementation):

```yaml
name: mujoco-mjx-sweep
backend: daytona
simulator: mujoco_mjx
task: Ant
algorithm: ppo
seeds: [11, 22, 33]
parameters:
  learning_rate: [0.0003, 0.001]
max_concurrency: 2
max_attempts: 3
checkpoint_interval_steps: 10000
max_runtime_minutes: 30
artifact_store: s3
```

## Testing expectations
- Unit tests with a fake Daytona provider for state transitions, concurrency limits, retry decisions, and cancellation.
- Failure-injection tests: sandbox disappears, API timeout, checkpoint upload interrupted, duplicate completion event, orchestrator restart, and worker exits nonzero.
- One integration smoke test against Daytona (opt-in, cost-bounded).
- Assert cleanup after success, failure, and cancellation.
- Clearly separate agent/workload failure from infrastructure failure in reports.

## Metrics to collect
- Provisioning latency; time until first successful command; dependency/image setup time.
- Training wall time; simulated environment steps per second; policy updates per second where available.
- Reward curves, final reward, total steps, random seed.
- Checkpoint size, upload duration, interval, and resume latency.
- Sandbox count, attempt count, transient/permanent failures, leaked-resource count.
- GPU model, GPU memory/utilization if accessible; CPU/RAM metrics if accessible.
- Estimated compute cost per successful run and per million simulation steps; do not conflate estimate with invoiced charges.

## Key engineering decisions to document
1. Why use application checkpoints rather than relying on sandbox snapshots?
2. How do you prevent duplicate job execution after ambiguous API failures?
3. What guarantees are possible for checkpoint durability and resumption?
4. How do you detect a dead sandbox versus a slow but healthy training run?
5. When does sandbox reuse save more than it costs in isolation/cleanup complexity?
6. How do provisioning overhead and concurrency affect time-to-result and spend?
7. Which parts of the implementation are simulator-specific versus provider-specific?

## Deliverables
- Runnable repository with pinned dependencies and concise setup instructions.
- One end-to-end MuJoCo/MJX training example on Daytona (or documented compatibility blocker plus CPU fallback).
- Durable scheduler and job-state implementation.
- Recovery demo with reproducible failure injection.
- Automated tests, sample config, and example generated report.
- `FINDINGS.md` describing measured behavior, limitations, and actionable Daytona product feedback.
- Optional Isaac Lab compatibility result.

## Definition of done
A reviewer can configure Daytona credentials and artifact storage, submit a small MuJoCo experiment sweep, observe bounded concurrent execution, kill one sandbox after a checkpoint, restart the orchestrator if desired, and obtain a final report showing that the interrupted job resumed from a committed checkpoint. Any unavailable cloud capability must be documented with a clear fallback and evidence.

## Guidance to coding agent
Implement incrementally, verify APIs from official documentation, and run tests after each phase. Prefer a narrow, working vertical slice to a broad unfinished framework. Keep a running log of assumptions and measured findings. Ask for credentials or spending approval when needed; never hard-code secrets or launch expensive GPU sweeps without explicit limits.
