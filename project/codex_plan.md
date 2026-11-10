# Daytona Experiment Runner: implementation plan

The delivery sequence below is the original broad plan. The later
[reduced V1 plan](codex_plan_2.md) governs implementation and requires confirmation
between numbered steps. Step 1 is complete on Python 3.14; step 2 awaits confirmation.

## 1. Objective and delivery boundaries

Build a Python application that demonstrates concurrent GPU experiment scheduling, reliable sandbox lifecycle management, durable training progress, and automatic recovery from spot termination.

MuJoCo/MJX provides a real physical AI workload. The scheduler and recovery system remain independent of the simulator.

The core demonstration will:

1. Submit four experiments with a concurrency limit of two.
2. Run two experiments simultaneously while the others remain queued.
3. Persist checkpoints and metrics to Daytona Volumes.
4. Deliberately delete one GPU sandbox after a checkpoint is committed.
5. Automatically launch a replacement and resume training.
6. Restart the orchestrator and reconnect to existing workers.
7. Finish the sweep, export results, and verify resource cleanup.

Agreed constraints

- Target 3–5 focused implementation days for the core deliverable.
- Plan for up to $150 in paid testing, with current rates and concrete test limits confirmed before execution.
- Use Daytona Volumes for durable artifacts.
- Verify spot eligibility, GPU credits, and capacity first.
- W&B is an optional follow-up.
- ClearML is an optional integration investigation.
- Isaac Lab follows those priorities and does not block delivery.
- No custom dashboard, multi-GPU distributed training, Kubernetes, or general workflow engine.

Daytona currently documents immediate spot termination without advance warning, Tier 2 eligibility, and immediate creation failure when spot capacity is unavailable. Recovery must therefore work without a shutdown hook. Daytona GPU documentation

## 2. Architecture and implementation decisions

### Control plane

Run one local scheduler process using Python 3.14, asyncio, the Daytona async SDK, and SQLite.

Use:

- Typer for the CLI.
- Pydantic and YAML for configuration validation.
- SQLite in WAL mode for submissions, attempts, events, and cancellation requests.
- A process lock to prevent two schedulers managing the same database.
- pytest, Ruff, and type checking for verification.
- Separate optional dependency groups for training and integrations, keeping the local controller lightweight.

Organize implementation around configuration, persistence, scheduling, the Daytona adapter, artifact handling, and workload adapters.

This Mermaid diagram uses features the terminal renderer doesn't support.
flowchart TD
    CLI[CLI] --> DB[(Local SQLite)]
    DB <--> Scheduler[Async scheduler]
    Scheduler --> Provider[Daytona adapter]
    Provider --> GPU1[GPU sandbox: experiment A]
    Provider --> GPU2[GPU sandbox: experiment B]
    GPU1 --> Volume[(Daytona Volume)]
    GPU2 --> Volume
    Scheduler --> Reader[CPU artifact reader sandbox]
    Reader --> Volume
    Reader --> Export[Local metrics and report exports]
    Export -. Optional .-> WB[Weights & Biases]

The CPU artifact reader allows the controller to inspect checkpoints and metrics after their GPU sandbox disappears. It uses Daytona filesystem/process APIs; it does not expose a public service.

Create at most one reader per running controller. Reuse it during reconciliation, recreate it if lost, and delete it on clean shutdown. A standalone export can create a temporary reader and delete it afterward.

Keep SQLite on the local machine. Daytona Volumes are FUSE mounts backed by object storage, and the documentation specifically excludes database-style block-storage use. Daytona Volumes

### Scheduling and identity

Use three distinct identities:

| Entity | Meaning |
|---|---|
| Sweep | One submitted collection of experiments |
| Experiment | One immutable configuration and training target |
| Attempt | One execution of an experiment in one sandbox |

An experiment retains its identity across retries. Each attempt receives a new identifier and records its sandbox, parent checkpoint, timestamps, exit information, and termination reason.

Use FIFO scheduling with bounded concurrency:

- Default to two concurrent GPU sandboxes.
- One experiment per sandbox and one GPU per experiment.
- Provisioning, running, ambiguous creation, and pending deletion all occupy a slot.
- Retries return to the queue after backoff.
- Capacity failures back off without consuming the training-attempt allowance.
- Default to three execution attempts per experiment.
- No automatic switch from spot to on-demand.

Persist provisioning intent before requesting a sandbox. Assign deterministic attempt-based names and ownership labels containing project, sweep, experiment, and attempt identifiers.

After an ambiguous API timeout, search for the existing sandbox before issuing another create request. If its existence remains uncertain, retain the reservation and reconcile; do not blindly create a replacement.

### State and recovery

Use this experiment state model:

QUEUED → PROVISIONING → STARTING → RUNNING → SUCCEEDED
             │             │         │
             └─────────────┴─────────┴→ RETRY_WAIT → QUEUED
                                               └→ FAILED

Any nonterminal state → CANCELLING → CANCELLED

Track cleanup separately from the experiment outcome. A completed workload with a sandbox still awaiting deletion must remain visible as a cleanup obligation.

Recovery rules:

- Confirm sandbox deletion or terminal worker failure before replacing an attempt.
- Treat API unavailability as uncertainty, not proof that a worker died.
- Treat stale heartbeats as a reason to inspect the worker, not immediately duplicate it.
- For a hung worker, terminate and confirm deletion before replacement.
- Resume from the newest complete, compatible checkpoint.
- If no checkpoint has ever been committed, restart from step zero and record the lost progress.
- If previously committed progress becomes unreadable, surface an artifact error rather than silently restarting from zero.
- Read durable completion records before retrying a disappeared sandbox; it may have finished just before deletion.
- Cancellation prevents new attempts and takes precedence over subsequent retry decisions.

The guarantee is at-least-once execution attempts with explicit history, not exactly-once training or deterministic trajectory replay.

### Durable artifacts

Use one project volume with a separate experiment subpath mounted into each GPU sandbox. The artifact reader mounts the project prefix.

Within an experiment, separate immutable configuration, attempt metrics, checkpoints, and completion records.

Checkpoint publication:

1. Serialize the checkpoint to local sandbox disk at a completed training-update boundary.
2. Copy it to a unique checkpoint location on the volume.
3. Close all payload files.
4. Publish a unique manifest containing identifiers, training progress, file sizes, hashes, schema version, and configuration fingerprint.
5. Accept the checkpoint only when the manifest and every referenced file validate.

Do not depend on atomic directory renames, shared append operations, or a mutable latest file. Recovery discovers immutable manifests and validates candidates newest first.

Use the separate reader mount to verify cross-sandbox visibility. Allow bounded visibility retries before rejecting a candidate. If volume durability fails the deletion/remount tests, that is a release blocker; do not silently replace the selected storage backend.

Metrics use immutable JSONL chunks:

- Identify records by attempt and monotonically increasing sequence.
- Include logical training step and timestamp.
- Flush periodically and before checkpoint publication.
- Include a metrics watermark in each checkpoint manifest.
- Preserve metrics from abandoned work, marking them as belonging to an interrupted attempt.
- Deduplicate repeated imports by record identity.

A sudden termination may lose the current uncommitted checkpoint or metrics chunk. Report the observed loss window explicitly.

### Example training workload

Use MuJoCo Playground with MJX-JAX and RSL-RL PPO, initially with Go1JoystickFlatTerrain.

Start from the upstream RSL-RL integration and a pinned compatible version set. RSL-RL’s inspected v3.1.0 implementation saves both model and optimizer state, providing a suitable basis for the recovery adapter. MuJoCo Playground training entrypoint, RSL-RL checkpoint implementation

The adapter must additionally preserve and restore:

- Completed update count and total environment steps.
- Current learning-rate or scheduler state.
- Observation normalization state, when enabled.
- Python, NumPy, PyTorch CPU/CUDA, and relevant JAX random state.
- Configuration and dependency compatibility metadata.

Restore the next update index, avoiding an off-by-one repeat of a completed update. Resume toward the original total training target.

Reset simulation episodes on recovery and document this boundary. The deliverable preserves learner progress, without claiming exact replay of interrupted trajectories.

Use a deterministic counter workload for fast orchestration tests. GPU training remains required for the final showcase.

### Public interfaces

Provide these CLI commands:

daytona-runner doctor
daytona-runner validate CONFIG
daytona-runner submit CONFIG
daytona-runner serve --until-idle
daytona-runner status [SWEEP_OR_EXPERIMENT] [--watch] [--json]
daytona-runner cancel SWEEP_OR_EXPERIMENT
daytona-runner artifacts export SWEEP --output DIRECTORY
daytona-runner report SWEEP --output DIRECTORY
daytona-runner cleanup [--apply]
daytona-runner demo interrupt EXPERIMENT --after-checkpoint

submit persists work and returns immediately. serve executes queued work. cleanup previews actions unless --apply is supplied.

Configuration covers workload, seeds, parameter combinations, image, GPU preferences, spot selection, volume, concurrency, retry limits, checkpoint cadence, deadlines, and estimated spending limits.

Use small internal interfaces:

| Interface | Responsibilities |
|---|---|
| SandboxProvider | Create, find, inspect, launch, inspect command, delete, confirm deletion |
| ArtifactStore | Publish/read manifests, list checkpoints, validate payloads, export metric chunks |
| WorkloadAdapter | Build command, emit metrics, save/restore training state |
| ExperimentTracker | Optional export of configuration, metrics, artifacts, and attempt status |

The worker writes progress independently of the scheduler. It receives its configuration and artifact paths, but does not receive the controller’s Daytona API key.

## 3. Core milestones

Each milestone should end with a working demonstration and focused tests. Estimates total approximately 30–40 hours, with live-service availability remaining an external dependency.

### M0 — Verify access and platform assumptions

Timebox: 2–3 hours · Day 1

- Check account eligibility, GPU credits, available spot GPU types, and current rates.
- Confirm SDK support for spot requests, ownership labels, volume subpaths, sessions, deletion observation, and wall-clock TTL.
- Select one NVIDIA GPU type for the initial tests; record the selection.
- Launch one explicitly bounded smoke test after approval.
- Verify GPU visibility and a background process surviving controller disconnection.
- Record actual provisioning latency and deletion behavior.

Acceptance: one GPU sandbox is created, inspected, and confirmed deleted. Access blockers are explicit. No training sweep starts yet.

### M1 — Establish the package and immutable configuration

Timebox: 2 hours · Day 1

- Create the Python package, CLI entrypoint, dependency groups, and development checks.
- Define validated sweep and experiment configuration.
- Expand seeds and parameter combinations deterministically.
- Generate experiment identifiers and configuration fingerprints.
- Add the deterministic counter workload and fake provider.

Acceptance: a four-experiment configuration validates and expands predictably without cloud access. Invalid limits fail before submission.

### M2 — Implement the durable job store

Timebox: 2–3 hours · Day 1

- Create versioned SQLite tables for sweeps, experiments, attempts, events, and cleanup obligations.
- Implement transactional submission and state transitions.
- Persist cancellation intent and retry eligibility.
- Enforce the single-scheduler process lock.
- Add submit, basic status, and structured JSON output.

Acceptance: submitted work survives process restart; a second scheduler is rejected; invalid transitions cannot corrupt state.

### M3 — Prove the volume artifact protocol

Timebox: 3 hours · Day 2

- Create the project volume and experiment subpaths.
- Implement the CPU artifact reader and its ownership/cleanup rules.
- Implement immutable checkpoint manifests and metrics chunks.
- Test incomplete writes, missing files, checksum failures, and delayed visibility.
- Delete a writer sandbox and read its committed artifacts from a fresh sandbox.

Acceptance: complete artifacts survive writer deletion; incomplete artifacts are never accepted. The reader can be deleted and recreated without losing data.

### M4 — Implement one complete sandbox attempt

Timebox: 2–3 hours · Day 2

- Build the Daytona provider adapter using the verified SDK.
- Persist create intent, apply labels, and reconcile ambiguous responses.
- Launch workers using background session execution.
- Capture process identity, exit status, stdout/stderr, and phase timestamps.
- Add confirmed deletion and durable cleanup retries.
- Run the counter workload through provisioning, execution, artifact publication, and teardown.

Daytona sessions support long-lived background processes, making them appropriate for workers that must continue independently of a particular controller request. Daytona process execution documentation

Acceptance: one submitted experiment reaches completion and leaves no owned sandbox running.

### M5 — Add bounded concurrent scheduling

Timebox: 2–3 hours · Day 2

- Implement the scheduler reconciliation loop.
- Enforce concurrency across provisioning, execution, and deletion.
- Add FIFO dispatch, capacity backoff, and runtime deadlines.
- Implement serve --until-idle and live CLI status.
- Ensure one slow API call cannot block monitoring of other attempts.

Acceptance: four counter experiments run with a limit of two; execution intervals overlap, a third allocation is never admitted early, and queued work starts as slots become available.

### M6 — Build and verify the MuJoCo worker

Timebox: 3–4 hours · Day 3

- Prepare a reproducible CUDA-compatible image with Python 3.14 and pinned training dependencies.
- Include model assets in the image to avoid downloads during recovery.
- Start with 256 parallel environments and a short fixed update target.
- Verify simulation and policy learning actually use the GPU.
- Add the RSL-RL adapter for structured metrics and full learner checkpoints.
- Control JAX GPU memory allocation so JAX and PyTorch can coexist.
- Run a save/load test and continue training past the saved update.

Acceptance: a real MuJoCo experiment performs policy updates on Daytona, exports metrics, and resumes with restored optimizer state and increasing logical progress.

No reward threshold or polished robot video is required for this gate.

### M7 — Recover from sandbox loss

Timebox: 3–4 hours · Day 3

- Detect deleted sandboxes and record preemption metadata when available.
- Separate capacity errors, transient API errors, workload errors, and sandbox loss.
- Apply bounded execution retries with exponential backoff and jitter.
- Launch a new attempt with the latest valid checkpoint.
- Verify checkpoint compatibility before starting training.
- Record the relationship between interrupted and replacement attempts.
- Implement failure injection after an independently validated nonzero checkpoint.

Acceptance: deleting one running GPU sandbox causes its experiment to resume in a different sandbox from a recorded checkpoint, while another experiment continues.

Label deliberate deletion as simulated spot loss, unless an actual eviction is observed.

### M8 — Recover the controller and handle ambiguous operations

Timebox: 3 hours · Day 4

- Reconcile persisted attempts against Daytona on startup.
- Reattach to healthy workers instead of relaunching them.
- Recover from crashes between create request, sandbox-ID persistence, launch, and completion processing.
- Discover completed work from durable completion manifests.
- Import missing metric chunks without duplicate rows.
- Recreate the artifact reader without affecting workers.
- Verify that checkpoint publication continues while the controller is offline.

Acceptance: force-kill and restart the controller during concurrent training. Existing workers are adopted, queued work remains intact, and no duplicate attempt is launched.

### M9 — Complete cancellation, cleanup, and resource limits

Timebox: 2–3 hours · Day 4

- Implement cancellation of queued, provisioning, and running experiments.
- On graceful controller interruption, stop dispatch and cancel managed nonterminal work, then clean up.
- Preserve crash-recovery behavior for abrupt process loss.
- Add ownership-scoped orphan discovery and cleanup.
- Retain training artifacts; cleanup deletes compute resources, not the project volume.
- Apply provider-side TTLs so resources remain bounded during controller outages.
- Persist cleanup failures and retry them on restart.

Acceptance: success, workload failure, cancellation, and controller crash all have a verified path to resource deletion. Cleanup never selects unrelated sandboxes.

### M10 — Produce operational evidence

Timebox: 2 hours · Day 5

- Export experiment summaries, attempt histories, events, and raw metrics as JSON/CSV.
- Generate a concise Markdown report.
- Include provisioning, startup/compilation, training, checkpoint, recovery, and cleanup timings.
- Report useful progress separately from repeated work after rollback.
- Include requested and observed GPU type, image identifier, dependency versions, and seed.
- Estimate cost from recorded resource durations and dated rates.
- Include reader-sandbox and failed-attempt costs.

Acceptance: a reviewer can explain what ran concurrently, why a retry occurred, which checkpoint was restored, how much progress was lost, and whether cleanup completed.

### M11 — Rehearse and package the interview demonstration

Timebox: 3 hours · Day 5

- Run four short MuJoCo experiments with concurrency two.
- Interrupt one after a committed checkpoint.
- Restart the controller while another worker remains active.
- Wait for completion and confirm zero remaining owned compute resources.
- Repeat the recovery scenario once to check reproducibility.
- Add quick-start instructions, architecture explanation, troubleshooting, and measured findings.
- Record a short terminal walkthrough and retain its real artifacts.

Acceptance: a fresh reviewer can follow the documented commands and reproduce the demonstration with their own credentials.

If time runs short, cut integrations, videos, and broader benchmarking first. Preserve concurrency, durable recovery, restart reconciliation, and cleanup.

## 4. Optional milestones after the core demonstration

### O1 — Weights & Biases tracking

Timebox: 3–4 hours

- Add W&B as an optional dependency and tracker implementation.
- Initially export from the controller’s durable artifact stream, keeping training independent of W&B availability.
- Use one W&B run per attempt, grouped by logical experiment and tagged with the sweep.
- Log training step, reward/loss metrics, infrastructure timings, checkpoint references, and interruption reasons.
- Persist tracking IDs and export cursors.
- Resume the same tracking run after exporter restart; create a new attempt run after training rollback.
- Preserve local/volume metrics if tracking is unavailable.

Separate attempt runs keep rollback and repeated training steps visible. W&B exposes run IDs, grouping, and resume controls suitable for this mapping. W&B SDK implementation

Acceptance: original and replacement attempts appear together, their history survives exporter restart, and a W&B outage does not stop training or recovery.

### O2 — ClearML feasibility investigation

Timebox: 3–4 hours; prototype separately if viable

Investigate a small bridge:

ClearML queue
    → Daytona bridge
    → GPU sandbox
    → ClearML Agent executes the task
    → ClearML receives logs and metrics

- Use an existing ClearML server/account; do not add server deployment to this project.
- Inspect queue ownership, task claiming, cancellation, and retry semantics.
- Test executing a specific task inside a Daytona sandbox using ClearML Agent in virtual-environment mode.
- Avoid requiring Docker inside the sandbox.
- Verify GPU visibility, credentials, environment reproduction, and task status reporting.
- Evaluate crash recovery between queue selection and agent startup.
- Establish which system owns retries so two schedulers cannot restart the same task independently.

ClearML documents explicit task execution and a --require-queue option intended for third-party scheduler integrations. This is a promising integration point, but queue-claim and crash semantics still require verification. ClearML Agent CLI

Acceptance: deliver a feasibility note with verified behavior, limitations, and a go/no-go recommendation. If feasible, demonstrate one queued task launching on Daytona and being cancelled and cleaned up through the bridge.

Generic checkpoint recovery still requires cooperation from the training workload; the bridge must not claim transparent recovery for arbitrary scripts.

### O3 — Isaac Lab compatibility

Timebox: 2–3 hours

- Verify the selected GPU, driver, graphics libraries, image access, and headless runtime requirements.
- Attempt one minimal task using the established workload/artifact interface.
- Record startup behavior and a small number of training updates.
- Produce a compatibility report if blocked.

Acceptance: either a reproducible minimal run or a precise documented blocker. No dependency changes may destabilize the working MuJoCo image.

## 5. Verification, defaults, and release criteria

### Default operational limits

| Setting | Initial default |
|---|---|
| GPU concurrency | 2 |
| GPUs per experiment | 1 |
| Execution attempts | 3 |
| Scheduler polling | 5 seconds |
| Worker heartbeat | 10 seconds |
| Heartbeat investigation threshold | 60 seconds |
| Checkpoint target | Every 60 seconds, at an update boundary |
| Metrics flush target | Every 10 seconds, plus checkpoint/completion |
| Worker startup allowance | 10 minutes |
| Training allowance per attempt | 15 minutes |
| GPU sandbox TTL | 30 minutes from creation |
| Experiment elapsed deadline | 60 minutes, including retries |
| Sweep elapsed deadline | 120 minutes |
| Artifact reader TTL | 150 minutes; delete earlier on clean exit |

Long compilation or update durations must be visible as phases. Checkpoint cadence is a target, not a promise that a long GPU operation can be interrupted safely.

Reserve approximately $30 of the $150 planning budget for unexpected retries. Stop admitting paid work when the conservative estimate plus remaining allocations would exceed the working budget. These estimates supplement TTLs and concurrency limits; they are not an exact billing cap.

### Required test scenarios

| Scenario | Expected result |
|---|---|
| More jobs than available slots | Queue remains bounded; no over-allocation |
| Spot capacity unavailable | Backoff without burning execution retries |
| Create succeeds but response is lost | Discover and adopt the sandbox |
| Worker launch response is lost | Inspect the existing session before relaunch |
| Sandbox dies before first checkpoint | Retry from zero with explicit history |
| Sandbox dies after checkpoint | Resume from the newest valid committed progress |
| Checkpoint upload is interrupted | Ignore incomplete publication |
| Newest checkpoint is corrupt | Restore an older valid checkpoint and record fallback |
| Artifact visibility is delayed | Retry reads before declaring loss |
| Controller crashes | Workers continue; restart reconciles |
| Completion occurs during controller outage | Recognize completion without rerunning |
| API is unavailable | Preserve uncertainty; do not duplicate workers |
| Workload exits with invalid configuration | Fail terminally |
| Cancellation races with retry | Do not launch another attempt |
| Delete request fails | Keep cleanup pending and retry |
| Cleanup sees unrelated sandboxes | Leave them untouched |
| Tracking service is unavailable | Durable metrics and training continue |

Run most scenarios with a fake provider and deterministic worker. Keep cloud tests opt-in, bounded, and focused on platform behavior that cannot be established locally.

### Final release criteria

The core project is ready when:

- Two real GPU experiments demonstrably overlap.
- Additional experiments wait and are dispatched correctly.
- Committed checkpoints and metrics survive sandbox deletion.
- An interrupted experiment restores learner state and completes in a replacement sandbox.
- Restarting the controller does not duplicate healthy work.
- Attempt history explains retries, restored progress, and discarded work.
- Success, failure, and cancellation leave no owned compute resources running.
- Artifacts remain available after cleanup.
- The repository includes reproducible commands, passing tests, recorded measurements, and honest limitations.

The local database must survive controller restarts; recovery from loss of the entire local machine is outside the initial release. GPU availability, training compatibility, and volume behavior remain explicit feasibility gates to verify during implementation.
