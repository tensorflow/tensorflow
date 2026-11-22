<!-- disableFinding("hang") -->
<!-- disableFinding("hung") -->
<!-- disableFinding(SNIPPET_EM_DASH) -->

# Diagnose and resolve TPU workload hangs

When a TPU workload stops making progress, the TPU runtime automatically detects
the stall, captures a snapshot of every core's execution state, and writes
diagnostic artifacts to disk. This guide explains how the detection system
works, how to configure it, and how to use the diagnostic artifacts to identify
and resolve the root cause.

## Before you begin

This guide covers `libtpu` 0.0.51. Minimum version requirements:

*   **`0.0.46+`**: Stall detection and diagnostic artifact collection (enabled
    by default on all TPU VMs).
*   **`0.0.47+`**: Python API (`libtpu.sdk.diagnostics`) for reading individual
    artifacts.
*   **`0.0.48+`**: Diagnostic report CLI (`python -m libtpu.diagnostics`).

To check which version you have, run:

```sh
pip show libtpu | grep Version
```

To install the latest `libtpu` package:

```sh
pip install --upgrade libtpu
```

No additional setup is needed. Stall detection and diagnostic artifact
collection are enabled by default on all TPU VMs.

## Quick start

If your workload is hung right now:

```sh
# Collect the diagnostic artifacts from your TPU VM and generate a report
mkdir -p ./diag
gcloud compute tpus tpu-vm scp \
  'TPU_NAME:/tmp/tpu_*.pb' ./diag/ \
  --zone=ZONE
python -m libtpu.diagnostics ./diag/
```

The report identifies which cores are stuck, what operation they're waiting on,
and which host is the likely cause.

Attach the files to your support case, see [File a support
case](./file_support_case.md). If no files are found, see [No diagnostic
artifacts found](#no-diagnostic-artifacts-found).

## How TPU hang detection works

The TPU runtime includes three independent watchdogs. Only the PJRT event
tracker is on by default; the other two are opt-in. When a watchdog detects a
problem, it posts a trigger event to the **diagnostic coordinator** on that
host, which aggregates the triggers, captures TPU state, and writes diagnostic
artifacts. By default, detection is report-only: it doesn't halt or kill your
workload. Each host runs its own coordinator and writes its own artifacts.

```
┌─────────────────────────────────────────────┐
│  WATCHDOGS (detect the hang)                │
│  ├── PJRT event tracker                     │
│  ├── Driver watchdog                        │
│  └── LLO (sync flag) watchdog               │
├─────────────────────────────────────────────┤
│  DIAGNOSTIC COORDINATOR (responds to it)    │
│  └── Aggregates triggers → captures state   │
│      → writes artifacts to disk             │
└─────────────────────────────────────────────┘
```

### Watchdog 1: PJRT event tracker

The PJRT event tracker monitors high-level operations (compute, transfers,
collectives) at the framework layer. If any operation takes longer than a
configurable threshold (default: 5 minutes), it fires a stall trigger.

This is the watchdog most likely to fire for typical training hangs, because it
operates at the level your framework sees — if your `jax.jit`-compiled step
isn't completing, this detector catches it. It is **enabled by default** — no
flags needed.

### Watchdog 2: Driver watchdog

The driver watchdog monitors TPU driver callbacks at the HAL (Hardware
Abstraction Layer). If a driver callback (such as a sync flag completion or
interrupt handler) does not return within the timeout, the watchdog fires a
stall trigger and then terminates the process with a descriptive error.

This watchdog catches low-level driver or hardware failures where the TPU chip
itself is non-responsive. On Cloud TPU VMs it is **disabled by default** — see
[Level 2 flags](#level-2-enable-when-investigating-a-hang) to enable it.

IMPORTANT: Enable it only if Cloud Support asks you to.

### Watchdog 3: LLO (sync flag) watchdog

The LLO watchdog operates at the lowest level — inside the compiled TPU program
itself. When you set a [sync flag wait timeout](./sync_flag_wait_timeout.md),
the compiler instruments every sync flag wait operation (used for collective
operations like all-reduce, all-gather, etc.) with a hardware timer. If the wait
exceeds the timeout, the TPU core executes a callback that fires a stall
trigger.

This is the most precise watchdog for collective hang detection, because it
fires on the exact core and at the exact instruction that is stuck waiting.

**NOTE:** The LLO watchdog requires recompilation after changing the flag,
because the instrumentation is baked into the compiled program. Changing the
timeout takes effect only for programs compiled after the flag change. See
[Level 2 flags](#level-2-enable-when-investigating-a-hang) for configuration.

### What you'll see in your logs

When a watchdog fires, you'll see these messages in your workload's stderr. They
tell you that diagnostics have been triggered and artifacts are being written.

**PJRT event tracker** (most common):

```
E0720 10:54:18.028528  746 pending_event_logger.cc:946] PendingEventLogger:
High-level software slow operation detected. Triggering diagnostic coordinator
via event bus.
```

**Diagnostic coordinator** (appears shortly after any watchdog fires):

```
E0720 10:54:18.231200  746 tpu_diagnostic_coordinator.cc:616]
TpuDiagnosticCoordinator: Stall detected on host 0. Starting diagnostic
aggregation window and queueing artifact harvest.
W0720 10:54:28.410112  746 tpu_diagnostic_coordinator.cc:1160]
TPU stall snapshot (PJRT_WATCHDOG): my-tpu-w0/746 host 0: 4 TensorCores: 4 in flight, 0 idle, 0 unreadable; top positions: HLO: all-reduce.42 x4
W0720 10:54:28.410150  746 tpu_diagnostic_coordinator.cc:1160]
TPU stall artifact: /tmp/tpu_stall_summary_host0_1722500096123456_my-tpu-w0_746.pb (+1 compiler metadata files)
W0720 10:54:28.410180  746 tpu_diagnostic_coordinator.cc:1160]
Next: copy tpu_stall_summary_* and tpu_compiler_metadata_* from ALL hosts into one directory and run: python -m libtpu.diagnostics <dir>. To snapshot this process again: send "trigger_dump" to /tmp/tpu_diag_trigger_746.socket (may be lost when the task or VM restarts; set --xla_tpu_impure_diagnostic_artifacts_dir to persistent storage)
```

**Driver watchdog** (if enabled — process will crash after this):

```
E0720 10:59:18.042100  746 real_program_continuator.cc:1631] Core 0,0,0:
TPU driver callback watchdog timeout detected. Triggering diagnostic
coordinator via event bus.
F0720 10:59:18.043200  746 real_program_continuator.cc:1657]
TPU driver callbacks are stuck for 5m; this is likely a driver or a
hardware issue
```

If you see the `PendingEventLogger` message but no artifacts appear, check the
[troubleshooting section](#troubleshooting).

### Diagnostic coordinator

The diagnostic coordinator sits between the watchdogs and the artifact output.
Each host runs its own coordinator, which only sees triggers from its own
process. When a watchdog fires (or you trigger it manually), the coordinator:

1.  **Aggregates** — Waits 10 seconds for additional triggers from other
    watchdogs on the same host that may fire moments apart.
2.  **Harvests** — Reads the execution state of every TPU core on the host
    (program counter, HLO operation, program ID) and compiler metadata for up to
    5 active programs.
3.  **Writes** — Saves the artifacts to the artifacts directory (local disk or
    Cloud Storage) and enters a 15-minute cooldown to prevent flooding.

Triggers that arrive during harvesting or cooldown are dropped — they're almost
certainly downstream effects of the same root cause. This includes manual
triggers.

### What gets collected

| Artifact              | Contents                                                                                                                                                           | Always collected?                                      |
| :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- |
| **Stall summary**     | Per-core snapshot: program counter, HLO operation name, program ID, sequencer type                                                                                 | Yes                                                    |
| **Compiler metadata** | For each active program: module name, program ID, executable fingerprint, topology, whether the program is haltless, plus the compiled HLO module and symbol table | Yes (up to 5 unique programs)                          |
| **Core dump**         | Full hardware register state, FIFO depths, analyzer reports                                                                                                        | Only if `--xla_tpu_diagnostic_collect_core_dumps=true` |

-----

## Step 0: Configure the diagnostic system

All flags are set via the `LIBTPU_INIT_ARGS` environment variable **before**
starting your workload:

```sh
export LIBTPU_INIT_ARGS="--flag1=value1 --flag2=value2"
```

Flags are organized into three levels. Start with Level 1 and add higher levels
as needed.

### Level 1: Always set

These flags have no performance cost and should be part of every TPU workload
configuration.

| Flag                                        | Default | What it does                                                                                                                 |
| :------------------------------------------ | :------ | :--------------------------------------------------------------------------------------------------------------------------- |
| `--xla_tpu_enable_diagnostic_coordinator`   | `true`  | Primary on/off switch for the diagnostic system. Leave this on.                                                              |
| `--xla_tpu_impure_diagnostic_artifacts_dir` | `/tmp`  | Where to write artifacts. Set to a `gs://` URI for multi-host Pods so you don't need to SSH into each host to collect files. |

The **PJRT event tracker** is enabled by default with a 5-minute threshold. No
flags needed — it just works.

**Example — multi-host Pod:**

```sh
export LIBTPU_INIT_ARGS="--xla_tpu_impure_diagnostic_artifacts_dir=gs://MY_BUCKET/tpu-diagnostics/"
```

### Level 2: Enable when investigating a hang

Set these flags when you've observed a hang and want to reproduce it with better
diagnostics. These flags either add compiler instrumentation or change process
behavior on detection.

| Flag                                            | Default        | What it does                                                                                                                                                                                                                                             |
| :---------------------------------------------- | :------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--xla_tpu_debug_sflag_wait_timeout_ms`         | `0` (disabled) | LLO watchdog. Timeout in milliseconds for TensorCore sync flag waits (collectives and host-to-chip transfers). Set to `600000` (10 min) to enable. Requires recompilation.                                                                               |
| `--xla_tpu_debug_sflag_wait_shalt_on_detection` | `false`        | If `true`, the TPU core halts when the LLO watchdog fires, so the coordinator captures precise hardware state. If `false`, it only logs via the LLO logger (requires `--xla_tpu_enable_log_recorder`).                                                   |
| `--tpu_driver_callback_watchdog_timeout`        | `inf`          | Driver watchdog. Maximum time to wait for a TPU driver callback before triggering diagnostics and crashing. Only enable it (for example, `5m`) if Cloud Support asks you to. The process exits right after the trigger, so artifacts may not be written. |
| `--xla_tpu_diagnostic_collect_core_dumps`       | `false`        | Capture hardware state snapshots (register state, FIFO depths, analyzer reports) on each stall event. Snapshots can be large, so make sure the artifacts directory has enough free space.                                                                |
| `--xla_tpu_diagnostic_cooldown_duration`        | `15m`          | Minimum gap between consecutive harvests. Reduce to `5m` if you need to capture intermittent hangs more frequently. Triggers during the cooldown, including manual ones, are dropped.                                                                    |

**Example — reproduce a collective hang with full diagnostics:**

```sh
export LIBTPU_INIT_ARGS=" \
  --xla_tpu_impure_diagnostic_artifacts_dir=gs://MY_BUCKET/tpu-diagnostics/ \
  --xla_tpu_debug_sflag_wait_timeout_ms=600000 \
  --xla_tpu_debug_sflag_wait_shalt_on_detection=true \
  --xla_tpu_diagnostic_cooldown_duration=5m"
```

### Level 3: Advanced

These flags are for specific scenarios. Most users will never need them.

| Flag                                                | Default               | When to use                                                                                                                                                                       |
| :-------------------------------------------------- | :-------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--xla_tpu_debug_sc_sflag_wait_timeout_ms`          | `auto` (disabled)     | Same as `sflag_wait_timeout_ms`, but for sparse core sync flag waits. Only relevant for sparse core workloads.                                                                    |
| `--xla_tpu_diagnostic_aggregation_window`           | `10s`                 | How long each host's coordinator waits after the first trigger before it captures state. Triggers are aggregated per host, not across the Pod, so you rarely need to change this. |
| `--xla_tpu_diagnostic_artifacts_max_files`          | `20`                  | Increase if you need to retain more stall events before rotation.                                                                                                                 |
| `--xla_tpu_diagnostic_artifacts_max_bytes`          | `268435456` (256 MiB) | Caps total artifact bytes per host while preserving each process's newest stall summary.                                                                                          |
| `--xla_tpu_diagnostic_core_dump_max_files`          | `5`                   | Increase if you need to keep more hardware state snapshots before rotation. The limit is per chip.                                                                                |
| `--pjrt_tpu_event_tracker_slow_operation_threshold` | `5m`                  | Adjust the PJRT event tracker threshold. Reduce for faster detection, increase to reduce noise.                                                                                   |
| `--pjrt_tpu_event_tracker_logging_frequency`        | `inf` (disabled)      | Enable periodic logging of the event tracker state. Only for debugging the tracker itself.                                                                                        |

### About artifact output paths

The `--xla_tpu_impure_diagnostic_artifacts_dir` flag controls where artifacts
are written:

| Path type           | Example                | Behavior                                                                                                                                                                                                |
| :------------------ | :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Local directory** | `/tmp` (default)       | Artifacts written directly to the host filesystem. You must SSH or SCP to collect them. Files are readable only by the user that ran the workload (mode `0600`).                                        |
| **Cloud Storage**   | `gs://my-bucket/path/` | Artifacts are uploaded to GCS automatically. No SSH needed. All hosts can write to the same bucket. **Recommended for multi-host Pods.** The TPU VM's service account needs write access to the bucket. |

Artifact filenames include the host ID, timestamp, hostname, and PID for
uniqueness:

*   `tpu_stall_summary_host0_1722500096123456_my-tpu-w0_12345.pb`
*   `tpu_compiler_metadata_host0_1722500096123456_prog17_my-tpu-w0_12345.pb`

Old files are automatically rotated when they exceed `max_files` or `max_bytes`.
Rotation is scoped per `(hostname, pid)` and `tpu_compiler_metadata_*.pb` is
deduplicated per `(program_id, fingerprint)`, so multiple hosts sharing a
`gs://` directory won't evict each other's files. Manually triggered captures
use the resolved `host<id>` (`host-1` in `libtpu` ≤ 0.0.50).

-----

## Trigger diagnostics on demand

If your workload appears hung but the stall detectors haven't fired yet (for
example, the hang just started), you can trigger artifact collection manually.
Manual triggers are ignored during the cooldown after a capture (15 minutes by
default).

### Using the Unix Domain Socket directly

The coordinator listens on a Unix Domain Socket. You can trigger it from any
process on the same host:

```sh
echo -n trigger_dump | nc -U /tmp/tpu_diag_latest.socket
```

### For multi-host Pods

Trigger on all workers, or pass `--worker=N` to target one host:

```sh
gcloud compute tpus tpu-vm ssh TPU_NAME \
  --zone=ZONE \
  --worker=all \
  --command="echo -n trigger_dump | nc -U /tmp/tpu_diag_latest.socket"
```

Wait 10–30 seconds for the aggregation window and collection to complete, then
check for artifacts.

-----

## Step 1: Check for diagnostic artifacts

Connect to your TPU VM and check whether the runtime has written diagnostic
files:

```sh
gcloud compute tpus tpu-vm ssh TPU_NAME \
  --zone=ZONE \
  --command="ls -lh /tmp/tpu_stall_summary_*.pb /tmp/tpu_compiler_metadata_*.pb 2>/dev/null"
```

For multi-host Pods, check all workers:

```sh
gcloud compute tpus tpu-vm ssh TPU_NAME \
  --zone=ZONE \
  --worker=all \
  --command="ls -lh /tmp/tpu_stall_summary_*.pb 2>/dev/null"
```

If no files are found, see [No diagnostic artifacts
found](#no-diagnostic-artifacts-found).

-----

## Step 2: Collect artifacts

Choose the method that matches your setup.

### Single-host TPU VM

```sh
mkdir -p ./diag
gcloud compute tpus tpu-vm scp \
  'TPU_NAME:/tmp/tpu_*' ./diag/ \
  --zone=ZONE
```

### Multi-host Pod — Cloud Storage (recommended)

If you configured Cloud Storage output:

```sh
gcloud storage cp -r gs://BUCKET_NAME/tpu-diagnostics/ ./diag/
```

### Multi-host Pod — manual collection

```sh
mkdir -p ./diag
for i in $(seq 0 NUM_WORKERS_MINUS_1); do
  gcloud compute tpus tpu-vm scp \
    "TPU_NAME:/tmp/tpu_*" ./diag/ \
    --zone=ZONE --worker=$i
done
```

### GKE

```sh
POD_NAME=$(kubectl get pods -l app=YOUR_APP \
  -o jsonpath='{.items[0].metadata.name}')
kubectl cp $POD_NAME:/tmp/ ./diag/
```

**TIP:** For GKE workloads, mount a Cloud Storage FUSE volume so artifacts
survive Pod eviction.

-----

## Step 3: Generate the diagnostic report

Generate a diagnostic report from the artifacts collected in [Step
2](#step-2-collect-artifacts) using the report CLI
(`python -m libtpu.diagnostics`, requires `libtpu` 0.0.48+).

### CLI

```sh
python -m libtpu.diagnostics ./diag/
```

The CLI only accepts local directories. If your artifacts are in Cloud Storage,
first copy them locally as shown in [Step 2](#step-2-collect-artifacts) and then
run `python -m libtpu.diagnostics ./diag/`:

```sh
gcloud storage cp -r gs://BUCKET_NAME/tpu-diagnostics/ ./diag/
python -m libtpu.diagnostics ./diag/
```

### Example report

```
=== TPU Diagnostic Report ===
Artifact directory: ./diag/

--- Stall Summaries (4 files) ---
File: tpu_stall_summary_host0_1722500096123456_my-tpu-w0_12345.pb
  Core 0: PC=0x4a20, HLO=all-reduce.42, program=17 (SEQUENCER_TYPE_TENSOR_CORE)
  Core 1: PC=0x4a20, HLO=all-reduce.42, program=17 (SEQUENCER_TYPE_TENSOR_CORE)
...

--- Compiler Metadata (1 files) ---
File: tpu_compiler_metadata_host0_1722500096123456_prog17_my-tpu-w0_12345.pb
  Module: jit_train_step (program_id=17)
  Fingerprint: a1b2c3d4...
  Topology: 4 replicas x 2 cores/replica (haltless)
  Binary: 1048576 bytes, 850 HLO instructions

--- Cross-Reference: Stalled Programs ---
  Program 'jit_train_step' (id=17, 4 replicas, haltless)

--- Diagnosis ---
  ■ [A2] 6/8 cores at program_id=17, hlo=all-reduce.42. 1 outlier host(s). (confidence: very_high)
    Suspect: host2
    host2: 2 core(s) at program_id=17, hlo=fusion.7/add
  ▪ [B1] 6/8 cores at all-reduce.42. Outlier host(s): ['host2']. (confidence: high)
    Suspect: host2
    host2/core4: fusion.7/add
    host2/core5: fusion.7/add

Total: 5 artifact(s) processed.
```

-----

## Step 4: Interpret the results and take action

Use the **Pattern** field in the diagnosis to determine your next steps. The
stuck cores show where the hang surfaced, which isn't always where it started.

### Collective hang

**Pattern:** `collective_deadlock` (all cores at the same collective) or
`collective_timeout` (majority at collective, with outlier hosts)

All (or most) cores are stuck at the same collective operation — `all-reduce`,
`all-gather`, `collective-permute`, or `reduce-scatter`. The operation requires
all participants; one or more cores haven't arrived.

**What to do:**

1.  If the report lists **outlier hosts**: those hosts haven't reached the
    collective. Check their system logs for errors (see [Straggler / mismatched
    host](#straggler-host)).
2.  If **all cores** are at the same collective (no outliers): this suggests a
    network-level issue between hosts. Check host syslog for ICI errors and
    contact [Cloud Support](https://cloud.google.com/support).
3.  Verify your topology and partition counts match your TPU configuration.

### Straggler / mismatched host {#straggler-host}

**Pattern:** `pc_outlier` (`[A2]`), `run_id_behind` (`[A3]`), `program_mismatch`
(`[A1]`), or `collective_timeout` (`[B1]`) with `Suspect:` hosts listed

Most cores are waiting at a synchronization point or advancing through steps,
but one or more hosts fell behind, are stuck at a different program counter
(PC), or are running a different program.

**What to do:**

1.  Identify the **suspect or outlier hosts** from the `Suspect:` line in the
    report.
2.  SSH into the suspect host and check for:
    -   **Out-of-memory errors:** `sudo dmesg | grep -i oom`
    -   **CPU throttling:** `sudo dmesg | grep -i throttl`
    -   **Disk errors:** `sudo dmesg | grep -i error`
3.  If `program_mismatch` (`[A1]`) is reported, verify that all hosts run the
    exact same model code and software version.
4.  If `run_id_behind` (`[A3]`) is reported, check whether that host's data
    pipeline is delivering batches slower than other hosts.
5.  Check whether your data pipeline delivers batches evenly across hosts.
6.  If a single host consistently causes hangs, contact [Cloud
    Support](https://cloud.google.com/support) to request a replacement.

### Data pipeline starvation

**Pattern:** `infeed_stall` (cores waiting on `infeed`)

Cores are stalled waiting for input data from the host. The host-side data
pipeline cannot feed batches fast enough to keep the TPU cores busy.

**What to do:**

1.  Check host-side data loading pipelines (`tf.data`, PyGrain, DataLoader).
2.  Verify Cloud Storage (GCS) read throughput, network bandwidth, or disk I/O
    on the hosts.
3.  Profile data loading and preprocessing steps to identify host CPU or I/O
    bottlenecks.

### Core / hardware error

**Pattern:** `core_error`

A TPU core has reported an error message or hardware-level failure (such as a
core wedge or fatal status). The application cannot recover from this.

**What to do:**

1.  Contact [Cloud Support](https://cloud.google.com/support) immediately.
2.  Attach the diagnostic report and the raw .pb artifact files.
3.  Request a TPU host replacement.

### Widespread PC dispersion

**Pattern:** `pc_dispersion`

The cores are scattered across many different program locations with no clear
majority, indicating a potential cluster-wide deadlock or bus stall.

**What to do:**

1.  Enable [core dump collection](#level-2-enable-when-investigating-a-hang) for
    more detailed hardware state.
2.  Check whether the hang reproduces consistently.
3.  Contact [Cloud Support](https://cloud.google.com/support) with the full
    diagnostic report.

-----

## Read individual artifacts with the Python API

For advanced analysis, you can read and inspect individual artifact files
programmatically with `libtpu.sdk.diagnostics` (requires `libtpu` 0.0.47+).

### Read stall summaries

```py
from libtpu import sdk

stall = sdk.diagnostics.read_stall_summaries(
    "tpu_stall_summary_host0_1234.pb"
)

for core_id, state in stall["core_states"].items():
    for seq in state["sequencer_info"]:
        print(
            f"Core {core_id}: "
            f"PC=0x{seq.get('pc', 0):x}, "
            f"HLO={seq.get('hlo_location', '?')}, "
            f"program={seq.get('program_id', '?')}"
        )
```

**Key signals:**

-   Same PC + collective HLO across many cores indicates a collective hang
-   Different PCs across cores indicates that it is not a collective issue;
    investigate individually
-   Use `program_id` to cross-reference with compiler metadata

### Read compiler metadata

```py
meta = sdk.diagnostics.read_compiler_metadata(
    "tpu_compiler_metadata_host0_456.pb"
)

print(f"Module:     {meta['module_name']}")
print(f"Program ID: {meta['program_id']}")
print(f"Topology:   {meta['replica_count']}R × {meta['num_cores_per_replica']}C")
print(f"Haltless:   {meta['is_haltless']}")
print(f"Size:       {meta['executable_size']:,} bytes")
```

**Key signals:**

-   `module_name` tells you which compiled function is hung (e.g.,
    `jit_train_step`)
-   `program_id` cross-references with the stall summary to link cores to
    programs
-   `is_haltless=True` means the program runs without halt instructions — hangs
    require external watchdog detection

### Read core dumps (advanced)

> **Note:** Core dumps are only available if you set
> `--xla_tpu_diagnostic_collect_core_dumps=true` before the hang occurred.

```py
dump = sdk.diagnostics.read_core_dump(
    "tpu_core_dump_host0_789.bin"
)

# Check for hardware errors
for cs in dump["core_states"]:
    tags = cs.get("tags", [])
    category = cs.get("category", "")
    if "WEDGE" in tags or "ERROR" in category:
        print(
            f"⚠️  Chip {cs['chip_id']} Core {cs['core_id']}: "
            f"{category} — {cs['description']}"
        )

# Analyzer reports often contain the most specific diagnosis
for report in dump.get("symbolized_analyzer_reports", []):
    print(report)
```

### Generate a report programmatically

```py
from libtpu.diagnostics import generate_report

report = generate_report("/path/to/artifacts/")
print(report)
```

-----

## Troubleshooting

### No diagnostic artifacts found

If no `.pb` files appear in the artifacts directory (`/tmp` by default) after a
hang:

1.  **Verify the coordinator is enabled.** Check that `LIBTPU_INIT_ARGS` does
    not contain `--xla_tpu_enable_diagnostic_coordinator=false`.
2.  **Confirm the workload actually hung** (rather than crashed). The stall
    detectors need the process to remain alive. If the process exited with an
    error, no artifacts are generated.
3.  **Wait for the detection threshold.** The PJRT event tracker (the primary
    watchdog) requires \~5 minutes by default before triggering. If you killed
    the workload quickly, the watchdog may not have fired yet. If you enabled
    the LLO watchdog, its timeout is whatever you set in
    `--xla_tpu_debug_sflag_wait_timeout_ms`.
4.  **Try triggering manually:**
    `echo -n trigger_dump | nc -U /tmp/tpu_diag_latest.socket`. See [Trigger
    diagnostics on demand](#trigger-diagnostics-on-demand).
5.  **Check the cooldown.** After a capture, the coordinator drops new triggers,
    including manual ones, for 15 minutes by default
    (`--xla_tpu_diagnostic_cooldown_duration`).
6.  **Check the artifacts directory.** If you set
    `--xla_tpu_impure_diagnostic_artifacts_dir` to a non-default path, artifacts
    won't be in `/tmp`. If writes fail, the workload's stderr shows
    `Failed to write artifact to disk`.

### Common errors

| Problem                                                 | Solution                                                                                                                                                                                                                                    |
| :------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Artifacts appear but are from a previous stall          | Check the timestamp in the filename. The coordinator rotates old files, but if `max_files` is high, old artifacts may persist. Compare timestamps against your hang time.                                                                   |
| No artifacts even after 5+ minutes of hang              | The stall may be at the application level (Python GIL, data pipeline) rather than the TPU level. The TPU stall detectors only monitor TPU core execution. Check host-side CPU and I/O.                                                      |
| Manual trigger writes nothing                           | The coordinator is in its cooldown after a recent capture (15 minutes by default). Wait, or start the workload with a shorter `--xla_tpu_diagnostic_cooldown_duration`.                                                                     |
| Permission denied when copying artifacts                | Artifact files are created with mode `0600` and owned by the user that runs the workload (for example, `root` in a container). Make them readable before copying, for example with `sudo chmod a+r /tmp/tpu_*`.                             |
| No artifacts in Cloud Storage                           | Look for `Failed to write artifact to disk` in the workload's stderr. The TPU VM's service account needs write access to the bucket.                                                                                                        |
| `ModuleNotFoundError: No module named 'libtpu'`         | Install `libtpu` 0.0.48 or later: `pip install --upgrade libtpu`                                                                                                                                                                            |
| `AttributeError: module has no attribute 'diagnostics'` | The diagnostic parsers need `libtpu` 0.0.47 or later (`0.0.48+` for the CLI): `pip install --upgrade libtpu`                                                                                                                                |
| `RuntimeError: /path: No such file`                     | The file path is incorrect, or artifacts were not generated for this host. Verify with `ls /tmp/tpu_*.pb`.                                                                                                                                  |
| `read_core_dump` returns a parse error                  | The hardware state snapshot may be truncated (for example, the VM died mid-write). Use the stall summaries instead.                                                                                                                         |
| Artifacts on Cloud Storage but parsers fail             | The CLI and Python `read_*` functions only accept local directories and files. Download the artifacts first (`gcloud storage cp -r gs://BUCKET_NAME/tpu-diagnostics/ ./diag/` as in Step 2) and run `python -m libtpu.diagnostics ./diag/`. |

### Can I run the diagnostic tool on a machine without a TPU?

Yes. The diagnostic parsers are pure offline readers. Copy the `.pb` and `.bin`
files to any machine with `libtpu` installed and run the analysis there.
