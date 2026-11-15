# Debug callback in libtpu.sdk.debug

<!-- disableFinding("hang") -->

The Python debug callback provides an asynchronous mechanism to receive runtime
diagnostics directly from libtpu during TPU execution.

When hardware stalls, synchronization timeouts, numeric anomalies, or device
crashes occur on the TPU, libtpu invokes your registered callback with
structured diagnostic data. This allows your training or inference script to log
events, record program state, or trigger alerts without parsing console logs or
polling device registers.

## Prerequisite

**You need `libtpu` 0.0.46 or later for the debug callback.** Install or upgrade
it, then check the version:

```sh
pip install --upgrade libtpu
python3 -c "import importlib.metadata as m; print(m.version('libtpu'))"
```

Registering a callback needs no extra setup. However, only stall events
(`STALL_EVENT`) are reported by default. Every other event is reported only
after you enable its instrumentation; the required flags are listed under
**Trigger condition** in [Supported events](#supported-events).

## Usage guidelines

### Create your callback function

Define the function to be invoked when an event occurs. Your callback function
must accept two mapping arguments: **`payload` and `metadata`**, and **return
`None`**.

Events are delivered to the callback through the mapping arguments:

*   **payload**: Contains large runtime artifacts, such as NumPy ndarrays
    (`literal`), raw artifact bytes (`serialized_artifact`), or parsed per-core
    state dumps (`core_states`).
*   **metadata**: Contains lightweight diagnostic attributes, such as boolean
    event flags (`is_sflag_timeout`, `is_sc_mismatch`), source locations
    (`core_location`), and tags (`user_tag`).

Neither mapping has an event-type field: the event names in this guide, such as
`SFLAG_TIMEOUT`, are labels only. Identify an event by the keys that are
present, for example `is_sflag_timeout` in metadata or `core_states` in payload,
as shown in the Example workflow.

### Register (and de-register) your function

You need to register the callback function before the workload execution using
`**libtpu.sdk.debug.register_debug_callback**(callback_fn) -> int`:

*   **Argument:** `callback_fn`, a callable taking (`payload: Mapping[str, Any],
    metadata: Mapping[str, Any]`) and returning `None`
*   **Return:** An integer `callback_id` representing the registration handle.
    Raises `RuntimeError` if registration fails.

Always unregister a previously registered callback after execution completes or
errors using `**libtpu.sdk.debug.unregister_debug_callback**(callback_id: int)
-> None`:

*   `callback_id`: The integer handle returned by `register_debug_callback`

<!-- end list -->

```py
from typing import Any, Mapping
import libtpu.sdk.debug as tpu_debug

# Define your callback function
def callback_fn(
    payload: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
  ...


def main():
  # Register the callback before starting the execution loop
  callback_id = tpu_debug.register_debug_callback(callback_fn)

  try:
    # Training or inference loop runs here
    pass
  finally:
    # Always unregister when finished or on error
    tpu_debug.unregister_debug_callback(callback_id)
if __name__ == "__main__":
  main()

```

IMPORTANT:

*   **Thread Safety:** Callbacks execute on a background thread managed by
    `libtpu`. Any state mutated in your Python callback should be protected with
    appropriate thread synchronization or restricted to thread-safe data
    structures.
*   **Keep Callbacks non-blocking:** Do not perform blocking operations or
    long-running computations inside the callback. If heavy processing is
    needed, push the event to a queue.Queue for a separate worker thread. The
    callback holds the Python GIL while it runs, so a slow callback also blocks
    your other Python threads.
*   **Exceptions:** Exceptions raised inside the callback are caught and logged
    by `libtpu`; they are not propagated to your program.

## Supported events

The following event types are supported by the debug callback mechanism.

| Event Type            | Trigger condition                                                                                                                                                                                                                                                                                                                                         | metadata Fields                                                                                                                                                                                                                                 | payload Fields                                                                                                                                     |
| :-------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| SFLAG_TIMEOUT        | A TensorCore wait on a sync flag (sflag), used by DMAs (intra-chip, inter-chip, or host-to-chip) and barriers, exceeded the timeout. Requires `--xla_tpu_debug_sflag_wait_timeout_ms` \> 0, `--xla_tpu_debug_sflag_wait_shalt_on_detection=false`, and `--xla_tpu_enable_log_recorder=true`. See [TPU hang: Sync flag wait timeout](./sync_flag_wait_timeout.md). | `is_sflag_timeout`: `True` <br> `user_tag`: `str` <br> `core_location`: `str`                                                                                                                                                                   | literal: `np.ndarray` <br> `expected_sflag_value`: `int` <br> `latest_sflag_value_seen`: `int` <br> `progress_tracker_current_depth`: `int` <br> `progress_tracker_meta_id`: `int` <br> `progress_tracker_levels`: `list[tuple[int, int]]` <br> `unfinished_send_sflags_count`: `int` <br> `unfinished_send_sflags`: `list[int]` <br> `errors`: `list[str]`                                                                             |
| BOUNDS_CHECK_FAIL   | Reserved; not emitted in `libtpu` 0.0.46. Bounds-check failures are reported in the program log instead.                                                                                                                                                                  | `is_bounds_check_fail`: `True` <br> `user_tag`: `str` <br> `core_location`: `str`                                                                                                                                                               | literal: `np.ndarray` (offending index/offset)                                                                                                     |
| STALL_EVENT          | The TpuDiagnosticCoordinator detected a stall (for example, a watchdog or sflag timeout) or received a manual trigger, and captured hardware state snapshots. On by default. Reported once per host about 10 seconds after the first trigger, then at most once every 15 minutes. See [TPU hang diagnostics](./hang_diagnostics.md).                         | `  source `: `int` <br> `host_id:` `int` <br> `description`: `str` (optional) <br> `timeout_ms`: `int` (optional) <br> `core_location`: `str` (always "unknown") <br> `launch_id`: `int` (optional)                                             | `core_states`: `dict` (absent if parsing fails) <br> `serialized_artifact`: bytes <br> `core_states_parsing_error`: `True` (only if parsing fails) |
| SC_MISMATCH          | SparseCore embedding lookup or scatter/gather mismatch.                                                                                                                                                                                                                                                                                                   | `is_sc_mismatch`: `True` <br> `user_tag`: `str` <br> `core_location`: `str`                                                                                                                                                                     | literal: `np.ndarray` (if enabled)                                                                                                                 |
| SDC_NAN_INF         | Op generated or received unexpected NaN or Inf values.                                                                                                                                                                                                  | `contains_nan`: `bool` <br> `contains_inf`: `bool` <br> `is_input_nan_or_inf`: `bool` <br> `deduplicated_names`: `list[str]`  `core_location`: `str` `is_collective_hlo`: `bool` `is_tpu_custom_call`: `bool` <br> `user_tag`: `str` (optional) | literal: `np.ndarray` (if enabled)                                                                                                                 |
| SDC_OUTPUT_MISMATCH | Redundant execution check found mismatched outputs.                                                                                                                                                                                | `is_input_mismatch`: `bool` <br> `deduplicated_names`: `list[str]` <br> `core_location`: `str` <br> `is_collective_hlo`: `bool` <br> `is_tpu_custom_call`: `bool` <br> `user_tag`: `str` (optional)                                             | literal: `np.ndarray` (if enabled)                                                                                                                 |
| HLO_LOGGER           | Instruction compiled with logging enabled finished execution.                                                                                                                                                                                                                                                                                             | `instruction_name`: `str` <br> `user_tag`: `str` <br> `extra_args`: `list[str]` (optional) <br> `core_location`: `str`                                                                                                                          | literal: `np.ndarray` (tensor values)                                                                                                              |
| FUSION_DEBUG         | Fused instruction compiled with debug instrumentation executed.                                                                                                                                                                                                                                                                                           | `instruction_name`: `str` <br> `user_tag`: `str` <br> `core_location`: `str`                                                                                                                                                                    | `duration_cycles`: `int` <br> literal: `np.ndarray` (if captured)                                                                                  |
| CORE_DUMP            | The runtime captured a hardware state snapshot (core dump) after the device halted on a fatal error. See [TPU crash: Analyze core dump](./analyze_core_dumps.md).                                                                                                                                                                                            | `instruction_name`: `str` (optional) <br> `core_dump_tc_values`: `list[dict]` <br> `core_dump_sc_values`: `list[dict]` <br> `core_location`: `str`                                                                                              | (empty)                                                                                                                                            |

`STALL_EVENT` is covered in [TPU hang diagnostics](./hang_diagnostics.md),
including how to trigger a snapshot on a running job. Each `SFLAG_TIMEOUT` event
also counts as a stall trigger, so expect a `STALL_EVENT` with `source=3`
(`SFLAG_TIMEOUT`) from the same host about 10 seconds later.

Check the [Reference](#reference-events-structure) section for more details
about the structure of the events.

## Example workflow

The following example script shows a complete workflow: how to register a
callback in a training script, handle common events, and cleanly unregister upon
completion.

```py
import sys
from typing import Any, Mapping
import libtpu.sdk.debug as tpu_debug


def on_tpu_debug_event(
    payload: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
  """Called asynchronously by libtpu when an event occurs."""


  # 1. Sync Flag Timeouts
  if metadata.get("is_sflag_timeout"):
    tag = metadata.get("user_tag", "N/A")
    core = metadata.get("core_location", "unknown")
    print(
        f"[TPU TIMEOUT] Core {core} timed out waiting on sync flag (tag: {tag})",
        file=sys.stderr,
        flush=True,
    )


  # 2. Out-of-bounds memory access (reserved; not emitted in libtpu 0.0.46)
  elif metadata.get("is_bounds_check_fail"):
    tag = metadata.get("user_tag", "N/A")
    core = metadata.get("core_location", "unknown")
    bad_offset = payload.get("literal")
    print(
        f"[TPU BOUNDS ERROR] Bounds violation at {core} (offset: {bad_offset}, tag: {tag})",
        file=sys.stderr,
        flush=True,
    )


  # 3. Hardware Stalls
  elif metadata.get("source") is not None or "core_states" in payload:
    host_id = metadata.get("host_id", "N/A")
    desc = metadata.get("description", "")
    source = metadata.get("source")  # 0 (UNKNOWN) for a manual trigger
    print(
        f"[TPU STALL] Host {host_id}, source {source}: {desc}",
        file=sys.stderr,
        flush=True,
    )


    core_states = payload.get("core_states", {})
    for core_id, state in core_states.items():
      for seq in state.get("sequencer_info", []):
        pc = seq.get("pc", 0)
        hlo_op = seq.get("hlo_location", "N/A")
        print(
            f"  -> Core {core_id} at PC {hex(pc)} in {hlo_op}",
            file=sys.stderr,
            flush=True,
        )


  # 4. Numerical Anomalies (NaN / Inf)
  elif metadata.get("contains_nan") or metadata.get("contains_inf"):
    ops = ", ".join(metadata.get("deduplicated_names", []))
    core = metadata.get("core_location", "unknown")
    is_input = metadata.get("is_input_nan_or_inf", False)
    kind = "/".join(
        name
        for key, name in (("contains_nan", "NaN"), ("contains_inf", "Inf"))
        if metadata.get(key)
    )
    print(
        f"[TPU NUMERICS] {kind} detected at {core} in ops: {ops} (input already"
        f" {kind}: {is_input})",
        file=sys.stderr,
        flush=True,
    )


  # 5. Everything else (HLO logger, fusion, SDC mismatch, SparseCore, core dump)
  else:
    print(f"[TPU EVENT] {dict(metadata)}", file=sys.stderr, flush=True)




def main():
  # Register the callback before starting the execution loop
  callback_id = tpu_debug.register_debug_callback(on_tpu_debug_event)
  print(f"Registered debug callback ID {callback_id}", flush=True)


  try:
    # Training or inference loop runs here
    pass
  finally:
    # Always unregister when finished or on error
    tpu_debug.unregister_debug_callback(callback_id)
    print(f"Unregistered debug callback ID {callback_id}", flush=True)


if __name__ == "__main__":
  main()

```

## Reference: Events structure

### Sync flag timeout (`SFLAG_TIMEOUT`)

```
# metadata
{
    "is_sflag_timeout": True,
    "user_tag": "Wait timeout on sflag {...} after 180000 milliseconds, ...",  # Timeout message
    "core_location": "core 0",
}

# payload
{
    "literal": np.array([...], dtype=np.int32),  # Optional array if literal capture is enabled
    "expected_sflag_value": 16,  # Target sync flag value waited on
    "latest_sflag_value_seen": 15,  # Last sync flag value seen before timeout
    # Optional: present when --xla_tpu_enable_progress_tracker > 0
    "progress_tracker_current_depth": 1,
    "progress_tracker_meta_id": -1,  # LLO metadata ID (-1 if unset)
    "progress_tracker_levels": [  # (iteration, counter) per level; (-1, -1)  if unused
        (0, 2),
        (-1, -1),
    ],
    # Optional: present when --xla_tpu_enable_dma_send_sync_flag_tracker=true
    "unfinished_send_sflags_count": 1,
    "unfinished_send_sflags": [53, -1, -1, -1, -1],  # -1 for unused slots
    # "errors": ["..."],  # Only if decomposing literal fails; some fields above may then be absent
}

```

### Memory bounds violation (`BOUNDS_CHECK_FAIL`)

```
# Reserved: not emitted in libtpu 0.0.46.
# metadata
{
    "is_bounds_check_fail": True,
    "user_tag": "bounds_check",
    "core_location": "core 1",
}

# payload
{
    "literal": np.array(
        [1048576], dtype=np.int64
    ),  # Offending memory index or offset
}

```

### TPU stall or hang (`STALL_EVENT`)

```
# metadata
{
    # StallSource enum: UNKNOWN=0, PJRT_WATCHDOG=1, HAL_WATCHDOG=2,
    #                   SFLAG_TIMEOUT=3, CORE_STALL=4, HARDWARE_WEDGE=5
    "source": 1,
    "host_id": 0,  # -1 for a manual trigger in libtpu ≤ 0.0.50
    "description": "...",  # Optional: summary of the triggers
    "timeout_ms": 300000,  # Optional
    "core_location": "unknown",  # Not set for stall events
    "launch_id": 12345,  # Optional: runtime launch identifier
}

# payload
{
    "core_states": {
        0: {  # Keyed by integer core ID
            "core_id": {
                "global_core_id": 0,
                "chip_id": 0,
                "core_on_chip": {"type": "TPU_CORE_TYPE_TENSOR_CORE", "index": 0},
            },
            "sequencer_info": [{
                "sequencer_type": "SEQUENCER_TYPE_V",
                "sequencer_index": 0,
                "pc": 4260,  # Program counter (0x10a4)
                "tag": 12,
                "tracemark": 0,
                "program_id": 100,
                "run_id": 1,
                "hlo_location": "collective-permute.4",
                "hlo_detailed_info": "collective-permute",
            }],
            "xdb_server_running": True,
            "program_fingerprint": b"...",
            "launch_id": 12345,
            "queued_program_info": [
                {"run_id": 2, "launch_id": 12346, "program_fingerprint": b"..."}
            ],
            "error_message": "...",
        }
    },
    "serialized_artifact": b"...",  # Raw bytes of AllCoreStateSummaries
    # "core_states_parsing_error": True,  # Only if parsing fails; core_states is then absent
}

```

### SparseCore mismatch (`SC_MISMATCH`)

```
# metadata
{
    "is_sc_mismatch": True,
    "user_tag": "embedding_scatter",
    "core_location": "core 0",
}

# payload
{
    "literal": np.array([...]),  # Optional array if literal capture is enabled
}
```

### Detected NaNs and Infs (`SDC_NAN_INF`)

```
# metadata
{
    "contains_nan": True,
    "contains_inf": False,
    "is_input_nan_or_inf": False,  # True if input was already NaN; False if op produced it
    "deduplicated_names": ["layer1/matmul", "layer1/bias_add"],
    "is_collective_hlo": False,
    "is_tpu_custom_call": False,
    "user_tag": "forward_pass",  # Optional: instruction tag
    "core_location": "core 0",
}

# payload
{
    "literal": np.array([...]),  # Optional tensor literal if captured
}

```

### SDC HLO output mismatch (`SDC_OUTPUT_MISMATCH`)

```
# metadata
{
    "core_location": "core 0",  # Core where mismatch was detected
    "deduplicated_names": ["conv.1"],  # Involved HLO instruction names
    "is_input_mismatch": False,  # True if input operands already mismatched
    "is_collective_hlo": False,  # True if operation was a collective op
    "is_tpu_custom_call": False,  # True if operation was a custom call
    "user_tag": "mismatch_tag",  # Optional: tag if specified
}
# payload
{
    "literal": np.array([...]),  # Optional array if literal capture is enabled
}
```

### HLO logger event (`HLO_LOGGER`)

```
# metadata
{
    "core_location": "core 0",  # Physical or logical core location
    "user_tag": "my_tag",  # User-defined tag on the HLO instruction
    "instruction_name": "custom-call.1",  # Name of the HLO instruction
    "tag": "hlo_logger",  # Compiler log record tag
    "extra_args": ["arg1", "arg2"],  # Optional: list of extra arguments
}

# payload
{
    "literal": np.array([...]),  # Logged tensor data as a NumPy ndarray
}
```

### Fusion debug event (`FUSION_DEBUG`)

```
# metadata
{
    "core_location": "core 0",
    "user_tag": "fusion_trace",
    "instruction_name": "fused_computation.3",  # Fused HLO instruction name
}

# payload
{
    "duration_cycles": 14250,  # Execution duration in hardware cycles
    "literal": np.array([...]),  # Optional tensor values if captured
}
```

### Hardware state snapshot (`CORE_DUMP`)

```
# metadata
{
    "core_location": "core 0",
    "instruction_name": "dense_layer.1",  # Optional: associated HLO instruction
    "core_dump_tc_values": [
        {"title": "PC", "values": ["0x10a0", "0x10b4"]},
        {"title": "STATUS", "values": ["HALTED"]},
    ],
    "core_dump_sc_values": [
        {
            "scs_info": [{"title": "SCS_STATE", "values": ["IDLE"]}],
            "tile_info": [{"title": "TILE_ERR", "values": ["0x0"]}],
            "misc_info": [{"title": "INFO", "values": ["OK"]}],
        }
    ],
}

# payload
{}  # Empty: core dump values are in metadata

```
