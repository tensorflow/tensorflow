<!-- disableFinding("hang") -->

# Debugging hangs: Sync flag wait timeout

This instrumentation can be useful when programs appear to be hanging. When that
happens, it is most likely the program is waiting on a Direct Memory Access
(DMA) or a barrier to finish.

For DMAs, there can be many reasons why the expected DMA didn’t finish. For
example, the wrong receiving chip is specified or the wrong receiving count is
computed so that it is larger than the amount that was sent. For barriers, they
act as synchronization points among participating devices, implemented in a
similar way to DMAs except that they don’t carry any payload and simply update
and wait on sync flags.

**Sync flag wait instrumentation as the name suggests instruments sync flag wait
instruction to report timeout if the wait takes longer than specified value.**

This covers both DMA wait and barrier wait as they are implemented using sync
flag wait instructions under the hood. This instrumentation also covers
sync-wait instructions issued from both TensorCore and
SparseCore.[^1]

## Basic configuration

To start sync flag wait instrumentation of your program, set the flag:

```sh
--xla_tpu_debug_sflag_wait_timeout_ms=180000
```

This reports a timeout event if the wait takes longer than specified value.

If you want to terminate the program as soon as a hang is detected, set flag:

```sh
--xla_tpu_debug_sflag_wait_shalt_on_detection=true
```

If the cause is a timeout, you can expect to see fatal errors reported in the
log. Key words like "wait timeout on op" will help narrow down the specific HLO
op as well as the low-level instruction that caused the timeout.

Example log entries:

```sh
0x0x0_SC0: [Physical location: pazw15:pe0:0] generic::internal: Program or fatal error occurred; computation may be invalid: Assertion args: INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc SparseCoreSequencer:0:0xf2 (from SparseCoreSequencer:0:0x132): Sflag wait timeout on op {sc_tpu.dma_wait sflag(%alloca_1 : memref<1xi32, #sc_tpu.memory_space<sflag_scs>>) [%c0] granules(%c16)} after 1 milliseconds.
```

**TIP:** This instrumentation can be used along with the [hierarchical program
progress
tracker](./hierarchical_progress_tracker.md)
instrumentation to add more details helping localize the primary hang cause.

## Configurations to log SparseCore and TensorCore events

If collective or compute is offloaded to SparseCore, it may report error on both
SparseCore and TensorCore, whichever happens first, to disambiguate it is
recommended to use silent reporting described below.

To rule out false positives, it is also recommended to run with
`xla_tpu_debug_sflag_wait_shalt_on_detection` off but turn on TC and SC logger:

```
--xla_tpu_debug_sflag_wait_shalt_on_detection=false
--xla_tpu_enable_log_recorder=true
--xla_tpu_enable_sc_log_recorder=true
```

It is possible that the logger would report some timeout initially but it will
not terminate or block the program in case the hanging chip eventually gets
unblocked and makes progress. This is specifically true for launch barriers when
some cores may start significantly later. Also note that even though the program
may be unblocked, this behavior is not something that should be tolerated in
real workloads. In this case, it is important at the end to check for the last
reported timeout as the previous timeout entries are false positives.

### Detect false-positives

Sometimes it is possible that the user-specified timeout value is too small
which could lead to false positives. Recommended timeout value is 3 min (180000
ms) to be safe for large slices (\>=64 devices).

### Identify cause of hang

It is likely that different chips hang in different places.

If all chips report **timeout inside a specific HLO and all the waits are
non-barrier waits**, this is a strong indication that the problem lies within
this HLO.

If **some chips hang at barriers while others at normal DMAs**,

*   Then the next step is identifying if the barriers hangs are a result of the
    normal hangs or the cause. To do so, analyse the execution order of HLOs
    reported in the hangs.
*   Note that timeouts do propagate especially among collective operations. Some
    chips may get a headstart but will eventually hang if some other chips in
    its communication chain hangs. Currently, this analysis is done manually.
*   The key is to figure out which hang happened first as the rest of the
    timeouts would be deemed as false positives. To do that, map all the timeout
    events to the program control flow graph and analyze to determine the
    primary sources of hang.

## Flags reference

*   `--xla_tpu_debug_sflag_wait_timeout_ms`: enables instrumentation if the
    value is greater than 0, default: 0, recommended: 180000
*   `--xla_tpu_debug_sc_sflag_wait_timeout_ms`: enables instrumentation on
    SparseCore if the value is greater than 0, default: 0
*   `--xla_tpu_debug_sflag_wait_shalt_on_detection`: if true crashes the program
    on timeout, if false reports timeout via event and continues, default: false
*   `--xla_tpu_enable_log_recorder`: if true enables silent reporting of
    TensorCore (TC) events
*   `--xla_tpu_enable_sc_log_recorder`: if true enables silent reporting of
    SparseCore (SC) events
*   `--xla_tpu_sflag_busy_wait_filter`: specifies category of the sync flags to
    be instrumented, most used categories are: 'local', 'remote', 'all'
    (default), and 'barriers'
*   `--xla_tpu_enable_progress_tracker`: enables [progress
    tracker](./hierarchical_progress_tracker.md)
    to be used to localize the primary culprit of the hang, options: N, where N
    specifies how many hierarchical levels to be expected, valid range: `[8, …)`

## Further reading

*   [Hierarchical program progress
    tracker](./hierarchical_progress_tracker.md)

[^1]: More specifically, on TensorCore only `kVectorWaitGe`, `kDmaDoneWait`,
    `kDmaDone` are instrumented, and on SparseCore all wait kinds are
    instrumented.
