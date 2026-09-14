<!-- linter style off -->

# Tuning MegaScale XLA Flags

This guide provides guidelines and best practices for configuring and tuning
MegaScaleXLA flags for users running multi-slice TPU workloads.

MegaScaleXLA (MXLA) coordinates communication across TPU slices over Data Center
Networks (DCN). While default settings are optimized for typical large language
model (LLM) and foundation model workloads, tuning specific flags can
significantly reduce collective communication overhead, alleviate host/network
bottlenecks, and enhance diagnostics during debugging.

[TOC]

## Host-Offloading

MegaScale supports two execution models for multi-slice reductions:

1. **Host-Offloaded Reductions (Default)**: Reduction computations are
   offloaded to host CPUs and host memory. Data is transferred via DMA from TPU
   to host, reduced across slices using CPU SIMD operations, and returned via
   DMA to TPU. This frees TPU matrix units (MXUs) and high-bandwidth memory
   (HBM) for pure model computation.
2. **TPU-Based Reductions**: Reductions are executed directly on the TPU cores.

### Flag Configuration

```bash
# To disable host-offloading and perform reductions on TPU cores:
--xla_tpu_use_megascale_host_reduction=false
```

---

## Collective Buffer Sizes

For small collectives, it is generally more efficient to package small TPU
send/receives and network send/receives into larger ones to reduce latency
overhead. For larger collectives, it may be advantageous to use a greater number
of smaller intermediate buffers. A number of tools exist which allow for
altering this behavior. TPU DMA reads may be coalesced or split for some
collectives by passing:

```bash
--megascale_target_dma_size=<value>
```

which will attempt to schedule DMA reads that are as close as possible to the
target size. By default, a 1 MB (or 8 MB in newer releases) default is used.
It's only recommended to adjust this value if a profiling trace shows
considerable delay between the time a DMA is initiated and the first network
transfer for a particular collective is initiated.

For small collectives, it is often beneficial to only use a subset of the
available participants in a collective for performing reduction operations. This
is known as "sparse reduction". The buffer size threshold for which sparse
reduction is enabled is controlled by the flag:

```bash
--megascale_sparse_reduction_threshold=<value>
```

By default, this value is 1 MB (or 256 KB in newer releases). If a profiler
shows considerable latency overhead for a reduction operation, it may be
beneficial to adjust this value upwards so that fewer reducers are used,
resulting in a smaller number of larger network transfers.

In some circumstances it may also be necessary to avoid large network transfers,
and instead use a greater number of smaller transfers. This can be adjusted with
the flag:

```bash
--megascale_max_reduction_shard_size=<value>
```

which will divide the buffer up into a greater number of smaller shards than is
otherwise needed by the collective. By default, this value is 8 MB. It should be
the case that `megascale_max_reduction_shard_size` >>
`megascale_sparse_reduction_threshold` since these have opposing effects.

Additionally, transfers on a host can be divided into chunks using:

```bash
--megascale_chunk_size=<value>
```

If not zero, this divides each transfer on a host into multiple chunks of
`megascale_chunk_size` bytes (default: 8 MB) to pipeline DMA and network
transfers.

---

## Memory Management & Zero-Copy Transfers

MegaScale network transfers are designed to be zero-copy: host memory regions
are pinned and premapped for DMA to and from the TPU. If premapped memory is
exhausted at runtime, the system falls back to on-demand mapping or dynamic
memory copies, severely impacting throughput.

### Pre-Mapped Memory Region

```bash
--megascale_grpc_premap_memory_bytes=17179869184  # Default: 16 GB (16LL << 30)
```

Controls the size of the host pinned-memory region allocated at initialization
for DMA transfers.

### Diagnostic Symptoms

1. **`MapDmaBuffer` in steady-state**: In the XProf Trace Viewer, search for
   `MapDmaBuffer`. These calls should only occur during startup. If they
   continue during training iterations, dynamic remapping is occurring.
2. **`Megascale: Memory Copy` traces**: If the Communication Transport track in
   XProf displays `Megascale: Memory Copy` events, memory buffers are being
   copied rather than transferred via zero-copy DMA.

### Resolution

Increase `--megascale_grpc_premap_memory_bytes` (e.g., to 24 GB or 32 GB,
depending on available host memory on your TPU VM instance) and restart the job.

---

## Diagnostic & Troubleshooting Flags

When diagnosing stalls, timeouts, or program straggler, the following flags
enable detailed error digests and prevent premature job crashes.

### Diagnostic Flags Reference

```bash
# Enable low-level logging for TensorCore and SparseCore execution states
--xla_tpu_enable_log_recorder=true
--xla_tpu_enable_sc_log_recorder=true

# Extend synchronization wait timeouts to prevent false-positive stall
# detections
--xla_tpu_debug_sflag_wait_timeout_ms=150000
--xla_tpu_debug_sc_sflag_wait_timeout_ms=150000

# Prevent immediate hard crash upon timeout detection so logs can be gathered
--xla_tpu_debug_sflag_wait_shalt_on_detection=false

# Enable hierarchical tracking of HLO progress
--xla_tpu_enable_progress_tracker=16

# Prevent the coordinator from aborting all workers prematurely on hang
--megascale_error_reporter_abort_on_hang=false
```

### Purpose and Usage

- **Preventing Premature Aborts**: By default, when a hang is detected, the
  coordinator immediately aborts all workers
  (`megascale_error_reporter_abort_on_hang=true`). Setting this flag to `false`
  keeps workers alive long enough to flush thread stack traces and detailed TPU
  states to Cloud Logging.
- **Investigating Slow Steps vs. Real Hangs**: For exceptionally large models or
  initialization steps that take longer than the default timeout (typically 60
  seconds), increasing `--xla_tpu_debug_sflag_wait_timeout_ms=150000` (150s)
  rules out false positives.
- **Inspecting Timeout Logs**: In Cloud Logging, search for:
  ```text
  Wait timeout on sflag
  ```
  Group matching logs by HLO opcode to isolate the first worker or core that
  failed to make progress, distinguishing the culprit from bystander workers
  waiting on dependencies.

---

<!-- linter style on -->
