<!-- linter style off -->

# Tuning Megascale XLA Flags

MegascaleXLA (MXLA) coordinates communication across TPU slices over Data Center
Networks (DCN). This guide provides guidelines and best practices for
configuring and tuning MegascaleXLA flags for users running multi-slice TPU
workloads. An effort has been made to set reasonable defaults for most cases.

## Collective Buffer Sizes

For small collectives, it is generally more efficient to package small TPU
send/receives and network send/receives into larger ones to reduce latency
overhead. For larger collectives, it may be advantageous to use a greater number
of smaller intermediate buffers. A number of tools exist which allow for
altering this behavior. TPU DMA reads may be coalesced or split for some
collectives by passing,

```bash
--megascale_target_dma_size=<value>
```

which will attempt to schedule DMA reads that are as close as possible to the
target size. By default, a 8 MB default is used. It's only recommended to adjust
this value if a profiling trace shows considerable delay between the time a DMA
is initiated and the first network transfer for a particular collective is
initiated.

For small collectives, it is often beneficial to only use a subset of the
available participants in a collective for performing reduction operations. This
is known as "sparse reduction". The buffer size threshold for which sparse
reduction is enabled is controlled by the flag,

```bash
--megascale_sparse_reduction_threshold=<value>
```

By default, this value is 256 KB. If a profiler shows considerable latency
overhead for a reduction operation, it may be beneficial to adjust this value
upwards so that fewer reducers are used, resulting in a smaller number of larger
network transfers.

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

For one-to-one transfers (such as `collective-permute`), transfers on a host can
be divided into chunks using:

```bash
--megascale_chunk_size=<value>
```

If not zero, this divides each one-to-one transfer on a host into multiple
chunks of `megascale_chunk_size` bytes (default: 8 MB) to pipeline DMA and
network transfers. All other collectives use `megascale_target_dma_size`,
`megascale_sparse_reduction_threshold`, and
`megascale_max_reduction_shard_size` to control sharding and chunking.

---

## Memory Management & Zero-Copy Transfers

Megascale network transfers are designed to be zero-copy: host memory regions
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

<!-- linter style on -->
