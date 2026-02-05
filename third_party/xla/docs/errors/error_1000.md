# Error code: 1000

**Category:** Compile Time: HBM OOM

This error indicates that the program requires more High Bandwidth Memory (HBM)
than is physically available on the TPU device.

**Sample Error Messages:**

```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 49.34G of 32.00G hbm. Exceeded hbm capacity by 17.34G.
```

```
RESOURCE_EXHAUSTED: TPU TensorCore Hbm usage: 34.82G, SparseCore Hbm usage 174.10G, exceeding available bytes: 95.74G
```

**XLA Backends:** TPU

## Overview

XLA performs checks to ensure that the aggregate size of all necessary static
allocations fit in the device's HBM.

The compiler manages the TPU's fixed HBM capacity for several types of
allocations:

* **Program Inputs and Outputs:** Training batches, optimizer states etc.
* **TensorCore + SparseCore Temporaries:** Dynamic memory required for
intermediate calculations (e.g. activations, gradients, etc).
* **Compiled Binary:** The machine code for both TensorCore (TC) and SparseCore
(SC).
* **System Overhead:** Reserved space for the XLA Runtime (e.g. infeed buffers
on older TPU generations).
* **Constants:** Constant values embedded in the HLO IR are allocated on HBM.
* **Compiler Internals:** Program level and per-HLO allocations (e.g. routing
info for nodes in the mesh)

This error occurs when the XLA compiler cannot fit all of the above allocations
into the device HBM.

## Debugging

Carefully analyze the error message and logs to determine which category of HBM
OOM below best describes your error:

* **TensorCore (TC) + SparseCore (SC) HBM Usage Exceeds Limit**:
If the error explicitly breaks down usage, e.g., *"TC Hbm usage: X, SC Hbm
usage Y"*. → Jump to
[Section 1. Balance TC and SC HBM usage](#section_1_balance_tc_and_sc_hbm_usage).
* **Unexpectedly Large Allocations**:
If the error reads *"Ran out of memory in memory space HBM"*, check the logs
for an enumeration of the largest allocations on HBM. In case, one or more
unexpectedly large tensors (e.g. > 50% of HBM limit) are present → Jump to
[Section 2. Unexpectedly Large Allocations](#section_2_unexpectedly_large_allocations).
* **Aggregate Allocations Exceed HBM Limit**:
If the error reads *"Ran out of memory in memory space HBM"* but no unexpectedly
large tensors are present in the logs → Jump to
[Section 3. Aggregate Allocations Exceed HBM Limit](#section_3_aggregate_allocations_exceed_hbm_limit).

---

### Section 1. Balance TC and SC HBM usage

If the error explicitly breaks down usage, e.g., *"TC Hbm usage: X, SC Hbm
usage Y"* compare the two values to identify the bottleneck

* **High SparseCore Usage:**
  * **Optimize HBM Stack Usage:** HBM stack memory consumption scales with
  `feature_width`, `max_unique_nz_per_row` and `logical_replica_count`. You
  can reduce peak stack usage by tuning the
  `--xla_sc_num_serialized_tables_to_optimize_hbm` flag which serializes the
  processing of tables. This comes at the cost of reduced parallelism.
  * **Check Padding Overhead:** SparseCore aligns embedding tables to 32B (8
floats). Tables with small feature widths (e.g., < 8 floats) incur
significant padding overhead, wasting HBM.
  * **Reduce Heap Usage:** High values for `maximum_parallel_iterations`
increase the amount of input data prefetched into the HBM heap. Lowering
this value can free up significant memory.
  * **Verify Sharding:** Ensure embedding tables are properly mod-sharded across
all chips. See
[How limits translate to tables](https://www.google.com/search?q=https://openxla.org/xla/sparsecore%237-how-limits-translate-to-tables-on-sparsecore).
  * Checkout [SC: Performance and memory bottlenecks](https://openxla.org/xla/sparsecore#10_performance_memory_bottlenecks) for more ideas.
* **High TensorCore Usage:**
  * Proceed to [Section 2](#section_2_unexpectedly_large_allocations).
* **Balanced**
  * If neither is individually excessive but the sum is too high, you are at the
chip's capacity. You must try lowering usage of both components. Follow
recommendations in all three sections.

### Section 2. Unexpectedly Large Allocations

If one or more unexpectedly large allocations are present in the logs
(> 50% of HBM limit), it is almost never a hardware capacity issue. It is
typically a configuration error. Check the XLA label (if present) of the large
allocations, for hints on their JAX source code.

* **Remove Debugging Artifacts:**
  * Using [jax.debug.print()](https://docs.jax.dev/en/latest/_autosummary/jax.debug.print.html)
  in large-scale runs can force the compiler to materialize the full tensor in
  HBM to transfer it to the CPU, breaking fusion and increasing peak memory
  usage. Remove any left-over `jax.debug.print()`s.
* **Fix Inefficient Mesh Shapes or Sharding:**
  * Incorrect mesh shapes or missing sharding annotations can cause the compiler
  to default to **Replication** - forcing the compiler to try to fit really
  large tensors on a single chip
  * Check the shapes of the large allocations and verify sharding is correctly
  specified and propagated by XLA.

### Section 3. Aggregate Allocations Exceed HBM Limit

If the program runs out of capacity due to the aggregate sum of allocations
exceeding the HBM limit, it is often helpful to visualize the memory profile to
identify the specific buffers contributing to the peak usage. See
[Debug OOM errors with XProf](https://openxla.org/xla/oom_debugging.md) for a
step-by-step guide on identifying peak memory contributors.

Once you have identified some of the top contributors, use the following steps
to optimize the memory footprint.

#### A. Check tensor padding and alignment

Inefficient tensor shapes are a common, silent cause of OOMs on TPUs. To get
peak performance on TPU's, XLA pads tensor dimensions—typically to multiples of
128 for the minor-most dimension and 8 for the second-minor. This padding
affects both input arrays and intermediate tensors (HLO temporaries),
potentially inflating memory usage significantly, especially with small
dimension sizes. See
[Array Layouts](https://docs.jax.dev/en/latest/pallas/tpu/details.html#array-layouts).

* **Audit shapes of large buffers:** (On TPU v5 with default layouts)
  * Hovering over a buffer in
  [Xprof Memory Viewer](https://openxla.org/xprof/memory_viewer#memory_viewer_components)
  brings up the buffer details card which contains buffer details including
  padding information.
  * *Example*: A shape of `(129, 1024)` might be padded to `(256, 1024)`,
  resulting in nearly 50% memory waste.
  * *Correction:* A shape of `(128, 1024)` requires no padding and incurs 0%
  memory waste.
* **Align dimensions:** Ensure all large tensor dimensions (batch size,
embedding dimension, hidden size) are multiples of 128.

#### B. Adjust configuration

You can often resolve OOMs with these configuration adjustments:

* **Reduce Batch Size:** The memory needed for intermediate activations and
gradients is directly proportional to the batch size. Reducing the batch size
can often help reduce memory usage.
* **Donate Input Buffers:** When using `jax.jit`, specify
[donate_argnums](https://docs.jax.dev/en/latest/buffer_donation.html) for your
model parameters. This allows XLA to overwrite the input memory with the output.
* **Enable Mixed Precision (bfloat16):** Use bfloat16 or quantization (int8 etc)
for the largest tensors in the program if the model architecture and quality
requirements allow.

#### C. Optimize architecture and sharding

If configuration changes are insufficient, the model topology might be too large
for the current hardware setup.

* **Use Newer TPU Generations:** Newer TPUs generally offer more HBM per chip;
switch to newer TPU generations if available.
* **Run on a larger chip topology:** If the model weights are too large for the
existing topology, you can try sharding them across more chips.
* **Implement advanced sharding techniques:**
  * Explore more advanced data, tensor, or pipeline parallelism approaches.
  * Specify [sharding hints](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#constraining-shardings-of-intermediates-in-jitted-code)
  for intermediate values and outputs.
* **Use JAX Host Offloading:** Offload large tensors to the host CPU memory.
e.g. [activation offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#activation-offloading)
and [optimizer state offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#optimizer-state-offloading).

#### D. Tune key memory impacting XLA flags:

[Key memory flags](https://openxla.org/xla/flags_guidance#memory_flags) can be
tuned to trade-off performance for lower memory usage. But these should be used
as a last resort measure because it can adversely affect performance.

#### E. Tune XLA Rematerialization Pass / Manual Checkpointing

If the model is close to fitting into memory, you can force the
`XLA::Rematerialization` pass to prioritize memory savings, potentially at the
cost of slower compilations:

| Flag | Description | Impact / Trade-off |
| --- | --- | --- |
| `--xla_tpu_max_hbm_size_mib` | Manually sets the limit on HBM size used by the Rematerialization pass. | Forces the compiler to work harder to fit the program into a limit smaller than the actual physical HBM. |
| `--xla_tpu_rematerialization_algo=PEAK_PRIORITY` | Focuses efforts at the points of peak memory usage. | Can be more efficient for aggressive memory reduction than the default algorithm. |
| `--xla_tpu_rematerialization_max_block_size_limit=32` | Controls the maximum number of instructions in a block that can be rematerialized at once. | Increasing this allows for memory savings at the cost of **significantly increases compile time**. |
| `--xla_tpu_rematerialization_block_effort_factor=10.0` | Defines the amount of effort (compile time) spent searching for blocks to rematerialize. | Higher values allow a more exhaustive search for memory savings at the cost of **increased compile times**. |
| `--xla_tpu_pre_fusion_remat=true` | Enables an additional Rematerialization pass *before* the fusion pass. | Can find more memory savings, but increases compile times and may **potentially impact numerical stability**. |

Alternatively, use the
[jax.checkpoint](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html)
decorator with `jax.grad` to manually control which intermediates are saved on
the forward pass versus recomputed on the backward pass, trading compute cycles
for HBM.

#### F. Use advanced profiling tools

[Debug OOM errors with XProf](https://openxla.org/xla/oom_debugging.md) provides
a tutorial on using the
[XProf Memory Viewer](https://openxla.org/xprof/memory_viewer) to visualize the
compiler's view of HBM usage.

This tool allows you to see peak memory allocation and buffer lifetimes, which
is crucial for understanding exactly what consumes HBM at the point of peak
utilization. For general profiling setup, see
[Getting started with Xprof](https://openxla.org/xprof#getting_started) and
[TensorBoard Profiling](https://docs.jax.dev/en/latest/profiling.html#xprof-tensorboard-profiling).
