# Error code: E1000

**Category:** Compile Time: HBM OOM

This error indicates that the program requires more High Bandwidth Memory (HBM)
than is physically available on the TPU device.

**Sample error messages:**

```
RESOURCE_EXHAUSTED: TPU TensorCore Hbm usage: 34.82G, SparseCore Hbm usage 174.10G, exceeding available bytes: 95.74G
```

```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 49.34G of 32.00G hbm. Exceeded hbm capacity by 17.34G.
```

**XLA backends:** TPU

## Overview

XLA performs checks to ensure that the aggregate size of all necessary static
allocations fit in the device's HBM.

The compiler manages the TPU's fixed HBM capacity for several types of
allocations:

-   **Program inputs and outputs:** Training batches, optimizer states etc.
-   **TPU temporaries:** Dynamic memory required for intermediate calculations
    (e.g. activations, gradients, etc).
-   **Compiled binary:** The machine code for both TensorCore (TC) and
    SparseCore (SC).
-   **System overhead:** Reserved space for the XLA Runtime (e.g. infeed buffers
    on older TPU generations).
-   **Constants:** Constant values embedded in the HLO IR are allocated on HBM.
-   **Compiler internals:** Program level and per-HLO allocations (e.g. routing
    info for nodes in the mesh).

This error occurs when the XLA compiler cannot fit all of the above allocations
into the device HBM.

## Debugging

Carefully analyze the error message and logs to determine which category of HBM
OOM below best describes your error:

-   **"TC Hbm usage: X, SC Hbm usage Y":** If the error explicitly breaks down
    HBM usage, the aggregate TensorCore (TC) + SparseCore (SC) usage exceeds the
    HBM limit. → Jump to
    [Scenario 1. Balance TC and SC HBM usage](#scenario_1_balance_tc_and_sc_hbm_usage).
-   **"Ran out of memory in memory space HBM"**: Check the logs for an
    enumeration of the largest allocations on HBM.
    -   In case one or more unexpectedly large tensors (e.g. > 50% of HBM limit)
        are present → Jump to
        [Scenario 2. Out of memory due to unexpectedly large allocations](#scenario_2_out_of_memory_due_to_unexpectedly_large_allocations).
    -   If no unexpectedly large tensors are present in the logs → Jump to
        [Scenario 3. Out of memory due to aggregate allocations](#scenario_3_out_of_memory_due_to_aggregate_allocations).

---

### Scenario 1. Balance TC and SC HBM usage

If the error explicitly breaks down usage, e.g., *"TC Hbm usage: X, SC Hbm usage
Y"*, this means the aggregate TensorCore (TC) + SparseCore (SC) usage exceeds
the HBM limit. Compare the two values to identify the bottleneck:

-   **High SparseCore usage**
    -   **Optimize HBM stack usage:** HBM stack memory consumption scales with
        `feature_width`, `max_unique_nz_per_row` and `logical_replica_count`.
        You can reduce peak stack usage by tuning the
        `--xla_sc_num_serialized_tables_to_optimize_hbm` flag which serializes
        the processing of tables. This comes at the cost of reduced parallelism.
    -   **Check padding overhead:** SparseCore aligns embedding tables to 32B (8
        floats). Tables with small feature widths (e.g., < 8 floats) incur
        significant padding overhead, wasting HBM.
    -   **Reduce heap usage:** High values for `maximum_parallel_iterations`
        increase the amount of input data prefetched into the HBM heap. Lowering
        this value can free up significant memory.
    -   **Verify sharding:** Ensure embedding tables are properly mod-sharded
        across all chips. See
        [How limits translate to tables](https://openxla.org/xla/sparsecore).
    -   Check out
        [SC: Performance and memory bottlenecks](https://openxla.org/xla/sparsecore#10_performance_memory_bottlenecks)
        for more ideas.
-   **High TensorCore usage**
    -   Proceed to
        [Scenario 2](#scenario_2_out_of_memory_due_to_unexpectedly_large_allocations).
-   **Balanced**
    -   If neither is individually excessive but the sum is too high, you are at
        the chip's capacity. You must try lowering usage of both components.
        Follow recommendations in all three sections.

### Scenario 2. Out of memory due to unexpectedly large allocations

If you observe the error message *"Ran out of memory in memory space HBM"* and
one or more unexpectedly large allocations are present in the logs (> 50% of HBM
limit), it is almost never a hardware capacity issue. It is typically a
configuration error. Check the XLA label (if present) of the large allocations
for hints on their JAX source code.

-   **Remove debugging artifacts**
    -   Using
        [jax.debug.print()](https://docs.jax.dev/en/latest/_autosummary/jax.debug.print.html)
        in large-scale runs can force the compiler to materialize the full
        tensor in HBM to transfer it to the CPU, breaking fusion and increasing
        peak memory usage. Remove any left-over `jax.debug.print()`s.
-   **Fix inefficient mesh shapes or sharding**
    -   Incorrect mesh shapes or missing sharding annotations can cause the
        compiler to default to **replication** - forcing the compiler to try to
        fit really large tensors on a single chip.
    -   Check the shapes of the large allocations and verify sharding is
        correctly specified and propagated by XLA.

### Scenario 3. Out of memory due to aggregate allocations

If you observe the error message *"Ran out of memory in memory space HBM"* and
no unexpectedly large tensors are present in the logs, the program runs out of
capacity due to the aggregate sum of allocations exceeding the HBM limit. In
this case, it is often helpful to visualize the memory profile to identify the
specific buffers contributing to the peak usage. See
[Debug OOM errors with XProf](https://openxla.org/xla/oom_debugging) for a
step-by-step guide on identifying peak memory contributors.

Once you have identified some of the top contributors, use the following steps
to optimize the memory footprint.

#### Scenario 3.A Adjust configuration

You can often resolve OOMs with these configuration adjustments:

-   **Reduce batch size:** The memory needed for intermediate activations and
    gradients is directly proportional to the batch size. Reducing the batch
    size can often help reduce memory usage, although you may need to retune
    your learning rate, momentum, or optimizer hyperparameters to maintain model
    stability.
-   **Donate input buffers:** When JAX executes a computation it uses buffers on
    the device for all inputs and outputs. If you know that one of the inputs is
    not needed after the computation, and if it matches the shape and element
    type of one of the outputs, you can specify that you want the corresponding
    input buffer to be donated to hold an output. This will reduce the memory
    required for the execution by the size of the donated buffer. You can
    achieve this by specifying
    [donate_argnums](https://docs.jax.dev/en/latest/buffer_donation.html)
    parameter as an argument when using `jax.jit`.
-   **Enable mixed precision (bfloat16):** Use bfloat16 or quantization (int8
    etc) for the largest tensors in the program if the model architecture and
    quality requirements allow. Note that this change can affect model behaviour
    and should be considered carefully.

##### Micro-batching (optional)

If reducing the global batch size or increasing the chip count is not viable,
and the batch size per chip is not already minimized, you can try a
micro-batching strategy:

-   Split each batch into `n` micro-batches;
-   For each micro-batch, process the forward and backward pass;
-   Once this is done, accumulate the gradients and update the weight as a
    whole.

This process reduces the activation memory as we divided each batch into `n`
micro-batches, so that the if the original batch had size `M`, the activation
memory size becomes `M/n`.

**Potential issues:** - This process increases step time as we have multiple
forward and backward passes. - If the sizes of the model and micro-batch are too
different, you may face convergence issues in your model.

#### Scenario 3.B Optimize architecture and sharding

If configuration changes are insufficient, the model topology might be too large
for the current hardware setup.

-   **Use newer TPU generations:** Newer TPUs generally offer more HBM per chip;
    switch to newer TPU generations if available.
-   **Run on a larger chip topology:** If the model weights are too large for
    the existing topology, you can try sharding them across more chips.
-   **Implement advanced sharding techniques:**

    -   Explore more advanced data, tensor, or pipeline parallelism approaches.
    -   Specify
        [sharding hints](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#constraining-shardings-of-intermediates-in-jitted-code)
        for intermediate values and outputs.

    Note that this may cause an increase in network communication overhead due
    to splitting tensors across multiple chips.

-   **Use JAX host offloading:** Host offloading techniques allow the user to
    offload large tensors to the host CPU memory (e.g.
    [activation offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#activation-offloading)
    and
    [optimizer state offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#optimizer-state-offloading)).
    Note that host offloading techniques can severely impact performance, since
    these operations will force the system to constantly move large tensors back
    and forth between TPU HBM and CPU RAM.

#### Scenario 3.C Check tensor padding and alignment

Inefficient tensor shapes are a common, silent cause of OOMs on TPUs. To get
peak performance on TPU's, XLA pads tensor dimensions—typically to multiples of
128 for the minor-most dimension and 8 for the second-minor. This padding
affects both input arrays and intermediate tensors (HLO temporaries),
potentially inflating memory usage significantly, especially with small
dimension sizes. See
[Array Layouts](https://docs.jax.dev/en/latest/pallas/tpu/details.html#array-layouts).

-   **Audit shapes of large buffers:** (On TPU v5 with default layouts)
    -   Hovering over a buffer in
        [Xprof Memory Viewer](https://openxla.org/xprof/memory_viewer#memory_viewer_components)
        brings up the buffer details card which contains buffer details
        including padding information.
    -   *Example*: A shape of `(129, 1024)` might be padded to `(256, 1024)`,
        resulting in nearly 50% memory waste.
    -   *Correction:* A shape of `(128, 1024)` requires no padding and incurs 0%
        memory waste.
-   **Align dimensions:** Ensure all large tensor dimensions (batch size,
    embedding dimension, hidden size) are multiples of 128. Note that this
    change can affect model behaviour and should be considered carefully.

#### Scenario 3.D Tune key memory impacting XLA flags

[Key memory flags](https://openxla.org/xla/flags_guidance#memory_flags) can be
tuned to trade-off performance for lower memory usage. However, this strategy
should be used as a last resort measure since it can adversely affect
performance.

#### Scenario 3.E Tune XLA rematerialization pass/manual checkpointing

If the model is close to fitting into memory, you can use the
[jax.checkpoint](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html)
decorator with `jax.grad` to manually control which intermediates are saved on
the forward pass versus recomputed on the backward pass. Note that this
operation may impact performance, since you are explicitly trading compute
cycles for HBM. Check out the JAX documentation for more information: -
[Gradient checkpointing with `jax.checkpoint` (`jax.remat`)](https://docs.jax.dev/en/latest/gradient-checkpointing.html) -
[Control autodiff’s saved values with `jax.checkpoint` (aka `jax.remat`)](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html) -
[JAX Memories and Host Offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)

Alternatively, you can force the `XLA::Rematerialization` pass to prioritize
memory savings, potentially at the cost of slower compilations:

| Flag | Description | Impact / Trade-off |
| --- | --- | --- |
| `--xla_tpu_max_hbm_size_mib` | Manually sets the limit on HBM size used by the Rematerialization pass. | Forces the compiler to work harder to fit the program into a limit smaller than the actual physical HBM. |
| `--xla_tpu_rematerialization_algo=PEAK_PRIORITY` | Focuses efforts at the points of peak memory usage. | Can be more efficient for aggressive memory reduction than the default algorithm. |
| `--xla_tpu_rematerialization_max_block_size_limit=32` | Controls the maximum number of instructions in a block that can be rematerialized at once. | Increasing this allows for memory savings at the cost of **significantly increases compile time**. |
| `--xla_tpu_rematerialization_block_effort_factor=10.0` | Defines the amount of effort (compile time) spent searching for blocks to rematerialize. | Higher values allow a more exhaustive search for memory savings at the cost of **increased compile times**. |
| `--xla_tpu_pre_fusion_remat=true` | Enables an additional Rematerialization pass *before* the fusion pass. | Can find more memory savings, but increases compile times and may **potentially impact numerical stability**. |

Note that making changes to XLA flags should be used as a last resort measure,
since it can adversely affect performance.

#### Scenario 3.F Use advanced profiling tools

[Debug OOM errors with XProf](https://openxla.org/xla/oom_debugging.md) provides
a tutorial on using the
[XProf Memory Viewer](https://openxla.org/xprof/memory_viewer) to visualize the
compiler's view of HBM usage.

This tool allows you to see peak memory allocation and buffer lifetimes, which
is crucial for understanding exactly what consumes HBM at the point of peak
utilization. For general profiling setup, see
[Getting started with Xprof](https://openxla.org/xprof#getting_started) and
[TensorBoard Profiling](https://docs.jax.dev/en/latest/profiling.html#xprof-tensorboard-profiling).

## Summary table

The following table contains a summary of potential interventions to solve OOM
errors and information that will help you decide what to do.

Intervention                              | Safe to do? (Will it change the behavior of the program?)                                                                                                                                       | Potential gains                                                                                                 | Telltale signs (is this actually the bottleneck that you're experiencing?)
----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------
Using advanced sharding techniques        | **Yes.** It almost never changes the numerical correctness of the experiment, although it can cause network communication overhead due to splitting tensors across multiple chips.              | **Massive gains** (up to a 256x reduction)                                                                      | Unexpectedly large individual allocations in the memory viewer (e.g., a single tensor replicated across all TPUs that is 256x bigger than the others). Active arrays showing as un-sharded in TensorBoard hooks.
Reducing batch size                       | **No.** It changes the training dynamics and usually requires retuning the learning rate. (Note: **Microbatching** is a safe alternative that reduces memory without changing behavior).        | **Massive gains** (can save a factor of thousands).                                                             | "Temporaries" failing to allocate during gradient calculations. Seeing "JVP" in the operation name, and encountering many batch-size-shaped tensors in the memory profile.
Enabling mixed Precision (e.g., Bfloat16) | **Risky.** It alters numerical precision, which can change experiment results or cause the model to fail to converge entirely.                                                                  | **Moderate gains** (typically a factor of 2x, as it halves memory usage).                                       | The memory viewer confirms that the largest tensors are currently utilizing 32-bit floats (`float32`).
Manual checkpointing (`jax.checkpoint`)   | **Yes.** It does not alter behavior; it merely trades computation time (flops) to save memory by recomputing tensors instead of storing them.                                                   | **Large gains** (e.g., can result in only half of the activations needing to exist in memory at the same time). | Multiple tensors of the exact same size filling up memory during a backward pass. Often accompanied by "JVP" in the operation name.
Donating input buffers (`donate_argnums`) | **Yes.** Maintains experiment integrity. If applied incorrectly, it will not corrupt data but will simply throw a clear error message.                                                          | **Marginal gains** (~1% memory savings).                                                                        | There is no specific telltale sign, but it is considered a "free win" that is always worth trying.
Changing model dimensions                 | **No.** Directly alters the model's behavior. Modifying input or output dimensions can completely break compatibility with the dataset.                                                         | **Variable gains** depending on how drastically hidden dimensions or layers are reduced.                        | The Xprof memory viewer shows a large amount of memory wasted on "padding" because array dimensions are not powers of two or multiples of 128 (e.g., a dimension of 2050 instead of 2048).
Host offloading (CPU)                     | **Yes (numerically)**, but **No (performance)**. While mathematically safe, it is considered a "foot gun" that may cause severe speed bottlenecks due to data transfer between the CPU and TPU. | **Marginal gains** (the CPU only has about 3x the memory of the TPU).                                           | Typically a last resort for massive optimizer states or for memory-heavy data preparation/pre-processing steps.
