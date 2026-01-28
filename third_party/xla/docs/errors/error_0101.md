# Error code: 0101

**Category:** Runtime: Program Allocation Failure

This error indicates that the XLA runtime on a TPU device failed to load a
compiled XLA program executable into the TPU's HBM.

**Sample Error Message:**

```
XlaRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit_embedding_pipeline_step_fn': Attempting to reserve 29.49G at the bottom of memory. That was not possible. There are 147.64M free, 0B reserved, and 147.64M reservable. Scope: unknown..: while running replica 0 and partition 34 of a replicated computation (other replicas may have failed as well).
```

**XLA backends:** TPU

## Overview

This error is typically caused by one of the following reasons:

- Program Size Exceeds Available HBM: The compiled XLA program, including its
instructions, static data, and any embedded constants, is larger than the total
amount of free HBM currently available on the specific TPU core(s) where the
program is being loaded.
- HBM Fragmentation: While the total free HBM on the device might be sufficient
in aggregate, it is not available in a single, contiguous block large enough to
fit the entire program.

It's important to understand how the TPU runtime prioritizes memory. Buffer
allocations are privileged over loaded programs. If a buffer allocation fails,
the runtime will evict already loaded programs from HBM to free up space. This
can lead to a situation where a program that loaded successfully before now
fails with an OOM error, because the HBM is now occupied with more data buffers.

## Debugging

-   Reduce Buffer Memory Footprint: Freeing up memory used by data buffers will
    leave more room for the program itself:
    -   Decrease Batch Size: This is one of the most effective ways to reduce
        the amount of memory used for activations.
    -   Parameter Sharding: For very large models, use model parallelism or
        sharding techniques (like FSDP or Megascale) to distribute the model's
        parameters and computation across multiple TPU cores or hosts.
    -   Shorten Sequence/Context Length: For models processing sequential data
        (e.g., NLP models), reducing the sequence length can significantly
        decrease memory usage.
    -   Buffer Donation: Use framework features (e.g., `jax.jit(...,
        donate_argnums=...)`) to allow XLA to reuse the memory of input buffers
        for storing output, reducing peak memory usage.
-   Reduce program’s memory requirements for temporaries:
    -   Reduce programs memory usage for temporaries by using the
        `tpu_shared_memory_percent` flag. Note that this might negatively affect
        performance.
-   Optimize Execution Strategy/Reduce Serving load:
    -   Manage Program Loading: If you are JIT-compiling multiple functions, be
        aware that each function can result in a program being loaded. Try to
        structure your workload to minimize the number of concurrently loaded
        programs.
-   Ensure no memory leaks:
    -   Ensure references to `jax.Array` objects are not being held longer than
        intended. Holding on to `jax.Array` objects might prevent automatic
        de-allocation even after program compilation is completed.

### Tooling

-   Enable the `tpu_log_allocations_on_oom` flag for which the allocator will
    dump a detailed report of all current allocations when an OOM occurs, which
    can be invaluable for debugging.
-   Profile Your Program: Use the JAX memory profiler or the TensorFlow profiler
    to get a detailed view of your program's memory usage over time. This can
    help identify unexpected peaks in memory consumption.
