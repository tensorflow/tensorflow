# Error code: E0100

**Category:** Runtime: Buffer Allocation Failure

This error indicates that XLA:TPU runtime’s memory allocator failed to find a
suitable block of memory on the accelerator’s HBM for the requested allocation.

**Sample error message:**

```
ValueError: RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 8.00M. That was not possible. There are 6.43M free.; (0x0x1_HBM0)
```

**XLA backends:** TPU

## Overview

This error is thrown on:

-   Failures of user-initiated buffer allocation via
    [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html)

or

-   Failures of user-scheduled program's output allocations.

These failures are typically caused due to a couple of reasons:

-   **Out of Memory (OOM):** The user is trying to allocate a chunk of memory
    that is larger than the total amount of free memory available on the TPU’s
    HBM.
-   **Memory fragmentation:** The allocation fails because **no single
    contiguous free block** in the memory space is large enough to satisfy the
    requested size. The total amount of free memory is sufficient for the
    allocation, but it is scattered across the memory space in small,
    non-contiguous blocks.

The TPU runtime has a number of mechanisms in-place to retry allocation failures
including:

-   If there are queued deallocations, the runtime retries failed allocations.
-   On OOMs caused by a fragmentation, the runtime can automatically trigger a
    defragmentation and a retry.
-   The TPU runtime prioritizes buffer allocations over keeping programs loaded.
    If a buffer allocation fails due to insufficient HBM, the system will evict
    loaded TPU programs until enough memory is available for the buffer.

So an error encountered after the above mitigations typically require user
action.

## Debugging

-   **Reduce your model's memory footprint**
    -   **Decrease batch size:** Reducing the batch size directly lowers memory
        usage.
    -   **Parameter sharding:** For very large models, use techniques like model
        parallelism or sharding to distribute parameters across the HBM of
        multiple TPU cores or hosts.
    -   **Shorten sequence/context length:** For models that operate on
        sequences (like language models), reducing the input sequence length can
        significantly decrease the memory footprint.
    -   **Buffer donation:** Utilize framework features (such as: `jax.jit(...,
        donate_argnums=...)`) to signal to XLA that certain input buffers can be
        overwritten and reused for outputs. Read
        [Buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)
        for more details.
    -   **Optimize checkpoint strategy:** Instead of saving the entire model
        state at once, consider saving only the model weights or using a sharded
        checkpointing strategy.
-   **Address memory layout and padding**
    -   TPU memory is allocated in chunks, and padding can increase the actual
        size of tensors.
-   **Ensure no memory leaks**
    -   Ensure references to `jax.Array` objects are not being held longer than
        intended. Holding on to `jax.Array` objects might prevent automatic
        de-allocation even after program compilation is completed.

See also [Error code: E1000](https://openxla.org/xla/errors/error_1000) for
other strategies you can use to reduce the amount of memory each program uses.

### Tooling

Enable the `tpu_log_allocations_on_oom` flag for which the allocator will dump a
detailed report of all current allocations when an OOM occurs, which can be
invaluable for debugging.
