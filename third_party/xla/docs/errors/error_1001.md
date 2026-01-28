# Error code: 1001

**Category:** Compile Time: Scoped Vmem OOM

This error indicates that the program requires more Scoped Vector Memory (Vmem)
than what was allocated.

**Sample Error Messages:**

```
RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem while allocating on stack for %my-custom-kernel = bf16[2048,4096]{1,0:T(8,128)(2,1)} custom-call(...) ...
```

**XLA Backends:** TPU

## Overview

TPUs have Vector Memory (VMEM) which is a local scratchpad memory used
exclusively by the TensorCore (TC). The compiler manages Vmem for different
types of allocations:

* **Instruction-scoped allocations:** Temporary storage in Vmem while executing
a single HLO instruction. This includes operand span buffer (e.g. for double
buffering) and register spills.
* **Program-scoped allocations:** Allocations that live beyond the scope of
a single HLO instruction. These are usually HLO temporaries and intermediate
results that are inputs and/or outputs of HLO instructions.

A Compile Time Scoped Vmem OOM occurs when the instruction-scoped allocations
exceed the allocation limit for that instruction. This limit is controlled

-   globally for the entire program via the flag
    [--xla_tpu_scoped_vmem_limit_kib](https://openxla.org/xla/flags_guidance)
    and
-   per custom kernel via
    [vmem_limit_bytes param](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams.vmem_limit_bytes).

These errors are typically caused by an internal compiler bug or by a
custom kernel exceeding its allocation limit.

## Debugging

Carefully analyze the error message to identify if the error stems from a
custom kernel or a standard HLO. An error due to a custom kernel should have
the following signature:

```
Ran out of memory in memory space vmem while allocating on stack for %my-custom-call = <output-shape> custom-call(<params>), custom_call_target="tpu_custom_call" ...
```

*   **Custom Kernel Scoped Vmem OOM**: If the error points to a custom kernel →
    Jump to [Retune the Kernel](#retune_the_kernel).
*   **Non-Kernel Vmem Issues**: If the Vmem OOM occurs due to a
    non-custom-kernel op, it is likely an internal compiler bug. Please file a
    bug on XLA with an HLO dump.

---

### Retune the Kernel

If the error originates from a custom kernel, use the following techniques to
reduce the kernel's memory requirement:

* **Adjust Block Sizes:** Reduce the block sizes (tile sizes) in your kernel
configuration, to lower Scoped Vmem usage.
* **Set Per-Kernel Scoped Vmem Limits:** Explicitly request the required amount
of memory for that specific kernel using the
[vmem_limit_bytes param](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams.vmem_limit_bytes)
* **Modify Memory Coloring:** Explicitly color/constrain the kernel's
inputs/outputs to VMEM using
[pallas.tpu.with_memory_space_constraint](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.with_memory_space_constraint.html). But be careful not to color too many inputs
outputs to Vmem, as that might cause an overall VMEM OOM.
* If kernel specific retuning is difficult or the issue affects many kernels,
you can adjust the global Vmem limit using the flag
[--xla_tpu_scoped_vmem_limit_kib](https://openxla.org/xla/flags_guidance).
