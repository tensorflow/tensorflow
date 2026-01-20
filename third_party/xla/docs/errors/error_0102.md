# Error code: 0102

**Category:** Runtime: Program input buffer mismatch

This error occurs when the XLA runtime detects a mismatch between the size of a
memory buffer expected by a compiled program and the size of the buffer that is
actually provided at execution time.

**Sample Error Message:**

```
XlaRuntimeError: INVALID_ARGUMENT: Executable(jit_embedding_pipeline_step_fn) expected parameter 2482 of size 5242880 (bf16[16,1280,40]{2,1,0:T(8,128)(2,1)}) but got buffer with incompatible size 1638400 (bf16[16,1280,40]{1,2,0:T(8,128)(2,1)}): while running replica 0 and partition 0 of a replicated computation (other replicas may have failed as well).
```

**XLA backends:** TPU

## Overview

The error message indicates both the expected and actual sizes, as well as the
tensor shapes and layouts. Note that these errors might occur even if two
tensors have the same shape but their size in memory can be different if their
physical layout (how the data is tiled and arranged on the hardware) is
different.

These errors are predominantly caused by:

- **Checkpoint and XLA configuration mismatch** - A model is trained and a
checkpoint is saved. The physical layout of the weights in that checkpoint is
determined by the exact XLA version and configuration (e.g. XLA flags) at that
time. Later, this checkpoint is loaded in a different environment where the
configuration has changed. A new flag, a different default value, or a change
in the model/XLA code can cause the runtime to expect a different physical
layout for the weights. When the old buffer from the checkpoint is passed to
the new compiled XLA program, the runtime throws an error.
- **Hardware/Topology-Specific Layouts** - The XLA compiler is free to
choose different physical layouts for tensors to optimize performance on
different hardware. A layout that is optimal for v4 TPU might be different from
a v5 TPU, or even for different pod slices of the same chip (e.g., 4x4x4 vs
4x8). The error occurs when a model is compiled with an assumption about one
topology's layout, but at runtime it is scheduled on a different topology, or
there is a bug in the compiler's layout logic for a specific piece of hardware.

## Debugging

-   Ensure configuration consistency between model export and re-runs from
    checkpoints:
    -   Avoid using old checkpoints with new code unless you are certain that no
        layout-affecting changes have been made.
    -   Re-export the Saved Model: If you suspect a checkpoint/configuration
        mismatch, the most reliable solution is to re-export the saved model
        using the exact same (and current) codebase and configuration that you
        are using for inference or fine-tuning.
    -   Check for configuration changes (e.g. XLA flags) between the two runs.
-   Hardware/Topology-Specific layouts:
    -   Check for hardware version and topology mismatches if switching hardware
        or topologies.
