# Error code: 1200

**Category:** Compile Time: Host Offload Output Mismatch

This error occurs when a tensor explicitly offloaded to host memory is returned
as a program output, but the program's output signature is not configured to
expect host memory.

**Sample Error Messages:**

```
INVALID_ARGUMENT: Tensor which is moved to host (starting from tuple.64) is returned from the entry computation but the layout for this output is not set to host memory.
```

**XLA Backends:** TPU, GPU

## Overview

When the compiler encounters an annotation to offload a tensor to the host
(CPU), it tracks that tensor's location through the computation graph until one
of three events occurs:

1.  **Move to Device:** A matching annotation moves the tensor back to the
    accelerator.
2.  **Host Computation:** The tensor is consumed by a host-side operation.
3.  **Program End:** The tensor reaches the end of the program and becomes an
    output.

This error is triggered in scenario **#3**. The tensor is physically located in
host memory at the end of the execution, but the XLA program's entry
computation signature defines that specific output as residing in **Device
Memory**. Because the compiler cannot implicitly change the entry computation's
interface, it raises an error.

## Debugging

To resolve this error, determine whether you intended for this tensor to be an
output on the Host or if it should have been moved back to the Device before
returning.

### Verify Output Intent & Trace Path

*   **Intended to return on Host:** If you explicitly want this tensor to be
    returned in host memory (avoiding a transfer back to device), you should
    explicitly set the output memory space of the entry computation to **Host
    Memory** for this specific output.

*   **Intended to return on Device:** If the tensor was meant to stay on the
    device or return to it before the program ends, you likely missed an
    annotation. Insert a matching annotation to move the tensor back to the
    device.

If the source of the offloaded tensor is unclear, or you cannot find where
the "move to device" annotation is missing, use XLA logging to trace the
instructions.

*   **Enable Logging:** If you are on Google Cloud TPU, rerun your program with
    the following flag: `--vmodule=host_offloader=1`.
*   **Analyze Logs:** Look for the "trace" output in the logs. This will show
    the path of the tensor starting from the offload instruction. Use this to
    pinpoint exactly where the tensor reaches the program boundary without being
    moved back to the device.
