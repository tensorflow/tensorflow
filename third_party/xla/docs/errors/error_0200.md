# Error Code: E0200

**Category:** Runtime: Core Halted Unexpectedly

This error indicates that a TPU core stopped executing
instructions prematurely. This is a fatal error state where the hardware forces
a halt due to an unrecoverable fault, a violation of hardware constraints, or a
deliberate interrupt triggered by compiler-generated runtime assertions.

**Sample Error Message:**

```
INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0x1d9 ...
```

**XLA backends:** TPU

## Overview

XLA compiles JAX programs into a sequence of low-level assembly instructions.
At runtime, the TPU device executes these instructions sequentially. A
"Core Halted Unexpectedly" error occurs when the TPU hardware encounters an
unrecoverable condition that prevents further execution, forcing the core into a
fatal "HALTED" state.

Because this error can stem from physical hardware failures, compiler bugs, or
user code issues (particularly in custom kernels), you must carefully analyze
the log messages to identify the specific cause.

## Debugging

To resolve this error, you must first identify which of the three specific
scenarios caused the unexpected halt. Check your logs for the specific text
signatures described below.

### Scenario 1: Infrastructure Failures (Hardware/Network/Power)

**Signature:** The logs explicitly state `observed errors are: [Network]` or
`observed errors are: [Power]` or `observed errors are: [Hardware]`.

This indicates a physical infrastructure failure unrelated to your software or
model logic. The TPU chip, the network fabric connecting the chips, or the
power supply has failed.

* **Retry the job:** If the issue was a transient voltage dip or network
flap, a simple retry may work.
* **Identify and Remove Bad Nodes:** If the error persists on the same
specific task or host, the hardware is likely defective. Use your cluster
management tools to "drain"/"cordon" the affected node and restart your
job on healthy nodes.

### Scenario 2: Hardware Constraint Violations

**Signature:** The logs state `observed errors are: [User]`.

This indicates that the XLA compiler generated an instruction that violated a
inviolable hardware constraint (e.g., an instruction attempting to access an
out-of-bounds memory address on HBM or Scratchpad memory). While labeled "User",
this is rarely caused by high-level user code.

* **File an XLA Bug:** This is likely a compiler bug, the compiler should never
emit instructions that violate hardware specs. Please file a bug report.

### Scenario 3: XLA compiler-generated Assertion Failures

**Signature:** The error message contains specific details on the
compiler-generated assertion that is failing. Look for the for the following
keywords:

* `BoundsCheck`, `scheckne`, `scheckeq`, `schecklt`, `scheckge`, `scheckbetween`

This indicates that a compiler-generated assertion in the compiled
program failed during execution. Analyze the specific
error message to determine the sub-type:

#### Scenario 3.A: Launch Group Mismatch

**Sample Error Message:**

```
Core halted unexpectedly: INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0x1d9 (from TensorCoreSequencer:1:0x309): scheckne: An unexpected leader shows up in the launch group with a different launch id than the current group leader.
```

**Cause:**
This error typically occurs in multi-host TPU environments. It indicates that
the TPU cores, which are expected to execute the same program in a synchronized
manner (as part of a "launch group"), have become out of sync. Specifically, a
TPU core joined a synchronization group with a different program identifier
than the current group leader, suggesting inconsistent programs across hosts.

* **Verify XLA Flags:** Ensure all hosts use the exact same `XLA_FLAGS`.
* **Consistent Jax Programs:** Check that all hosts are executing identical
Jax program. verify docker images, libtpu versions etc.

#### Scenario 3.B: Bounds Check Failure

**Sample Error Message:**

```
Core halted unexpectedly: INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:23:0x292 (from TensorCoreSequencer:23:0xd74a): BoundsCheck 92 [deref of %s931] for %937 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s931, /*size_in_granules=*/16384, /*vmem=*/%s935, /*dst_syncflagno=*/%s860, /*src_stride=*/512, /*dst_stride=*/128, /*steps_per_stride=*/8
```

**Cause:** The program tried to access memory outside of allocated bounds. The
error message often includes details about the memory access type (e.g.,
dma.hbm_to_vmem) and the address calculation.

*   **Debug Custom Kernels:** If using Pallas, check your index calculations.
    Use
    [`pl.debug_print`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.debug_print.html)
    or
    [`checkify`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.checkify.check.html)
    to validate tensor indices.
*   **Check Sharding:** Ensure sharding annotations are consistent with tensor
    shapes.

#### Scenario 3.C: Mosaic/Pallas Synchronization

**Sample Error Message:**

```
Core halted unexpectedly: INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:21:0xae5 (from TensorCoreSequencer:21:0x54c5): Semaphore (scratch argument 1) has a nonzero value upon exit from a Mosaic kernel. Make sure every DMA is awaited, and every semaphore signal is paired with a wait.
```

**Cause:**
This error is specific to code generated by the Mosaic compiler (used by Pallas
JAX). It indicates a synchronization issue within a custom kernel. TPUs use
semaphores to manage dependencies (e.g., ensuring a DMA is complete before
use). This error suggests a signal on a semaphore was not properly waited upon.

* **Audit Synchronization:** Ensure every `dma_start` has a corresponding
`dma_wait`.
* **Check Semaphores:** Verify that semaphore signals and waits are strictly
paired.

### Uncategorized Issues

If your error log does not match Scenario 1, 2, or 3 (i.e., no "observed
errors", no "scheck" tags, and no specific bounds/semaphore messages):

* **Action:** This is likely an internal XLA bug. Please file a bug report.
