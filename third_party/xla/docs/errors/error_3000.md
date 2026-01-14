# Error code: E3000

**Category:** Compile Time: SparseCore Allocation Failure

This error occurs when the `XLA:SparseCore` compiler is unable to allocate a
contiguous block of memory in the specified memory space required by the current
SparseCore program.

**Sample Error Messages:**

```
INTERNAL:Failed to run pass pipeline. Hlo-Op: result.1:279:1: error: 'memref.alloca' op current allocation offset upper bound (140704 words) exceeds the legitimate user allocatable offset upper bound (131071 words) in memory space 201 when allocating 23440 words. result.1:279:1: note: see current operation: %232 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<23440xf32, 201>
```

**XLA Backends:** TPU

## Overview

The SparseCore (SC) is a specialized processor for sparse workloads. It relies
on specific memory hierarchies to manage data movement efficiently. The XLA
compiler attempts to statically size and allocate buffers based on hardware
limits and user-defined shapes. This error indicates an **Out of Memory (OOM)**
condition during this allocation phase.

The error message typically specifies a memory space ID. Below are the common
memory spaces and their integer encodings:

| Memory Space ID | Name | Description |
| --- | --- | --- |
| **0** | **Smem** | Local Scalar Memory. Used for scalar registers and control flow. |
| **201** | **TileSpmem** | Tile-specific Scratchpad Memory. Fast, local SRAM available to a specific SC tile. |
| **202** | **Spmem** | Shared Scratchpad Memory. Used to opportunistically stage data (inputs, outputs, intermediates) to hide HBM latency. |
| **203** | **HBM** | High Bandwidth Memory. Large, shared memory used for embedding tables, heaps, and stacks. |
| **204** | **Sync Flags** | Synchronization primitives used for coordination. |

For a deep dive into SC and its memory hierarchy, refer to the
[SparseCore documentation](https://openxla.org/xla/sparsecore).

## Debugging

The resolution depends on which memory space failed the allocation.

### Scenario 1. HBM Allocation Failures

Memory Space ID: 203

This error occurs if a single temporary allocation requested by the SparseCore
program is too large to fit in the available HBM. In standard embedding
workloads and SC offloaded collectives extremely large per-core partitions
or incorrect sharding specifications can force the compiler to request massive
buffers.

**Recommended Actions:**

* **Check Sharding:** Ensure your embedding tables and SC input/output tensors
are partitioned/sharded correctly. If a single core is responsible for too
much data, the allocation might fail.
* **Adjust Limits:** Review `max_ids_per_partition` and
`max_unique_ids_per_partition`. If these are set unnecessarily high, the
compiler reserves more memory than needed. Refer to
[How limits translate to tables](https://openxla.org/xla/sparsecore#7_how_limits_translate_to_tables_on_sparsecore).

### Scenario 2. Internal Memory Failures

Memory Space IDs: 0, 201, 202, 204

Allocation failures in **Smem**, **TileSpmem**, **Spmem**, or **Sync Flags**
typically occur due to compiler bugs or limitations in the allocation strategy,
where the compiler fails to account for all memory
requirements.

**Recommended Actions:**

1. **Isolate the Failing XLA Operation:** To identify the specific SC HLO or
Mosaic kernel causing the failure, generate the intermediate compiler
representations:
  * **Dump SparseCore MLIR:** Set the flag
  `--xla_sc_dump_mlir_to=/path/to/dump`. This generates the MLIR of the
  SparseCore program, allowing you to see which allocation size matches the
  error message.
  * **Dump Mosaic LLO:** For custom kernels, use
  `--xla_mosaic_dump_to=/path/to/dump` to inspect all Low Level Optimizer
  (LLO) programs emitted by Mosaic.
2. **Reduce Scratch Sizes (Pallas Users):** If the failure occurs within a
Mosaic kernel, review your `scratch_shapes` configuration. Ensure that your
`pltpu.SMEM` requests fit within the hardware specifications for your specific
TPU generation.
3. **Disable Collective Offload:** If the error arises from a SC offloaded
collective operations, try disabling the SC offloading features:
  * `--xla_tpu_enable_sparse_core_collective_offload_all_gather=false`
  * `--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false`
4. **File a Bug:**
If the above steps do not resolve the issue, it is likely a compiler bug.
Please file a bug report.
