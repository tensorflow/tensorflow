# Error code: E3001

**Category:** CompileTime: SparseCore No Viable Logical Replica Count

This error occurs when the `XLA:SparseCore` compiler fails to determine a
valid logical replica count configuration that allows the workload to fit within
the SparseCore's local scratchpad memory (Tilespmem).

**Sample Error Messages:**

```
XLA:TPU compile permanent error. Compilation failure: No viable logical replica count for the embedding table with metadata: max_nz_per_row = 141352, max_unique_nz_per_row = 8, feature_width = 8, sample_count = 204800 (last tried split factor for vector splitting = 1, last tried split factor for sample dimension splitting = 1, fixed_size_allocation_bytes = 410880, row_dependent_size_allocation_bytes = 1696224, total_spmem_size_bytes = 524288) ...
```

**XLA Backends:** TPU

## Overview

This error is specific to **SparseCore** use cases, particularly for Large
Embedding Models (LEMs).

The **logical replica count** is an internal compiler parameter that determines
how input batches are partitioned to manage scratchpad allocation pressure. The
compiler attempts to split the workload into smaller chunks (replicas) so that
the intermediate buffers required for each chunk fit into the SparseCore's
limited **Scratchpad Memory**. Generally, a higher logical replica count reduces
allocation pressure by processing smaller batches of data at a time.

This error indicates that even after attempting various splitting
configurations, the compiler could not find a setup where the required buffers
fit in the Tilespmem memory. The allocation size is determined by a combination
of:

* **`sample_count`**: The number of embedding lookup IDs assigned to each
SparseCore (derived from batch size).
* **`feature_width`**: The size of the embedding dimension.
* **`max_nz_per_row`**: The maximum number of non-unique embedding lookup IDs
across all SparseCores.
* **`max_unique_nz_per_row`**: The maximum number of unique embedding lookup
IDs.

## Debugging

To resolve this error, you need to reduce the memory pressure on the SparseCore
scratchpad.

### 1. Improve Metadata Estimations

The compiler allocates memory based on `max_nz_per_row` and
`max_unique_nz_per_row`. If these values are estimated conservatively (i.e., set
much higher than the actual data requires), the compiler will reserve
unnecessary space, causing this error. Ensure these parameters accurately
reflect the actual ID distribution of your dataset.

You can consider applying **Feedback-Directed Optimization (FDO)** to determine
optimal values for these parameters.

### 2. Reduce Batch Size

The `sample_count` is directly derived from your global batch size. Reducing the
batch size decreases the amount of data each SparseCore must process per step,
thereby reducing the size of the required scratchpad buffers.
