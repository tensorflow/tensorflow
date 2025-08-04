# A deep dive into SparseCore for Large Embedding Models (LEM)

SparseCore is a specialized tiled processor engineered for high-performance
acceleration of workloads that involve irregular, sparse memory access and
computation, particularly on large datasets stored in High Bandwidth Memory
(HBM). While it excels at tasks like embedding lookups, its capabilities extend
to accelerating a variety of other dynamic and sparse workloads.
## 1. Introduction to SparseCore

Key architectural features:

* **Tiled architecture**: Comprises multiple compute tiles (each tile is a complete dataflow unit with its own local memory and processing unit) allowing for parallel processing.
* **Dynamic execution**: Natively supports data-dependent control flow and memory accesses, crucial for sparse data.
* **Vector processing**: Utilizes small-vector tasks (8-element or 16-element, depending on the hardware version) for efficient computation.
* **Centralized control**: A single SparseCore sequencer orchestrates tasks across all tiles, ensuring synchronized operations.
* **Data summarization support**: Includes specialized cross-lane operations beneficial for tasks such as sorting, filtering, and prefix sums.
* **Memory hierarchy**: Strategically leverages HBM for storing large datasets and local scratchpad memory (SPMEM) for staging frequently accessed data, significantly reducing HBM latency.

### Specifications at a glance:

| Attribute          | TPU v4  | TPU v5p  | Trillium            |
| :----------------- | :------ | :------- | :------------------ |
| **SparseCores/Chip** | 4       | 4        | 2                   |
| **Tiles/SparseCore** | 16      | 16       | 16                  |
| **SIMD Width** | 8       | 8        | 8 (F32) 16 (BF16)   |
| **HBM Capacity** | 32 GiB  | 96 GiB   | 32 GiB              |

## 2. SparseCore host preprocessing

Effective data preparation is paramount for SparseCore performance, and this is
where host preprocessing plays a vital role. It encompasses several key
functionalities:

* **Data transformation**:
    * Apply necessary transformations to raw input data.
    * Manage ID transformations, which is particularly important when dealing with feature or table stacking.
    * Convert input data into the Coordinate (COO) sparse format, detailed in the following section.
    * Partition the data for efficient distribution across the different SparseCores available on the chip.
* **Limit validation**:
    * Ensure that the characteristics of the input data (for example, number of
      IDs) conform to the predefined operational limits of the SparseCore, such
      as `max_ids_per_partition` and `max_unique_ids_per_partition`.
    * If the input data exceeds these limits, your host preprocessing layer can attempt to segment the data into smaller mini-batches that do fit within the constraints.
* **Data transfer**:
    * Efficiently copy the processed and validated data to the TPU's High Bandwidth Memory (HBM), making it ready for SparseCore execution.

### Understanding table stacking:

Table stacking is a significant optimization technique where multiple
embedding tables are logically combined to enhance embedding lookup efficiency.
This process is typically handled automatically by the underlying ML framework.

*   **Feature stacking**: This occurs when multiple distinct features share the
    same underlying embedding table. A common example is using a single
    embedding dictionary for various categorical features like zip codes from
    different contexts.
*   **Table stacking**: In this scenario, multiple distinct embedding tables are
    stacked together. Tables that share the same embedding dimension and
    optimizer configuration are often grouped.

The primary advantage of table stacking is the creation of a larger effective
batch size for the operations on these stacked tables. This reduces
computational overhead and can be effective in hiding inter-chip communication
(ICI) latencies. For optimal performance, a moderate number of stacked tables
(generally in the range of 5 to 100) is recommended.

## 3. Conversion to COO tensors

Before data can be processed by SparseCore, it's commonly converted into a
Coordinate (COO) sparse tensor format. The COO format is a way to represent
sparse matrices efficiently typically using three arrays:

*   `row_ids`: An array containing the row indices for each non-zero element. In
    the context of batch processing, this often corresponds to the batch
    dimension.
*   `col_ids`: An array containing the column indices for each non-zero element.
    For embeddings, these are often the feature or ID values.
*   `values` (optional): An array holding the actual values of the non-zero
    elements at the corresponding (`row`, `col`) coordinates. For limit
    calculations (discussed later) related to ID counts, these values (gains)
    are often not considered.

### Illustrative example:

Consider an input sparse matrix representing batches of IDs:

    
    [
        [id_A],                 // Sample 0
        [id_A, id_B, id_C],     // Sample 1
        [id_B, id_B, id_D],     // Sample 2 (note duplicate id_B)
    ]
    

After conversion to COO format (and potentially after deduplicating IDs within
the same sample):

    row_ids = [0, 1, 1, 1, 2, 2]
    col_ids = [id_A, id_A, id_B, id_C, id_B, id_D]

This conversion is fundamental to how SparseCore processes and distributes work.
The `col_ids` in particular, are crucial for determining which specific
SparseCore partition an ID belongs to, enabling efficient sharding and lookup.

## 4. SparsecoreConfig: The high-level API

### Framework specific embedding APIs:

* **JAX**: https://github.com/jax-ml/jax-tpu-embedding
* **TensorFlow**: https://www.tensorflow.org/recommenders/api_docs/python/tfrs/layers/embedding/TPUEmbedding
* **Keras**: https://keras.io/keras_rs/api/embedding_layers/distributed_embedding

The `SparsecoreConfig`, or equivalent mechanisms such as XLA flags, serves as a
high-level interface for controlling a wide array of SparseCore behaviors. A
thorough understanding of these parameters is vital for effective performance
tuning and ensuring the correct operation of your models.

*   **`disable_table_stacking: bool = False`**
    *   **Explanation**: This flag controls whether automatic table stacking is
        prevent the framework from stacking tables, potentially leading to
        reduced performance due to increased overheads and a diminished ability
        to hide Inter-Chip Interconnect (ICI) latency.
    *   **Default**: `False` (implying table stacking is generally enabled by
        default where the framework supports it).
*   **`max_ids_per_chip_per_sample: int = 64`**
    *   **Explanation**: This parameter establishes a global upper limit on the
        total number of embedding IDs that a single chip can process from one
        sample in the input batch, aggregated across all tables. It is a
        mechanism for managing resources at the chip level, before more granular
        per-table or per-partition limits are taken into account. Fine-tuning
        this value typically depends on specific model characteristics and
        overall system capacity.
    *   **Default**: `64`.
*   **`max_ids_per_table: Optional[Dict[str, int]] = None`**
    *   **Explanation**: This parameter specifies the maximum number of
        embedding IDs (which can include duplicates) that can be processed for
        each logical table, considering all its partitions across all
        SparseCores. This is a broader limit than `max_ids_per_partition`. If a
        table `T` is divided into `P` partitions, this limit applies to the sum
        of IDs directed to all P partitions. It's often related to
        `max_ids_per_partition_per_sample` and the overall batch size.
    *   **Setting**: Typically configured using a limits file (for example,
        using the `xla_sparse_core_max_ids_file` flag), where
        `max_ids_per_partition` is defined. This table-level concept is a method
        to set those partition-level limits (`max_ids` and `max_uniques`).
    *   **Default**: `None` (the value may be inferred from per-partition limits
        or other configurations if not explicitly provided).
* **`max_unique_ids_per_table: Optional[Dict[str, int]] = None`**
    * **Explanation**: Analogous to `max_ids_per_table`, but this parameter
      specifies the maximum number of unique IDs for each logical table. This is
      a critical setting for appropriately sizing on-device buffers used in
      unique ID processing and subsequent vector operations.
    * **Setting**: Also commonly defined in a limits file or derived from `max_unique_ids_per_partition_per_sample`.
    * **Default**: `None`.
* **`allow_id_dropping: bool = False`**
    * **Explanation**: This boolean flag controls ID dropping when the number of
      IDs encountered in the input data (observed limits) surpasses the limits
      set during compilation (for example, `max_ids_per_partition`).
        * **If `True`**: IDs that would cause the limits to be exceeded are
          silently dropped. Typically, IDs within a partition are processed in a
          sorted order, and any ID that would push the running count over the
          limit for its designated mini-batch is discarded. This allows the
          program to continue execution but may have an adverse impact on model
          accuracy.
        * **If `False`**: An error is triggered, and the process will likely
          terminate if observed limits go beyond compiled limits. This approach
          ensures all data is processed but requires limits to be configured
          more conservatively.
    * **Default**: `False` (causing an error on overflow rather than silent data dropping).
* **`initialize_tables_on_host: bool = True`**
    * **Explanation**: This flag determines whether embedding tables are
      initialized on the host CPU before being subsequently transferred to the
      TPU's High Bandwidth Memory (HBM). Standard practice is for tables to be
      initialized on the host. Setting this to `True` follows this convention.
      If it were set to `False`, it would imply an on-device initialization
      mechanism, which could have different performance implications or specific
      initialization prerequisites.

* **`enable_fast_table_initialization: bool = False`**
    * **Explanation**: Initializes the tables directly on the TPU. This can help reduce model startup times.

## 5. Pipelining for performance

Pipelining is a performance optimization technique that enables the
simultaneous execution of operations on the TensorCore (TC) and the SparseCore
(SC). By overlapping these computations, overall throughput can be
significantly improved.

*   **Mechanism**: In a standard training step that involves sparse embedding
    lookups (handled by SC) and dense layer computations (handled by TC),
    pipelining allows the SC to work on its part of step `i` (for example,
    forward or backward pass) while the TC is concurrently processing a
    different part of the same step `i`, or even parts of adjacent steps like
    `i-1` or `i+1`.
*   **Impact on gradients**: The SparseCore might operate on "stale" gradients.
    For instance, the gradients calculated during the backpropagation phase of
    step `i` might not be fully updated and visible to the SC until step `i+2`.
*   **Performance vs. numerics trade-off**: This overlapping execution can lead
    to substantial speedups, potentially up to a 2x improvement in device step
    time.
    However, the subtle changes in numerics (embedding_weights) resulting from
    the use of stale gradients might influence model convergence behavior or the
    final achieved accuracy. The acceptability of this trade-off is highly
    model-dependent and often requires empirical validation.
*   **Control flag**: Pipelining can be controlled by
    `tf_xla_disable_full_embedding_pipelining`. Setting this flag to `true`
    disables full pipelining (overlapping TensorCore and SparseCore
    computation), whereas setting it to `false` (or if the flag's semantics
    imply enabling when false) activates it.

### Conceptual pipelining flow:

* **Without pipelining (simplified sequential flow)**:

    `Loop: SC/F_i -> TC/F_i -> TC/B_i -> SC/B_i`
* **With pipelining (simplified overlapped flow)**:
    ```
    Time ->
    Step i:   SC/F_i | TC/F_i | TC/B_i | SC/B_i
    Step i+1:          SC/F_i+1| TC/F_i+1| TC/B_i+1| SC/B_i+1
    ```
    **Note**: The actual pipelining stages implemented in the hardware and
    compiler can be more intricate, often involving pre-loops, main execution
    loops, and post-loops to manage data dependencies and ensure correctness.

## 6. The role of XLA

XLA (Accelerated Linear Algebra) is the domain-specific compiler that translates
high-level computational graphs, typically from frameworks like TensorFlow, into
highly optimized machine code tailored for TPUs. This includes generating the
instructions for operations destined for the SparseCore.

### Key functions in the SparseCore context:

*   **Compilation of sparse operations**: XLA is responsible for compiling
    embedding lookup operations (such as the `SparseDenseMatmulOp`) and other
    sparse computations into low-level, executable SparseCore programs.
*   **Integration of limits**: It utilizes the configured operational limits
    (for example, `max_ids_per_partition`, `max_unique_ids_per_partition`, often
    provided via a limits file specified by flags like
    `xla_sparse_core_max_ids_file`) to statically determine the sizes of and
    allocate on-device memory buffers, particularly within the SPMEM.
*   **Targeted optimizations**: XLA performs a suite of optimizations
    specifically designed for the SparseCore architecture. These can include
    instruction scheduling, memory layout transformations, and fusion of
    operations to maximize efficiency.
*   **Control using flags**: Many aspects of SparseCore behavior, tuning
    parameters, and optimization strategies are exposed and controlled through
    XLA flags (for example, `xla_sparse_core_estimate_max_ids` for limit
    estimation, or `xla_sc_detect_nan` for debugging).

### Open source status:
Currently Sparsecore Implementation is internal and served using `libtpu.so`.

### Error reporting and diagnostics:
Compilation failures related to SparseCore configurations or resource
constraints often manifest as `XLA:TPU` compile-time errors. These error
messages can provide valuable insights into issues such as limits being set too
high for the available SPMEM, or the use of unsupported configurations.

## 7. How limits translate to tables on SparseCore

On SparseCore, "limits" are fundamental configuration parameters that primarily
refer to two per-partition settings for each table that is sharded (distributed)
across the available SparseCores:

*   **`max_ids_per_partition`**: This defines the maximum number of total IDs
    (including duplicates) that any single SparseCore is expected to send to, or
    process for, a specific partition of a given table within a single
    computational step.
*   **`max_unique_ids_per_partition`**: This defines the maximum number of
    unique IDs that any single SparseCore is expected to send to, or process
    for, a

### Translation to physical table layout and processing:

*   **Table sharding strategy**: Embedding tables are typically "mod-sharded"
    across all SparseCores in the system. This means each SparseCore becomes
    responsible for a distinct subset of the vocabulary (rows) of each table. An
    ID `j` would generally be assigned to `SparseCore_k` based on a formula like
    `k = j % num_total_sparse_cores`.
    <!-- disableFinding(LINE_OVER_80) -->
* **Definition of a "partition"**: In this context, a "partition" refers to the specific segment of an embedding table for which a single SparseCore handles lookups.
* **SPMEM buffer allocation**: These limits are used by the XLA compiler to
  statically size and allocate buffers within the on-device scratchpad memory
  (SPMEM). Buffers are dimensioned such that all necessary data related to the
  IDs for a given partition (up to the specified `max_ids` and
  `max_unique_ids` limits) can be loaded into SPMEM for processing. This is
  particularly crucial for non-elementwise computations, such as reducing
  duplicate IDs within a partition (for example, , when creating a Compressed
  Sparse Row (CSR) representation), where the entire relevant dataset for that
  partition's IDs needs to be readily available in fast memory.
* **Compiled limits versus observed limits**:

    * **Observed limits**: These are the actual number of IDs encountered for each partition during runtime, based on the input data being processed.
    * If observed limits exceed compiled limits, it can lead to ID dropping (if `allow_id_dropping` is enabled) or errors.
* **Calculating limits**: The process of determining appropriate limits involves a careful analysis of the input data distribution. For any given table (let's call it `T1`, which might itself be part of a larger stacked table `T`):
    1.  The input batch (for example, a 2D `SparseTensor` of shape `[BatchSize, MaxSequenceLength]`) is initially split across the available SparseCores. For instance, if a TensorCore is paired with 2 SparseCores, each SparseCore might receive a sub-batch of shape `[BatchSize/2, MaxSequenceLength]`.
    2.  This sub-batch is then converted into the COO format, yielding `row_ids` and `col_ids`.
    3.  Duplicate IDs within the same sample (i.e., entries with the same `row_id` and `col_id`) are removed.
    4.  For each remaining unique `col_id` (within a sample), the target SparseCore responsible for this ID is determined using the mod-sharding rule: `target_sc_id = col_id % num_total_sparse_cores`.
    5.  A count is maintained of the total number of IDs (`ids_per_sparse_core[target_sc_id]++`) and the number of unique IDs (`unique_ids_per_sparse_core[target_sc_id]++`, after ensuring uniqueness for that specific `target_sc_id`) that are destined for each `target_sc_id`.
    6.  The `max_ids_per_partition` for table `T1` is then set to `max(ids_per_sparse_core_array)`.
    7.  Similarly, the `max_unique_ids_per_partition` for table `T1` is set to `max(unique_ids_per_sparse_core_array)`.
    8. If table `T1` is a component of a stacked table, additional transformations like rotations or shifts might be applied to the ID distributions before summing statistics from all constituent tables. This helps in balancing the load across chips.

Setting these limits correctly is a balancing act: lower limits can potentially lead to higher performance (as less data needs to be processed per step and SPMEM pressure is reduced), but if set too low, they can result in excessive mini-batching or undesirable ID dropping.

## 8. How each SparseCore communicates

SparseCore communication, particularly in the context of processing a list of IDs for embedding lookups, relies on several coordinated mechanisms:

* **Mod sharding and implicit routing**:
    * Embedding tables are mod-sharded across all SparseCores in the system.
    * When the host provides a batch of input data (which is subsequently preprocessed into COO format, including `col_ids`), the `col_id` value is used to determine which SparseCore is responsible for that specific ID: `target_sc_id = col_id % num_total_sparse_cores`.
    * Each SparseCore effectively receives and processes only the subset of IDs that map to its assigned vocabulary partitions. The host preprocessing stage is crucial for preparing the data in such a way that each SparseCore can readily identify and operate on its relevant IDs.
* **Data distribution by host**:
    * The host preprocessing logic partitions the overall input batch and distributes the pertinent portions of the `row_ids` and `col_ids` (along with any associated features or weights if applicable) either to memory (HBM) directly accessible by each SparseCore or to a shared HBM from which the SparseCores will fetch their required data.
* **Intra-SparseCore processing**:
    * Once a SparseCore has received its designated set of IDs for a given table partition, it performs operations such as deduplication of these IDs and gathering the corresponding embedding vectors. These are primarily local computations executed within the SparseCore's own tiles and utilizing its local SPMEM.
* **Inter-SparseCore communication (All-to-All)**:
    * After the initial processing phase (like embedding lookups), an "all-to-all" communication pattern may be used to combine or redistribute results across SparseCores (for example, before feeding activations into a TensorCore layer that expects input corresponding to all original sample positions). This is vital for reconstructing the complete set of activations if the original input batch was distributed for parallel processing.
* **Communication with TensorCores**:
    * SparseCores communicate with TensorCores to send embedding activations (during the forward pass) and to receive gradients (during the backward pass). This interaction is orchestrated by the XLA-compiled program and frequently involves HBM as an intermediary buffer. The pipelining strategy (discussed earlier) heavily influences the timing and synchronization of this SC-TC communication.

In essence, the initial "distribution" of IDs to the appropriate SparseCores is largely handled by the sharding scheme and the host preprocessing steps. Subsequent communication involves SparseCores operating on their local data, potentially followed by collective communication operations like all-to-all if data needs to be globally exchanged or reordered across SparseCores before further processing by the TensorCores.

## 9. SparseCore memory management

Each SparseCore efficiently manages several distinct types of memory to perform its computations:

* **Scratchpad memory (SPMEM)**:
    * **Nature**: A relatively small, but very fast, local SRAM that is available exclusively to each SparseCore. It is important to note that SPMEM is not a cache; its usage is explicitly managed and orchestrated by the XLA compiler.
    * **Purpose**: SPMEM is used to "opportunistically stage data." This includes inputs, outputs, and intermediate results that are required for ongoing SC computations. Staging data in SPMEM significantly reduces the high latency typically associated with accessing HBM.
    * **Sizing**: As discussed in the section on "Limits," SPMEM buffers are statically sized at compile time. This sizing is based on parameters like `max_ids_per_partition` and `max_unique_ids_per_partition`. This static allocation ensures that for any given operation on a table partition (such as CSR reduction), all the necessary data for that partition's IDs (up to the defined limits) can fit into SPMEM.
    * **Compiler optimizations**: The XLA compiler incorporates sophisticated optimizations to determine precisely how much data, and which specific data elements, need to be staged in SPMEM to effectively hide HBM latency and maximize performance.
    * **Dynamic allocation constraint**: The SparseCore compiler does not currently support dynamic scratchpad allocation. This highlights the critical importance of static sizing through the careful configuration of limits.
* **High bandwidth memory (HBM)**:
    * **Nature**: A large, shared memory resource accessible by all SparseCores, TensorCores, and the host system. Primary embedding tables are stored in HBM.
    * **Stack usage**: SparseCore operations often require temporary storage in HBM for intermediate results that either do not fit into the limited SPMEM or need to be passed between larger stages of the processing pipeline. HBM stack usage during both forward and backward passes can be estimated as follows -
        * Forward Pass HBM Stack (single table) ≈ (2 \* `feature_width` + 1) \* `max_unique_nz_per_row` \* `logical_replica_count` \* 4 bytes
        * Backward Pass HBM Stack (single table) ≈ 3 \* `feature_width` \* `max_unique_nz_per_row` \* `logical_replica_count` \* 4 bytes
    * **Heap usage**: HBM also accommodates the heap, which is managed by the host. The heap stores data such as dense layer weights, constants used by the model, and prefetched input data. Heap usage tends to increase with the number of steps for which the host prefetches data (controlled by the `maximum_parallel_iterations` flag). While more prefetching can improve performance by overlapping host-to-device transfers with device computation, it also consumes more HBM.
    * **Serialization for HBM optimization**: The flag `xla_sc_num_serialized_tables_to_optimize_hbm` provides a mechanism to control how many tables' data is kept "live" in HBM stack memory at any given time. Increasing this number effectively serializes the processing for more tables, which can reduce peak HBM stack usage but may come at the cost of performance due to reduced parallelism.
* **Vector memory (VMEM)**:
    * VMEM is a local scratchpad memory used exclusively by the TC (TensorCore). While VMEM is not directly managed by the SparseCore, it is an integral part of the memory ecosystem with which the SC interacts, primarily through the TensorCore.

### Overall memory management strategy:
The core memory management strategy for SparseCore revolves around using the small, fast SPMEM for the "hot" data that is actively being processed by a SparseCore tile, thereby minimizing accesses to the slower HBM.
The configured limits are the primary mechanism for ensuring that SPMEM does not overflow.
HBM is utilized for storing large embedding tables and temporary data that either exceeds SPMEM capacity or needs to be shared across different processing units or pipeline stages.
The XLA compiler is responsible for orchestrating all data movement and buffer allocation based on these architectural principles and the user-configured limits.

## 10. Performance & memory bottlenecks

Achieving optimal performance with SparseCore necessitates a clear understanding of potential bottlenecks and how to address them. These can arise on the host, within the SparseCore itself, or in its interaction with the TensorCores.

### Common performance bottlenecks:

* **Host bottleneck**:
    * **Issue**: The host CPU may fail to preprocess data and feed it to the TPU rapidly enough, leading to underutilization of SparseCores and TensorCores. This is a frequent performance limiter.
    * **Mitigation**: Monitor host CPU utilization and input pipeline metrics. Optimize host-side data loading and preprocessing routines (refer to COO conversion tips). Adjust the `maximum_parallel_iterations` flag to fine-tune data prefetching.
* **Suboptimal TC/SC synchronization (lack of pipelining)**:
    * **Issue**: If pipelining between the TensorCore and SparseCore is disabled or not operating efficiently, one unit may spend significant time waiting for the other, thereby reducing overall system throughput.
    * **Mitigation**: Ensure pipelining is enabled (for example, `tf_xla_disable_full_embedding_pipelining = false` or its equivalent).
* **Limit-induced bottlenecks**:
    * **Issue**:
        * **Limits too low**: Can trigger excessive mini-batching (splitting input batches into numerous smaller sub-batches to meet the tight limits). While this maintains correctness, each mini-batch introduces some processing overhead, potentially slowing down the overall execution. If `allow_id_dropping` is true, overly low limits can also lead to ID dropping, which impacts model accuracy.
        * **Limits too high (but still fit)**: While very high limits might prevent mini-batching, they could increase SPMEM pressure unnecessarily if the actual data characteristics rarely approach these peak values. They could also lead to larger HBM stack usage than strictly needed.
        * **Compilation failures**: If the configured limits require more SPMEM or HBM stack than the available physical memory, the compilation will fail.
    * **Mitigation**: Ensure limits are set correctly.
* **Data distribution skew**:
    * **Issue**: If certain SparseCore partitions consistently receive a disproportionately larger number of IDs compared to others (indicating a poor ID distribution), those overloaded SparseCores will become performance bottlenecks.
    * **Mitigation**: ID shuffling during the mini-batching process can help alleviate this for stacked tables, especially those with "hot" user tables. Analyze ID distributions carefully to set appropriate and balanced per-table limits.
* **Table stacking issues**:
    * **Issue**:
        * **Too few tables stacked**: May not be sufficient to effectively hide ICI latency or adequately reduce processing overheads.
        * **Too many tables stacked**: Could result in the creation of very large logical tables that become unwieldy to manage or might exceed available resource limits.
    * **Mitigation**:
        * Ensure optimal number of tables for stacking. A general guideline suggests a "sweet spot" of 5-100 tables for stacking.
* **Inefficient numerics/quantization**:
    * **Issue**: Using full FP32 precision when lower precision formats like BF16 or quantized integers would suffice (and offer faster computation) can be a performance bottleneck.
    * **Mitigation**: Explore lower precision options. However, be aware that quantization itself has some overhead and might require careful tuning of quantization parameters to maintain model accuracy.
* **HBM bandwidth saturation**:
    * **Issue**: Excessive data movement to and from HBM, potentially caused by very small feature widths (leading to high padding overhead), inefficient memory access patterns, or an extremely large number of lookups, can saturate the available HBM bandwidth.
    * **Mitigation**: Scaling the number of TPUs can help with HBM bandwidth saturation.

### Common memory bottlenecks:

* **SPMEM overflow (compilation failure)**:
    * **Issue**: If `max_ids_per_partition` and `max_unique_ids_per_partition` are set too high, the XLA compiler may be unable to allocate sufficient SPMEM, resulting in compilation errors like: `"Fixed size allocations (...) do not fit in TileSpmem (...)"`. Additionally, if the term `(sample_count * feature_width) / kNumTiles` (where `kNumTiles` is the number of tiles per SC) is too large for staging gather operands within the tile SPMEM, errors such as `"Gather operand too large..."` can occur.
    * **Mitigation**: Reduce the batch size or increase the number of chips used for processing.
* **HBM stack overflow (runtime or compilation)**:
    * **Issue**: If the combination of `feature_width`, `max_unique_nz_per_row`, and `logical_replica_count` leads to HBM stack memory requirements that exceed the available HBM, this can cause Out-Of-Memory (OOM) errors either at runtime or during compilation.
    * **Mitigation**: Tune the `xla_sc_num_serialized_tables_to_optimize_hbm` flag to reduce HBM stack usage by serializing the processing of tables (this usually comes at a performance cost).
* **HBM heap exhaustion**:
    * **Issue**: Primarily caused by very large dense layer weights, numerous constants stored in memory, or overly aggressive input prefetching (high `maximum_parallel_iterations`).
    * **Mitigation**: Monitor heap usage using tools like XProf Memory Viewer.
* **Padding overhead**:
    * **Issue**: Embedding tables are padded to be 32B-aligned (equivalent to 8 floats) in the feature dimension. Consequently, small feature widths (for example, 1 float) incur significant padding overhead (for example, 7/8 of the allocated buffer space is padding), leading to wasted HBM. The vocabulary dimension of tables is also padded to be a multiple of the number of SparseCores in the system; however, this impact is usually negligible for tables with a sufficiently high vocabulary size.

### General factors impacting performance and memory:

* **Topology**: The number of chips available and their interconnection architecture.
* **Batch sized**: Directly affects the `sample_count` per SparseCore, which in turn influences memory consumption and compute load.
* **Data formatting**: Ensuring an efficient on-device data layout is crucial for optimal performance.

## 11. Analyzing a SparseCore profile

Analyzing a performance profile is a key step in identifying bottlenecks and uncovering opportunities for optimization within your SparseCore workloads.

1.  **Obtain a trace**:
    * Utilize profiling tools, such as [XProf](https://github.com/openxla/xprof), to capture a detailed execution trace while your model is training or running inference. This trace will provide a timeline of operations occurring on the host, the TensorCores, and the SparseCores.
2.  **Examine the Trace Viewer (for example, in XProf or TensorBoard)**:
    * **Host activity**: Scrutinize the host's activity. Are there significant gaps in TPU activity? Such gaps might indicate that the host is a bottleneck, failing to feed data quickly enough. Analyze the performance of your input pipeline.
    * **TensorCore (TC) and SparseCore (SC) activity**:
        * Look at the execution timelines for both TC and SC. Are they operating in parallel, indicating effective pipelining? Or are there extended periods where one unit is idle, waiting for the other?
        * Identify the operations that consume the most time (longest-running operations) on both the SC and TC.
        * Visual trace outputs (often showing colored blocks representing different operations over time, like the `TPU:0 SparseCore 1 (pid 1005)`) are invaluable for visually identifying dominant operations and idle periods.
    * **Step time analysis**: Observe the overall step time and understand how it is distributed or broken down between host processing, SC computation, and TC computation.
3.  **Memory analysis (XProf Memory Viewer)**:
    * **Heap usage**: Use tools like XProf's "Memory Viewer" tab to inspect HBM heap usage. This can help determine if large model weights, constants, or overly aggressive input prefetching are consuming an excessive amount of HBM. Enabling flags like `--vmodule=best_fit_allocator=1` might provide logs of peak heap usage.
    * **Stack usage (indirect)**: While direct HBM stack profiling can be complex, if you encounter Out-Of-Memory errors and heap usage appears reasonable, HBM stack exhaustion (often due to overly large limits or feature widths) is a strong suspect. The formulas provided for HBM stack usage can help in estimating this.
4.  **Look for specific patterns**:
    * **Mini-batching**: If limits are frequently being exceeded, you might observe evidence of mini-batching in the trace (for example, a higher number of smaller SC operations than expected for the global batch size). This can often be inferred from logs or by observing the invocation counts of certain operations.
    * **ID dropping**: If ID dropping is enabled and occurring, system logs might provide indications of this. This would also be a clear sign that the configured limits are too restrictive for the input data.
    * **Compilation times**: Extended recompilation times, particularly if Feedback Directed Optimization (FDO) is enabled and frequently adjusting limits, can add significant overhead to the overall training time.
5.  **Correlate with flags and configuration**:
    * Relate the observed behavior in the profile back to your SparseCore configurations (settings in limits files, XLA flags). For example, if `xla_sc_num_serialized_tables_to_optimize_hbm` is set to a high value, you might expect slower SC performance but lower HBM stack consumption.
6.  **Iterative process**:
    * Profiling is often an iterative refinement process. Make a specific change (adjust a limit, enable or disable a feature), capture a new profile, and then compare it against the previous profile to see the impact of your modification.

## 12. General debugging flags

Several flags can be enabled to assist in debugging issues related to SparseCore execution. It's important to note that enabling these checks often incurs a performance penalty and, therefore, they should typically be disabled for production runs.

* **ID checks (out-of-range)**:
    * **Flag**: `xla_sparse_core_enable_id_bound_check = true`
    * **Purpose**: Enables checks on the host system to detect if any embedding IDs in the input data fall outside the valid vocabulary range defined for a given embedding table. This helps catch issues related to incorrect or corrupted input data.
* **NaN checker**:
    * **Flag**: `xla_sc_detect_nan = true`
    * **Purpose**: Enables the detection of NaN (Not a Number) values within floating-point data processed on the SparseCore. If a NaN is detected in the inputs or outputs of various compiler passes, this flag will cause an error to be raised. Such errors typically provide information about where the NaN was encountered.
* **Bounds checker (memory access)**:
    * **Flag**: `xla_sc_assert_level=bounds`
    * **Purpose**: This flag enables an ASAN (AddressSanitizer)-style tool that rewrites memory-accessing instructions (such as VMEM loads/stores and DMA operations) to include dynamic checks. These checks verify if the memory access is within the allocated bounds of the target memory region.
    * **Behavior**: If an out-of-bounds memory access is detected, the execution will fail.
    * **Caution**: It's possible for this checker to produce false positives, for example, due to complex strided access patterns that are not fully comprehended by the checker. This transformation is applied at a late stage in the backend compilation process.
* **Buffer checker (memory corruption)**:
    * **Flags**:
        * `xla_tpu_buffer_contents_sanitizer_config='cores_to_sanitize: [TC, SC_SCS, SC_TILE], sanitizer_mode: LOCAL_ONLY'`
        * `xla_tpu_verify_launch_id_across_cores=true`
    * **Purpose**: These flags help ensure that memory buffers are not being inadvertently corrupted or overwritten by unrelated operations. The Buffer Sanitizer checks the contents of buffers to verify that they are not changing unexpectedly.

## 13. Quantization support

SparseCore's `SparseDenseMatmulOp` is designed to support operations on embedding tables using both 32-bit floating-point (FP32) and integer data types. While model training is typically performed using FP32 precision for embedding tables, post-training quantization (PTQ) can be applied. PTQ allows the use of lower-precision datatypes (like 8-bit integers) for inference, which can potentially lead to improved performance and a reduced memory footprint.

### Simulated Quantization:

The `SparseDenseMatmulOp` can be configured to perform "simulated quantization." In this operational mode, embedding vectors are first quantized to a lower precision and then dequantized back to a higher precision (for example, FP32) before they are used in subsequent computations. This technique allows models to be trained while accounting for the effects of quantization noise. Training with simulated quantization can improve the accuracy of the final model when it is fully quantized for inference.

### Configuration attributes for `SparseDenseMatmulOp` (for quantization):

* **`quantization_config_num_buckets = 256`**
    * This attribute specifies the number of discrete buckets or levels into which a 32-bit floating-point number will be quantized. For example, when quantizing to 8-bit integers, one would typically specify 2^8 =256 buckets.
* **`quantization_config_low = -X.X`**
    * This attribute defines the minimum floating-point value in the quantization range. Any input values below this specified minimum will be clipped to this minimum value during quantization.
* **`quantization_config_high = Y.Y`**
    * This attribute defines the maximum floating-point value in the quantization range. Any input values above this specified maximum will be clipped to this maximum value during quantization.

### Numerics and pipelining interaction:

The numerical behavior of the model can change depending on whether pipelining between the TensorCore and SparseCore is enabled. If pipelining is active, gradients processed by the SparseCore might be "stale" (from a previous iteration). This can interact with the quantization process and potentially affect model training dynamics or final accuracy.

## 14. Upcoming features and recent improvements

The SparseCore ecosystem is subject to continuous development and enhancement.

### Roadmap:

* **Sample-dimension mini-batching**:
    * This is planned as a complementary feature to the existing vocabulary-dimension mini-batching capabilities.
    * It would allow for further partitioning of embedding inputs along the sample dimension. This would be achieved by introducing on-device loops that can filter and process lookups from a subset of samples at a time. Such a feature could be beneficial for managing very large per-sample ID counts or for improving load balancing across processing units.
* **Improved support for embedding with less than 8 integers per row (small feature widths)**:
    * The current design often uses significant padding for embedding feature widths that are less than 8 floats (which corresponds to 32 bytes). This padding can lead to wasted HBM and potentially underutilized compute resources. Future improvements aim to mitigate this inefficiency for tables with small feature dimensions.

### Recent improvements:

* **Staging of gather operands in HBM**:
    * This optimization helps to reduce pressure on the shared scratchpad memory (SPMEM) by allowing some gather operation inputs or outputs to be staged in the larger HBM.
* **Reduction in stack memory usage**:
    * Enhancements have been implemented to reduce HBM stack memory consumption during SparseCore operations, ideally without negatively impacting overall performance or throughput.

These enhancements are focused on improving SparseCore's performance, memory efficiency, and operational flexibility for an even wider range of sparse workloads.