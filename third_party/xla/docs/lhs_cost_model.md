# LHS Cost Model

---

## tldr;

This page describes the internals of the cost model used by Latency Hiding
Scheduler. If you are interested in tuning the model go straight to the
[Tuning section](#tuning).

The Latency Hiding Scheduler (LHS) is a compiler pass that schedules a HLO DAG
in a way that minimizes wall time.

Its decisions are guided by the unified cost model, which uses a mixture of
performance tables and analytical models. In particular XLA embeds performance
tables for a GEMMs and fast-interconnect collectives, and uses analytical
networking and fusion cost model for other cases. The rest of the document
describes the inner workings of these on a high level.

---

## Performance tables – ICI collectives

Performance table consist of two main components: a collector and an
interpolator.

### Collector

[The **collector**](https://github.com/openxla/xla/blob/e6a0a911eb79f540d501458f953393ede9e0048c/xla/tools/collective_perf_table_gen_main.cc#L1) is a C++ tool responsible for generating the performance
tables for collective operations. It measures the performance of individual HLO
ops (e.g., `all-gather`, `all-reduce`) across a statically defined parameter
space.

#### How It Works

The tool performs a sweep over a range of collective ops, transfer sizes, and
transfer schemes for a given cluster. It uses the existing multi-host HLO runner
infrastructure and `ExecutionProfile` data to run the generated HLO and gather
performance metrics.

#### Data Collection Parameters

Latency tables are collected for a cross-product of the following parameters:

* **Collective Type**:
    * `all-reduce`
    * `all-gather`
    * `reduce-scatter`
* **Transfer Size**:
    * Logarithmic scale from 1024B up to 2GiB (e.g., 1024B, 2048B, 4096B, ...)
* **Transfer Scheme**:
    * `rail-aligned`
    * `non-rail-aligned`

This sweep is run for intra-node clusters with **2, 4, and 8 devices**.

#### Output

The result of a collection run is a latency table in `.pbtxt` format
(approximately 116 KB per platform).

### Interpolator

[The **interpolator**](https://github.com/openxla/xla/blob/e6a0a911eb79f540d501458f953393ede9e0048c/xla/service/gpu/model/collective_interpolator.h#L40) is the compiler component that consumes the generated
performance tables to provide runtime estimates during compilation.

#### Internal Data Structure

On initialization, the Interpolator processes the performance table into a map.
This map uses a tuple of `(collective_type, transfer_scheme)` as its **key**.

The **value** associated with each key is a 2D Euclidean plane. This plane
indexes the **network throughput** (measured by the Collector) based on two
axes:
1.  Transfer size.
2.  Number of devices involved.

#### Lookup and Interpolation

When the compiler encounters a collective operation, the Interpolator performs
the following steps:

1.  It identifies the correct 2D throughput plane using the operation's `(collective_type, transfer_scheme)` as the map key.
2.  It then uses a **weighted average retrieval** (based on Euclidean distance) within that 2D plane, using the operation's `(transfer_size, num_devices)` as the query point.
3.  The result of this lookup is a single, unique **network throughput** value.

### Rationale: Throughput and Extrapolation

The system is designed to store **network throughput** rather than raw latency.
This design choice significantly simplifies extrapolating performance for
transfer sizes not explicitly present in the table.

If the latency tables capture network bandwidth saturation at a collective size
`S`, the throughput `T` at that point is considered the maximum. For any new
collective of size `S'` > `S`, the runtime can be estimated as:

<!-- mdformat off(disable mdformat for proper MathJax formatting) -->

$$\text{EstimatedTime}(S') = \frac{S'}{T_{\text{saturated}}}$$

<!-- mdformat on -->

This allows the model to estimate performance for collectives of any size, even
those larger than the 2GiB maximum measured by the Collector.

**Important:** This extrapolation model relies on the assumption that the
generated latency tables **capture true network bandwidth saturation**.
If the tables do not contain measurements at or beyond the saturation point, the
interpolator will:

* **Underestimate** the maximum throughput.
* Consequently, **overestimate** the runtime for large transfers.

In general XLA:GPU teams maintains performance tables, but in cases user decide
to provide their own, it is the responsibility of the user generating the tables
to ensure they are representative and include measurements in the
bandwidth-saturated region for the target hardware.

---

## Performance tables – GEMMs

Similar to the system for collectives, GEMM latency tables are supported by two
components: a **collector** and an **interpolator**.

### Collector

[The **collector**](https://github.com/openxla/xla/blob/e6a0a911eb79f540d501458f953393ede9e0048c/xla/tools/matmul_perf_table_gen_main.cc) is a C++ tool that computes performance tables for General
Matrix Multiplications (GEMMs). It measures the performance of matrix
multiplications at the HLO `dot` op level.

#### How It Works

The tool performs a sweep over a static space of GEMM dimensions (batch,
two non-contracting, and one contracting dimension) and data types.

* **Default Data Types:** `LHS = bf16,f32`, `RHS = bf16,f32`, `OUT = bf16,f32`.
* **Infrastructure:** Re-uses the HLO op profiler.

#### Collection Parameters

Latency tables are collected for a cross-product of the following dimensions:

* **batch:** `{1, 2, 4}`
* **m (non-contracting):** `{256, 512, ..., 4096}`
* **n (non-contracting):** `{256, 512, ..., 4096}`
* **k (contracting):** `{256, 512, ..., 4096}`

#### Output and Storage

A full sweep generates a `.pbtxt` latency table, ready to be consumed by
interpolator.

### Interpolator

[The **interpolator**](https://github.com/openxla/xla/blob/e6a0a911eb79f540d501458f953393ede9e0048c/xla/service/gpu/model/matmul_interpolator.h#L35) is the compiler component that uses the generated tables to estimate GEMM performance.

#### Rationale: FLOPS Saturation

The collected latency tables allow the interpolator to reconstruct **FLOPS** for
each entry:

<!-- mdformat off(disable mdformat for proper MathJax formatting) -->

$$\text{FLOPS} = \frac{2 \times b \times m \times n \times k}{\text{runtime}}$$

<!-- mdformat on -->

A key insight is that FLOPS **saturate** at a certain point; that is, the
hardware reaches peak FLOPS beyond a certain matrix shape. This saturation
allows the use of the same extrapolation method employed for collectives.

#### Lookup and Interpolation

The interpolator builds a **4D Euclidean space** from the table data. To provide
a performance estimate, it performs a **weighted-average interpolation** within
this 4D space. If there's no table for a certain data type, as a heuristic each
dimension is normalized to the number of bytes.

---

## Analytical Cost Model - DCN

### S-curve Collective Cost Model

The **S-curve** model is a fully analytical networking roofline model.

#### Overview

The model is designed to estimate the performance of collective operations based
on a set of fixed network properties.

#### Model Inputs

The model requires two categories of inputs:

1.  **Fixed Network Properties (User-Defined):**
    * Collective launch overhead
    * NIC speed
    * RTT (round trip time)

    By default, XLA auto-detects a platform and uses values for the most common
    architectures. These properties are configurable by the user. See
    [Tuning section](#tuning) for details.

2.  **Per-Collective Inputs:**
    * Collective type (e.g., `AllGather`, `ReduceScatter`)
    * Transfer size
    * Number of nodes involved in the communication

#### Integration

The S-curve model is integrated into `XLA:GPU` and is being used on Hopper, and
Blackwell.

---

## Analytical Cost Model - Fusions

For other kernels we rely on the [GPU performance cost model](https://github.com/openxla/xla/blob/e6a0a911eb79f540d501458f953393ede9e0048c/xla/service/gpu/model/gpu_performance_model.h) to estimate the
right runtimes. You can read more about it [here](https://github.com/openxla/xla/discussions/10065).

---

## Tuning

S-curve model can be tuned by issuing right XLA flags. Default configuration
should be good enough in majority of cases, but the model control is exposed in
other cases.

```
export NIC_SPEED_GBPS=... # NIC speed per GPU in Gigabytes
export GPUS_PER_NODE=... # Num of GPUs per cluster interconnected with fast network (e.g. NVLINK)
export XLA_FLAGS=--xla_gpu_analytical_latency_estimator_options="nic_speed_gbps=$NIC_SPEED_GBPS,gpus_per_node=$GPUS_PER_NODE"
```
