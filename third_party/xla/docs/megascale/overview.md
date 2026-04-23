<!-- linter style off -->

# Megascale XLA

MegascaleXLA is a compiler + runtime system that powers large-scale TPU
training. It implements collective communication primitives that allow multiple
TPU slices to communicate, which allows running training jobs that span beyond
the limits of a single ICI domain.

The [debugging guide](./debugging_workflow.md) discusses how to identify and
diagnose sources of performance issues such as slowness, hangs or errors in a
multi-slice job driven by Megascale.

## Terminology

-   **Slice**

    -   A slice is a collection of chips all located inside the same TPU Pod
        connected by high-speed inter chip interconnects (ICI). Slices are
        described in terms of chips or TensorCores, depending on the TPU
        version.
    -   Multislice is a group of slices, extending TPU connectivity beyond the
        inter-chip interconnect (ICI) connections and leveraging the data-center
    network (DCN) for transmitting data beyond a slice. Data within each slice
        is still transmitted by ICI. Using this hybrid connectivity, Multislice
    enables parallelism across slices and lets you use a greater number of TPU
        cores for a single job than what a single slice can accommodate.
    -   TPUs can be used to run a job either on a single slice or multiple
        slices.

-   **RapidEye**

    -   RapidEye is a system that aims to provide global ML debugging
        infrastructure to quickly identify and root cause issues caused by bad
        hardware or software bugs. It monitors MegaScale jobs to automatically
        detect, analyze, and classify hang events. The process involves
        collecting data from all job workers, coordinating responses when a hang
    occurs, and generating a summary digest file for each event.
    -   RapidEye is enabled by default for all multislice workloads. The
        diagnosis can be found under the Pathways resource manager or MXLA
        coordinator (slice0 task0 for multi-controller JAX workload). RapidEye
        is also used for auto-removing bad TPUs and NICs based on fleetwide
        rapideye data.

-   **Megascale Collective**

    -   XLA collectives are supported via the Megascale XLA (MXLA) primitives
        which are not directly usable by the end user. At the time of writing
        this the MXLA primitives include collectives include AllGather,
        AllReduce, AllToAll, ReduceScatter and OneToOne. The reduction
        operations that are currently supported include summation, max/min and
        product.

<!-- linter style on -->
