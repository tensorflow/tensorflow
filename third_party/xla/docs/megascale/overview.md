<!-- linter style off -->

# Megascale XLA

MegascaleXLA is a compiler + runtime system that powers large-scale TPU
training. It implements collective communication primitives that allow multiple
TPU slices to communicate, which allows running training jobs that span beyond
the limits of a single ICI domain.

To learn more about Megascale, check out the terminologies section below. This
[guide](./debugging_workflow.md) discusses how to identify and diagnose sources
of performance issues such as slowness, hangs or errors in a multi-slice job
driven by Megascale.

## Prerequisite

1.  Use JAX 0.6 or up, and enable JAX distributed service. This version of JAX
    contains additional logging that can help identify which workers are
    experiencing issues.
2.  Generate an HLO dump using the --xla_dump_to flag when initializing your
    workload. This is discussed in the [XLA
    documentation](https://openxla.org/xla/hlo_dumps).
3.  Set --vmodule=real_program_continuator=1 to enable verbose logging for the
    TPU program execution status.

## Terminology

There are several terms that are used in the context of Megascale, XLA, and
other related technologies. Below is a partial list of these terms and their
definitions.

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

-   **XProf**

    -   XProf is a Trace Viewer tool that is used to visualize the timeline of
        events that occurred during the profiling session. It displays the
        durations of operations executed by your model on different parts of the
        system, such as the host (CPU) and accelerators (GPUs or TPUs). This
        enables you to understand how your model utilizes hardware resources,
        identify performance bottlenecks, and optimize your model for faster
        execution.
    -   There are several ways to capture an XProf session. See the [XProf
        documentation](https://openxla.org/xprof/trace_viewer) for more details.

-   **HLO Dump**

    -   An HLO dumpis a textual representation of the HLO modules at different
        stages of the computation. It is useful for debugging, and you often
        need to include it in bug reports. This is typically a human-readable
        text file that lists the HLO instructions and their properties.
    -   You can use XLA flags to specify and get dumps. In most cases, you can
        set it with an environment variable. JAX also offers a programmatic way
        to print the HLO dump. Please follow [these
        steps](https://openxla.org/xla/hlo_dumps) to dump HLO to the local
        filesystem on the TPU worker.

-   **Collectives**

    -   Collectives are data synchronization/movement building blocks to
        facilitate communication across TPU or host devices. Tasks within a
        collective are performed on an executor, such as TPU or host device. For
        example, All-Reduce, is a collective operation which divides data on
        each node into shards, distributes them, runs computation across
        replicas and gathers them back.

<!-- linter style on -->
