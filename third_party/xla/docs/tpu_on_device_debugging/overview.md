# TPU on-device debugging

<!-- disableFinding("hang") -->

This set of documentation describes debugging techniques when your *TPU programs
do not work as expected*. This can include program hangs and crashes,
non-determinism in outputs or observable intermediate results, unexpected
NaNs/Inf, wrong outputs, and more.

## Overview

Debugging TPU program executed on device usually includes three stages:

*   ***Localization*** of the issue to a single culprit HLO (or a group of HLOs)
    inside the HLO module. Localization approach heavily depends on observed
    symptoms.
*   ***Debugging*** of the culprit HLO(s), depends on the results of
    localization and HLO category (collective, fusion kind, custom kernel,
    etc,.).
*   ***Fix*** the issue by changing source code or adjusting the model.

This document mostly focuses on localization since it is usually the most
difficult part of the debugging.

Most of the localization and debugging techniques are based on instrumentation
of XLA code with extra checks, logging, or other changes in the emitted code
with the intention of identifying the issue. Some most common instrumentations
required for localization or debugging are already present in XLA code and can
be enabled and configured via compiler flags. Many other instrumentations are
very specific to the issue being debugged, in which case, the lifespan of such
instrumentations ends with the end of debugging.

Hence, localization and debugging of many issues requires amending XLA code and
rerunning the reproducer with new recompiled binary.[^1]

## Typical workflow: Debugging XLA compiler flags

Currently, most of the debugging events are reported as: program assertions,
and/or logging messages.

XLA compiler provides few debugging instrumentations (via XLA flags) intended to
help with debugging different issues with the TPU program executed on device.

The common debugging workflow would involve:

*   Decide what debugging instrumentations to run
*   Provide a set of relevant flags to the compiler
*   Rerunning the reproducible snippet (with flags set) to collect debugging
    events
*   Analyze the events to localize the issue to an HLO or LLO, un-reliable TPU
    core, etc.

Both collecting and analysis of debugging events is done manually, and in some
cases automatically.

For programmatic debugging, you can also use the [**Debug
callback**](./debug_callback.md), to invoke callback functions when anomalies
like hangs, crashes, numerical inconsistencies, and more occur.

## Common symptoms & localization approach

For large-scale TPU program debugging the localization is usually the most
complicated part, involving different techniques depending on the category of
symptoms observed. The most common ones are:

*   **Crash: Hardware fatal error**

    *   Hardware reported an unrecoverable user fatal error during TPU program
        execution.
    *   Next steps:
        *   [Collect and process core dumps](./analyze_core_dumps.md) from all
            the chips from the slice (Localize by scalar pc location based on
            LLO dump.)

*   **Crash: XLA generated assertion**

    *   TPU program stopped at location of one of the assertions inserted by XLA
        compiler (e.g. bounds checker assertion)
    *   Next steps:
        *   Collect and analyze assertion information from logs.

*   **TPU program HANG (single-slice or multi-host Pod)**

    *   TPU program stops making progress and each core gets stuck at some
        location
    *   Next steps:
        *   Use [hang diagnostics](./hang_diagnostics.md)
            (`TpuDiagnosticCoordinator`, enabled by default since `libtpu`
            0.0.46) to automatically detects stalls, captures per-core execution
            snapshots, and generates diagnostic reports.
        *   Use the [sync flag wait timeout](./sync_flag_wait_timeout.md), with
            a [hierarchical program progress
            tracker](./hierarchical_progress_tracker.md). to detect a stuck
            Direct Memory Access (DMA) or a barrier to finish.
        *   Register a [debug callback](./debug_callback.md) to receive
            `STALL_EVENT` and `SFLAG_TIMEOUT` events in process.
        *   [Use TPUZ
            Monitoring]()
            to inspect live runtime status for all TPU cores on a host.
        *   [File a support case](./file_support_case.md) to get help from Cloud
            Support.

[^1]: This limitation is relaxed for Pallas kernels, since while the users don't
    control low-level debug primitives, they still can make changes in their
    kernels and use it for debugging.
