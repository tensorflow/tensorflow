# Reporting JAX/XLA compile-time crashes on TPU

## Introduction

This guide provides instructions on how to report a compile-time crash while
running JAX on TPU. Following these steps will help you gather the necessary
information to report it effectively.

The crash report should be submitted to the JAX GitHub repository
([https://github.com/google/jax/issues](https://github.com/google/jax/issues))
or, for XLA-specific issues, to the OpenXLA GitHub repository
([https://github.com/openxla/xla/issues](https://github.com/openxla/xla/issues)). However, if you suspect the cause of the problem to be in your own code, you can
try to [investigate the issue further](#analyzing-issue).

## Initial steps when a crash occurs {#initial-steps}

In order to report the issue effectively, make sure you capture the full error
message and stack trace:

*   Copy and paste the *entire* console output, including the full error message
    and stack trace, in the bug report. This is often the most critical piece of
    information, e.g. which HLO pipeline or pass was running, or which HLO
    module was being lowered when a crash happened. You can save this to a
    `.txt` file and share it in your bug report.

## Gathering debug information

Include the following information on your bug report.

### 1. HLO dump (most important for XLA bugs)

XLA operates on HLO (High Level Optimizer) intermediate representation. An HLO
dump allows the XLA team to reproduce compiler behavior often without needing
your full JAX code or data.

**How to dump:** See
[Dump HLO Computations | OpenXLA Project](https://openxla.org/xla/hlo_dumps) for
details and examples. This generates several text files in the output directory,
including one with all the environment variables (look for files ending with
`tpu_comp_env.txt`); all are necessary for reporting a crash. If you are running
your program on a Cloud TPU environment, you may need to set up a remote storage
to save your HLO dump. You can share this directory as a zipped attachment or a
link to your cloud storage.

### 2. Minimal reproducer

If possible, provide a small, self-contained JAX code snippet that reliably
triggers the crash. You can try:

*   Isolating the `@jax.jit`, or other JAX transformation causing the issue;
*   Replacing data dependencies with mock data (e.g., using `jax.random`).

Creating a minimal reproducer can be challenging when working in complex,
integrated codebases and is not required for the crash report, but it is highly
recommended.

## Reporting the crash

*   Report issues to the JAX GitHub repository
    ([https://github.com/google/jax/issues](https://github.com/google/jax/issues))
    or, for XLA-specific issues, the OpenXLA GitHub repository
    ([https://github.com/openxla/xla/issues](https://github.com/openxla/xla/issues)).
*   **What to include in reports:**
    1.   Clear description of the issue.
    2.   FULL error message and stack trace, as described in the [Initial steps
         when a crash occurs section](#initial-steps).
    3.   The [HLO Dump](https://openxla.org/xla/hlo_dumps) directory (e.g.,
         zipped attachment or link to cloud storage).
    4.   The minimal reproducer code (if available).

## Analyzing the issue (optional) {#analyzing-issue}

Advanced users who want to analyze the issue further before filing a bug, for
example, to produce a minimal reproducer, or try to find a workaround, can try
the following tools. Otherwise, skip this section.

### Isolating problematic HLO instructions

If you have an HLO dump and suspect a specific instruction or section within the
HLO module is causing a crash or numerical mismatch, but you are not sure which
one it is, you can use the `hlo_isolation_test_base` tool. This tool takes an
HLO graph and runs each of the instructions individually on two user-specified
runners (default of TPU and HloEvaluator). It reports any mismatches between the
results of the environments, and dumps the isolated "problem" HLO instruction(s)
and a numerical report on the magnitude of the mismatch. See [HLO isolation user
guide](https://github.com/openxla/xla/blob/86758660db0ed0d3676fa7b70d67e9173fcaebd5/docs/hlo_isolation.md)
for more information and examples.

### Using optimization barrier as a workaround

Sometimes, XLA optimizations (like instruction fusion, reordering, or layout
changes) can trigger bugs or unexpected behavior. To test this hypothesis or as
a temporary workaround, you can insert optimization barriers into your JAX code.

An `OptimizationBarrier` op prevents the compiler from moving operations across
it and can inhibit certain fusions or reassociations. By strategically placing
barriers, you can pinpoint if a specific optimization interaction is causing the
crash. If adding a barrier makes the crash disappear, this is valuable
information for the bug report.

The [JAX
Documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.optimization_barrier.html)
contains examples of how to use OptimizationBarrier operations in your code. You
can also find more information on the [OpenXLA
Documentation](https://openxla.org/xla/operation_semantics#optimizationbarrier).
