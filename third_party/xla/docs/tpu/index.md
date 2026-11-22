# TPU Debugging Overview

This section includes a collection of guides and reference docs for debugging
TPU programs. Both JAX and TPU-SDK provide an extensive set of debugging tools
and techniques, and choosing the right one is a technical decision itself.

Debugging can be divided into several levels, where each group offers
diagnostics for a specific topic or area.

## JAX

When starting debugging your JAX model, JAX offers a high-level and intuitive
API and flags for logging and troubleshooting your model. If you are just
starting, consider it as your first choice for any investigation, before jumping
directly to a more complex tooling:

*   [JAX’s official debugging docs
    page](https://docs.jax.dev/en/latest/debugging.html)

## HLO debugging

For TPU, the JAX program is lowered to HLO, which can be extensively inspected
for errors. HLO dumps provide an insight into generated modules:

*   [Dump HLO & LLO computations](/xla/docs/hlo_dumps)

## Domain-specific debugging

For a range of domain-specific errors, check out the following documentation on
some common cases:

-   [01 - Megascale Error Handling & Hangs](/xla/docs/megascale/debugging_workflow)
-   [02 - Megascale Stragglers](/xla/docs/megascale/debugging_workflow#debugging_stragglers_in_megascale)
-   [03 - Megascale Performance](/xla/docs/megascale/debugging_workflow#performance)
-   [04 - OOM + HBM OOM (Error code
    1000)](/xla/docs/oom_debugging)
-   [05 - TPU Program On-device Debugging ](/xla/docs/on_device_debugging/overview)
-   [06 - Reporting JAX/XLA Compile-time Crashes on
    TPU](/xla/docs/jax_compile_crash)
-   07 - Pallas/Mosaic Kernel Debugging (coming soon)
-   08 - Data Non-determinism (coming soon)
-   09 - Numerics-related Errors (coming soon)
-   10 - SparseCore Debugging Tools
    -   [SparseCore Mismatch Detector](/xla/docs/sparsecore_mismatch_detector)
    -   [XLA Flags: SparseCore](/xla/docs/sparsecore_flags)
-   [11 - HLO & LLO Dumps](/xla/docs/hlo_dumps)
