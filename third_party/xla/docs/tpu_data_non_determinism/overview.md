# Debugging data non-determinism

This set of documentation describes debugging techniques when your *TPU programs
do not work as expected*. This can include program non-determinism in outputs or
observable intermediate results, unexpected NaNs/Inf, wrong outputs, and more.

The localization and debugging workflow involves amending XLA code and
rerunning the reproducer with new recompiled binary,
similar to [on-device debugging workflow](/tpu_on_device_debugging/overview).

## Common symptoms & localization approach

For large-scale TPU program debugging the localization is usually the most
complicated part, involving different techniques depending on the category of
symptoms observed, most common ones are:

*   **Data non-determinism observed**

    *   Any of the observed program outputs (e.g. loss) are non-deterministic,
        i.e. diverge across several runs on the same inputs
    *   XSC reported non-determinism
    *   Next steps: Use the [XLA Silent Data Corruption Checker (XSC) with HLO
        re-run and extra debugging flags](./xsc_debug_nondeterminism)

*   **NaN/Inf observed**

    *   Unexpected NaNs or Infinities are observed in program output or in
        internal state/computation result
    *   Next steps: Use the [XLA Silent Data Corruption Checker (XSC) for
        NaN/Inf Tracing](./xsc_debug_nans)
