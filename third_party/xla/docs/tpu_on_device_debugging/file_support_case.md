<!-- disableFinding("hang") -->
<!-- disableFinding("hung") -->

# TPU hang: File a support case

When contacting [Cloud Support](https://cloud.google.com/support) about a TPU
hang, include the following:

## Required

1.  [**Diagnostic artifacts**](./hang_diagnostics.md) from every host:\
    `tpu_stall_summary_*.pb`,\
    `tpu_compiler_metadata_*.pb`,\
    `tpu_core_dump_*.bin` (if you enabled
    `--xla_tpu_diagnostic_collect_core_dumps=true`), and\
    output of `python -m libtpu.diagnostics ./diag/` (`libtpu` ≥ 0.0.48).
2.  **TPU type and topology** (e.g., `v5litepod-16`, `v4-32`)
3.  **Framework and libtpu versions:**
    ```sh
    python -c "import jax; print(jax.__version__)"
    pip show libtpu | grep Version
    ```

## Helpful

1.  **Workload stderr** from around the hang, including the `PendingEventLogger`
    and the 3-line `TpuDiagnosticCoordinator` `WARNING` digest
    (`TPU stall snapshot (...)`, `TPU stall artifact: ...`, `Next: ...`).
2.  **`LIBTPU_INIT_ARGS`** used for the job. When reproducing a hang for
    Support, we recommend:
    ```sh
    export LIBTPU_INIT_ARGS="--xla_tpu_impure_diagnostic_artifacts_dir=gs://MY_BUCKET/tpu-diagnostics/ --xla_tpu_debug_sflag_wait_timeout_ms=600000 --xla_tpu_diagnostic_cooldown_duration=5m"
    ```
3.  **Training script** or minimal reproducer
4.  **Timing information** — when it happened, what training step, whether it
    reproduces

## Template

```
Title: TPU hang on [TPU type]

Environment:
  TPU: v5litepod-16
  JAX: 0.11.1
  libtpu: 0.0.51

What happened:
  Training hung at step 15,000 after ~2 hours. No error in stderr.

Stall detection log lines:
  [paste the PendingEventLogger and TpuDiagnosticCoordinator lines]

Attached: tpu_stall_summary_*.pb, tpu_compiler_metadata_*.pb
```

**Important:** The artifacts are more than telemetry. Compiler metadata files
include your compiled HLO module and symbol table, and hardware state snapshots
include register contents. Review them against your data-handling policies
before you share them.
