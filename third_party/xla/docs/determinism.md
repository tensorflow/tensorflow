# Determinism (GPU)

## Compilation

XLA compilation is deterministic if
[persisted autotuning](./persisted_autotuning) is used to perform autotuning
once and avoid it in subsequent compilations. Otherwise due to fluctuations in
measurements different kernels can be picked as the fastest ones in different
compilation runs.

`--xla_gpu_require_complete_aot_autotune_results` can be used to ensure that no
autotuning happens on repeated compilations - they either reuse compatible
results of previous runs or fail.

## Execution

Programs compiled by XLA can be non-deterministic on operations like scatter,
select-and-scatter, GEMMs, convolutions, multi-headed attention. The flag
`--xla_gpu_exclude_nondeterministic_ops` switches these operations to
deterministic and potentially slower implementations and makes compilation fail
on select-and-scatter which does not have a deterministic implementaiton.
