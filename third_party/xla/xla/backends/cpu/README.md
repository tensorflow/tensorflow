# OSS HLO Benchmarks
This directory contains HLO benchmarks for open source models.

Below is a sample command for running an individual HLO benchmark:

```bash
bazel run //xla/backends/cpu/benchmarks:hlo_benchmark_test -- --hlo_paths=xla/backends/cpu/benchmarks/hlo/gemma3_1b_flax_call.hlo
```

Another option is to run already defined Bazel targets. For example you, can
get a list of targets using this simple command:

```bash
bazel query //xla/backends/cpu/benchmarks/... | grep hlo_benchmark
```

This will reveal the list of HLO benchmark targets available.