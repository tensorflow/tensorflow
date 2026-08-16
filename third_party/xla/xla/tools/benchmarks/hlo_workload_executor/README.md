# XLA HLO Workload Executor

This composite GitHub Action executes OpenXLA HLO (`.hlo`) microbenchmarks on
target CPU and GPU runners under the [Benchmarking Automation Platform (BAP)](https://github.com/google-ml-infra/bap).

## Overview

When triggered by BAP's dynamic action executor, this action performs the
following pipeline:

1. **Source Resolution**: Resolves the `openxla/xla` checkout directory (whether
   invoked internally from `openxla/xla` or from an external repository).
2. **Binary Building**: Builds required benchmarking binaries
   (`hlo_runner_main`, `compute_xspace_stats_main`) via Bazel using
   `.github/workflows/benchmarks/build_binaries.sh`.
3. **Artifact Preparation**: Downloads or verifies the target HLO artifact
   (`.hlo`) using `.github/workflows/benchmarks/prepare_artifact.sh`. Supports
   both local repository paths (`xla/tools/benchmarks/hlo/...`) and GCS URIs
   (`https://storage.googleapis.com/...`).
4. **Execution**: Runs the HLO benchmark
   (`.github/workflows/benchmarks/run_benchmark.sh`) across specified runtime
   flags (`--num_repeats=5`), capturing execution profiles (`_xspace.pb`) and
   writing raw parsed results to `$WORKLOAD_ARTIFACTS_DIR/results.json`.
5. **TensorBoard Export**: Converts the parsed metrics in `results.json` into
   TensorBoard scalar events (`$TENSORBOARD_OUTPUT_DIR`) via
   `xla/tools/benchmarks/json_to_tensorboard.py` for BAP metric harvesting
   (`tb_parser`).

## Inputs

<!-- disableFinding(LINE_OVER_80) -->

| Input | Description | Required | Default |
| :--- | :--- | :---: | :---: |
| `hlo_path` | Path to the HLO artifact. Can be a GCS URI (`https://...`) or a local path.<br>For local paths, prefix with the appropriate source directory:<br>• Use `user_repo/...` if calling from within the `openxla/xla` repository.<br>• Use `xla_src/...` if calling from an external repository. | Yes | - |
| `hardware_category` | Target hardware category (e.g. `CPU_X86`, `GPU_L4`, `GPU_B200`). | Yes | - |
| `backend_flags` | Space-separated list of flags passed directly to the XLA compiler. | No | `""` |
| `runtime_flags` | Space-separated list of flags passed to `multihost_hlo_runner` (e.g. `--num_repeats=5`). | No | `""` |
| `xla_ref` | Git branch, tag, or SHA of `openxla/xla` to build from (when invoked externally). | No | `"main"` |

## Usage Example (BAP `benchmark_registry.pbtxt`)

```protobuf
workload {
  action: "./user_repo/xla/tools/benchmarks/hlo_workload_executor"
  action_inputs {
    key: "hlo_path"
    value: "user_repo/xla/tools/benchmarks/hlo/hlo_gemma4_2b_bf16.hlo"
  }
  action_inputs {
    key: "runtime_flags"
    value: "--num_repeats=5"
  }
}
```
