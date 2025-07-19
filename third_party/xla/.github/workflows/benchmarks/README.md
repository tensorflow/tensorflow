# Onboard New Microbenchmarks to OpenXLA

This guide provides step-by-step instructions for contributing new
microbenchmarks to OpenXLA microbenchmarking infrastructure.

## Overview

The OpenXLA microbenchmarking system is designed to automatically detect
performance regressions and track performance trends across different hardware
backends (CPU, GPU) in presubmit, postsubmit and nightly workflows. By adding
your microbenchmark, you help ensure XLA's performance remains robust for your
specific use cases.

The process involves:

1.  **Preparing your Benchmark Artifact:** Ensuring your HLO file is OSS-friendly.
2.  **Defining the Benchmark Configuration:** Adding an entry to a benchmark registry file.
3.  **Establishing a Baseline:** Adding initial performance thresholds if your benchmark will run in presubmit/postsubmit/nightly jobs.

## Prerequisites

*   **Benchmark Artifact (HLO):** You should have your benchmark ready as either
    an HLO text file (`.hlo`) (Note: StableHLO MLIR text file (`.mlir`) will be
    supported later).
    *   For small artifacts, you can place them in `xla/tools/benchmarks/hlo/`.
    *   **[Not yet supported]** For larger artifacts, upload them to a GCS
        bucket (e.g., `gs://xla-benchmarking-temp/your-benchmark.hlo`) and
        ensure it's publicly readable.
*   **GitHub Access:** You'll need to create a Pull Request (PR) to the OpenXLA repository.

## Step-by-Step Guide

### Step 1: Prepare Your Benchmark Artifact

Ensure your HLO file is ready and accessible.

*   **Store in the XLA Repository**
    1.  Place your `.hlo` file in the `xla/xla/tools/benchmarks/hlo/` directory.
    2.  Make sure your hlo benchmarks run < 15min/20min/30min for
        presubmit/postsubmit/nightly workflows.

### Step 2: Define the Benchmark Configuration

You'll need to add a new entry to a benchmark registry YAML file. For most
community contributions, this will be
`xla/xla/tools/benchmarks/registries/default_registry.yml`.

Each benchmark configuration is a YAML object with the following key fields:

*   `name`: A unique, descriptive name for your benchmark (e.g., `"my_model_attention_layer"`).
*   `description`: A brief explanation of what the benchmark measures.
*   `owner`: Your GitHub handle or relevant team alias (e.g., `"your-github-username@"`).
*   `input_artifact`:
    *   `input_format`: Currently we support `HLO_TEXT`, and `STABLEHLO_MLIR` will be supported in the future.
    *   `artifact_path`: (If stored in repo) Relative path from `xla`, e.g., `xla/tools/benchmarks/hlo/my_new_benchmark.hlo`.
    *   `artifact_gcs_bucket_path`: (If stored in GCS) Full GCS URL.
*   `model_source_info`: A list of strings describing the origin of the benchmark (e.g., `["Gemma2 2B"]`).
*   `hardware_targets`: A list defining on which hardware configurations this
    benchmark should run. Each target has:
    *   `hardware_category`: e.g., `GPU_L4`, `CPU_X86`, `GPU_B200`.
    *   `topology`:
        *   `num_hosts`: Number of hosts (default: 1).
        *   `num_devices_per_host`: Number of devices per host (default: 1).
        *   `multi_host`: `true` or `false`.
    *   `multi_device`: `true` or `false`.
    *   `target_metrics`: A list of metrics to collect, e.g., `[GPU_DEVICE_TIME, PEAK_GPU_MEMORY]`.
    *   `run_frequencies`: When to run this benchmark, e.g., `[PRESUBMIT, POSTSUBMIT]`, `[SCHEDULED]`.
*   `update_frequency_policy`: How often this benchmark definition should
    be reviewed, e.g., `QUARTERLY`.
*   `xla_compilation_flags` (Optional): List of XLA flags, e.g.,
    `["--xla_gpu_enable_cudnn_fusion=false"]`.
*   `runtime_flags` (Optional): List of flags for the
    `multihost_hlo_runner`, e.g., `["--num_repeats=5"]`.
*   `github_labels` (Optional): GitHub labels to manually trigger this
    specific benchmark.

**Example: Adding "gemma3\_1b\_flax\_sample\_loop" to `default_registry.yml`**

```yaml
# xla/xla/tools/benchmarks/registries/default_registry.yml
benchmarks: [
  # ... existing benchmarks ...
  {
    name: "gemma3_1b_flax_sample_loop"
    description: "Gemma3 1B in Flax Sample Loop."
    owner: "company-A@" # Replace with your GitHub handle or team
    input_artifact: {
      input_format: HLO_TEXT, # Or STABLEHLO_MLIR
      artifact_path: "xla/tools/benchmarks/hlo/gemma3_1b_flax_sample_loop.hlo"
      # Option 2 (for large hlo):
      #`artifact_gcs_bucket_path`: (If stored in GCS) Full GCS URL (not supported yet).
    }
    model_source_info: ["Gemma3 1B"] # Describe the source of your HLO
    hardware_targets: [{
      hardware_category: GPU_L4
      topology: { num_hosts: 1, num_devices_per_host: 1, multi_host: false, multi_device: false }
      target_metrics: [GPU_DEVICE_TIME, GPU_DEVICE_MEMCPY_TIME]
      run_frequencies: [PRESUBMIT, POSTSUBMIT] # Run on PRs in presubmit and postsubmit
      runtime_flags: ["--num_repeats=5"] # Example: run 5 times to reduce noise
    },
    {
      hardware_category: CPU_X86
      topology: { num_hosts: 1, num_devices_per_host: 1, multi_host: false, multi_device: false }
      target_metrics: [CPU_TIME, WALL_TIME]
      run_frequencies: [PRESUBMIT] # Only run on PRs for presubmit
      runtime_flags: ["--num_repeats=5"]
    }]
    update_frequency_policy: QUARTERLY # Review this benchmark definition quarterly
  }
]
```

### Step 3: Establish a Baseline

1.  **Determine Baseline Values:**
    *   The best way to get initial baseline values is to run your benchmark
    manually on the target hardware or let it run once in postsubmit after your
    initial PR (without presubmit blocking) is merged.
    *   Promote your benchmarks from postsubmit to presubmit once you get
        stable results for baseline values.
    *   Run the benchmark multiple times (e.g., using `--num_repeats=5` or
        more) and take the median or a stable average.
    *   Benchmarks must run < 15min for presubmit, < 20min for postsubmit and < 30min for nightly.

2.  **Add to `presubmit_baseline.yml`:**
    Edit the file `xla/xla/tools/benchmarks/baseline/presubmit_baseline.yml`. The
    key for each entry is the `config_id`.

    **Note on `config_id` generation:**
    `config_id` follows the below pattern:
    `"{benchmark_name}_{hardware_category_simplified}_{topology_simplified}_{workflow_type}"`.
    *   `hardware_category_simplified`: e.g., `l4` (for `GPU_L4`), `b200` (for `GPU_B200`), `x86` (for `CPU_X86`).
    *   `topology_simplified`: e.g., `1h1d` for 1 host, 1 device. <!-- disableFinding(LINE_OVER_80) -->
    *   `workflow_type`: e.g., `presubmit`, `postsubmit`, `scheduled`.

    If unsure, you can check the GitHub Actions workflow logs for the
    `generate_benchmark_matrices.py` script output, which will show the generated
    `config_id`s.

    For each metric you want to track in presubmit (must be in `target_metrics`
    in the registry):

    *   `baseline_ms`: The baseline performance in milliseconds.
    *   `threshold`: The maximum allowed regression percentage (e.g.,
        `0.30` for 30%).
    *   **Note on metrics:** Currently, we support `GPU_DEVICE_TIME`,
        `GPU_DEVICE_MEMCPY_TIME` for GPU, and `CPU_TIME`, `WALL_TIME` for CPU.

**Example: Adding baseline for "gemma3\_1b\_flax\_sample\_loop"**

Assuming the `name` is `"gemma3_1b_flax_sample_loop"`:

*   For `GPU_L4`, `1` host, `1` device: `config_id` becomes
    `gemma3_1b_flax_sample_loop_l4_1h1d_presubmit`
*   For `CPU_X86`, `1` host, `1` device: `config_id` becomes
    `gemma3_1b_flax_sample_loop_x86_1h1d_presubmit`

```yaml
# xla/xla/tools/benchmarks/baseline/presubmit_baseline.yml
{
  # ... existing baselines ...

  "gemma3_1b_flax_sample_loop_l4_1h1d_presubmit": {
    "GPU_DEVICE_TIME": {
      "baseline_ms": 4,  # Your measured baseline
      "threshold": 0.30
    },
    "GPU_DEVICE_MEMCPY_TIME": {
      "baseline_ms": 10, # Your measured baseline
      "threshold": 0.30
    }
  },
  "gemma3_1b_flax_sample_loop_x86_1h1d_presubmit": {
    "CPU_TIME": {
      "baseline_ms": 8000, # Your measured baseline
      "threshold": 0.30
    },
    "WALL_TIME": {
      "baseline_ms": 1300, # Your measured baseline
      "threshold": 0.30
    }
  }
}
```

### Step 4: Create a Pull Request

1.  Commit your changes:
    *   The HLO file (if added to the repo).
    *   The updated benchmark registry file (e.g., `default_registry.yml`).
    *   The updated `presubmit_baseline.yml` (if applicable).
2.  Push your branch and open a Pull Request against the `openxla/xla` main branch.
3.  A member of the OpenXLA repository or organization will need to review your
    PR for safety before the CI system is invoked.
     * **Note**: This step happens
    automatically for organization members and most Googlers, but require
    manual review for external contributors.
4.  Once approved, the CI system will pick up your new benchmark configuration.
    *   If it's a `PRESUBMIT` benchmark, it will run against your PR and check for regressions based on the baseline you provided.
    *   If it's `POSTSUBMIT` or `SCHEDULED`, it will run after your PR is merged.
5.  Monitor the CI checks. If the presubmit check fails due to your new
    benchmark (e.g., performance is significantly different from your initial
    baseline), you might need to adjust the baseline values in
    `presubmit_baseline.yml` and update your PR.

## Best Practices

*   **Establish Baselines First:** Since, a baseline value per metric is required, always add the benchmark with only
    `POSTSUBMIT` or `SCHEDULED` frequency first to establish stable baseline
    values. Once it runs a few times and you have stable performance data, you
    can add `PRESUBMIT` and the corresponding baseline entry in a follow-up PR.
*   **Meaningful Names and Descriptions:** Make it easy for others to
    understand what your benchmark does.
*   **Targeted Metrics:** Only include relevant metrics in `target_metrics`.
*   **Noise Reduction:** Use `runtime_flags: ["--num_repeats=X"]` (e.g., X=5 or
    10) to run the benchmark multiple times within a single execution, which
    helps in getting more stable measurements. The runner typically reports the
    median or average.
*   **Keep Baselines Updated:** If your benchmark's performance characteristics
    change significantly (due to XLA improvements or changes in the benchmark
    itself), the baseline values in `presubmit_baseline.yml` will need to be
    updated. This is usually done by the benchmark owner or XLA maintainers.

## Troubleshooting

*   **Workflow Failures:** Check the GitHub Actions logs for detailed error
    messages. The logs for the "Compare Benchmarks" step are particularly useful
    for presubmit issues.
*   **Incorrect `config_id`:** If your presubmit benchmark isn't being picked up
    or matched to a baseline, double-check the `config_id` format in
    `presubmit_baseline.yml`.
*   **Performance Fluctuations:** Microbenchmarks can be sensitive to noise.
    Ensure you're using `--num_repeats` and that your baseline reflects typical
    performance.

If you encounter issues, feel free to ask for help on the OpenXLA communication
channels or tag the juliagmt-google@ on your PR.
