# OpenXLA Benchmarking Architecture & Onboarding Guide

This document outlines the architecture and maintenance process for OpenXLA's
open-source microbenchmarking suite. We leverage the [Benchmarking Automation
Platform (BAP)](https://github.com/google-ml-infra/bap) via GitHub Actions to
execute workloads across physical hardware and publish metrics for performance
regression tracking.

---

## Quick Links

* [OpenXLA Benchmark Registry
  (`benchmark_registry.pbtxt`)](https://github.com/openxla/xla/blob/main/xla/tools/benchmarks/benchmark_registry.pbtxt)
* [Postsubmit Workflow
  (`postsubmit_benchmark.yml`)](https://github.com/openxla/xla/blob/main/.github/workflows/postsubmit_benchmark.yml)
* [Nightly Workflow (`nightly_benchmarks.yml`)](https://github.com/openxla/xla/blob/main/.github/workflows/nightly_benchmarks.yml)
* [XLA HLO Workload Executor
  Documentation](https://github.com/openxla/xla/blob/main/xla/tools/benchmarks/hlo_workload_executor/README.md)
* [BAP (Benchmarking Automation Platform)
  Repository](https://github.com/google-ml-infra/bap)
* [BAP General Onboarding
  Guide](https://github.com/google-ml-infra/bap/blob/main/docs/onboarding.md)
* [BAP Registry Definition
  (`benchmark_registry.proto`)](https://github.com/google-ml-infra/bap/blob/main/bap_proto/benchmark_registry.proto)

---

## Overview

The OpenXLA microbenchmarking suite is designed to automatically detect
performance regressions on pull requests (`presubmit`) and track performance
regressions across continuous post-merge (`postsubmit`) and nightly
(`scheduled`) runs on CPU and GPU hardware backends.

Under BAP, all benchmark configurations, hardware variants, execution action
paths, and static regression thresholds are centralized in a single declarative
Protocol Buffer text file:
[`xla/tools/benchmarks/benchmark_registry.pbtxt`](https://github.com/openxla/xla/blob/main/xla/tools/benchmarks/benchmark_registry.pbtxt).

---

## System Architecture

Our benchmarking pipeline relies on a push-based execution and ingestion model
consisting of four main components:

1. **Benchmark Registry**
   ([`benchmark_registry.pbtxt`](https://github.com/openxla/xla/blob/main/xla/tools/benchmarks/benchmark_registry.pbtxt)):
   The declarative source of truth for all workloads, target metrics, stat
   definitions (e.g., `MEDIAN`), and environment configurations (e.g., `gpu_l4`,
   `gpu_b200`, `cpu_x86`).
2. **GitHub Actions Workflows**: The continuous integration pipelines
   [`postsubmit_benchmark.yml`](https://github.com/openxla/xla/blob/main/.github/workflows/postsubmit_benchmark.yml),
   [`nightly_benchmarks.yml`](https://github.com/openxla/xla/blob/main/.github/workflows/nightly_benchmarks.yml) that invoke BAP's core
   [`run-benchmarks.yaml`](https://github.com/google-ml-infra/bap/blob/main/.github/workflows/run-benchmarks.yaml)
   engine. BAP dynamically generates matrix jobs, delegates execution to our
   local composite action
   [`hlo_workload_executor`](https://github.com/openxla/xla/blob/main/xla/tools/benchmarks/hlo_workload_executor),
   parses scalar metrics (via `tb_parser`), performs static regression checking
   directly inside the workflow, displays a formatted **benchmark report**
   inside the GitHub Actions workflow run summary, and packages the **benchmark
   result** (`results.json`), markdown summary report, execution logs, and
   profiling traces (`_xspace.pb`) into a downloadable [GitHub Actions Artifact
   Bundle](https://github.com/google-ml-infra/bap/blob/main/docs/onboarding.md#artifact-bundling).
3. **Pub/Sub Metric Publishing (Optional)**: BAP can optionally be configured to
   push telemetry downstream upon run completion. When enabled, BAP takes the
   structured JSON result payloads generated in step 2 and publishes them
   directly to a Google Cloud Pub/Sub topic (e.g., `public-results-prod` with
   attribute `repo="openxla/xla"`).
4. **Downstream Ingestion & Dashboards (Optional)**: When Pub/Sub publishing is
   enabled, downstream ingestion pipelines drain the topic, format metric
   payloads, and export telemetry to visualization and regression tracking
   platforms (e.g., internal dashboards like MLCompass or external Looker Studio
   dashboards).

---

## Running Benchmarks

### Automatic Triggers

The benchmarking suite executes automatically across three continuous CI
workflows:

* **Presubmit (`tag_filter: "presubmit"`)**: Runs on open Pull Requests
  targeting the `main` branch. Executes baseline-checked benchmarks and blocks
  CI if a metric regresses beyond its configured `threshold` tolerance.
* **Postsubmit (`tag_filter: "postsubmit"`)**: Runs immediately on push after a
  PR is merged into `main`. Records official post-merge baseline telemetry.
* **Scheduled / Nightly (`tag_filter: "scheduled"`)**: Runs nightly via cron
  across all scheduled benchmarks.

### Testing / Ad-hoc Runs

If you need to execute benchmarks on demand—such as testing a registry change on
a feature branch without merging, or reproducing a regression—you can trigger
the pipeline manually via `workflow_dispatch` in the GitHub Actions UI:

1. **Push to a remote branch**: Commit your changes to
   `benchmark_registry.pbtxt` and push your feature branch to your repository.
2. **Trigger the workflow**: Navigate to the `Postsubmit - Run Benchmarks`
   workflow inside the GitHub Actions UI and click **Run workflow** against your
   branch.
3. **Inspect Results**: Look at the benchmark report inside the workflow run
   summary or inspect the downloaded artifact bundle to verify parsed scalar
   values and static regression comparisons.

---

## Adding a New Benchmark or Hardware Platform

Adding a new microbenchmark or expanding an existing benchmark to run on a new
hardware platform requires zero updates to downstream comparison scripts or YAML
workflows.

### Step 1: Prepare Your Benchmark Artifact
You can host your HLO file either directly inside the repository (recommended
for small microbenchmarks) or in a Google Cloud Storage (GCS) bucket
(recommended for larger artifacts):

* **Option A: Store in the XLA Repository**
  1. Place your `.hlo` file in the `xla/tools/benchmarks/hlo/` directory.
* **Option B: Store in a GCS Bucket**
  1. Upload your `.hlo` artifact to a GCS bucket
     (e.g., `https://storage.googleapis.com/your-bucket/your-benchmark.hlo`).
  2. **Important:** Ensure the GCS bucket/object is publicly readable (or that
     the GitHub Actions runners have the required permissions to access it), as
     the CI runner downloads the artifact via `wget`.

**Important:** Regardless of which hosting option you choose, ensure your HLO
execution finishes within our CI runner timeouts (`< 15 min` for presubmit,
`< 20 min` for postsubmit/nightly).

### Step 2: Define the Benchmark Configuration in `benchmark_registry.pbtxt`
Add a new `benchmarks { ... }` block to
`xla/tools/benchmarks/benchmark_registry.pbtxt`.

The schema for this file is defined by BAP's [`BenchmarkConfig` and
`EnvironmentConfig` protobuf
messages](https://github.com/google-ml-infra/bap/blob/main/bap_proto/benchmark_registry.proto).

For standard HLO microbenchmarks, set `workload.action` to
[`./user_repo/xla/tools/benchmarks/hlo_workload_executor`](https://github.com/openxla/xla/blob/main/xla/tools/benchmarks/hlo_workload_executor)
and specify `hlo_path` alongside optional `runtime_flags` (`--num_repeats=5`).
For detailed parameter specifications and path prefixing rules
(`user_repo/...`), refer to the [XLA HLO Workload Executor
README](https://github.com/openxla/xla/blob/main/xla/tools/benchmarks/hlo_workload_executor/README.md).

### Step 3: Establish Baselines & Thresholds
To enable automatic CI regression checking when your benchmark runs in CI
(`postsubmit` or `scheduled`), embed static
`comparison { baseline { ... } threshold { ... } }` blocks inside the respective
metric definitions under each `environment_configs` block.

**Tip:** When onboarding a new benchmark for the very first time, we recommend
adding it with `postsubmit` or `scheduled` tags and omitting the `comparison`
blocks. Once it runs cleanly on main and you observe its stable median numbers
in BAP's outputs, lock in the exact `baseline` and `threshold` tolerance.

#### Example Full `benchmark_registry.pbtxt` Entry
```protobuf
benchmarks {
  name: "gemma3_1b_flax_sample_loop"
  description: "Gemma3 1B in Flax Sample Loop."
  owner: "your-team@"

  workload {
    action: "./user_repo/xla/tools/benchmarks/hlo_workload_executor"
    action_inputs {
      key: "hlo_path"
      value: "xla/tools/benchmarks/hlo/gemma3_1b_flax_sample_loop.hlo"
    }
    action_inputs {
      key: "runtime_flags"
      value: "--num_repeats=5"
    }
  }

  environment_configs {
    id: "gpu_l4"
    runner_label: "linux-x86-g2-16-l4-1gpu"
    container_image: "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build-cuda12.8-cudnn9.8:latest"
    workload_action_inputs {
      key: "hardware_category"
      value: "GPU_L4"
    }
    tags: "presubmit"
    tags: "postsubmit"

    metrics {
      name: "GPU_DEVICE_TIME"
      unit: "ms"
      stats {
        stat: MEDIAN
        comparison {
          baseline { value: 15.0 }
          threshold { value: 0.30 } # Allow up to 30% regression before failing CI
          improvement_direction: LESS
        }
      }
    }
    metrics {
      name: "GPU_DEVICE_MEMCPY_TIME"
      unit: "ms"
      stats {
        stat: MEDIAN
        comparison {
          baseline { value: 0.3 }
          threshold { value: 0.30 }
          improvement_direction: LESS
        }
      }
    }
  }

  environment_configs {
    id: "cpu_x86"
    runner_label: "linux-x86-n2-128"
    container_image: "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"
    workload_action_inputs {
      key: "hardware_category"
      value: "CPU_X86"
    }
    tags: "presubmit"

    metrics {
      name: "CPU_TIME"
      unit: "ms"
      stats {
        stat: MEDIAN
        comparison {
          baseline { value: 10000.0 }
          threshold { value: 0.30 }
          improvement_direction: LESS
        }
      }
    }
    metrics {
      name: "WALL_TIME"
      unit: "ms"
      stats {
        stat: MEDIAN
        comparison {
          baseline { value: 3000.0 }
          threshold { value: 0.30 }
          improvement_direction: LESS
        }
      }
    }
  }
}
```

---

## Removing a Benchmark

Removing an obsolete benchmark or hardware configuration is straightforward:
1. **Update the Registry**: Simply delete the corresponding
   `benchmarks { ... }` block (or specific `environment_configs { ... }` block)
   from `xla/tools/benchmarks/benchmark_registry.pbtxt`.
2. **Commit and Merge**: Once merged, BAP will immediately stop synthesizing
   matrix jobs and publishing telemetry for that workload.

---

## BAP Documentation & Troubleshooting

For deeper dives into the underlying platform tooling or troubleshooting, refer
to:

* [BAP General Onboarding
  Guide](https://github.com/google-ml-infra/bap/blob/main/docs/onboarding.md)
