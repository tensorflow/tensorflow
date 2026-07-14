# HLO Isolation User Guide

This document explains how to install and use the HLO Isolation API and CLI.
The HLO isolation tooling helps developers and researchers isolate, verify, and
debug numeric mismatches and stability issues in compiled HLO modules.

## Installation

You can use HLO isolation components through the standard OpenXLA/TensorFlow
build mechanism (Bazel) or via binary distribution.

### Source and Bazel Setup

When building the compiler and tools stack from source, include or depend on the
following libraries:

-   API Target:
    `//third_party/tensorflow/compiler/xla/tools/hlo_isolation:hlo_isolation_api`
-   CLI Target:
    `//third_party/tensorflow/compiler/xla/tools/hlo_isolation:hlo_isolation_test`

To build the standalone CLI tool:

```bash
bazel build -c opt //third_party/tensorflow/compiler/xla/tools/hlo_isolation:hlo_isolation_test
```

## Command-Line Interface (CLI)

The `hlo_isolation_test` CLI allows you to isolate and run numeric mismatch and
stability checks against compiled HLO modules directly from a terminal. It
compares the execution results across different environments (e.g., TPU vs.
Defused TPU / CPU / Interpreter).

### Flag Reference

The CLI supports the following flags:

-   `--hlo_file`: Path to the input `.hlo` or `.pbtxt` file to load. Can be text
    or proto format (required).
-   `--test_platform`: Target platform to run the primary test on (e.g., `cpu`,
    `gpu`, `tpu`). Defaults to `cpu`.
-   `--reference_platform`: Reference platform for baseline comparison (e.g.,
    `interpreter`). If empty, reference comparison is disabled.
-   `--filter_by_name`: Regular expression to match the module name. Only
    matching modules will be run. Defaults to `.*`.
-   `--skip_by_name`: Regular expression to match the module name. Matching
    modules will be skipped.
-   `--filter_by_opcode`: Regular expression to match instruction opcodes. Only
    modules containing at least one matching opcode will be run. Defaults to
    `.*`.
-   `--skip_by_opcode`: Regular expression to match instruction opcodes. Modules
    containing any matching opcode will be skipped.
-   `--abs_error_bound`: Absolute error bound used for comparison. Defaults to
    `0.01`.
-   `--rel_error_bound`: Relative error bound used for comparison. Defaults to
    `0.1`.
-   `--run_hlo_passes`: Boolean flag to determine whether to run standard HLO
    passes on the submodules. Defaults to `false`.
-   `--shard_index`: The specific shard index to run (zero-based). Defaults to
    `-1` (disabled).
-   `--num_shards`: The total number of shards. Defaults to `1`.

### Basic Invocation

```bash
./hlo_isolation_test \
  --hlo_file=/path/to/failing_fusion.hlo \
  --test_platform=gpu \
  --reference_platform=interpreter
```

## Result and Artifact Dumps

When a submodule encounters a numeric mismatch or other failure during isolation
testing, the tool automatically serializes debug artifacts to disk for deeper
inspection.

### Dump Contents

On numeric mismatch, the tool writes the following debug artifacts:

1.  The failed HLO submodule text (`failed-module-<module_name>.txt`).
2.  The expected output literal (`failed-<module_name>-expected.txt`).
3.  The actual output literal (`failed-<module_name>-actual.txt`).
4.  The mismatching elements summary (`failed-<module_name>-mismatches.txt`).

### Dump Target Location

-   **Test Environment:** If run via `bazel test` or an environment defining the
    `TEST_UNDECLARED_OUTPUTS_DIR` environment variable, the results are placed
    directly in that directory with the exact file names listed above (e.g.,
    `failed-<module_name>-expected.txt`).
-   **Standard/Manual Run:** When executed manually via the command line, the
    artifacts are written to the operating system's temporary directory (e.g.,
    `/tmp`), preserving the exact same unified file naming conventions (e.g.,
    `/tmp/failed-<module_name>-expected.txt`).

## C++ Integration and API

For developers building custom compiler passes, testing rigs, or automated
pipelines, the C++ API provides a direct way to integrate isolation tests.

### Using the API Directly

The core API provides functional interfaces to run modules and fetch structured
reports:

```cpp
#include "third_party/tensorflow/compiler/xla/tools/hlo_isolation/hlo_isolation_api.h"

xla::hlo_isolation::PipelineIsolationOptions options;
options.module_options.abs_error_bound = 0.01;
options.module_options.rel_error_bound = 0.1;
// Filter specific opcodes programmatically
options.filter_by_opcode = "exponential";

absl::StatusOr<std::vector<xla::HloIsolationTestResult>> results =
    xla::hlo_isolation::RunIsolationPipeline(
        input_hlo_module,
        &my_test_runner,
        &my_reference_runner,
        options);
```

### Using the Test Mixin

When writing GoogleTest C++ test suites, you can inherit from
`HloIsolationTestMixin` for built-in assertion handling. The base class must
provide both a test runner and a reference runner (e.g., via
`HloPjRtInterpreterReferenceMixin`):

```cpp
#include "third_party/tensorflow/compiler/xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "third_party/tensorflow/compiler/xla/tools/hlo_isolation/hlo_isolation_test_base.h"

class MyCustomPassIsolationTest : public xla::hlo_isolation::HloIsolationTestMixin<
    xla::HloPjRtInterpreterReferenceMixin<xla::HloPjRtTestBase>> {};

TEST_F(MyCustomPassIsolationTest, ChecksMyFusionSanity) {
  RunAndVerifyIsolationTest(my_failing_module);
}
```

## Sharded Execution (K8s/Slurm)

For large modules or heavy test matrices, you can partition execution across
multi-device clusters (such as Google Kubernetes Engine or Slurm) using the
sharding flags. Each `shard_index` deterministically runs an isolated subset of
the decomposed submodules. This allows for reproducible distributed verification
and targeted re-execution of failing partitions.

### Example: Kubernetes Job

Each test shard is executed as a separate Kubernetes pod using `completionMode:
Indexed`. The `JOB_COMPLETION_INDEX` is passed directly to the CLI's
`--shard_index` flag.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hlo-isolation-job
spec:
  completions: 50
  parallelism: 50
  completionMode: Indexed
  template:
    spec:
      containers:
      - name: test-runner
        image: gcr.io/my-project/hlo-isolation-tools:latest
        command: ["/bin/sh", "-c"]
        args:
        - |
          ./hlo_isolation_test \
            --num_shards=50 \
            --shard_index=$JOB_COMPLETION_INDEX \
            --hlo_file=/data/path/to/hlo.hlo
        volumeMounts:
        - name: hlo-data-volume
          mountPath: /data
      volumes:
      - name: hlo-data-volume
        csi:
          driver: gcsfuse.csi.storage.gke.io
          volumeAttributes:
            bucketName: my-xla-debug-bucket
```

## Key Capabilities

-   **Portability:** Decouples internal test wrappers from the standalone API,
    making it easy to debug HLO mismatches locally.
-   **Granularity:** Granular opcode and name filtering improves the debugging
    loop when interacting with massive HLO dumps.
-   **Extensibility:** Custom runner execution callbacks and data injectors
    (`make_fake_arguments_fn`) permit full customization for advanced
    verification workflows.

