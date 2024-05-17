/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTION_OPTIONS_H_
#define TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTION_OPTIONS_H_

#include <functional>
#include <optional>
#include <ostream>
#include <string>

#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"

namespace tensorflow {
namespace tfrt_stub {

class Runtime;

// General options for graph execution.
struct GraphExecutionOptions {
  explicit GraphExecutionOptions(const tensorflow::tfrt_stub::Runtime* rt)
      : runtime(rt) {
    DCHECK(runtime);
  }

  // If true, when creating an optimized subgraph, Placer and Grappler will
  // also run on the functions.
  bool run_placer_grappler_on_functions = false;

  // If true, the function optimizer in the grappler will be enabled, and
  // optimizations like function inlining will be applied.
  bool enable_grappler_function_optimizer = false;

  // Whether to enable TFRT GPU.
  bool enable_tfrt_gpu = false;

  // The number of virtual GPUs to create on a physical GPU.
  int tfrt_gpu_parallelism = 1;

  // if not zero, override the reserved memory space for gpu system.
  int gpu_system_memory_size_in_mb = 0;

  // Whether to use gpurt.compile_and_execute for GPU.
  // TODO(b/294895431): Remove the flag and default to the fused op.
  bool tfrt_use_fused_gpu_op = false;

  // Runtime configuration. Refer to tensorflow::tfrt_stub::Runtime class for
  // more details. It must not be nullptr;
  const tensorflow::tfrt_stub::Runtime* runtime = nullptr;

  // Model metadata used for monitoring and tracing.
  tensorflow::SessionMetadata model_metadata;

  // The model-specific runtime configurations.
  tensorflow::tfrt_stub::RuntimeConfig runtime_config;

  // TODO(b/266251216): Maybe flip the default value.
  [[deprecated(
      "Use CostAnalysisOptions's `CostAnalysisOptions::ONCE` instead")]] bool
      enable_online_cost_analysis = false;

  // Determines how often op costs are recorded, and how often these costs
  // are used to re-compile the executable. Note to users: CostAnalysisOptions
  // is overwritten when `enable_online_cost_analysis = true`.
  struct CostAnalysisOptions {
    enum CostAnalysisVersion {
      kDisabled,
      kOnce,  // Cost recording and recompilation occurs on the first run only.
      kPeriodic,  // This is experimental.
    };
    CostAnalysisVersion version = kDisabled;

    // Time between resets in Op cost estimates. Upon reset, the executable
    // will be recompiled.
    // However, a reset always occurs after the first execution.
    absl::Duration reset_interval = absl::ZeroDuration();

    // Number of times to record costs before resetting Op cost estimates.
    // However, a reset always occurs after the first execution.
    int updates_per_interval = 1;
  };

  CostAnalysisOptions cost_analysis_options;

  // If true, the MLRT interpreter will be used instead of the BEF executor.
  // This option is experimental.
  bool enable_mlrt = false;

  // If true, the IFRT will be used instead of the TPU Runner.
  // This option is experimental.
  bool use_ifrt = false;

  tensorflow::TfrtCompileOptions compile_options;
};

std::ostream& operator<<(std::ostream& os,
                         const GraphExecutionOptions& options);

// Per-request options for graph execution.
struct GraphExecutionRunOptions {
  std::optional<std::chrono::system_clock::time_point> deadline;

  // Priority of the request. Larger number means higher priority.
  int priority = 0;

  // If true, the input specs will be checked before running, and an error
  // will be raised upon mismatch.
  bool validate_input_specs = false;

  // TODO(b/279197040) Remove after b/279197040 is fixed.
  // If true, the input specs will be checked before running, and an error
  // will be logged upon mismatch.
  bool validate_input_specs_dry_run = false;

  // The thread pool used for this run. If it is nullptr, a default one set
  // in the tensorflow::tfrt_stub::Runtime will be used.
  tensorflow::tfrt_stub::WorkQueueInterface* work_queue = nullptr;

  // If true, just-in-time host compilation is disabled, and then if the
  // specified graph is not compiled, the execution will return an error.
  bool disable_compilation = false;

  std::function<void(absl::flat_hash_map<std::string, tensorflow::Tensor>)>
      streamed_output_callback;

  // The optional name for debugging purposes. If empty, the runtime will pick a
  // name e.g. the joined string of input names and output names.
  std::string name;
};

// Creates the default `SessionOptions` from a `GraphExecutionOptions`.
// The created `SessionOptions` contains the Grappler configs.
tensorflow::SessionOptions CreateDefaultSessionOptions(
    const GraphExecutionOptions& options);

// Updates TPU target to fallback if bridge uncompatible, otherwise TPU runtime.
void UpdateTpuTargetByBridgeCompatibility(
    tensorflow::tfrt_stub::GraphExecutionOptions& options,
    const tensorflow::GraphDef& graph_def);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTION_OPTIONS_H_
