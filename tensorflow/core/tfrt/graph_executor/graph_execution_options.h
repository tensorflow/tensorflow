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

#include <optional>
#include <ostream>

#include "absl/types/optional.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"

namespace tensorflow {
namespace tfrt_stub {

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

  // Runtime configuration. Refer to tensorflow::tfrt_stub::Runtime class for
  // more details. It must not be nullptr;
  const tensorflow::tfrt_stub::Runtime* runtime = nullptr;

  // Model metadata used for monitoring and tracing.
  tensorflow::SessionMetadata model_metadata;

  // If true, for each client graph, the op costs of the first request will be
  // recorded and used to re-compile the client graph.
  // TODO(b/266251216): Maybe flip the default value or remote it.
  bool enable_online_cost_analysis = false;

  // If true, the MLRT interpreter will be used instead of the BEF executor.
  // This option is experimental.
  bool enable_mlrt = false;

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

  // TODO(b/239749833) Remove after b/239749833 is fixed.
  // If true, the input specs will be checked before running, and an error
  // will be logged upon mismatch.
  bool validate_input_specs_dry_run = false;

  // The thread pool used for this run. If it is nullptr, a default one set
  // in the tensorflow::tfrt_stub::Runtime will be used.
  tensorflow::tfrt_stub::WorkQueueInterface* work_queue = nullptr;

  // If true, just-in-time host compilation is disabled, and then if the
  // specified graph is not compiled, the execution will return an error.
  bool disable_compilation = false;
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
