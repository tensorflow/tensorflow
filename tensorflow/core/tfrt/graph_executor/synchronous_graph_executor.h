/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_SYNCHRONOUS_GRAPH_EXECUTOR_H_
#define TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_SYNCHRONOUS_GRAPH_EXECUTOR_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/value.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// This class manages a GraphDef based "session" and allows clients to run
// graphs via the TFRT synchronous interpreter.
// This class is thread-safe.
class SynchronousGraphExecutor {
 public:
  // Creates and returns a SynchronousGraphExecutor for the given `graph`.
  static absl::StatusOr<std::unique_ptr<SynchronousGraphExecutor>> Create(
      const tensorflow::GraphDef& graph);

  // Runs the graph identified by `graph_name` using the input `inputs` and
  // stores the output of the execution in `outputs`. It is the client's
  // responsibility to ensure `graph_name` corresponds to logically different
  // graphs, since this name is used to lookup compiled graphs in the cache. The
  // graph is run synchronously with the TFRT interpreter.
  absl::Status Run(const std::string& graph_name,
                   absl::Span<tfrt::Value*> input_values,
                   absl::Span<const std::string> input_names,
                   absl::Span<const tensorflow::DataType> input_dtypes,
                   absl::Span<const std::string> output_tensor_names,
                   absl::Span<const std::string> target_tensor_names,
                   absl::Span<tfrt::Value*> outputs);

  // Returns the TFRT host context for allocating tensors.
  // TODO(rohitju): This should ideally not be exposed to the client.
  tfrt::HostContext* host_context() {
    return graph_executor_->runtime().core_runtime()->GetHostContext();
  }

 private:
  SynchronousGraphExecutor(
      std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime,
      std::unique_ptr<tensorflow::tfrt_stub::FallbackState> fallback_state,
      std::unique_ptr<tensorflow::tfrt_stub::GraphExecutor> graph_executor)
      : runtime_(std::move(runtime)),
        fallback_state_(std::move(fallback_state)),
        graph_executor_(std::move(graph_executor)) {}

  // Various state required to use the graph executor.
  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime_;
  std::unique_ptr<tensorflow::tfrt_stub::FallbackState> fallback_state_;
  std::unique_ptr<tensorflow::tfrt_stub::GraphExecutor> graph_executor_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_SYNCHRONOUS_GRAPH_EXECUTOR_H_
