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
#ifndef TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
#define TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/tpu/tpu_resources.h"
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// Contains request related info.
struct RequestInfo {
  tfrt::RCReference<tfrt::RequestContext> tfrt_request_context;
  std::unique_ptr<WorkQueueInterface> request_queue;
  std::function<void(std::function<void()>)> runner;
};

// Creates a `RequestInfo` given relative data.
StatusOr<std::unique_ptr<RequestInfo>> SetUpRequestContext(
    const GraphExecutionRunOptions& run_options,
    const SessionMetadata& model_metadata, tfrt::HostContext* host,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    tfrt::ResourceContext* resource_context,
    const FallbackState& fallback_state);

// Runs on a function given input/output and other info.
tensorflow::Status GraphExecutionRunOnFunction(
    const GraphExecutionOptions& options,
    const GraphExecutionRunOptions& run_options,
    absl::string_view signature_name, const tfrt::Function& func,
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const tensorflow::Tensor> captures,
    std::vector<tensorflow::Tensor>* outputs,
    tfrt::ResourceContext* resource_context, const Runtime& runtime,
    const FallbackState& fallback_state,
    tfrt::RequestDeadlineTracker& req_deadline_tracker);

// Creates a ResourceContext and populate it with per model resource from
// Runtime. If `tpu_target` is set to kTpurt, also call a special
// `AddTpuResources` function to populate TPU related resources for tpurt.
//
// TODO(b/178227859): Remove the need for the special handling for TPU here.
std::unique_ptr<tfrt::ResourceContext> CreateResourceContext(
    const Runtime& runtime, tfrt::tpu::TpuModelResource* tpu_model_resource,
    tensorflow::TfrtTpuInfraTarget tpu_target);

// Loads (if not yet) and runs a subgraph in a graph as per each request.
class GraphExecutor {
 public:
  using Options = GraphExecutionOptions;
  using RunOptions = GraphExecutionRunOptions;

  // The loading result of a `ClientGraph`.
  struct LoadedClientGraph {
    std::string name;
    tfrt::BefBuffer bef;
    tfrt::RCReference<tfrt::BEFFile> bef_file;
    std::unique_ptr<tfrt::ResourceContext> resource_context;
  };

  // A subgraph constructed by specifying input/output tensors.
  struct ClientGraph {
    // A unique name by joining all the input/output/target names.
    std::string name;
    // The feed nodes for the corresponding inputs, but they might not be in the
    // original order and if there are more than one original inputs mapped to
    // the same feed node, only one is picked here.
    tensorflow::GraphImportConfig::InputArrays input_nodes;
    // The fetch nodes for the outputs, which should be in the original order.
    std::vector<std::string> output_nodes;
    // The target nodes that should be run but not returned as outputs.
    std::vector<std::string> target_nodes;
  };

  // Creates a `GraphExecutor` given the args.
  static StatusOr<std::unique_ptr<GraphExecutor>> Create(
      Options options, const FallbackState& fallback_state,
      tfrt::tpu::TpuModelResource* tpu_model_resource,
      tensorflow::GraphDef graph_def);

  // Ctor. Public for `Create()`. Do not use directly.
  GraphExecutor(Options options, const FallbackState& fallback_state,
                tfrt::tpu::TpuModelResource* tpu_model_resource,
                std::unique_ptr<tensorflow::tfrt_stub::TfrtGraphExecutionState>
                    graph_execution_state)
      : options_(std::move(options)),
        fallback_state_(fallback_state),
        tpu_model_resource_(tpu_model_resource),
        graph_execution_state_(std::move(graph_execution_state)),
        req_deadline_tracker_(
            options_.runtime->core_runtime()->GetHostContext()) {}

  // Runs on the graph according to given input/output.
  tensorflow::Status Run(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names,
      std::vector<tensorflow::Tensor>* outputs);

  // Extends the current graph by `graph`.
  tensorflow::Status Extend(const GraphDef& graph);

  tensorflow::tfrt_stub::TfrtGraphExecutionState& graph_execution_state()
      const {
    return *graph_execution_state_;
  }

  // Compiles and returns a graph that is specified by `client_graph`.
  StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
  ImportAndCompileClientGraph(const GraphExecutor::ClientGraph& client_graph);

  // Returns the underlying runtime.
  const tensorflow::tfrt_stub::Runtime& runtime() const {
    DCHECK(options_.runtime);
    return *options_.runtime;
  }

 private:
  // A set of methods to load a client graph.
  StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>> LoadClientGraph(
      const GraphExecutor::ClientGraph& client_graph);
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
  ImportClientGraphToMlirModule(const GraphExecutor::ClientGraph& client_graph,
                                mlir::MLIRContext* context) const;
  StatusOr<tfrt::BefBuffer> CompileMlirModuleToBef(mlir::ModuleOp module) const;
  tensorflow::Status InitBef(tfrt::BEFFile* bef_file,
                             tfrt::ResourceContext* resource_context);

  // Returns a `LoadedClientGraph` given input/output tensor info. If there is
  // no existing one yet, creates one first.
  StatusOr<std::reference_wrapper<const GraphExecutor::LoadedClientGraph>>
  GetOrCreateLoadedClientGraph(
      absl::Span<const std::string> input_tensor_names,
      absl::Span<const tensorflow::DataType> input_tensor_dtypes,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names)
      TF_LOCKS_EXCLUDED(loaded_client_graphs_mu_);

  Options options_;
  std::reference_wrapper<const FallbackState> fallback_state_;
  tfrt::tpu::TpuModelResource* tpu_model_resource_;  // NOT owned.

  std::unique_ptr<tensorflow::tfrt_stub::TfrtGraphExecutionState>
      graph_execution_state_;

  tfrt::RequestDeadlineTracker req_deadline_tracker_;

  tensorflow::mutex loaded_client_graphs_mu_;
  // Caches `LoadedClientGraph` by the joined name.
  // For pointer stability of values in `absl::flat_hash_map<>`, additional
  // `std::unique_ptr<>` is necessary. (See https://abseil.io/tips/136.)
  absl::flat_hash_map<std::string /*joined_name*/,
                      std::unique_ptr<LoadedClientGraph>>
      loaded_client_graphs_ TF_GUARDED_BY(loaded_client_graphs_mu_);
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
