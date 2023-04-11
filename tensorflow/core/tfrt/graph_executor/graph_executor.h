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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "learning/brain/experimental/tfrt/native_lowering/kernels/sync_context.h"
#include "learning/infra/mira/mlrt/bytecode/bytecode.h"
#include "learning/infra/mira/mlrt/bytecode/executable.h"
#include "learning/infra/mira/mlrt/interpreter/context.h"
#include "absl/base/call_once.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"
#include "tensorflow/tsl/platform/thread_annotations.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// Contains request related info.
struct RequestInfo {
  tfrt::RCReference<tfrt::RequestContext> tfrt_request_context;
  // If this request needs to create a new queue, it is stored here. Otherwise,
  // it can be nullptr.
  std::unique_ptr<WorkQueueInterface> request_queue_owner;
  // The inter-op thread pool to be used for this request, and it must not be
  // nullptr. If `request_queue_owner` is not nullptr, then `request_queue` is
  // the raw pointer inside `request_queue_owner`.
  WorkQueueInterface* request_queue = nullptr;
  // The task runner used by tensorflow::OpKernel.
  std::function<void(std::function<void()>)> runner;
};

// Creates a `RequestInfo` given relative data.
// Note: `resource_context` is per-graph-executor and
// `client_graph_resource_context` is per-loaded-client-graph. See the comment
// above `GraphExecutor::resource_context_` about the todo to merge these two.
StatusOr<std::unique_ptr<RequestInfo>> CreateRequestInfo(
    const GraphExecutionRunOptions& run_options,
    const SessionMetadata& model_metadata, const Runtime& runtime,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    tfrt::ResourceContext* resource_context,
    tfrt::ResourceContext* client_graph_resource_context,
    OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array,
    const FallbackState& fallback_state, CostRecorder* cost_recorder = nullptr);

// Runs on a function given input/output and other info.
// Note: `resource_context` is per-graph-executor and
// `client_graph_resource_context` is per-loaded-client-graph. See the comment
// above `GraphExecutor::resource_context_` about the todo to merge these two.
tensorflow::Status GraphExecutionRunOnFunction(
    const GraphExecutionOptions& options,
    const GraphExecutionRunOptions& run_options,
    absl::string_view signature_name, const tfrt::Function* func,
    const mlrt::LoadedExecutable* loaded_executable,
    absl::Span<const tensorflow::Tensor> inputs,
    std::vector<tensorflow::Tensor>* outputs,
    tfrt::ResourceContext* resource_context,
    tfrt::ResourceContext* client_graph_resource_context,
    OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array, const Runtime& runtime,
    const FallbackState& fallback_state,
    tfrt::RequestDeadlineTracker* req_deadline_tracker,
    CostRecorder* cost_recorder = nullptr);

// Compiles MLIR in TF executor dialect to MLRT bytecode executable.
StatusOr<mlrt::bc::Buffer> CompileMlirModuleToByteCode(
    const TfrtCompileOptions& options, mlir::ModuleOp module,
    const CostRecorder* cost_recorder = nullptr,
    mlir::OwningOpRef<mlir::ModuleOp>* module_with_op_keys = nullptr);

// Runs a MLRT function for executing tensorflow graphs.
tensorflow::Status RunMlrtFunction(
    mlrt::bc::Function function,
    const mlrt::LoadedExecutable& loaded_executable,
    const tsl::RCReference<tfrt::RequestContext>& request_context,
    tfrt::ConcurrentWorkQueue& work_queue,
    absl::Span<const tensorflow::Tensor> inputs,
    std::vector<tensorflow::Tensor>* outputs);

// Loads (if not yet) and runs a subgraph in a graph as per each request.
class GraphExecutor {
 public:
  using Options = GraphExecutionOptions;
  using RunOptions = GraphExecutionRunOptions;

  // Stores executable-related data.
  struct ExecutableContext {
    ExecutableContext(
        mlrt::bc::Buffer bytecode_buffer,
        std::unique_ptr<mlrt::LoadedExecutable> bytecode_executable)
        : bytecode_buffer(std::move(bytecode_buffer)),
          bytecode_executable(std::move(bytecode_executable)) {}

    ExecutableContext(tfrt::BefBuffer bef,
                      tfrt::RCReference<tfrt::BEFFile> bef_file)
        : bef(std::move(bef)), bef_file(std::move(bef_file)) {}

    bool IsForMlrt() const { return bytecode_executable != nullptr; }

    // Only one set of values will be filled.

    // For the MLRT path.
    mlrt::bc::Buffer bytecode_buffer;
    std::unique_ptr<mlrt::LoadedExecutable> bytecode_executable = nullptr;

    // For the TFRT path.
    tfrt::BefBuffer bef;
    tfrt::RCReference<tfrt::BEFFile> bef_file;

    // There are some resources that need re-creating when the executable is
    // re-created, so a resource context is stored along with the executable.
    // This resource context is meant to be passed to the op kernels for their
    // references. See the comment above `GraphExecutor::resource_context_`
    // about the todo to merge that resource context with this one.
    tfrt::ResourceContext resource_context;
  };

  // The loading result of a `ClientGraph`.
  class LoadedClientGraph {
   public:
    LoadedClientGraph(std::string name, GraphExecutor* graph_executor,
                      std::unique_ptr<mlir::MLIRContext> mlir_context,
                      mlir::OwningOpRef<mlir::ModuleOp> tf_mlir_with_op_keys,
                      mlir::OwningOpRef<mlir::ModuleOp> tfrt_mlir,
                      std::shared_ptr<ExecutableContext> executable_context)
        : name_(std::move(name)),
          graph_executor_(graph_executor),
          mlir_context_(std::move(mlir_context)),
          tf_mlir_with_op_keys_(std::move(tf_mlir_with_op_keys)),
          tfrt_mlir_(std::move(tfrt_mlir)),
          executable_context_(std::move(executable_context)) {}

    // Returns a `CostRecorder` if none has been created before for this
    // `LoadedClientGraph`.
    std::unique_ptr<CostRecorder> MaybeCreateCostRecorder() const;

    // Updates the op cost values in this `LoadedClientGraph` with records from
    // `cost_recorder`.
    Status UpdateCost(const CostRecorder& cost_recorder,
                      const Runtime& runtime);

    // Getters.
    std::shared_ptr<ExecutableContext> executable_context() const {
      tensorflow::mutex_lock lock(executable_context_mu_);
      return executable_context_;
    }
    absl::string_view name() const { return name_; }

    OpKernelRunnerTable& runner_table() { return runner_table_; }
    tfd::FallbackResourceArray& resource_array() { return resource_array_; }

   private:
    std::string name_;
    GraphExecutor* graph_executor_ = nullptr;
    OpKernelRunnerTable runner_table_;
    tfd::FallbackResourceArray resource_array_;
    std::unique_ptr<mlir::MLIRContext> mlir_context_;
    // Thread-safety resulted from `create_cost_recorder_once_`.
    mlir::OwningOpRef<mlir::ModuleOp>
        tf_mlir_with_op_keys_;                     // For recompilation in MLRT.
    mlir::OwningOpRef<mlir::ModuleOp> tfrt_mlir_;  // For recompilation in TFRT.
    mutable tensorflow::mutex executable_context_mu_;
    // Can be updated if online cost analysis is enabled.
    std::shared_ptr<ExecutableContext> executable_context_
        TF_GUARDED_BY(executable_context_mu_);
    mutable absl::once_flag create_cost_recorder_once_;
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
      tensorflow::GraphDef graph_def,
      std::unique_ptr<mlrt::KernelRegistry> kernel_registry);

  // Ctor. Public for `Create()`. Do not use directly.
  GraphExecutor(Options options, const FallbackState& fallback_state,
                std::unique_ptr<tensorflow::tfrt_stub::TfrtGraphExecutionState>
                    graph_execution_state,
                std::unique_ptr<mlrt::KernelRegistry> kernel_registry);

  // Runs on the graph according to given input/output.
  tensorflow::Status Run(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names,
      std::vector<tensorflow::Tensor>* outputs);

  // Runs the graph identified by `graph_name` using the input `inputs` and
  // stores the output of the execution in `outputs`. It is the client's
  // responsibility to ensure `graph_name` corresponds to logically different
  // graphs, since this name is used to lookup compiled graphs in the cache. The
  // graph is run synchronously with the TFRT interpreter.
  tensorflow::Status RunWithSyncInterpreter(
      const std::string& graph_name, absl::Span<mlrt::Value> input_values,
      absl::Span<const std::string> input_names,
      absl::Span<const tensorflow::DataType> input_dtypes,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names,
      absl::Span<mlrt::Value> outputs);

  // Extends the current graph by `graph`.
  tensorflow::Status Extend(const GraphDef& graph);

  tensorflow::tfrt_stub::TfrtGraphExecutionState& graph_execution_state()
      const {
    return *graph_execution_state_;
  }

  // Returns the underlying runtime.
  const tensorflow::tfrt_stub::Runtime& runtime() const {
    DCHECK(options_.runtime);
    return *options_.runtime;
  }

  tfrt::ResourceContext& resource_context() { return resource_context_; }

 private:
  // A set of methods to load a client graph.
  StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>> LoadClientGraph(
      const GraphExecutor::ClientGraph& client_graph,
      tensorflow::tfrt_stub::WorkQueueInterface* work_queue);
  StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
  ImportAndCompileClientGraph(const GraphExecutor::ClientGraph& client_graph);
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
  ImportClientGraphToMlirModule(const GraphExecutor::ClientGraph& client_graph,
                                mlir::MLIRContext* context) const;
  StatusOr<tfrt::BefBuffer> CompileMlirModuleToBef(mlir::ModuleOp module) const;

  tensorflow::Status InitBef(
      LoadedClientGraph* loaded_client_graph,
      tensorflow::tfrt_stub::WorkQueueInterface* work_queue);

  tensorflow::Status InitBytecode(LoadedClientGraph* loaded_graph);

  // Returns a `LoadedClientGraph` given input/output tensor info. If there is
  // no existing one yet, creates one first.
  StatusOr<std::reference_wrapper<GraphExecutor::LoadedClientGraph>>
  GetOrCreateLoadedClientGraph(
      const RunOptions& run_options,
      absl::Span<const std::string> input_tensor_names,
      absl::Span<const tensorflow::DataType> input_tensor_dtypes,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names,
      tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
      std::optional<const std::string> graph_name = std::nullopt)
      TF_LOCKS_EXCLUDED(loaded_client_graphs_mu_);

  Options options_;
  std::reference_wrapper<const FallbackState> fallback_state_;

  // TODO(juanlishen): Maybe remove this per-model resource context and delegate
  // to the one in each `LoadedClientGraph` instead. Ideally, only one resource
  // context should be passed to the op kernels.
  tfrt::ResourceContext resource_context_;

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

  std::unique_ptr<mlrt::KernelRegistry> kernel_registry_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
