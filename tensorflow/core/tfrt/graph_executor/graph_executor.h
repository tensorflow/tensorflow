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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tfrt/backend_compiler.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/graph_executor/executable_context.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/sync_resource_state.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/function.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/runtime/stream.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"
#include "tsl/lib/monitoring/sampler.h"
#include "tsl/platform/thread_annotations.h"
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

  tensorflow::CancellationManager cancellation_manager;
};

struct SymbolUids {
  std::string tf_symbol_uid;
  std::string tfrt_symbol_uid;
};

// Creates a `RequestInfo` given relative data.
// Note: `resource_context` is per-graph-executor and
// `client_graph_resource_context` is per-loaded-client-graph. See the comment
// above `GraphExecutor::resource_context_` about the todo to merge these two.
absl::StatusOr<std::unique_ptr<RequestInfo>> CreateRequestInfo(
    const GraphExecutionOptions& options,
    const GraphExecutionRunOptions& run_options,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    tfrt::ResourceContext* resource_context,
    tfrt::ResourceContext* client_graph_resource_context,
    OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array, FallbackState& fallback_state,
    const ProcessFunctionLibraryRuntime& process_function_library_runtime,
    CostRecorder* cost_recorder = nullptr);

// Runs on a function given input/output and other info.
// Note: `resource_context` is per-graph-executor and
// `client_graph_resource_context` is per-loaded-client-graph. See the comment
// above `GraphExecutor::resource_context_` about the todo to merge these two.
//
// TODO(chky): Refactor this function to take `LoadedClientGraph` instead of
// having a long list of parameters.
tensorflow::Status GraphExecutionRunOnFunction(
    const GraphExecutionOptions& options,
    const GraphExecutionRunOptions& run_options,
    absl::string_view signature_name, const SymbolUids& symbol_uids,
    const tfrt::Function* func, const mlrt::LoadedExecutable* loaded_executable,
    absl::Span<const tensorflow::Tensor> inputs,
    std::vector<tensorflow::Tensor>* outputs,
    tfrt::ResourceContext* resource_context,
    tfrt::ResourceContext* client_graph_resource_context,
    OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array, const Runtime& runtime,
    FallbackState& fallback_state,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime,
    tfrt::RequestDeadlineTracker* req_deadline_tracker,
    std::optional<StreamCallbackId> stream_callback_id,
    CostRecorder* cost_recorder = nullptr);

// Runs a MLRT function for executing tensorflow graphs.
tensorflow::Status RunMlrtFunction(
    mlrt::bc::Function function,
    const mlrt::LoadedExecutable& loaded_executable,
    const tsl::RCReference<tfrt::RequestContext>& request_context,
    tfrt::ConcurrentWorkQueue& work_queue,
    absl::Span<const tensorflow::Tensor> inputs,
    std::vector<tensorflow::Tensor>* outputs,
    SyncResourceState* sync_resource_state);

// Loads (if not yet) and runs a subgraph in a graph as per each request.
class GraphExecutor {
 public:
  using Options = GraphExecutionOptions;
  using RunOptions = GraphExecutionRunOptions;

  // The loading result of a `ClientGraph`.
  class LoadedClientGraph {
   public:
    LoadedClientGraph(std::string name, SymbolUids symbol_uids,
                      GraphExecutor* graph_executor,
                      std::unique_ptr<mlir::MLIRContext> mlir_context,
                      mlir::OwningOpRef<mlir::ModuleOp> tf_mlir_with_op_keys,
                      mlir::OwningOpRef<mlir::ModuleOp> tfrt_mlir,
                      std::shared_ptr<ExecutableContext> executable_context,
                      std::optional<StreamCallbackId> stream_callback_id,
                      bool is_restore, FunctionLibraryDefinition flib_def,
                      tsl::monitoring::SamplerCell* latency_sampler);

    // Returns this instance's CostRecorder if it is time to update costs,
    // else returns nullptr. Only allows one non-null return value at a time
    // in order to provide thread-safety. If do_recompilation becomes `true`,
    // then recompiles using updated costs occurs.
    CostRecorder* MaybeGetCostRecorder(absl::Time now, bool* do_recompilation);
    // Updates the op cost values in this `LoadedClientGraph` with records from
    // `cost_recorder`.
    Status UpdateCost(const CostRecorder& cost_recorder,
                      const Runtime& runtime);
    // Updates `cost_analysis_data_` to make it accurate for the next execution.
    // Assumes a cost update occurred this cycle.
    void UpdateCostAnalysisData(absl::Time now, bool do_recompilation);
    // Getters.
    std::shared_ptr<ExecutableContext> executable_context() const {
      tensorflow::mutex_lock lock(executable_context_mu_);
      return executable_context_;
    }
    absl::string_view name() const { return name_; }
    const SymbolUids& symbol_uids() const { return symbol_uids_; }

    OpKernelRunnerTable& runner_table() { return runner_table_; }
    tfd::FallbackResourceArray& resource_array() { return resource_array_; }
    SyncResourceState& sync_resource_state() { return sync_resource_state_; }

    std::optional<StreamCallbackId> stream_callback_id() const {
      return stream_callback_id_;
    }

    bool is_restore() const { return is_restore_; }

    const ProcessFunctionLibraryRuntime& process_function_library_runtime()
        const {
      return pflr_;
    }
    tsl::monitoring::SamplerCell* latency_sampler() { return latency_sampler_; }

   private:
    std::string name_;
    SymbolUids symbol_uids_;
    GraphExecutor* graph_executor_ = nullptr;

    // `mlir_context_` is declared here because the resources declared later may
    // hold references to the MLIR objects.
    std::unique_ptr<mlir::MLIRContext> mlir_context_;

    struct CostAnalysisData {
      mutable tensorflow::mutex mu;
      // Ensures only one GraphExecutor thread updates costs at a time.
      bool is_available TF_GUARDED_BY(mu) = false;
      // Maintains the book-keeping of op costs.
      std::unique_ptr<CostRecorder> cost_recorder;
      // For recompilation in MLRT, TFRT respectively.
      mlir::OwningOpRef<mlir::ModuleOp> tf_mlir_with_op_keys;
      mlir::OwningOpRef<mlir::ModuleOp> tfrt_mlir;
      // Start of current cost measurement cycle.
      absl::Time start_time TF_GUARDED_BY(mu) = absl::Now();
      // Cost recordings within the current measurement cycle.
      int num_cost_updates TF_GUARDED_BY(mu) = 0;
    };
    CostAnalysisData cost_analysis_data_;

    OpKernelRunnerTable runner_table_;
    tfd::FallbackResourceArray resource_array_;
    mutable tensorflow::mutex executable_context_mu_;
    // Can be updated if online cost analysis is enabled.
    std::shared_ptr<ExecutableContext> executable_context_
        TF_GUARDED_BY(executable_context_mu_);
    SyncResourceState sync_resource_state_;

    std::optional<StreamCallbackId> stream_callback_id_;
    bool is_restore_;
    FunctionLibraryDefinition flib_def_;
    ProcessFunctionLibraryRuntime pflr_;
    tsl::monitoring::SamplerCell* latency_sampler_;
  };

  // A subgraph constructed by specifying input/output tensors.
  struct ClientGraph {
    // The human-readable name for the graph, e.g. the signature_name in the
    // saved model.
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
  static absl::StatusOr<std::unique_ptr<GraphExecutor>> Create(
      Options options, std::unique_ptr<FallbackState> fallback_state,
      std::unique_ptr<tfrt::ResourceContext> resource_context,
      tensorflow::GraphDef graph_def,
      std::unique_ptr<mlrt::KernelRegistry> kernel_registry);

  // Ctor. Public for `Create()`. Do not use directly.
  GraphExecutor(Options options, std::unique_ptr<FallbackState> fallback_state,
                std::unique_ptr<tfrt::ResourceContext> resource_context,
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

  tfrt::ResourceContext& resource_context() { return *resource_context_; }

  const Options& options() const { return options_; }
  const FallbackState& fallback_state() const { return *fallback_state_; }
  FallbackState& fallback_state() { return *fallback_state_; }

  // Compiles graph for `graph_name` and runs any initializers.
  tensorflow::Status CompileGraph(
      const std::string& graph_name,
      absl::Span<const std::string> input_tensor_names,
      absl::Span<const tensorflow::DataType> input_tensor_dtypes,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names);

  const mlrt::KernelRegistry& kernel_registry() const {
    return *kernel_registry_;
  }

 private:
  // A set of methods to load a client graph.
  absl::StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
  LoadClientGraph(
      const GraphExecutor::ClientGraph& client_graph,
      tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs);
  absl::StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
  ImportAndCompileClientGraph(
      const GraphExecutor::ClientGraph& client_graph,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs);
  absl::StatusOr<
      std::pair<FunctionLibraryDefinition, mlir::OwningOpRef<mlir::ModuleOp>>>
  ImportClientGraphToMlirModule(const GraphExecutor::ClientGraph& client_graph,
                                mlir::MLIRContext* context) const;
  absl::StatusOr<tfrt::BefBuffer> CompileMlirModuleToBef(
      mlir::ModuleOp module) const;

  tensorflow::Status InitBef(
      LoadedClientGraph* loaded_client_graph,
      tensorflow::tfrt_stub::WorkQueueInterface* work_queue);

  tensorflow::Status InitBytecode(LoadedClientGraph* loaded_graph);

  // Returns a `LoadedClientGraph` given input/output tensor info. If there is
  // no existing one yet, creates one first.
  absl::StatusOr<std::reference_wrapper<GraphExecutor::LoadedClientGraph>>
  GetOrCreateLoadedClientGraph(
      const RunOptions& run_options,
      absl::Span<const std::string> input_tensor_names,
      absl::Span<const tensorflow::DataType> input_tensor_dtypes,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_tensor_names,
      tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
      absl::string_view graph_name = "",
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs = {})
      TF_LOCKS_EXCLUDED(loaded_client_graphs_mu_);

  Options options_;
  std::unique_ptr<FallbackState> fallback_state_;

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

  std::unique_ptr<tfrt::ResourceContext> resource_context_;

 protected:
  // For testing basic Cost Analysis functionality.
  absl::Duration simulated_duration_ = absl::ZeroDuration();
  tensorflow::mutex num_recompilations_mu_;
  int num_recompilations_ TF_GUARDED_BY(num_recompilations_mu_) = 0;
};

void RegisterMlirDialect(mlir::DialectRegistry& registry,
                         tensorflow::BackendCompiler* backend_compiler);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
