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
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/update_op_cost_in_tfrt_mlir.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_utils.h"
#include "tensorflow/core/tfrt/common/metrics.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/graph_executor/executable_context.h"
#include "tensorflow/core/tfrt/graph_executor/export_mlir.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/sync_resource_state.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/function.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/runtime/step_id.h"
#include "tensorflow/core/tfrt/runtime/stream.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/stubs/tfrt_native_lowering_stub.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tsl/concurrency/async_value_ref.h"
#include "tsl/lib/monitoring/sampler.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/refcount.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

constexpr char kDeadlineExceededMessage[] = "Deadline exceeded.";
constexpr char kTensorNameJoiningDelimiter[] = "-";
constexpr char kArgumentTypeJoiningDelimiter[] = "^";
constexpr char kFallbackInitFunction[] = "_tfrt_fallback_init";
constexpr char kResourceInitFunction[] = "_tfrt_resource_init";

StepId GetNextStepId() {
  static StepIdGenerator gen;
  return gen.GetNextStepId();
}

auto* graph_executor_mode = monitoring::Gauge<std::string, 2>::New(
    "/tfrt/graph_executor/mode",
    "Record the total number of imported savedmodel using different graph "
    "executor modes (BEF vs MLRT interpreter)",
    "model_name", "model_version");

}  // namespace

tensorflow::Status RunMlrtFunction(
    mlrt::bc::Function function,
    const mlrt::LoadedExecutable& loaded_executable,
    const tsl::RCReference<tfrt::RequestContext>& request_context,
    tfrt::ConcurrentWorkQueue& work_queue,
    absl::Span<const tensorflow::Tensor> inputs,
    std::vector<tensorflow::Tensor>* outputs,
    SyncResourceState* sync_resource_state) {
  DCHECK(function);
  const auto* fallback_request_state =
      request_context->GetDataIfExists<tfd::KernelFallbackCompatRequestState>();
  DCHECK(fallback_request_state);

  mlrt::ExecutionContext execution_context(&loaded_executable);
  execution_context.set_work_queue(&work_queue);

  // Set up tfrt::SyncContext which is used for vrooml only.
  //
  // TODO(chky, rohitju): Unify tfrt::SyncContext with tf_mlrt::Context.
  tfrt::ExecutionContext exec_ctx(request_context);
  AddSyncContext(execution_context, *request_context->host(),
                 sync_resource_state);

  // Set up tf_mlrt::Context which is used for executing tensorflow::OpKernel.
  execution_context.AddUserContext(std::make_unique<tf_mlrt::Context>(
      fallback_request_state, request_context->resource_context(),
      request_context->cancellation_context().get()));

  absl::InlinedVector<mlrt::Value, 4> mlrt_inputs;
  mlrt_inputs.reserve(inputs.size());

  for (const auto& input : inputs) {
    mlrt_inputs.emplace_back(FallbackTensor(input));
  }

  absl::InlinedVector<mlrt::Value, 4> mlrt_outputs(
      function.output_regs().size());

  // Set up exit handler. We are using tsl::AsyncValue here because we need to
  // use ConcurrentWorkQueue::Await() to wait for the execution.
  // ConcurrentWorkQueue::Await() may be implemented in a special way instead of
  // blocking, e.g. tfrt::SingleThreadedWorkQueue.
  tsl::RCReference<tsl::AsyncValue> chain =
      tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
  execution_context.set_exit_handler(
      [chain = chain.get()]() { chain->SetStateConcrete(); });

  execution_context.CallByMove(function, absl::MakeSpan(mlrt_inputs),
                               absl::MakeSpan(mlrt_outputs));

  // TODO(chky): Set up cancellation.

  work_queue.AddTask(
      [&execution_context]() { mlrt::Execute(execution_context); });

  work_queue.Await(chain);

  if (!execution_context.status().ok()) {
    outputs->resize(mlrt_outputs.size(), tensorflow::Tensor());
    return execution_context.status();
  }

  for (auto& mlrt_output : mlrt_outputs) {
    DCHECK(mlrt_output.HasValue());
    outputs->push_back(std::move(mlrt_output.Get<FallbackTensor>().tensor()));
  }

  return tensorflow::OkStatus();
}

absl::StatusOr<std::unique_ptr<RequestInfo>> CreateRequestInfo(
    const GraphExecutionOptions& options,
    const GraphExecutionRunOptions& run_options,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    tfrt::ResourceContext* resource_context,
    tfrt::ResourceContext* client_graph_resource_context,
    OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array,
    tensorflow::tfrt_stub::FallbackState& fallback_state,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime,
    CostRecorder* cost_recorder) {
  auto request_info = std::make_unique<RequestInfo>();

  DCHECK(options.runtime);
  const Runtime& runtime = *options.runtime;

  // Set the request queue.
  // TODO(tfrt-devs): Consider using an ID unique within each model to reduce
  // contention.
  int64_t request_id = 0;
  if (work_queue != nullptr) {
    // If the user provides a work_queue, we use it for inter-op tasks.
    request_id = work_queue->id();
    // If the user does not provide a valid id, we need to generate one.
    if (request_id == 0) request_id = GetNextStepId().id;
    request_info->request_queue = work_queue;
  } else {
    request_id = GetNextStepId().id;
    // Otherwise we use the global queue in `runtime`.
    TF_ASSIGN_OR_RETURN(request_info->request_queue_owner,
                        runtime.CreateRequestQueue(request_id));
    request_info->request_queue = request_info->request_queue_owner.get();
  }
  auto* request_queue = request_info->request_queue;

  // Create a `tensorflow::Executor::Args::Runner` with the above request queue.
  request_info->runner = [request_queue](std::function<void()> f) {
    request_queue->AddTask(std::move(f));
  };

  // Create a request context builder.
  tfrt::RequestContextBuilder request_context_builder(
      runtime.core_runtime()->GetHostContext(), resource_context, request_id);

  // Set up the request contexts in the builder.
  // Note: if the intra-op thread pool from the request queue is null, the
  // thread pool in `tensorflow::Device` will be used.
  DCHECK(runner_table);
  DCHECK(resource_array);
  auto& fallback_request_state =
      request_context_builder.context_data()
          .emplace<tfd::KernelFallbackCompatRequestState>(
              &request_info->runner, &fallback_state.device_manager(),
              request_context_builder.id(), runner_table, resource_array,
              request_queue->GetIntraOpThreadPool(), options.model_metadata,
              &process_function_library_runtime);

  fallback_request_state.set_cost_recorder(cost_recorder);
  fallback_request_state.set_client_graph_resource_context(
      client_graph_resource_context);
  fallback_request_state.set_runtime_config(&options.runtime_config);
  fallback_request_state.set_cancellation_manager(
      &request_info->cancellation_manager);

  // Set priority in the builder.
  tfrt::RequestOptions request_options;
  request_options.priority = run_options.priority;
  request_context_builder.set_request_options(request_options);
  // Create the request context with the builder.
  auto expected_req_ctx = std::move(request_context_builder).build();
  if (!expected_req_ctx) {
    return tensorflow::errors::Internal(
        tfrt::StrCat(expected_req_ctx.takeError()));
  }
  request_info->tfrt_request_context = std::move(expected_req_ctx.get());

  return request_info;
}

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
    CostRecorder* cost_recorder) {
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(options, run_options, run_options.work_queue,
                        resource_context, client_graph_resource_context,
                        runner_table, resource_array, fallback_state,
                        process_function_library_runtime, cost_recorder));

  int64_t request_id = request_info->tfrt_request_context->id();
  // The top level traceme root for this request. The thread pool used later
  // will add TraceMeProducer and TraceMeConsumer to connect async tasks.
  tsl::profiler::TraceMe traceme(
      [request_id, signature_name, &options, symbol_uids] {
        return tensorflow::profiler::TraceMeEncode(
            "TfrtModelRun",
            {{"_r", 1},
             {"id", request_id},
             {"signature", signature_name},
             {"model_id", absl::StrCat(options.model_metadata.name(), ":",
                                       options.model_metadata.version())},
             {"tf_symbol_uid", symbol_uids.tf_symbol_uid},
             {"tfrt_symbol_uid", symbol_uids.tfrt_symbol_uid}});
      });

  // Only configure timer when the deadline is set.
  if (run_options.deadline.has_value()) {
    auto deadline = run_options.deadline.value();
    if (absl::ToChronoTime(absl::Now()) > deadline) {
      return tensorflow::errors::DeadlineExceeded(kDeadlineExceededMessage);
    }
    if (req_deadline_tracker == nullptr) {
      return tensorflow::errors::InvalidArgument(
          "req_deadline_tracker must be non-null");
    }
    req_deadline_tracker->CancelRequestOnDeadline(
        deadline, request_info->tfrt_request_context);
  }

  ScopedStreamCallback scoped_stream_callback;

  if (run_options.streamed_output_callback && !stream_callback_id.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Signature '", signature_name, "' does not support streaming."));
  }

  if (stream_callback_id.has_value()) {
    if (!run_options.streamed_output_callback) {
      return absl::InvalidArgumentError(
          absl::StrCat("Signature '", signature_name,
                       "' contains streaming ops but is called using Predict "
                       "without the streamed callback."));
    }
  }

  if (run_options.streamed_output_callback) {
    if (!stream_callback_id.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Signature ", signature_name, " does not support streaming."));
    }

    auto streamed_output_callback = run_options.streamed_output_callback;

    TF_ASSIGN_OR_RETURN(
        scoped_stream_callback,
        GetGlobalStreamCallbackRegistry().Register(
            options.model_metadata.name(), *stream_callback_id,
            StepId(request_id), std::move(streamed_output_callback)));
  }

  if (loaded_executable) {
    auto function = loaded_executable->GetFunction(signature_name);
    if (!function) {
      return errors::InvalidArgument(absl::StrCat(
          "Function not found in MLRT executable: ", signature_name));
    }

    return RunMlrtFunction(function, *loaded_executable,
                           request_info->tfrt_request_context,
                           *request_info->request_queue, inputs, outputs,
                           /*sync_resource_state=*/nullptr);
  }

  DCHECK(func);

  tfrt::ExecutionContext exec_ctx{request_info->tfrt_request_context};
  if (run_options.work_queue) {
    // TODO(b/198671794): Avoid creating `request_queue` when the `work_queue`
    // in `run_options` is specified.
    exec_ctx.set_work_queue(run_options.work_queue);
  } else if (request_info->request_queue) {
    exec_ctx.set_work_queue(request_info->request_queue);
  } else {
    exec_ctx.set_work_queue(runtime.work_queue());
  }

  llvm::SmallVector<tfrt::AsyncValue*, 4> arguments;
  auto cleanup = tensorflow::gtl::MakeCleanup([&]() {
    for (auto* argument : arguments) argument->DropRef();
  });

  // The first argument is a chain for side-effects. Since SavedModel::Run()
  // only returns when side-effects are visible, we can use a ready chain here.
  arguments.push_back(tfrt::GetReadyChain().release());

  for (const auto& input : inputs) {
    arguments.push_back(
        tfrt::MakeAvailableAsyncValueRef<FallbackTensor>(input).release());
  }

  if (arguments.size() != func->argument_types().size())
    return tensorflow::errors::Internal("incorrect number of inputs.");

  llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 4> chain_and_results;
  chain_and_results.resize(func->result_types().size());

  // Hand over the execution to thread pool.
  std::array<tfrt::RCReference<tfrt::AsyncValue>, 1> executed = {
      EnqueueWork(exec_ctx, [&]() -> tfrt::Chain {
        func->Execute(exec_ctx, arguments, chain_and_results);
        return {};
      })};

  // Wait for the function execution before checking chain and results.
  exec_ctx.work_queue().Await(executed);

  // Wait for all results including the side-effect chain. This ensures that all
  // side-effects are visible when SavedModel::Run() returns.
  exec_ctx.work_queue().Await(chain_and_results);

  DCHECK(!chain_and_results.empty());

  tfrt::RCReference<tfrt::AsyncValue>& chain = chain_and_results[0];
  auto results = llvm::drop_begin(chain_and_results, 1);

  tensorflow::StatusGroup status_group;

  if (chain->IsError()) {
    status_group.Update(chain->GetError());
  }

  for (tfrt::RCReference<tfrt::AsyncValue>& result : results) {
    DCHECK(result->IsAvailable());

    if (result->IsError()) {
      status_group.Update(result->GetError());
      outputs->push_back(tensorflow::Tensor());
      continue;
    }

    // The result must be a host tensor. This is guaranteed as the compiler
    // will insert necessary device transfer operations in the graph.
    DCHECK(result->IsType<FallbackTensor>());
    const auto& host_tensor = result->get<FallbackTensor>().tensor();
    // Make a copy of tensor here as the different result AsyncValues might
    // point to the same underlying tensor.
    outputs->push_back(host_tensor);
  }

  // Check if error is due to cancellation.
  // TODO(tfrt-devs): report cancellation reason from runtime.
  if (request_info->tfrt_request_context->IsCancelled()) {
    // Currently a request can only be cancelled by an expired timer.
    return tensorflow::errors::DeadlineExceeded(kDeadlineExceededMessage);
  }

  return status_group.as_summary_status();
}

GraphExecutor::GraphExecutor(
    Options options, std::unique_ptr<FallbackState> fallback_state,
    std::unique_ptr<tfrt::ResourceContext> resource_context,
    std::unique_ptr<tensorflow::tfrt_stub::TfrtGraphExecutionState>
        graph_execution_state,
    std::unique_ptr<mlrt::KernelRegistry> kernel_registry)
    : options_(std::move(options)),
      fallback_state_(std::move(fallback_state)),
      graph_execution_state_(std::move(graph_execution_state)),
      req_deadline_tracker_(options_.runtime->core_runtime()->GetHostContext()),
      kernel_registry_(std::move(kernel_registry)),
      resource_context_(std::move(resource_context)) {
  DCHECK(resource_context_);
  SetSessionCreatedMetric();
}

absl::StatusOr<std::unique_ptr<GraphExecutor>> GraphExecutor::Create(
    Options options, std::unique_ptr<FallbackState> fallback_state,
    std::unique_ptr<tfrt::ResourceContext> resource_context,
    tensorflow::GraphDef graph_def,
    std::unique_ptr<mlrt::KernelRegistry> kernel_registry) {
  if (options.runtime == nullptr) {
    return errors::InvalidArgument("options.runtime must be non-null ");
  }
  if (options.enable_online_cost_analysis) {
    // Overrides cost_analysis_options.
    options.cost_analysis_options.version = Options::CostAnalysisOptions::kOnce;
  }
  TfrtGraphExecutionState::Options graph_execution_state_options;
  graph_execution_state_options.run_placer_grappler_on_functions =
      options.run_placer_grappler_on_functions;

  options.compile_options.fuse_get_resource_ops_in_hoisting =
      !options.enable_mlrt;
  graph_executor_mode
      ->GetCell(options.model_metadata.name(),
                absl::StrCat(options.model_metadata.version()))
      ->Set(options.enable_mlrt ? "mlrt" : "bef");
  TF_ASSIGN_OR_RETURN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(graph_execution_state_options,
                                      std::move(graph_def), *fallback_state));
  return std::make_unique<GraphExecutor>(
      std::move(options), std::move(fallback_state),
      std::move(resource_context), std::move(graph_execution_state),
      std::move(kernel_registry));
}

namespace {

// Sort the strings in `names` and store the results in `sorted_names`. In
// addition, the original index in `names` for the item `sorted_names[i]` is
// stored in `original_indices[i]`.
void CreateSortedNamesAndOriginalIndices(absl::Span<const std::string> names,
                                         std::vector<std::string>& sorted_names,
                                         std::vector<int>& original_indices) {
  DCHECK(sorted_names.empty());
  DCHECK(original_indices.empty());

  // Generate indices.
  original_indices.resize(names.size());
  std::iota(original_indices.begin(), original_indices.end(), 0);

  // Sort indices by comparing the corresponding names.
  std::sort(original_indices.begin(), original_indices.end(),
            [&](int x, int y) { return names[x] < names[y]; });

  // Use sorted indices to generate sorted names.
  sorted_names.reserve(names.size());
  for (int original_index : original_indices) {
    DCHECK_LT(original_index, names.size());
    sorted_names.push_back(names[original_index]);
  }
}

}  // namespace

tensorflow::Status GraphExecutor::Run(
    const RunOptions& run_options,
    absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
    absl::Span<const std::string> output_tensor_names,
    absl::Span<const std::string> target_tensor_names,
    std::vector<tensorflow::Tensor>* outputs) {
  // TODO(b/192498110): Validate input type.

  // Sort the input/output names to have a stable order, so that the
  // `joined_name`, which is used as the cache key, will be the same as long as
  // the same set of inputs/outputs are specified.
  std::vector<std::string> input_names;
  input_names.reserve(inputs.size());
  for (const auto& p : inputs) input_names.push_back(p.first);
  std::vector<std::string> sorted_input_names;
  std::vector<int> input_original_indices;
  CreateSortedNamesAndOriginalIndices(input_names, sorted_input_names,
                                      input_original_indices);
  // We also need to create sorted input dtypes as they are needed for the
  // compilation.
  std::vector<tensorflow::DataType> sorted_input_dtypes;
  sorted_input_dtypes.reserve(inputs.size());
  for (int original_index : input_original_indices) {
    sorted_input_dtypes.push_back(inputs.at(original_index).second.dtype());
  }

  std::vector<std::string> sorted_output_names;
  std::vector<int> output_original_indices;
  CreateSortedNamesAndOriginalIndices(output_tensor_names, sorted_output_names,
                                      output_original_indices);

  // For target node names, we only need to sort them. The original indices are
  // not needed.
  std::vector<std::string> sorted_target_node_names(target_tensor_names.begin(),
                                                    target_tensor_names.end());
  std::sort(sorted_target_node_names.begin(), sorted_target_node_names.end());

  // Load the client graph.
  TF_ASSIGN_OR_RETURN(
      LoadedClientGraph & loaded_client_graph,
      GetOrCreateLoadedClientGraph(
          run_options, sorted_input_names, sorted_input_dtypes,
          sorted_output_names, sorted_target_node_names, run_options.work_queue,
          /*graph_name=*/{}, inputs));

  // Get a shared_ptr of the executable so that during the current request the
  // executable to use is guaranteed to be alive.
  auto executable_context = loaded_client_graph.executable_context();
  const mlrt::LoadedExecutable* loaded_executable = nullptr;
  const tfrt::Function* func = nullptr;
  if (executable_context->IsForMlrt()) {
    loaded_executable = executable_context->bytecode_executable.get();
  } else {
    func =
        executable_context->bef_file->GetFunction(loaded_client_graph.name());
  }
  DCHECK(func || loaded_executable);

  // Create the actual arguments to the compiled function, which are sorted
  // according to the input tensor names.
  std::vector<tensorflow::Tensor> flat_inputs;
  if (!loaded_client_graph.is_restore()) {
    flat_inputs.reserve(inputs.size());
    for (int original_index : input_original_indices) {
      flat_inputs.push_back(inputs.at(original_index).second);
    }
  }

  // Possibly record costs, depending on the particular setting of
  // `CostAnalysisOptions`. As of this comment, for correctness of that feature,
  // the time needs to be created after the client graph is created
  //
  // To reduce system calls, this value is also used for timing the duration of
  // `::Run`.
  auto now = absl::Now() + simulated_duration_;
  bool do_recompilation;
  CostRecorder* cost_recorder =
      loaded_client_graph.MaybeGetCostRecorder(now, &do_recompilation);

  std::vector<tensorflow::Tensor> flat_outputs;
  TF_RETURN_IF_ERROR(GraphExecutionRunOnFunction(
      options_, run_options, loaded_client_graph.name(),
      loaded_client_graph.symbol_uids(), func, loaded_executable, flat_inputs,
      &flat_outputs, resource_context_.get(),
      &executable_context->resource_context,
      &loaded_client_graph.runner_table(),
      &loaded_client_graph.resource_array(), runtime(), fallback_state(),
      loaded_client_graph.process_function_library_runtime(),
      &req_deadline_tracker_, loaded_client_graph.stream_callback_id(),
      cost_recorder));

  if (do_recompilation) {
    TF_RETURN_IF_ERROR(
        loaded_client_graph.UpdateCost(*cost_recorder, runtime()));
    tensorflow::mutex_lock l(num_recompilations_mu_);
    num_recompilations_ += 1;
  }
  if (cost_recorder != nullptr) {
    loaded_client_graph.UpdateCostAnalysisData(now, do_recompilation);
  }
  // Create the outputs from the actual function results, which are sorted
  // according to the output tensor names.
  auto flat_output_iter = flat_outputs.begin();
  outputs->resize(flat_outputs.size());
  for (int original_index : output_original_indices) {
    (*outputs)[original_index] = std::move(*flat_output_iter);
    ++flat_output_iter;
  }
  absl::Time end = absl::Now() + simulated_duration_;
  absl::Duration elapsed_duration = end - now;
  loaded_client_graph.latency_sampler()->Add(
      absl::ToDoubleMicroseconds(elapsed_duration));
  return OkStatus();
}

tensorflow::Status GraphExecutor::Extend(const GraphDef& graph) {
  return graph_execution_state_->Extend(graph);
}

absl::StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
GraphExecutor::ImportAndCompileClientGraph(
    const GraphExecutor::ClientGraph& client_graph,
    absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs) {
  // Step 1 of loading: Import the client graph from proto to an MLIR module.
  auto import_start_time = absl::Now();
  mlir::DialectRegistry registry;
  RegisterMlirDialect(registry, options_.compile_options.backend_compiler);
  // Disable multi-threading in lazy loading as the thread pool it uses is out
  // of our control and this affects serving performance.
  //
  // TODO(chky): Consider using a custom thread pool with limited threads for
  // compilation.
  auto context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  ASSIGN_OR_RETURN_IN_IMPORT(
      auto flib_def_and_module,
      ImportClientGraphToMlirModule(client_graph, context.get()));
  auto& [flib_def, module] = flib_def_and_module;

  // If the module contains a Restore op, then there should be one input,
  // and it should specify the checkpoint for variable restore.
  std::string checkpoint_path;
  if (options_.compile_options.backend_compiler &&
      mlir::tf_saved_model::IsRestoreGraph(module.get())) {
    if (inputs.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected 1 input for restore graph, but got ", inputs.size(), "."));
    }
    const tensorflow::Tensor& input = inputs[0].second;
    if (input.dtype() != tensorflow::DT_STRING) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected string input for restore graph, but got ",
                       input.dtype(), "."));
    }
    checkpoint_path = input.scalar<tstring>()();
  }

  TF_ASSIGN_OR_RETURN(
      auto stream_callback_id,
      CreateStreamCallbackId(options().model_metadata.name(), module.get()));

  // TODO(b/278143179): Upload module w/o control flow.
  SymbolUids symbol_uids;
  symbol_uids.tf_symbol_uid = MaybeUploadMlirToXsymbol(module.get());

  auto import_duration = absl::Now() - import_start_time;
  LOG(INFO) << "TFRT finished importing client graph (" << &client_graph
            << "). Took " << absl::ToInt64Milliseconds(import_duration)
            << " ms. Client graph name: " << client_graph.name;

  // Step 2 of loading: Compile the MLIR module from TF dialect to TFRT dialect
  // (in BEF).
  // TODO(b/229261464): Unify the sync and async lowering passes so we do not
  // need this branch.
  auto compile_start_time = absl::Now();
  mlir::OwningOpRef<mlir::ModuleOp> module_with_op_keys;
  std::shared_ptr<ExecutableContext> executable_context = nullptr;

  ModelRuntimeContext model_context(&options_,
                                    options_.compile_options.saved_model_dir,
                                    resource_context_.get());
  // Do not export to flib_def; restore graph may contain non-TF mlir ops.
  // TODO: Make restore graph compatible with flib, remove if statement.
  if (checkpoint_path.empty()) {
    model_context.set_function_library_definition(&flib_def);
  }
  model_context.set_checkpoint_path(checkpoint_path);

  if (options_.compile_options.compile_to_sync_tfrt_dialect) {
    if (kernel_registry_ == nullptr) {
      return tensorflow::errors::Internal("Missing kernel registry in MLRT.");
    }
    ASSIGN_OR_RETURN_IN_COMPILE(
        executable_context,
        tfrt::BuildExecutableContext(module.get(), *kernel_registry_));

  } else if (options_.enable_mlrt) {
    if (kernel_registry_ == nullptr) {
      return tensorflow::errors::Internal("Missing kernel registry in MLRT.");
    }

    ASSIGN_OR_RETURN_IN_COMPILE(
        auto bytecode_buffer,
        tensorflow::mlrt_compiler::ConvertTfMlirToBytecode(
            options_.compile_options, fallback_state(), module.get(),
            model_context, &module_with_op_keys));
    mlrt::bc::Executable executable(bytecode_buffer.data());
    auto bytecode_executable =
        std::make_unique<mlrt::LoadedExecutable>(executable, *kernel_registry_);
    executable_context = std::make_shared<ExecutableContext>(
        std::move(bytecode_buffer), std::move(bytecode_executable));
  } else {
    tfrt::BefBuffer bef;
    TF_RETURN_IF_ERROR(tensorflow::ConvertTfMlirToBef(
        options_.compile_options, module.get(), &bef, model_context));
    ASSIGN_OR_RETURN_IN_COMPILE(
        auto bef_file, tfrt::CreateBefFileFromBefBuffer(runtime(), bef));
    executable_context = std::make_shared<ExecutableContext>(
        std::move(bef), std::move(bef_file));
  }
  symbol_uids.tfrt_symbol_uid = MaybeUploadMlirToXsymbol(module.get());

  auto compile_duration = absl::Now() - compile_start_time;
  LOG(INFO) << "TFRT finished compiling client graph (" << &client_graph
            << "). Took " << absl::ToInt64Milliseconds(compile_duration)
            << " ms. Client graph name: " << client_graph.name;
  auto* latency_sampler =
      tensorflow::tfrt_metrics::GetTfrtGraphExecutorLatencySampler(
          options_.model_metadata.name(), options_.model_metadata.version(),
          client_graph.name);
  return std::make_unique<LoadedClientGraph>(
      client_graph.name, std::move(symbol_uids), this, std::move(context),
      std::move(module_with_op_keys), std::move(module),
      std::move(executable_context), stream_callback_id,
      !checkpoint_path.empty(), std::move(flib_def), latency_sampler);
}

absl::StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
GraphExecutor::LoadClientGraph(
    const GraphExecutor::ClientGraph& client_graph,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs) {
  LOG(INFO) << "TFRT loading client graph (" << &client_graph << ") "
            << client_graph.name;
  TF_ASSIGN_OR_RETURN(auto loaded_client_graph,
                      ImportAndCompileClientGraph(client_graph, inputs));
  // Step 3 of loading: Initialize runtime states using special BEF functions.
  auto init_start_time = absl::Now();
  if (loaded_client_graph->executable_context()->IsForMlrt()) {
    RETURN_IF_ERROR_IN_INIT(InitBytecode(loaded_client_graph.get()));
  } else {
    RETURN_IF_ERROR_IN_INIT(InitBef(loaded_client_graph.get(), work_queue));
  }
  auto init_duration = absl::Now() - init_start_time;
  LOG(INFO) << "TFRT finished initializing client graph (" << &client_graph
            << "). Took " << absl::ToInt64Milliseconds(init_duration)
            << " ms. Client graph name: " << client_graph.name;

  return loaded_client_graph;
}

absl::StatusOr<
    std::pair<FunctionLibraryDefinition, mlir::OwningOpRef<mlir::ModuleOp>>>
GraphExecutor::ImportClientGraphToMlirModule(
    const GraphExecutor::ClientGraph& client_graph,
    mlir::MLIRContext* context) const {
  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.graph_func_name = client_graph.name;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  graph_import_config.inputs = client_graph.input_nodes;
  graph_import_config.outputs = client_graph.output_nodes;
  graph_import_config.control_outputs = client_graph.target_nodes;
  graph_import_config.set_original_tf_func_name = true;

  // Optimize the graph.
  TF_ASSIGN_OR_RETURN(
      auto optimized_graph,
      graph_execution_state_->CreateOptimizedGraph(graph_import_config));

  LOG(INFO) << "TFRT import client graph (" << &client_graph
            << "): Functionalization took "
            << absl::ToInt64Milliseconds(
                   optimized_graph.functionalization_duration)
            << " ms. Client graph name: " << client_graph.name;
  LOG(INFO) << "TFRT import client graph (" << &client_graph
            << "): Grappler took "
            << absl::ToInt64Milliseconds(optimized_graph.grappler_duration)
            << " ms. Client graph name: " << client_graph.name;

  // Convert the optimized graph to an MLIR module.
  TF_ASSIGN_OR_RETURN(
      auto module,
      tensorflow::ConvertGraphToMlir(*optimized_graph.graph, /*debug_info=*/{},
                                     optimized_graph.graph->flib_def(),
                                     graph_import_config, context));

  return std::make_pair(std::move(*optimized_graph.graph->mutable_flib_def()),
                        std::move(module));
}

tensorflow::Status GraphExecutor::InitBef(
    LoadedClientGraph* loaded_client_graph,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue) {
  auto* bef_file = loaded_client_graph->executable_context()->bef_file.get();
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(
          options_, /*run_options=*/{}, work_queue, resource_context_.get(),
          /*client_graph_resource_context=*/nullptr,
          &loaded_client_graph->runner_table(),
          &loaded_client_graph->resource_array(), fallback_state(),
          loaded_client_graph->process_function_library_runtime()));

  tfrt::ExecutionContext exec_ctx(request_info->tfrt_request_context);

  // Run "_tfrt_fallback_init" first to initialize fallback-specific states. It
  // is the special function created by compiler, which calls a sequence of
  // tfrt_fallback_async.createop to create all fallback ops used in this BEF.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, kFallbackInitFunction));

  // After we initialized all the resources in the original graph, we can run
  // the "_tfrt_resource_init" function to set these resources in runtime
  // states, so that later it can be efficiently retrieved without any locking.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, kResourceInitFunction));

  return OkStatus();
}

tensorflow::Status GraphExecutor::InitBytecode(
    LoadedClientGraph* loaded_graph) {
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(options_, /*run_options=*/{},
                        options_.runtime->work_queue(), resource_context_.get(),
                        /*client_graph_resource_context=*/nullptr,
                        &loaded_graph->runner_table(),
                        &loaded_graph->resource_array(), fallback_state(),
                        loaded_graph->process_function_library_runtime()));

  const auto* loaded_executable =
      loaded_graph->executable_context()->bytecode_executable.get();
  DCHECK(loaded_executable);

  std::vector<tensorflow::Tensor> outputs;
  if (auto function = loaded_executable->GetFunction(kFallbackInitFunction)) {
    TF_RETURN_IF_ERROR(RunMlrtFunction(
        function, *loaded_executable, request_info->tfrt_request_context,
        *request_info->request_queue, {}, &outputs,
        &loaded_graph->sync_resource_state()));
  }

  if (auto function = loaded_executable->GetFunction(kResourceInitFunction)) {
    TF_RETURN_IF_ERROR(RunMlrtFunction(
        function, *loaded_executable, request_info->tfrt_request_context,
        *request_info->request_queue, {}, &outputs,
        &loaded_graph->sync_resource_state()));
  }

  return OkStatus();
}

absl::StatusOr<std::reference_wrapper<GraphExecutor::LoadedClientGraph>>
GraphExecutor::GetOrCreateLoadedClientGraph(
    const RunOptions& run_options,
    absl::Span<const std::string> input_tensor_names,
    absl::Span<const tensorflow::DataType> input_tensor_dtypes,
    absl::Span<const std::string> output_tensor_names,
    absl::Span<const std::string> target_tensor_names,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    absl::string_view graph_name,
    absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs) {
  // The format of the joined name is illustrated as in the following example:
  // input1-input2^output1-output2^target1-target2
  const std::string joined_name =
      !graph_name.empty()
          ? std::string(graph_name)
          : absl::StrCat(
                absl::StrJoin(input_tensor_names, kTensorNameJoiningDelimiter),
                kArgumentTypeJoiningDelimiter,
                absl::StrJoin(output_tensor_names, kTensorNameJoiningDelimiter),
                kArgumentTypeJoiningDelimiter,
                absl::StrJoin(target_tensor_names,
                              kTensorNameJoiningDelimiter));

  tensorflow::mutex_lock l(loaded_client_graphs_mu_);

  // Cache hit; return immediately.
  const auto iter = loaded_client_graphs_.find(joined_name);
  if (iter != loaded_client_graphs_.end()) return {*iter->second};

  if (run_options.disable_compilation) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("GraphExecutor: compilation is disabled in execution but "
                     "the compiled graph is not found for ",
                     joined_name));
  }

  // Cache miss; populate a `ClientGraph` and load it.
  tensorflow::GraphImportConfig::InputArrays input_nodes;
  DCHECK_EQ(input_tensor_names.size(), input_tensor_dtypes.size());
  for (int i = 0; i < input_tensor_names.size(); ++i) {
    const auto& input_name = input_tensor_names[i];
    auto input_dtype = input_tensor_dtypes[i];

    tensorflow::ArrayInfo array_info;
    array_info.imported_dtype = input_dtype;
    array_info.shape.set_unknown_rank(true);
    input_nodes[input_name] = array_info;
  }
  ClientGraph client_graph{
      run_options.name.empty() ? joined_name : run_options.name,
      std::move(input_nodes),
      {output_tensor_names.begin(), output_tensor_names.end()},
      {target_tensor_names.begin(), target_tensor_names.end()}};
  TF_ASSIGN_OR_RETURN(auto loaded_client_graph,
                      LoadClientGraph(client_graph, work_queue, inputs));

  // Store the new loaded client graph in cache and return.
  auto* loaded_client_graph_ptr = loaded_client_graph.get();
  loaded_client_graphs_[joined_name] = std::move(loaded_client_graph);
  return {*loaded_client_graph_ptr};
}

tensorflow::Status GraphExecutor::RunWithSyncInterpreter(
    const std::string& graph_name, absl::Span<mlrt::Value> input_values,
    absl::Span<const std::string> input_names,
    absl::Span<const tensorflow::DataType> input_dtypes,
    absl::Span<const std::string> output_tensor_names,
    absl::Span<const std::string> target_tensor_names,
    absl::Span<mlrt::Value> outputs) {
  TF_ASSIGN_OR_RETURN(
      LoadedClientGraph & loaded_client_graph,
      GetOrCreateLoadedClientGraph(
          /*run_options=*/{}, input_names, input_dtypes, output_tensor_names,
          target_tensor_names,
          /*work_queue=*/nullptr,
          graph_name.empty() ? output_tensor_names[0] : graph_name));

  // Get a shared_ptr of the executable so that during the current request the
  // executable to use is guaranteed to be alive.
  auto executable_context = loaded_client_graph.executable_context();
  mlrt::ExecutionContext execution_context(
      executable_context->bytecode_executable.get());

  AddSyncContext(execution_context,
                 *options_.runtime->core_runtime()->GetHostContext(),
                 &loaded_client_graph.sync_resource_state());

  tensorflow::tfd::KernelFallbackCompatRequestState kernel_fallback_state(
      tfd::GetDefaultRunner(), &fallback_state().device_manager(),
      /*step_id=*/0, &loaded_client_graph.runner_table(),
      &loaded_client_graph.resource_array(),
      /*user_intra_op_threadpool=*/nullptr, /*model_metadata=*/std::nullopt,
      &loaded_client_graph.process_function_library_runtime());

  auto tf_context = std::make_unique<tensorflow::tf_mlrt::Context>(
      &kernel_fallback_state, resource_context_.get());
  execution_context.AddUserContext(std::move(tf_context));

  auto serving_function = executable_context->bytecode_executable->GetFunction(
      loaded_client_graph.name());
  DCHECK(serving_function);

  execution_context.CallByMove(serving_function, input_values, outputs);
  mlrt::Execute(execution_context);
  return execution_context.status();
}

CostRecorder* GraphExecutor::LoadedClientGraph::MaybeGetCostRecorder(
    absl::Time now, bool* do_recompilation) {
  *do_recompilation = false;
  tensorflow::mutex_lock l(cost_analysis_data_.mu);
  if (!cost_analysis_data_.is_available) {
    return nullptr;
  }
  const auto& options = graph_executor_->options().cost_analysis_options;
  absl::Duration elapsed_duration = now - cost_analysis_data_.start_time;
  double intended_num_updates = absl::ToDoubleSeconds(elapsed_duration) /
                                absl::ToDoubleSeconds(options.reset_interval) *
                                options.updates_per_interval;
  // Compare with the actual number of cost updates to decide whether or not to
  // record costs for this particular execution.
  if (intended_num_updates - cost_analysis_data_.num_cost_updates >= 1) {
    cost_analysis_data_.is_available = false;
    *do_recompilation = 1 + cost_analysis_data_.num_cost_updates >=
                        options.updates_per_interval;
    return cost_analysis_data_.cost_recorder.get();
  }
  return nullptr;
}

Status GraphExecutor::LoadedClientGraph::UpdateCost(
    const CostRecorder& cost_recorder, const Runtime& runtime) {
  LOG(INFO) << "TFRT updating op costs of loaded client graph (" << this << ") "
            << name_;
  std::shared_ptr<ExecutableContext> new_executable_context = nullptr;
  if (executable_context()->IsForMlrt()) {
    auto tf_mlir_with_op_keys = ::mlir::OwningOpRef<mlir::ModuleOp>(
        cost_analysis_data_.tf_mlir_with_op_keys.get().clone());
    // Recompile from the TF MLIR with recorded costs (skipping
    // AssignOpKeyPass), during which Stream Analysis is redone.
    TF_ASSIGN_OR_RETURN(
        auto bytecode_buffer,
        tensorflow::mlrt_compiler::ConvertTfMlirWithOpKeysToBytecode(
            graph_executor_->options().compile_options,
            graph_executor_->fallback_state(), tf_mlir_with_op_keys.get(),
            cost_recorder));
    mlrt::bc::Executable executable(bytecode_buffer.data());
    auto bytecode_executable = std::make_unique<mlrt::LoadedExecutable>(
        executable, *graph_executor_->kernel_registry_);
    new_executable_context = std::make_shared<ExecutableContext>(
        std::move(bytecode_buffer), std::move(bytecode_executable));
  } else {
    // Update costs in TFRT MLIR.
    auto tfrt_mlir = ::mlir::OwningOpRef<mlir::ModuleOp>(
        cost_analysis_data_.tfrt_mlir.get().clone());
    mlir::StatusScopedDiagnosticHandler diag_handler(
        tfrt_mlir.get().getContext());
    tfrt_compiler::UpdateOpCostInTfrtMlir(tfrt_mlir.get(), cost_recorder);
    // Recompile from the updated TFRT MLIR, during which Stream Analysis is
    // redone.
    auto bef = tfrt::ConvertMLIRToBEF(tfrt_mlir.get(),
                                      /*disable_optional_sections=*/true);
    if (bef.empty()) {
      return diag_handler.Combine(
          tensorflow::errors::Internal("failed to convert MLIR to BEF."));
    }
    bef.shrink_to_fit();
    TF_ASSIGN_OR_RETURN(auto bef_file,
                        tfrt::CreateBefFileFromBefBuffer(runtime, bef));
    new_executable_context = std::make_shared<ExecutableContext>(
        std::move(bef), std::move(bef_file));
  }
  {
    // Swap in the new `ExecutableContext`.
    tensorflow::mutex_lock lock(executable_context_mu_);
    // TODO(b/259602527): Add test cases that fail when code is changed. E.g.,
    // add a test kernel that examines the cost.
    executable_context_ = std::move(new_executable_context);
  }
  return OkStatus();
}

GraphExecutor::LoadedClientGraph::LoadedClientGraph(
    std::string name, SymbolUids symbol_uids, GraphExecutor* graph_executor,
    std::unique_ptr<mlir::MLIRContext> mlir_context,
    mlir::OwningOpRef<mlir::ModuleOp> tf_mlir_with_op_keys,
    mlir::OwningOpRef<mlir::ModuleOp> tfrt_mlir,
    std::shared_ptr<ExecutableContext> executable_context,
    std::optional<StreamCallbackId> stream_callback_id, bool is_restore,
    FunctionLibraryDefinition flib_def,
    tsl::monitoring::SamplerCell* latency_sampler)
    : name_(std::move(name)),
      symbol_uids_(std::move(symbol_uids)),
      graph_executor_(graph_executor),
      mlir_context_(std::move(mlir_context)),
      executable_context_(std::move(executable_context)),
      stream_callback_id_(stream_callback_id),
      is_restore_(is_restore),
      flib_def_(std::move(flib_def)),
      pflr_(&graph_executor->fallback_state().device_manager(),
            graph_executor->fallback_state().session_options().env,
            &graph_executor->fallback_state().session_options().config,
            TF_GRAPH_DEF_VERSION, &flib_def_,
            graph_executor->fallback_state()
                .session_options()
                .config.graph_options()
                .optimizer_options(),
            /*thread_pool=*/nullptr, /*parent=*/nullptr,
            /*session_metadata=*/nullptr,
            Rendezvous::Factory{[](int64_t, const DeviceMgr* device_mgr,
                                   tsl::core::RefCountPtr<Rendezvous>* r) {
              *r = tsl::core::RefCountPtr<Rendezvous>(
                  new IntraProcessRendezvous(device_mgr));
              return OkStatus();
            }}),
      latency_sampler_(latency_sampler) {
  const auto& options = graph_executor_->options().cost_analysis_options;
  if (options.version != Options::CostAnalysisOptions::kDisabled) {
    // Initialize in a way that ensures recompilation on the first run.
    cost_analysis_data_.start_time = absl::Now() - options.reset_interval;
    cost_analysis_data_.is_available = true;
    cost_analysis_data_.num_cost_updates = options.updates_per_interval - 1;
    cost_analysis_data_.cost_recorder = std::make_unique<CostRecorder>();
    if (executable_context_->IsForMlrt()) {
      cost_analysis_data_.tf_mlir_with_op_keys =
          std::move(tf_mlir_with_op_keys);
    } else {
      cost_analysis_data_.tfrt_mlir = std::move(tfrt_mlir);
    }
  }
}

void GraphExecutor::LoadedClientGraph::UpdateCostAnalysisData(
    absl::Time now, bool do_recompilation) {
  tensorflow::mutex_lock lock(cost_analysis_data_.mu);
  if (!do_recompilation) {
    cost_analysis_data_.num_cost_updates += 1;
    cost_analysis_data_.is_available = true;
    return;
  }
  if (graph_executor_->options().cost_analysis_options.version ==
      Options::CostAnalysisOptions::kOnce) {
    // Free the cost analysis data if it will not be used again.
    cost_analysis_data_.is_available = false;
    cost_analysis_data_.tfrt_mlir = nullptr;
    cost_analysis_data_.tf_mlir_with_op_keys = nullptr;
    cost_analysis_data_.cost_recorder = nullptr;
  } else {
    // Update cost analysis data.
    cost_analysis_data_.cost_recorder = std::make_unique<CostRecorder>();
    cost_analysis_data_.is_available = true;
    cost_analysis_data_.start_time = now;
    cost_analysis_data_.num_cost_updates = 0;
  }
}

tensorflow::Status GraphExecutor::CompileGraph(
    const std::string& graph_name,
    absl::Span<const std::string> input_tensor_names,
    absl::Span<const tensorflow::DataType> input_tensor_dtypes,
    absl::Span<const std::string> output_tensor_names,
    absl::Span<const std::string> target_tensor_names) {
  return GetOrCreateLoadedClientGraph(
             /*run_options=*/{}, input_tensor_names, input_tensor_dtypes,
             output_tensor_names, target_tensor_names,
             /*work_queue=*/nullptr, graph_name)
      .status();
}

void RegisterMlirDialect(mlir::DialectRegistry& registry,
                         tensorflow::BackendCompiler* backend_compiler) {
  registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect>();
  mlir::RegisterAllTensorFlowDialects(registry);
  if (backend_compiler) {
    backend_compiler->GetDependentDialects(registry);
  }
}

}  // namespace tfrt_stub
}  // namespace tensorflow
