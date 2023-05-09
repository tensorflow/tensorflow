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

#include "learning/brain/experimental/tfrt/mlrt/application/tensorflow/compiler/transforms/import_model.h"
#include "learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel/context.h"
#include "learning/brain/experimental/tfrt/native_lowering/kernels/sync_context.h"
#include "learning/brain/experimental/tfrt/native_lowering/saved_model/saved_model_translate.h"
#include "learning/infra/mira/mlrt/bytecode/bytecode.h"
#include "learning/infra/mira/mlrt/bytecode/executable.h"
#include "learning/infra/mira/mlrt/interpreter/context.h"
#include "learning/infra/mira/mlrt/interpreter/execute.h"
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_request_context.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/update_op_cost_in_tfrt_mlir.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/sync_resource_state.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
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
  execution_context.AddUserContext(
      std::make_unique<tfrt::SyncContext>(&exec_ctx, sync_resource_state));

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
    return tsl::FromAbslStatus(execution_context.status());
  }

  for (auto& mlrt_output : mlrt_outputs) {
    DCHECK(mlrt_output.HasValue());
    outputs->push_back(std::move(mlrt_output.Get<FallbackTensor>().tensor()));
  }

  return tensorflow::OkStatus();
}

StatusOr<std::unique_ptr<RequestInfo>> CreateRequestInfo(
    const GraphExecutionOptions& options,
    const GraphExecutionRunOptions& run_options,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    tfrt::ResourceContext* resource_context,
    tfrt::ResourceContext* client_graph_resource_context,
    OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array,
    const tensorflow::tfrt_stub::FallbackState& fallback_state,
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
    if (request_id == 0) request_id = tfrt::GetUniqueInt();
    request_info->request_queue = work_queue;
  } else {
    // Otherwise we use the global queue in `runtime`.
    request_id = tfrt::GetUniqueInt();
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
              &fallback_state.process_function_library_runtime());

  fallback_request_state.set_cost_recorder(cost_recorder);
  fallback_request_state.set_client_graph_resource_context(
      client_graph_resource_context);
  fallback_request_state.set_model_config(&options.model_config);

  TF_RETURN_IF_ERROR(
      tensorflow::SetUpTfJitRtRequestContext(&request_context_builder));
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
    CostRecorder* cost_recorder) {
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(options, run_options, run_options.work_queue,
                        resource_context, client_graph_resource_context,
                        runner_table, resource_array, fallback_state,
                        cost_recorder));

  tensorflow::profiler::TraceMeProducer traceme(
      // To TraceMeConsumers in RunHandlerThreadPool::WorkerLoop.
      [request_id = request_info->tfrt_request_context->id(), signature_name,
       &options] {
        return tensorflow::profiler::TraceMeEncode(
            "TfrtModelRun",
            {{"_r", 1},
             {"id", request_id},
             {"signature", signature_name},
             {"model_id", absl::StrCat(options.model_metadata.name(), ":",
                                       options.model_metadata.version())}});
      },
      tensorflow::profiler::ContextType::kTfrtExecutor,
      request_info->tfrt_request_context->id());

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
    status_group.Update(FromAbslStatus(chain->GetError()));
  }

  for (tfrt::RCReference<tfrt::AsyncValue>& result : results) {
    DCHECK(result->IsAvailable());

    if (result->IsError()) {
      status_group.Update(FromAbslStatus(result->GetError()));
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
    Options options, const FallbackState& fallback_state,
    std::unique_ptr<tensorflow::tfrt_stub::TfrtGraphExecutionState>
        graph_execution_state,
    std::unique_ptr<mlrt::KernelRegistry> kernel_registry)
    : options_(std::move(options)),
      fallback_state_(fallback_state),
      graph_execution_state_(std::move(graph_execution_state)),
      req_deadline_tracker_(options_.runtime->core_runtime()->GetHostContext()),
      kernel_registry_(std::move(kernel_registry)) {
  SetSessionCreatedMetric();
  // Creates a ResourceContext and populate it with per model resource from
  // Runtime.
  options_.runtime->CreateRuntimeResources(options_, &resource_context_);
}

StatusOr<std::unique_ptr<GraphExecutor>> GraphExecutor::Create(
    Options options, const FallbackState& fallback_state,
    tensorflow::GraphDef graph_def,
    std::unique_ptr<mlrt::KernelRegistry> kernel_registry) {
  if (options.runtime == nullptr) {
    return errors::InvalidArgument("options.runtime must be non-null ");
  }

  TfrtGraphExecutionState::Options graph_execution_state_options;
  graph_execution_state_options.run_placer_grappler_on_functions =
      options.run_placer_grappler_on_functions;
  graph_execution_state_options.enable_tfrt_gpu = options.enable_tfrt_gpu;
  graph_execution_state_options.use_bridge_for_gpu =
      options.compile_options.use_bridge_for_gpu;

  options.compile_options.fuse_get_resource_ops_in_hoisting =
      !options.enable_mlrt;

  TF_ASSIGN_OR_RETURN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(graph_execution_state_options,
                                      std::move(graph_def), fallback_state));
  return std::make_unique<GraphExecutor>(std::move(options), fallback_state,
                                         std::move(graph_execution_state),
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
  TF_ASSIGN_OR_RETURN(LoadedClientGraph & loaded_client_graph,
                      GetOrCreateLoadedClientGraph(
                          run_options, sorted_input_names, sorted_input_dtypes,
                          sorted_output_names, sorted_target_node_names,
                          run_options.work_queue));

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
  flat_inputs.reserve(inputs.size());
  for (int original_index : input_original_indices) {
    flat_inputs.push_back(inputs.at(original_index).second);
  }

  // Conduct cost analysis for the first request on this `loaded_client_graph`.
  std::unique_ptr<CostRecorder> cost_recorder;
  if (options_.enable_online_cost_analysis) {
    cost_recorder = loaded_client_graph.MaybeCreateCostRecorder();
  }

  std::vector<tensorflow::Tensor> flat_outputs;
  TF_RETURN_IF_ERROR(GraphExecutionRunOnFunction(
      options_, run_options, loaded_client_graph.name(), func,
      loaded_executable, flat_inputs, &flat_outputs, &resource_context_,
      &executable_context->resource_context,
      &loaded_client_graph.runner_table(),
      &loaded_client_graph.resource_array(), runtime(), fallback_state_,
      &req_deadline_tracker_, cost_recorder.get()));

  if (cost_recorder != nullptr) {
    TF_RETURN_IF_ERROR(
        loaded_client_graph.UpdateCost(*cost_recorder, runtime()));
  }

  // Create the outputs from the actual function results, which are sorted
  // according to the output tensor names.
  auto flat_output_iter = flat_outputs.begin();
  outputs->resize(flat_outputs.size());
  for (int original_index : output_original_indices) {
    (*outputs)[original_index] = std::move(*flat_output_iter);
    ++flat_output_iter;
  }

  return OkStatus();
}

tensorflow::Status GraphExecutor::Extend(const GraphDef& graph) {
  return graph_execution_state_->Extend(graph);
}

StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
GraphExecutor::ImportAndCompileClientGraph(
    const GraphExecutor::ClientGraph& client_graph) {
  // Step 1 of loading: Import the client graph from proto to an MLIR module.
  auto import_start_time = absl::Now();
  auto context = std::make_unique<mlir::MLIRContext>();
  ASSIGN_OR_RETURN_IN_IMPORT(
      auto module, ImportClientGraphToMlirModule(client_graph, context.get()));
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
  if (options_.compile_options.compile_to_sync_tfrt_dialect) {
    if (kernel_registry_ == nullptr) {
      return tensorflow::errors::Internal("Missing kernel registry in MLRT.");
    }

    ASSIGN_OR_RETURN_IN_COMPILE(
        auto bytecode_buffer,
        tfrt::CompileTfMlirModuleToBytecode(module.get()));
    mlrt::bc::Executable executable(bytecode_buffer.data());
    auto bytecode_executable =
        std::make_unique<mlrt::LoadedExecutable>(executable, *kernel_registry_);
    executable_context = std::make_shared<ExecutableContext>(
        std::move(bytecode_buffer), std::move(bytecode_executable));
  } else if (options_.enable_mlrt) {
    if (kernel_registry_ == nullptr) {
      return tensorflow::errors::Internal("Missing kernel registry in MLRT.");
    }

    ASSIGN_OR_RETURN_IN_COMPILE(
        auto bytecode_buffer,
        CompileMlirModuleToByteCode(module.get(), &module_with_op_keys));
    mlrt::bc::Executable executable(bytecode_buffer.data());
    auto bytecode_executable =
        std::make_unique<mlrt::LoadedExecutable>(executable, *kernel_registry_);
    executable_context = std::make_shared<ExecutableContext>(
        std::move(bytecode_buffer), std::move(bytecode_executable));
  } else {
    ASSIGN_OR_RETURN_IN_COMPILE(auto bef, CompileMlirModuleToBef(module.get()));
    ASSIGN_OR_RETURN_IN_COMPILE(
        auto bef_file, tfrt::CreateBefFileFromBefBuffer(runtime(), bef));
    executable_context = std::make_shared<ExecutableContext>(
        std::move(bef), std::move(bef_file));
  }
  auto compile_duration = absl::Now() - compile_start_time;
  LOG(INFO) << "TFRT finished compiling client graph (" << &client_graph
            << "). Took " << absl::ToInt64Milliseconds(compile_duration)
            << " ms. Client graph name: " << client_graph.name;

  return std::make_unique<LoadedClientGraph>(
      client_graph.name, this, std::move(context),
      std::move(module_with_op_keys), std::move(module),
      std::move(executable_context));
}

StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
GraphExecutor::LoadClientGraph(
    const GraphExecutor::ClientGraph& client_graph,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue) {
  LOG(INFO) << "TFRT loading client graph (" << &client_graph << ") "
            << client_graph.name;
  TF_ASSIGN_OR_RETURN(auto loaded_client_graph,
                      ImportAndCompileClientGraph(client_graph));

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

tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
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
  return tensorflow::ConvertGraphToMlir(
      *optimized_graph.graph, /*debug_info=*/{},
      optimized_graph.graph->flib_def(), graph_import_config, context);
}

StatusOr<tfrt::BefBuffer> GraphExecutor::CompileMlirModuleToBef(
    mlir::ModuleOp module) const {
  tfrt::BefBuffer bef;
  TF_RETURN_IF_ERROR(
      tensorflow::ConvertTfMlirToBef(options_.compile_options, module, &bef));
  return bef;
}

StatusOr<mlrt::bc::Buffer> GraphExecutor::CompileMlirModuleToByteCode(
    mlir::ModuleOp module,
    mlir::OwningOpRef<mlir::ModuleOp>* module_with_op_keys) const {
  return tensorflow::mlrt_compiler::ConvertTfMlirToBytecode(
      options_.compile_options, module, module_with_op_keys);
}

StatusOr<mlrt::bc::Buffer> GraphExecutor::CompileMlirModuleWithOpKeysToByteCode(
    mlir::ModuleOp module, const CostRecorder& cost_recorder) const {
  return tensorflow::mlrt_compiler::ConvertTfMlirWithOpKeysToBytecode(
      options_.compile_options, module, cost_recorder);
}

tensorflow::Status GraphExecutor::InitBef(
    LoadedClientGraph* loaded_client_graph,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue) {
  auto* bef_file = loaded_client_graph->executable_context()->bef_file.get();
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(
          options_, /*run_options=*/{}, work_queue, &resource_context_,
          /*client_graph_resource_context=*/nullptr,
          &loaded_client_graph->runner_table(),
          &loaded_client_graph->resource_array(), fallback_state_));

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
                        options_.runtime->work_queue(), &resource_context_,
                        /*client_graph_resource_context=*/nullptr,
                        &loaded_graph->runner_table(),
                        &loaded_graph->resource_array(), fallback_state_));

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

StatusOr<std::reference_wrapper<GraphExecutor::LoadedClientGraph>>
GraphExecutor::GetOrCreateLoadedClientGraph(
    const RunOptions& run_options,
    absl::Span<const std::string> input_tensor_names,
    absl::Span<const tensorflow::DataType> input_tensor_dtypes,
    absl::Span<const std::string> output_tensor_names,
    absl::Span<const std::string> target_tensor_names,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    std::optional<const std::string> graph_name) {
  // The format of the joined name is illustrated as in the following example:
  // input1-input2^output1-output2^target1-target2
  const auto joined_name =
      graph_name
          ? *graph_name
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
      joined_name,
      std::move(input_nodes),
      {output_tensor_names.begin(), output_tensor_names.end()},
      {target_tensor_names.begin(), target_tensor_names.end()}};
  TF_ASSIGN_OR_RETURN(auto loaded_client_graph,
                      LoadClientGraph(client_graph, work_queue));

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

  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(options_, /*run_options=*/{},
                        options_.runtime->work_queue(), &resource_context_,
                        /*client_graph_resource_context=*/nullptr,
                        &loaded_client_graph.runner_table(),
                        &loaded_client_graph.resource_array(),
                        fallback_state_));
  tfrt::ExecutionContext exec_ctx{request_info->tfrt_request_context};

  // Get a shared_ptr of the executable so that during the current request the
  // executable to use is guaranteed to be alive.
  auto executable_context = loaded_client_graph.executable_context();
  mlrt::ExecutionContext execution_context(
      executable_context->bytecode_executable.get());

  auto sync_context = std::make_unique<tfrt::SyncContext>(
      &exec_ctx, &loaded_client_graph.sync_resource_state());
  execution_context.AddUserContext(std::move(sync_context));

  auto tf_context = std::make_unique<tensorflow::tf_mlrt::Context>(
      &request_info->tfrt_request_context
           ->GetData<tensorflow::tfd::KernelFallbackCompatRequestState>(),
      request_info->tfrt_request_context->resource_context());
  execution_context.AddUserContext(std::move(tf_context));

  auto serving_function = executable_context->bytecode_executable->GetFunction(
      loaded_client_graph.name());
  DCHECK(serving_function);

  execution_context.CallByMove(serving_function, input_values, outputs);
  mlrt::Execute(execution_context);
  return tsl::FromAbslStatus(execution_context.status());
}

std::unique_ptr<CostRecorder>
GraphExecutor::LoadedClientGraph::MaybeCreateCostRecorder() const {
  std::unique_ptr<CostRecorder> cost_recorder;
  absl::call_once(create_cost_recorder_once_,
                  [&]() { cost_recorder = std::make_unique<CostRecorder>(); });
  return cost_recorder;
}

Status GraphExecutor::LoadedClientGraph::UpdateCost(
    const CostRecorder& cost_recorder, const Runtime& runtime) {
  LOG(INFO) << "TFRT updating op costs of loaded client graph (" << this << ") "
            << name_;
  // Move to function scope to reduce memory footprint.
  auto tfrt_mlir = std::move(tfrt_mlir_);
  auto tf_mlir_with_op_keys = std::move(tf_mlir_with_op_keys_);
  mlir::StatusScopedDiagnosticHandler diag_handler(
      tfrt_mlir.get().getContext());
  std::shared_ptr<ExecutableContext> new_executable_context = nullptr;
  if (executable_context()->IsForMlrt()) {
    // Recompile from the TF MLIR with recorded costs (skipping
    // AssignOpKeyPass), during which Stream Analysis is redone.
    TF_ASSIGN_OR_RETURN(auto bytecode_buffer,
                        graph_executor_->CompileMlirModuleWithOpKeysToByteCode(
                            tf_mlir_with_op_keys.get(), cost_recorder));
    mlrt::bc::Executable executable(bytecode_buffer.data());
    auto bytecode_executable = std::make_unique<mlrt::LoadedExecutable>(
        executable, *graph_executor_->kernel_registry_);
    new_executable_context = std::make_shared<ExecutableContext>(
        std::move(bytecode_buffer), std::move(bytecode_executable));
  } else {
    // Update costs in TFRT MLIR.
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
  // Swap in the new `ExecutableContext`.
  tensorflow::mutex_lock lock(executable_context_mu_);
  // TODO(b/259602527): Add test cases that fail when code is changed. E.g.,
  // add a test kernel that examines the cost.
  executable_context_ = std::move(new_executable_context);
  return OkStatus();
}

}  // namespace tfrt_stub
}  // namespace tensorflow
