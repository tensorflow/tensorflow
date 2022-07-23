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

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_request_context.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/utils.h"
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

}  // namespace

StatusOr<std::unique_ptr<RequestInfo>> SetUpRequestContext(
    const GraphExecutionRunOptions& run_options,
    const SessionMetadata& model_metadata, tfrt::HostContext* host,
    tensorflow::tfrt_stub::WorkQueueInterface* work_queue,
    tfrt::ResourceContext* resource_context,
    const tensorflow::tfrt_stub::FallbackState& fallback_state) {
  DCHECK(host);
  DCHECK(work_queue);
  // Create request context and prepare deadline tracker.
  // TODO(tfrt-devs): Consider using an ID unique within each model to reduce
  // contention.
  int64_t request_id = work_queue->id();
  if (request_id == 0) request_id = tfrt::GetUniqueInt();
  tfrt::RequestContextBuilder request_context_builder(host, resource_context,
                                                      request_id);

  // TODO(b/198671794): `intra_op_threadpool` should be passed through Run()
  // directly.
  tensorflow::thread::ThreadPoolInterface* intra_op_threadpool = nullptr;

  // TODO(b/198671794): The per-request queue should be passed through Run()
  // directly.
  TF_ASSIGN_OR_RETURN(auto request_queue,
                      work_queue->InitializeRequest(&request_context_builder,
                                                    &intra_op_threadpool));

  auto request_info = std::make_unique<RequestInfo>();

  // If a per-request queue is not provided, use the original queue in the
  // tensorflow::Executor::Args::Runner.
  auto* inter_op_queue = request_queue ? request_queue.get() : work_queue;
  request_info->runner = [inter_op_queue](std::function<void()> f) {
    inter_op_queue->AddTask(std::move(f));
  };

  request_info->request_queue = std::move(request_queue);

  TF_RETURN_IF_ERROR(tensorflow::tfd::SetUpKernelFallbackCompatRequestContext(
      &request_context_builder, &fallback_state.device_manager(),
      &fallback_state.process_function_library_runtime(), intra_op_threadpool,
      model_metadata, &request_info->runner));

  TF_RETURN_IF_ERROR(
      tensorflow::SetUpTfJitRtRequestContext(&request_context_builder));
  tfrt::RequestOptions request_options;
  request_options.priority = run_options.priority;
  request_context_builder.set_request_options(request_options);

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
    absl::string_view signature_name, const tfrt::Function& func,
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const tensorflow::Tensor> captures,
    std::vector<tensorflow::Tensor>* outputs,
    tfrt::ResourceContext* resource_context, const Runtime& runtime,
    const FallbackState& fallback_state,
    tfrt::RequestDeadlineTracker& req_deadline_tracker) {
  auto* host = runtime.core_runtime()->GetHostContext();

  TF_ASSIGN_OR_RETURN(
      auto request_info,
      SetUpRequestContext(run_options, options.model_metadata, host,
                          run_options.work_queue ? run_options.work_queue
                                                 : runtime.work_queue(),
                          resource_context, fallback_state));

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
    req_deadline_tracker.CancelRequestOnDeadline(
        deadline, request_info->tfrt_request_context);
  }

  tfrt::ExecutionContext exec_ctx{request_info->tfrt_request_context};
  if (run_options.work_queue) {
    // TODO(b/198671794): Avoid creating `request_queue` when the `work_queue`
    // in `run_options` is specified.
    exec_ctx.set_work_queue(run_options.work_queue);
  } else if (request_info->request_queue) {
    exec_ctx.set_work_queue(request_info->request_queue.get());
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

  DCHECK(captures.empty()) << "signature should have no captures, which is "
                              "guaranteed by the compiler";

  if (arguments.size() != func.argument_types().size())
    return tensorflow::errors::Internal("incorrect number of inputs.");

  llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 4> chain_and_results;
  chain_and_results.resize(func.result_types().size());

  // Hand over the execution to thread pool.
  std::array<tfrt::RCReference<tfrt::AsyncValue>, 1> executed = {
      EnqueueWork(exec_ctx, [&]() -> tfrt::Chain {
        func.Execute(exec_ctx, arguments, chain_and_results);
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
    status_group.Update(CreateTfErrorStatus(chain->GetError()));
  }

  for (tfrt::RCReference<tfrt::AsyncValue>& result : results) {
    DCHECK(result->IsAvailable());

    if (result->IsError()) {
      status_group.Update(CreateTfErrorStatus(result->GetError()));
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

  // TODO(b/171926578): Explicitly clear the context data. Remove it after the
  // b/171926578 is fixed.
  exec_ctx.request_ctx()->ClearData();

  // Check if error is due to cancellation.
  // TODO(tfrt-devs): report cancellation reason from runtime.
  if (request_info->tfrt_request_context->IsCancelled()) {
    // Currently a request can only be cancelled by an expired timer.
    return tensorflow::errors::DeadlineExceeded(kDeadlineExceededMessage);
  }

  return status_group.as_summary_status();
}

std::unique_ptr<tfrt::ResourceContext> CreateResourceContext(
    const tensorflow::tfrt_stub::Runtime& runtime,
    tfrt::tpu::TpuModelResource* tpu_model_resource,
    tensorflow::TfrtTpuInfraTarget tpu_target) {
  auto resource_context = std::make_unique<tfrt::ResourceContext>();
  runtime.CreateRuntimeResources(resource_context.get());

  // TODO(b/178227859): We should make TPU resource init code pluggable, as
  // opposed to linking it in. We can do this by adding a callback with
  // `Runtime::AddCreateRuntimeResourceFn`.
  if (tpu_target == tensorflow::TfrtTpuInfraTarget::kTpurt) {
    AddTpuResources(resource_context.get(), tpu_model_resource);
  }
  return resource_context;
}

StatusOr<std::unique_ptr<GraphExecutor>> GraphExecutor::Create(
    Options options, const FallbackState& fallback_state,
    tfrt::tpu::TpuModelResource* tpu_model_resource,
    tensorflow::GraphDef graph_def) {
  if (options.runtime == nullptr) {
    return errors::InvalidArgument("options.runtime must be non-null ");
  }

  TfrtGraphExecutionState::Options graph_execution_state_options;
  graph_execution_state_options.run_placer_grappler_on_functions =
      options.run_placer_grappler_on_functions;
  graph_execution_state_options.enable_tfrt_gpu = options.enable_tfrt_gpu;

  TF_ASSIGN_OR_RETURN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(graph_execution_state_options,
                                      std::move(graph_def), fallback_state));
  return std::make_unique<GraphExecutor>(std::move(options), fallback_state,
                                         tpu_model_resource,
                                         std::move(graph_execution_state));
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
  TF_ASSIGN_OR_RETURN(const LoadedClientGraph& loaded_client_graph,
                      GetOrCreateLoadedClientGraph(
                          sorted_input_names, sorted_input_dtypes,
                          sorted_output_names, sorted_target_node_names));

  const auto* func = loaded_client_graph.bef_file->GetFunction(
      tensorflow::kImportModelDefaultGraphFuncName);
  DCHECK(func);

  // Create the actual arguments to the compiled function, which are sorted
  // according to the input tensor names.
  std::vector<tensorflow::Tensor> flat_inputs;
  flat_inputs.reserve(inputs.size());
  for (int original_index : input_original_indices) {
    flat_inputs.push_back(inputs.at(original_index).second);
  }

  std::vector<tensorflow::Tensor> flat_outputs;
  TF_RETURN_IF_ERROR(GraphExecutionRunOnFunction(
      options_, run_options, loaded_client_graph.name, *func, flat_inputs,
      /*captures=*/{}, &flat_outputs,
      loaded_client_graph.resource_context.get(), runtime(), fallback_state_,
      req_deadline_tracker_));

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
  auto loaded_client_graph = std::make_unique<LoadedClientGraph>();
  loaded_client_graph->name = client_graph.name;
  loaded_client_graph->resource_context = CreateResourceContext(
      runtime(), tpu_model_resource_, options_.compile_options.tpu_target);

  // Step 1 of loading: Import the client graph from proto to an MLIR module.
  auto import_start_time = absl::Now();
  mlir::MLIRContext context;
  ASSIGN_OR_RETURN_IN_IMPORT(
      auto module, ImportClientGraphToMlirModule(client_graph, &context));
  auto import_duration = absl::Now() - import_start_time;
  LOG(INFO) << "TFRT finished importing client graph (" << &client_graph
            << "). Took " << absl::ToInt64Milliseconds(import_duration)
            << " ms. Client graph name: " << client_graph.name;

  // Step 2 of loading: Compile the MLIR module from TF dialect to TFRT dialect
  // (in BEF).
  auto compile_start_time = absl::Now();
  ASSIGN_OR_RETURN_IN_COMPILE(loaded_client_graph->bef,
                              CompileMlirModuleToBef(module.get()));
  auto compile_duration = absl::Now() - compile_start_time;
  LOG(INFO) << "TFRT finished compiling client graph (" << &client_graph
            << "). Took " << absl::ToInt64Milliseconds(compile_duration)
            << " ms. Client graph name: " << client_graph.name;

  return loaded_client_graph;
}

StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
GraphExecutor::LoadClientGraph(const GraphExecutor::ClientGraph& client_graph) {
  LOG(INFO) << "TFRT loading client graph (" << &client_graph << ") "
            << client_graph.name;
  TF_ASSIGN_OR_RETURN(auto loaded_client_graph,
                      ImportAndCompileClientGraph(client_graph));

  // Step 3 of loading: Initialize runtime states using special BEF functions.
  auto init_start_time = absl::Now();
  ASSIGN_OR_RETURN_IN_INIT(
      loaded_client_graph->bef_file,
      tfrt::CreateBefFileFromBefBuffer(runtime(), loaded_client_graph->bef));
  RETURN_IF_ERROR_IN_INIT(InitBef(loaded_client_graph->bef_file.get(),
                                  loaded_client_graph->resource_context.get()));
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

tensorflow::Status GraphExecutor::InitBef(
    tfrt::BEFFile* bef_file, tfrt::ResourceContext* resource_context) {
  auto* host = runtime().core_runtime()->GetHostContext();
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      SetUpRequestContext(/*run_options=*/{}, /*model_metadata=*/{}, host,
                          runtime().work_queue(), resource_context,
                          fallback_state_));

  tfrt::ExecutionContext exec_ctx(request_info->tfrt_request_context);

  // Run "_tfrt_fallback_init" first to initialize fallback-specific states. It
  // is the special function created by compiler, which calls a sequence of
  // tfrt_fallback_async.createop to create all fallback ops used in this BEF.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, "_tfrt_fallback_init"));

  // After we initialized all the resources in the original graph, we can run
  // the "_tfrt_resource_init" function to set these resources in runtime
  // states, so that later it can be efficiently retrieved without any locking.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, "_tfrt_resource_init"));

  return OkStatus();
}

StatusOr<std::reference_wrapper<const GraphExecutor::LoadedClientGraph>>
GraphExecutor::GetOrCreateLoadedClientGraph(
    absl::Span<const std::string> input_tensor_names,
    absl::Span<const tensorflow::DataType> input_tensor_dtypes,
    absl::Span<const std::string> output_tensor_names,
    absl::Span<const std::string> target_tensor_names) {
  // The format of the joined name is illustrated as in the following example:
  // input1-input2^output1-output2^target1-target2
  const auto joined_name = absl::StrCat(
      absl::StrJoin(input_tensor_names, kTensorNameJoiningDelimiter),
      kArgumentTypeJoiningDelimiter,
      absl::StrJoin(output_tensor_names, kTensorNameJoiningDelimiter),
      kArgumentTypeJoiningDelimiter,
      absl::StrJoin(target_tensor_names, kTensorNameJoiningDelimiter));

  tensorflow::mutex_lock l(loaded_client_graphs_mu_);

  // Cache hit; return immediately.
  const auto iter = loaded_client_graphs_.find(joined_name);
  if (iter != loaded_client_graphs_.end()) return {*iter->second};

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
  TF_ASSIGN_OR_RETURN(auto loaded_client_graph, LoadClientGraph(client_graph));

  // Store the new loaded client graph in cache and return.
  const auto* loaded_client_graph_ptr = loaded_client_graph.get();
  loaded_client_graphs_[joined_name] = std::move(loaded_client_graph);
  return {*loaded_client_graph_ptr};
}

}  // namespace tfrt_stub
}  // namespace tensorflow
