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

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_request_context.h"
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
  tfrt::RequestContextBuilder request_context_builder(host, resource_context,
                                                      tfrt::GetUniqueInt());

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
       options] {
        return tensorflow::profiler::TraceMeEncode(
            "TfrtModelRun",
            {{"_r", 1},
             {"id", request_id},
             {"signature", signature_name},
             {"model_id", absl::StrCat(options.model_metadata.name(),
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

}  // namespace tfrt_stub
}  // namespace tensorflow
