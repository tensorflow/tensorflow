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
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/runtime/fallback_batch_kernel.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::tfrt::ArrayRef;
using ::tfrt::AsyncValue;
using ::tfrt::HostContext;
using ::tfrt::RCReference;

Status GetTfrtExecutionContext(OpKernelContext* c,
                               const tfrt::ExecutionContext** exec_ctx) {
  // ExecutionContext's address is passed in as an I64 input. exec_ctx is only
  // valid during the period of one bef execution. It should not be stored and
  // accessed after bef execution completes.
  const Tensor* tensor;
  TF_RETURN_IF_ERROR(c->input("tfrt_exec_ctx", &tensor));
  int64_t exec_ctx_intptr = *reinterpret_cast<const int64_t*>(tensor->data());
  *exec_ctx = absl::bit_cast<const tfrt::ExecutionContext*>(exec_ctx_intptr);
  return absl::OkStatus();
}

class FallbackBatchResource : public tensorflow::serving::BatchResourceBase {
 public:
  using BatchFunctionType = tsl::RCReference<const tfrt::Function>;

  struct FallbackBatchTask : BatchTask {
    explicit FallbackBatchTask(const tfrt::ExecutionContext& tfrt_exec_ctx)
        : tfrt_exec_ctx(tfrt_exec_ctx) {}
    tfrt::ExecutionContext tfrt_exec_ctx;

   private:
    std::unique_ptr<BatchTask> CreateDerivedTask() override {
      return std::make_unique<FallbackBatchTask>(this->tfrt_exec_ctx);
    }
  };

  static absl::StatusOr<tfrt::ResourceContext*> GetClientGraphResourceContext(
      OpKernelContext* context) {
    const tfrt::ExecutionContext* exec_ctx = nullptr;
    TF_RETURN_IF_ERROR(GetTfrtExecutionContext(context, &exec_ctx));
    const auto* fallback_request_state =
        exec_ctx->request_ctx()
            ->GetDataIfExists<tfd::KernelFallbackCompatRequestState>();
    // If `client_graph_resource_context` is null, it implies that it's safe to
    // fall back to the per-model resource context.
    return fallback_request_state->client_graph_resource_context() != nullptr
               ? fallback_request_state->client_graph_resource_context()
               : exec_ctx->resource_context();
  }

  static absl::StatusOr<std::unique_ptr<BatchTask>> CreateBatchTask(
      OpKernelContext* context) {
    const tfrt::ExecutionContext* exec_ctx = nullptr;
    TF_RETURN_IF_ERROR(GetTfrtExecutionContext(context, &exec_ctx));
    return {std::make_unique<FallbackBatchTask>(*exec_ctx)};
  }

  static absl::string_view GetBatchFunctionName(
      const BatchFunctionType& batch_function) {
    return batch_function->name();
  }

  static Status Create(OpKernelContext* c,
                       const serving::BatchResourceOptions& options,
                       tsl::RCReference<const tfrt::Function> bef_func,
                       bool enable_large_batch_splitting, bool disable_padding,
                       std::unique_ptr<FallbackBatchResource>* resource) {
    const tfrt::ExecutionContext* exec_ctx = nullptr;
    TF_RETURN_IF_ERROR(GetTfrtExecutionContext(c, &exec_ctx));

    BatcherT::Options batcher_options;
    batcher_options.num_batch_threads = options.num_batch_threads;
    std::shared_ptr<BatcherT> batcher;
    TF_RETURN_IF_ERROR(BatcherT::Create(batcher_options, &batcher));

    const auto* fallback_request_state =
        exec_ctx->request_ctx()
            ->GetDataIfExists<tfd::KernelFallbackCompatRequestState>();
    if (!fallback_request_state) {
      return tensorflow::errors::Internal(
          "KernelFallbackCompatRequestState not found in RequestContext.");
    }

    resource->reset(new FallbackBatchResource(
        *exec_ctx, *fallback_request_state, std::move(bef_func),
        std::move(batcher),
        GetBatcherQueueOptions(
            options.num_batch_threads, options.max_batch_size,
            options.batch_timeout_micros, options.max_enqueued_batches,
            options.allowed_batch_sizes, enable_large_batch_splitting,
            disable_padding, options.low_priority_max_batch_size,
            options.low_priority_batch_timeout_micros,
            options.low_priority_max_enqueued_batches,
            options.low_priority_allowed_batch_sizes,
            options.mixed_priority_batching_policy),
        options.allowed_batch_sizes));
    return absl::OkStatus();
  }

  static Status Create(
      OpKernelContext* c,
      AdaptiveBatcherT::Options adaptive_shared_batch_scheduler_options,
      int32_t max_batch_size, int32_t batch_timeout_micros,
      int32_t max_enqueued_batches, ArrayRef<int32_t> allowed_batch_sizes,
      tsl::RCReference<const tfrt::Function> bef_func, bool disable_padding,
      std::unique_ptr<FallbackBatchResource>* resource) {
    const tfrt::ExecutionContext* exec_ctx = nullptr;
    TF_RETURN_IF_ERROR(GetTfrtExecutionContext(c, &exec_ctx));

    std::shared_ptr<AdaptiveBatcherT> batcher;
    TF_RETURN_IF_ERROR(AdaptiveBatcherT::Create(
        adaptive_shared_batch_scheduler_options, &batcher));

    const auto* fallback_request_state =
        exec_ctx->request_ctx()
            ->GetDataIfExists<tfd::KernelFallbackCompatRequestState>();
    if (!fallback_request_state) {
      return tensorflow::errors::Internal(
          "KernelFallbackCompatRequestState not found in RequestContext.");
    }

    resource->reset(new FallbackBatchResource(
        *exec_ctx, *fallback_request_state, std::move(bef_func),
        std::move(batcher),
        GetAdaptiveBatcherQueueOptions(max_batch_size, batch_timeout_micros,
                                       max_enqueued_batches,
                                       true /* enable large batch split */,
                                       allowed_batch_sizes, disable_padding),
        allowed_batch_sizes));
    return absl::OkStatus();
  }

  string DebugString() const final { return "FallbackBatchResource"; }

  const tsl::RCReference<const tfrt::Function>& batch_function() const {
    return bef_func_;
  }

  static tsl::RCReference<const tfrt::Function> CastHandleToFunction(
      int64_t handle) {
    // BEF function's address is passed in as an I64 attribute.
    return tfrt::FormRef(absl::bit_cast<const tfrt::Function*>(handle));
  }

 private:
  FallbackBatchResource(
      const tfrt::ExecutionContext& exec_ctx,
      const tfd::KernelFallbackCompatRequestState& fallback_request_state,
      RCReference<const tfrt::Function> bef_func,
      std::shared_ptr<BatcherT> batcher,
      const BatcherT::QueueOptions& batcher_queue_options,
      ArrayRef<int32_t> allowed_batch_sizes)
      : BatchResourceBase(
            /*has_process_batch_function=*/true, std::move(batcher),
            batcher_queue_options,
            std::vector<int32_t>(allowed_batch_sizes.begin(),
                                 allowed_batch_sizes.end())),
        host_ctx_(exec_ctx.host()),
        resource_context_(exec_ctx.resource_context()),
        runner_table_(fallback_request_state.runner_table()),
        resource_array_(fallback_request_state.resource_array()),
        bef_func_(std::move(bef_func)) {}

  FallbackBatchResource(
      const tfrt::ExecutionContext& exec_ctx,
      const tfd::KernelFallbackCompatRequestState& fallback_request_state,
      RCReference<const tfrt::Function> bef_func,
      std::shared_ptr<AdaptiveBatcherT> batcher,
      const AdaptiveBatcherT::QueueOptions& batcher_queue_options,
      ArrayRef<int32_t> allowed_batch_sizes)
      : BatchResourceBase(
            /*has_process_batch_function=*/true, std::move(batcher),
            batcher_queue_options,
            std::vector<int32_t>(allowed_batch_sizes.begin(),
                                 allowed_batch_sizes.end())),
        host_ctx_(exec_ctx.host()),
        resource_context_(exec_ctx.resource_context()),
        runner_table_(fallback_request_state.runner_table()),
        resource_array_(fallback_request_state.resource_array()),
        bef_func_(std::move(bef_func)) {}

  void ProcessFuncBatchImpl(
      const BatchTask& last_task, absl::Span<const Tensor> inputs,
      std::vector<Tensor>* combined_outputs,
      std::function<void(const Status&)> done) const override;

  HostContext* const host_ctx_;
  tfrt::ResourceContext* const resource_context_;
  tfrt_stub::OpKernelRunnerTable* runner_table_;
  tfd::FallbackResourceArray* resource_array_;
  RCReference<const tfrt::Function> bef_func_;
};

tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> TFTensorToFallbackTensor(
    const tensorflow::Tensor& tf_tensor) {
  return tfrt::MakeAvailableAsyncValueRef<tfrt_stub::FallbackTensor>(tf_tensor);
}

Status SetUpKernelFallbackCompatRequestContextForBatch(
    tfrt::RequestContextBuilder* builder,
    tfrt_stub::OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array,
    tfrt::RequestContext& src_req_ctx) {
  DCHECK(builder);

  const auto* src_fallback_request_state =
      src_req_ctx.GetDataIfExists<tfd::KernelFallbackCompatRequestState>();
  if (!src_fallback_request_state) {
    return tensorflow::errors::Internal(
        "KernelFallbackCompatRequestState not found in RequestContext.");
  }

  auto* intra_op_threadpool = src_fallback_request_state->intra_op_threadpool();

  const auto& session_metadata = src_fallback_request_state->session_metadata();

  const auto* device_manager = &src_fallback_request_state->device_manager();

  const auto* pflr =
      &src_fallback_request_state->process_function_library_runtime();

  return SetUpKernelFallbackCompatRequestContext(
      builder, device_manager, pflr, runner_table, resource_array,
      intra_op_threadpool, session_metadata,
      src_fallback_request_state->runner(),
      src_fallback_request_state->cost_recorder(),
      src_fallback_request_state->client_graph_resource_context(),
      src_fallback_request_state->cancellation_manager(),
      src_fallback_request_state->runtime_config());
}

absl::StatusOr<RCReference<tfrt::RequestContext>> SetUpRequestContext(
    HostContext* host_ctx, tfrt::ResourceContext* resource_context,
    tfrt_stub::OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array,
    tfrt::RequestContext* src_req_ctx) {
  // Connect to the batch step id propagated from batch task.
  int64_t step_id = src_req_ctx->id();

  tfrt::RequestContextBuilder request_context_builder(
      host_ctx, resource_context, step_id);

  TF_RETURN_IF_ERROR(SetUpKernelFallbackCompatRequestContextForBatch(
      &request_context_builder, runner_table, resource_array, *src_req_ctx));

  auto expected_req_ctx = std::move(request_context_builder).build();
  if (!expected_req_ctx) {
    return tensorflow::errors::Internal(
        tfrt::StrCat(expected_req_ctx.takeError()));
  }

  return std::move(expected_req_ctx.get());
}

void FallbackBatchResource::ProcessFuncBatchImpl(
    const BatchTask& last_task, absl::Span<const Tensor> inputs,
    std::vector<Tensor>* combined_outputs,
    std::function<void(const Status&)> done) const {
  std::vector<tsl::RCReference<AsyncValue>> arguments;
  arguments.reserve(inputs.size() + 1);
  // The first argument is a Chain.
  arguments.push_back(tfrt::GetReadyChain());
  for (auto& input : inputs) {
    arguments.push_back(TFTensorToFallbackTensor(input));
  }
  llvm::SmallVector<RCReference<AsyncValue>, 4> results;
  results.resize(bef_func_->result_types().size());
  assert(results.size() > 1);
  assert(bef_func_->result_types().front().GetName() == "!tfrt.chain");
  auto& exec_ctx = down_cast<const FallbackBatchTask&>(last_task).tfrt_exec_ctx;

  auto statusor =
      SetUpRequestContext(host_ctx_, resource_context_, runner_table_,
                          resource_array_, exec_ctx.request_ctx());
  if (!statusor.ok()) {
    done(statusor.status());
    return;
  }
  auto req_ctx = std::move(statusor).value();

  int64_t id = req_ctx->id();
  tsl::profiler::TraceMeProducer activity(
      // To TraceMeConsumers in WorkQueue.
      [id] {
        return tsl::profiler::TraceMeEncode("RunBefFunction",
                                            {{"id", id}, {"_r", 1}});
      },
      tsl::profiler::ContextType::kTfrtExecutor, id,
      tsl::profiler::TraceMeLevel::kInfo);

  tfrt::ExecutionContext batch_exec_ctx(std::move(req_ctx));
  batch_exec_ctx.set_work_queue(&exec_ctx.work_queue());
  batch_exec_ctx.set_location(exec_ctx.location());

  bef_func_->ExecuteAsync(batch_exec_ctx, std::move(arguments), results);
  // There is a comment in tensorflow/core/kernels/batch_kernels.cc
  // counterpart of this method that blocking here seems to improve
  // latency/throughput in practice with how the batching library manage
  // threading, although this doesn't match TFRT's threading model. Keeping
  // this behavior for now, should reconsider when we redo the batching
  // kernels.
  batch_exec_ctx.work_queue().Await(results);

  // The first result is a Chain.
  combined_outputs->reserve(results.size() - 1);
  llvm::SmallVector<const absl::Status*, 3> errors;
  for (int i = 1, e = results.size(); i != e; ++i) {
    combined_outputs->emplace_back();
    auto& result = results[i];
    if (auto* error = result->GetErrorIfPresent()) {
      errors.push_back(error);
      continue;
    }
    combined_outputs->back() =
        result->get<tfrt_stub::FallbackTensor>().tensor();
  }
  // Aggregate errors.
  Status final_status;
  if (!errors.empty()) {
    if (errors.size() > 1) {
      auto last = std::unique(errors.begin(), errors.end());
      errors.erase(last, errors.end());
    }

    // If there is only 1 error after deduplication, we emit the error with
    // proper error code mapping from TFRT to TF.
    if (errors.size() == 1) {
      final_status = *errors[0];
    } else {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      for (auto* error : errors) {
        os << error->message() << ";\n";
      }
      final_status = errors::Internal(std::move(os.str()));
    }
  }
  done(final_status);
}

REGISTER_KERNEL_BUILDER(
    Name("_BatchFunctionFallback").Device(DEVICE_CPU),
    tfrt_stub::BatchFunctionFallbackKernel<FallbackBatchResource>);

// Identical to BatchFunction except it has 2 extra TFRT attributes and it does
// not have `f` attribute. Users will not invoke this op directly.
REGISTER_OP("_BatchFunctionFallback")
    .Input("in_tensors: Tin")
    .Input("captured_tensors: Tcaptured")
    // TFRT ExecutionContext pointer.
    .Input("tfrt_exec_ctx: int64")
    .Output("out_tensors: Tout")
    .Attr("num_batch_threads: int")
    .Attr("max_batch_size: int")
    .Attr("batch_timeout_micros: int")
    .Attr("max_enqueued_batches: int = 10")
    .Attr("allowed_batch_sizes: list(int) = []")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("batching_queue: string = ''")
    // A separate set of batch options for the low priority requests, which is
    // used for priority queue batching.
    .Attr("low_priority_max_batch_size: int = 0")
    .Attr("low_priority_batch_timeout_micros: int = 0")
    .Attr("low_priority_allowed_batch_sizes: list(int) = []")
    .Attr("low_priority_max_enqueued_batches: int = 0")
    // Policy that determines the mixed priority batching behavior when low
    // priority batch parameters are present.
    //
    // low_priority_padding_with_next_allowed_batch_size: If high priority
    // batches time out without reaching the max batch size, low priority inputs
    // pad the high priority batches up to the next allowed batch size. A low
    // priority only batch gets schedule only when the low priority input times
    // out or reaches the max batch size while there is no high priority input
    // waiting to be processed.
    // low_priority_padding_with_max_batch_size: Same as above but pad up to the
    // max batch size.
    // priority_isolation: High priority and low priority inputs never share the
    // same batch, i.e., no low priority input padding high priority batches.
    // Low priority inputs get scheduled only as part of low priority only
    // batches as described above.
    .Attr(
        "mixed_priority_policy: "
        "{'low_priority_padding_with_max_batch_size', "
        "'low_priority_padding_with_next_allowed_batch_size', "
        "'priority_isolation'} = 'low_priority_padding_with_max_batch_size'")
    .Attr("Tin: list(type)")
    .Attr("Tcaptured: list(type) >= 0")
    .Attr("Tout: list(type)")
    .Attr("enable_large_batch_splitting: bool = false")
    .Attr("disable_padding: bool = false")
    // An opaque function handle for the batch function.
    .Attr("opaque_function_handle: int")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
