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
#include <cstdlib>
#include <memory>
#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"
#include "tensorflow/core/kernels/batching_util/bounded_executor.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
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
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::tfrt::ArrayRef;
using ::tfrt::AsyncValue;
using ::tfrt::HostContext;
using ::tfrt::RCReference;

// Op attributes.
constexpr char kEnableAdaptiveSchedulerAttr[] = "_enable_adaptive_scheduler";
constexpr char kMinInflightBatchesAttr[] = "_min_inflight_batches";
constexpr char kInitialInflightBatchesAttr[] = "_initial_inflight_batches";
constexpr char kMaxInflightBatchesAttr[] = "_max_inflight_batches";
constexpr char kBatchesToAverageOverAttr[] = "_batches_to_average_over";
// Default thread count in the per-process batching thread pool.
// The value is the same as the TF batch kernel BatchKernel.
constexpr int64_t kBatchThreadPoolSize = 128;

Status GetTfrtExecutionContext(OpKernelContext* c,
                               const tfrt::ExecutionContext** exec_ctx) {
  // ExecutionContext's address is passed in as an I64 input. exec_ctx is only
  // valid during the period of one bef execution. It should not be stored and
  // accessed after bef execution completes.
  const Tensor* tensor;
  TF_RETURN_IF_ERROR(c->input("tfrt_exec_ctx", &tensor));
  int64_t exec_ctx_intptr = *reinterpret_cast<const int64_t*>(tensor->data());
  *exec_ctx = absl::bit_cast<const tfrt::ExecutionContext*>(exec_ctx_intptr);
  return OkStatus();
}

int32 NumBatchThreadsFromEnvironmentWithDefault(int default_num_batch_threads) {
  int32_t num;
  const char* val = std::getenv("TF_NUM_BATCH_THREADS");

  return (val && strings::safe_strto32(val, &num)) ? num
                                                   : default_num_batch_threads;
}

}  // namespace

thread::ThreadPool* GetOrCreateBatchThreadsPool() {
  static thread::ThreadPool* shared_thread_pool = [&]() -> thread::ThreadPool* {
    serving::BoundedExecutor::Options options;

    options.num_threads =
        NumBatchThreadsFromEnvironmentWithDefault(kBatchThreadPoolSize);

    options.thread_name = std::string("adaptive_batch_threads");

    auto status_or_executor = serving::BoundedExecutor::Create(options);
    if (!status_or_executor.ok()) {
      LOG(WARNING) << "Failed to create a batch threads pool with error "
                   << status_or_executor.status();
      return nullptr;
    }
    static serving::BoundedExecutor* executor =
        status_or_executor.value().release();
    return new thread::ThreadPool(executor);
  }();
  return shared_thread_pool;
}

namespace {

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

  static StatusOr<std::unique_ptr<BatchTask>> CreateBatchTask(
      OpKernelContext* context) {
    const tfrt::ExecutionContext* exec_ctx = nullptr;
    TF_RETURN_IF_ERROR(GetTfrtExecutionContext(context, &exec_ctx));
    return {std::make_unique<FallbackBatchTask>(*exec_ctx)};
  }

  static absl::string_view GetBatchFunctionName(
      const BatchFunctionType& batch_function) {
    return batch_function->name();
  }

  static Status Create(OpKernelContext* c, int32_t num_batch_threads,
                       int32_t max_batch_size, int32_t batch_timeout_micros,
                       int32_t max_enqueued_batches,
                       ArrayRef<int32_t> allowed_batch_sizes,
                       tsl::RCReference<const tfrt::Function> bef_func,
                       bool enable_large_batch_splitting, bool disable_padding,
                       std::unique_ptr<FallbackBatchResource>* resource) {
    const tfrt::ExecutionContext* exec_ctx = nullptr;
    TF_RETURN_IF_ERROR(GetTfrtExecutionContext(c, &exec_ctx));

    BatcherT::Options batcher_options;
    batcher_options.num_batch_threads = num_batch_threads;
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
        GetBatcherQueueOptions(num_batch_threads, max_batch_size,
                               batch_timeout_micros, max_enqueued_batches,
                               allowed_batch_sizes,
                               enable_large_batch_splitting, disable_padding),
        allowed_batch_sizes));
    return OkStatus();
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
    return OkStatus();
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

}  // namespace

BatchFunctionFallbackKernelBase::BatchFunctionFallbackKernelBase(
    OpKernelConstruction* c)
    : AsyncOpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("container", &container_));
  OP_REQUIRES_OK(c, c->GetAttr("shared_name", &shared_name_));
  OP_REQUIRES_OK(c, c->GetAttr("batching_queue", &batcher_queue_));
  OP_REQUIRES_OK(c, c->GetAttr("num_batch_threads", &num_batch_threads_));
  OP_REQUIRES_OK(c, c->GetAttr("max_batch_size", &max_batch_size_));
  OP_REQUIRES_OK(c, c->GetAttr("batch_timeout_micros", &batch_timeout_micros_));
  OP_REQUIRES_OK(c, c->GetAttr("max_enqueued_batches", &max_enqueued_batches_));
  OP_REQUIRES_OK(c, c->GetAttr("allowed_batch_sizes", &allowed_batch_sizes_));

  if (shared_name_.empty()) {
    // If shared_name is not supplied, use name instead (prevent collisions by
    // default).
    shared_name_ = name();
  }

  VLOG(1) << "BatchFunctionFallbackKernel(" << this
          << ") container attribute: \"" << container_
          << "\", shared_name attribute: \"" << shared_name_
          << "\", batching_queue attribute: \"" << batcher_queue_ << "\"";

  if (c->HasAttr("enable_large_batch_splitting")) {
    OP_REQUIRES_OK(c, c->GetAttr("enable_large_batch_splitting",
                                 &enable_large_batch_splitting_));
  } else {
    enable_large_batch_splitting_ = false;
  }

  if (c->HasAttr("disable_padding")) {
    OP_REQUIRES_OK(c, c->GetAttr("disable_padding", &disable_padding_));
  } else {
    disable_padding_ = false;
  }

  // Helper function `SetAdaptiveBatchSchedulerOptions` calls
  // `OP_REQUIRES_OK`, which exits the current function upon error.
  // So validate status of `op-kernel-construction`.
  SetAdaptiveBatchSchedulerOptions(c, num_batch_threads_);
  if (!c->status().ok()) {
    return;
  }

  if (enable_adaptive_batch_threads_) {
    // One scheduler instance contains a couple of queue instances,
    // `batcher_queue_` is the key to find queue for this batch-op in the
    // graph.
    // Use `shared_name_` and name() as prefix for `batcher_queue_`.
    // Note name() is unique per session (from session metadata).
    batcher_queue_ = name() + "/" + shared_name_ + batcher_queue_;
  }

  OP_REQUIRES_OK(c, ValidateAllowedBatchSizes());
}

Status BatchFunctionFallbackKernelBase::ValidateAllowedBatchSizes() const {
  if (allowed_batch_sizes_.empty()) {
    return OkStatus();
  }
  int32_t last_size = 0;
  for (size_t i = 0; i < allowed_batch_sizes_.size(); ++i) {
    const int32_t size = allowed_batch_sizes_.at(i);
    if (i > 0 && size <= last_size) {
      return errors::InvalidArgument(
          "allowed_batch_sizes entries must be monotonically increasing");
    }

    if ((!enable_large_batch_splitting_) &&
        (i == allowed_batch_sizes_.size() - 1) && (size != max_batch_size_)) {
      return errors::InvalidArgument(
          "final entry in allowed_batch_sizes must equal max_batch_size when "
          "enable_large_batch_splitting is False");
    }

    last_size = size;
  }
  return OkStatus();
}

void BatchFunctionFallbackKernelBase::SetAdaptiveBatchSchedulerOptions(
    OpKernelConstruction* c, int32_t num_batch_threads) {
  if (c->HasAttr(kEnableAdaptiveSchedulerAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kEnableAdaptiveSchedulerAttr,
                                 &enable_adaptive_batch_threads_));
  }

  if (num_batch_threads <= 0) {
    enable_adaptive_batch_threads_ = true;
  }

  if (!enable_adaptive_batch_threads_) {
    // adaptive_batch_scheduler_options_ is nullopt.
    return;
  }

  // adaptive_batch_scheduler_options_ is not nullopt
  AdaptiveBatchSchedulerOptions options;

  if (c->HasAttr(kBatchesToAverageOverAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kBatchesToAverageOverAttr,
                                 &options.batches_to_average_over));
  }

  if (c->HasAttr(kMinInflightBatchesAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kMinInflightBatchesAttr,
                                 &options.min_in_flight_batches_limit));
  }

  if (c->HasAttr(kInitialInflightBatchesAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kInitialInflightBatchesAttr,
                                 &options.initial_in_flight_batches_limit));
  }

  if (c->HasAttr(kMaxInflightBatchesAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kMaxInflightBatchesAttr,
                                 &options.max_in_flight_batches_limit));
  }

  // At this point, the batch kernel is configured to use adaptive scheduling.
  // To validate or return error at kernel construction time, invokes
  // `GetOrCreateBatchThreadsPool` and validates returned `thread_pool` is
  // valid.
  // Note`GetOrCreateBatchThreadsPool` creates the thread pool once and
  // re-uses the thread-pool instance afterwards.
  thread::ThreadPool* thread_pool = tfrt_stub::GetOrCreateBatchThreadsPool();
  OP_REQUIRES(
      c, thread_pool != nullptr,
      errors::FailedPrecondition("Failed to create batch threads pool"));

  adaptive_batch_scheduler_options_ = options;
}

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
      intra_op_threadpool, session_metadata, /*runner=*/nullptr);
}

StatusOr<RCReference<tfrt::RequestContext>> SetUpRequestContext(
    HostContext* host_ctx, tfrt::ResourceContext* resource_context,
    tfrt_stub::OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array,
    tfrt::RequestContext* src_req_ctx) {
  // Using the same logic as in the c'tor of FunctionLibraryRuntime::Options,
  // to avoid clash with any Session-generated step ID. DirectSession and
  // MasterSession generates non-negative step IDs.
  int64_t step_id = -std::abs(static_cast<int64_t>(random::New64()));

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
  llvm::SmallVector<AsyncValue*, 8> arguments;
  arguments.reserve(inputs.size() + 1);
  // The first argument is a Chain.
  arguments.push_back(tfrt::GetReadyChain().release());
  for (auto& input : inputs) {
    arguments.push_back(TFTensorToFallbackTensor(input).release());
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
  tensorflow::profiler::TraceMeProducer activity(
      // To TraceMeConsumers in WorkQueue.
      [id] {
        return tensorflow::profiler::TraceMeEncode("RunBefFunction",
                                                   {{"id", id}, {"_r", 1}});
      },
      tensorflow::profiler::ContextType::kTfrtExecutor, id,
      tensorflow::profiler::TraceMeLevel::kInfo);

  tfrt::ExecutionContext batch_exec_ctx(std::move(req_ctx));
  batch_exec_ctx.set_work_queue(&exec_ctx.work_queue());
  batch_exec_ctx.set_location(exec_ctx.location());

  bef_func_->ExecuteAsync(batch_exec_ctx, arguments, results);
  // There is a comment in tensorflow/core/kernels/batch_kernels.cc
  // counterpart of this method that blocking here seems to improve
  // latency/throughput in practice with how the batching library manage
  // threading, although this doesn't match TFRT's threading model. Keeping
  // this behavior for now, should reconsider when we redo the batching
  // kernels.
  batch_exec_ctx.work_queue().Await(results);
  for (AsyncValue* arg : arguments) {
    arg->DropRef();
  }

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
      final_status = FromAbslStatus(*errors[0]);
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

namespace {

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
