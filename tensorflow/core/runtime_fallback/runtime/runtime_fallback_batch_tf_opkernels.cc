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
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"
#include "tensorflow/core/kernels/batching_util/bounded_executor.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
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
namespace tfd {
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
  // ExecutionContext's address is passed in as an I64 input.
  const Tensor* tensor;
  TF_RETURN_IF_ERROR(c->input("tfrt_exec_ctx", &tensor));
  int64_t exec_ctx_intptr = *reinterpret_cast<const int64_t*>(tensor->data());
  *exec_ctx = absl::bit_cast<const tfrt::ExecutionContext*>(exec_ctx_intptr);
  return OkStatus();
}

// TODO(zce): Move to a util header that can depend on both KernelFallbackTensor
// and RuntimeFallbackTensor. Currently fallback_tensor_util is a dependency of
// KernelFallback, so we can't depend on RuntimeFallbackTensor yet.
llvm::Expected<tensorflow::Tensor> ConvertTFRTTensorToTFTensor(
    const tfrt::Tensor& tensor, HostContext* host) {
  if (auto* rtfbt = llvm::dyn_cast<RuntimeFallbackTensor>(&tensor)) {
    const tensorflow::Tensor* tf_tensor;
    Status s = rtfbt->GetTensorHandle()->Tensor(&tf_tensor);
    if (!s.ok()) {
      return tfrt::MakeStatusError(s);
    }
    return *tf_tensor;
  }
  return tfrt::TFRTTensorToTFTensor(tensor, host);
}

int32 NumBatchThreadsFromEnvironmentWithDefault(int default_num_batch_threads) {
  int32_t num;
  const char* val = std::getenv("TF_NUM_BATCH_THREADS");

  return (val && strings::safe_strto32(val, &num)) ? num
                                                   : default_num_batch_threads;
}

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

class FallbackBatchResource : public tensorflow::serving::BatchResourceBase {
 public:
  static Status Create(int32_t num_batch_threads, int32_t max_batch_size,
                       int32_t batch_timeout_micros,
                       int32_t max_enqueued_batches,
                       ArrayRef<int32_t> allowed_batch_sizes,
                       RCReference<const tfrt::Function> bef_func,
                       bool enable_large_batch_splitting,
                       const tfrt::ExecutionContext& exec_ctx,
                       std::unique_ptr<FallbackBatchResource>* resource) {
    BatcherT::Options batcher_options;
    batcher_options.num_batch_threads = num_batch_threads;
    std::shared_ptr<BatcherT> batcher;
    TF_RETURN_IF_ERROR(BatcherT::Create(batcher_options, &batcher));

    resource->reset(new FallbackBatchResource(
        exec_ctx, std::move(bef_func), std::move(batcher),
        GetBatcherQueueOptions(num_batch_threads, max_batch_size,
                               batch_timeout_micros, max_enqueued_batches,
                               allowed_batch_sizes,
                               enable_large_batch_splitting),
        allowed_batch_sizes));
    return OkStatus();
  }

  static Status Create(
      AdaptiveBatcherT::Options adaptive_shared_batch_scheduler_options,
      int32_t max_batch_size, int32_t batch_timeout_micros,
      int32_t max_enqueued_batches, ArrayRef<int32_t> allowed_batch_sizes,
      RCReference<const tfrt::Function> bef_func,
      const tfrt::ExecutionContext& exec_ctx,
      std::unique_ptr<FallbackBatchResource>* resource) {
    std::shared_ptr<AdaptiveBatcherT> batcher;
    TF_RETURN_IF_ERROR(AdaptiveBatcherT::Create(
        adaptive_shared_batch_scheduler_options, &batcher));

    resource->reset(new FallbackBatchResource(
        exec_ctx, std::move(bef_func), std::move(batcher),
        GetAdaptiveBatcherQueueOptions(
            max_batch_size, batch_timeout_micros, max_enqueued_batches,
            true /* enable large batch split */, allowed_batch_sizes),
        allowed_batch_sizes));
    return OkStatus();
  }

  string DebugString() const final { return "FallbackBatchResource"; }

  const tfrt::Function* bef_func() const { return bef_func_.get(); }

 private:
  struct FallbackBatchTask : BatchTask {
    explicit FallbackBatchTask(const tfrt::ExecutionContext& tfrt_exec_ctx)
        : tfrt_exec_ctx(tfrt_exec_ctx) {}
    tfrt::ExecutionContext tfrt_exec_ctx;

   protected:
    std::unique_ptr<BatchTask> CreateDerivedTask() override {
      return std::make_unique<FallbackBatchTask>(this->tfrt_exec_ctx);
    }
  };

  FallbackBatchResource(const tfrt::ExecutionContext& exec_ctx,
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
        bef_func_(std::move(bef_func)) {}

  FallbackBatchResource(
      const tfrt::ExecutionContext& exec_ctx,
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
        bef_func_(std::move(bef_func)) {}

  void ProcessFuncBatchImpl(
      const BatchTask& last_task, absl::Span<const Tensor> inputs,
      std::vector<Tensor>* combined_outputs,
      std::function<void(const Status&)> done) const override;

  Status CreateBatchTask(OpKernelContext* c,
                         std::unique_ptr<BatchTask>* output) const override {
    const tfrt::ExecutionContext* exec_ctx = nullptr;
    TF_RETURN_IF_ERROR(GetTfrtExecutionContext(c, &exec_ctx));
    *output = std::make_unique<FallbackBatchTask>(*exec_ctx);
    return OkStatus();
  }

  HostContext* const host_ctx_;
  tfrt::ResourceContext* const resource_context_;
  RCReference<const tfrt::Function> bef_func_;
};

// Legacy TF kernel which is a variant of tf.BatchFunction.
class BatchFunctionFallbackKernel : public AsyncOpKernel {
 public:
  explicit BatchFunctionFallbackKernel(OpKernelConstruction* c);

  bool IsExpensive() override { return false; }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final;

  // Validates 'allowed_batch_sizes_'. The entries must increase monotonically,
  // and the last one must equal 'max_batch_size_'.
  Status ValidateAllowedBatchSizes() const;

 private:
  // Initialize vars by reading from op-kernel-construction.
  // Vars
  // - enable_adaptive_batch_threads_
  //   true if value of attribute `kEnableAdaptiveSchedulerAttr` is true, or
  //   if `num_batch_threads` is not positive.
  // - adaptive_batch_scheduler_options_
  //   Read from corresponding attributes as long as they are set.
  void SetAdaptiveBatchSchedulerOptions(OpKernelConstruction* c,
                                        int32_t num_batch_threads);

  string container_;
  string shared_name_;
  string batcher_queue_;
  int32 num_batch_threads_;
  int32 max_batch_size_;
  int32 batch_timeout_micros_;
  int32 max_enqueued_batches_;
  std::vector<int32> allowed_batch_sizes_;
  bool enable_large_batch_splitting_;
  RCReference<const tfrt::Function> bef_func_;

  // Parameters for adaptive batch scheduler only.
  // Note 'num_batch_threads_' above is shared by two implementations of batch
  // scheduler.
  // Per-model inflight batches parameters.
  static constexpr int64_t kMinInflightBatches = 16;
  static constexpr int64_t kInitialInflightBatches = 16;
  static constexpr int64_t kBatchesToAverageOver = 10;
  static constexpr int64_t kMaxInflightBatches = 64;
  bool enable_adaptive_batch_threads_ = false;
  struct AdaptiveBatchSchedulerOptions {
    int32 min_in_flight_batches_limit = kMinInflightBatches;
    int32 initial_in_flight_batches_limit = kInitialInflightBatches;
    int32 max_in_flight_batches_limit = kMaxInflightBatches;
    int32 batches_to_average_over = kBatchesToAverageOver;
  };
  absl::optional<AdaptiveBatchSchedulerOptions>
      adaptive_batch_scheduler_options_ = absl::nullopt;
};

BatchFunctionFallbackKernel::BatchFunctionFallbackKernel(
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

  // BEF function's address is passed in as an I64 attribute.
  {
    int64_t bef_func_intptr;
    OP_REQUIRES_OK(c, c->GetAttr("tfrt_bef_func", &bef_func_intptr));
    bef_func_ =
        tfrt::FormRef(absl::bit_cast<const tfrt::Function*>(bef_func_intptr));
  }

  DCHECK(!shared_name_.empty());
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

void BatchFunctionFallbackKernel::ComputeAsync(OpKernelContext* c,
                                               DoneCallback done) {
  FallbackBatchResource* br;
  std::function<Status(FallbackBatchResource**)> creator;
  if (adaptive_batch_scheduler_options_ != absl::nullopt) {
    creator = [this, c](FallbackBatchResource** r) {
      serving::AdaptiveSharedBatchScheduler<
          serving::BatchResourceBase::BatchTask>::Options
          adaptive_shared_batch_scheduler_options;
      adaptive_shared_batch_scheduler_options.thread_pool_name =
          "adaptive_batch_threads";
      adaptive_shared_batch_scheduler_options.num_batch_threads =
          adaptive_batch_scheduler_options_->max_in_flight_batches_limit;
      adaptive_shared_batch_scheduler_options.thread_pool =
          GetOrCreateBatchThreadsPool();
      // adaptive_shared_batch_scheduler_options.full_batch_scheduling_boost_micros
      // is 0 (default value) intentionally, so tasks are scheduled in a FIFO
      // way.
      // Two rationales to use default value (zero) for
      // `full_batch_scheduling_boost_micros`
      // 1) In this way, tasks scheduling policy is FIFO. Compared with round
      // robin (what shared batch scheduler does), FIFO ensures that model
      // with low QPS (i.e., models enqueue fewer tasks in the shared queue)
      // will be processed timely.
      // 2) If set, `full_batch_scheduling_boost_micros` should be of order
      // the batch processing latency (which varies on a model basis).
      // If a non-zero value is not set properly, it harms tail latency.
      adaptive_shared_batch_scheduler_options.min_in_flight_batches_limit =
          adaptive_batch_scheduler_options_->min_in_flight_batches_limit;
      adaptive_shared_batch_scheduler_options.initial_in_flight_batches_limit =
          adaptive_batch_scheduler_options_->initial_in_flight_batches_limit;
      adaptive_shared_batch_scheduler_options.batches_to_average_over =
          adaptive_batch_scheduler_options_->batches_to_average_over;
      adaptive_shared_batch_scheduler_options.fifo_scheduling = true;
      const tfrt::ExecutionContext* exec_ctx = nullptr;
      TF_RETURN_IF_ERROR(GetTfrtExecutionContext(c, &exec_ctx));
      std::unique_ptr<FallbackBatchResource> new_resource;
      TF_RETURN_IF_ERROR(FallbackBatchResource::Create(
          adaptive_shared_batch_scheduler_options, max_batch_size_,
          batch_timeout_micros_, max_enqueued_batches_, allowed_batch_sizes_,
          bef_func_, *exec_ctx, &new_resource));
      *r = new_resource.release();
      return OkStatus();
    };
  } else {
    creator = [this, c](FallbackBatchResource** r) {
      const tfrt::ExecutionContext* exec_ctx = nullptr;
      TF_RETURN_IF_ERROR(GetTfrtExecutionContext(c, &exec_ctx));
      std::unique_ptr<FallbackBatchResource> new_resource;
      TF_RETURN_IF_ERROR(FallbackBatchResource::Create(
          num_batch_threads_, max_batch_size_, batch_timeout_micros_,
          max_enqueued_batches_, allowed_batch_sizes_, bef_func_,
          enable_large_batch_splitting_, *exec_ctx, &new_resource));
      *r = new_resource.release();
      return OkStatus();
    };
  }
  OP_REQUIRES_OK_ASYNC(c,
                       c->resource_manager()->LookupOrCreate(
                           container_, shared_name_, &br, creator),
                       done);
  // TODO(b/187173237): When we can guarantee only 1 copy of BEF function is
  // generated for the batched function, we can assert the pointers are equal
  OP_REQUIRES_ASYNC(
      c, br->bef_func()->name() == bef_func_.get()->name(),
      errors::InvalidArgument(tfrt::StrCat(
          "Provided BEF function doesn't match with FallbackBatchResource. "
          "Expected:",
          bef_func_.get()->name(), " Received:", br->bef_func()->name())),
      done);
  Status status = br->RegisterInput(random::New64(), c, batcher_queue_, done);
  br->Unref();
  OP_REQUIRES_OK_ASYNC(c, status, done);
  // Assume br calls done, so nothing to do here.
}

Status BatchFunctionFallbackKernel::ValidateAllowedBatchSizes() const {
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

void BatchFunctionFallbackKernel::SetAdaptiveBatchSchedulerOptions(
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
  thread::ThreadPool* thread_pool = GetOrCreateBatchThreadsPool();
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
    tfrt::RequestContextBuilder* builder, tfrt::RequestContext& src_req_ctx) {
  DCHECK(builder);

  const auto* src_fallback_request_state =
      src_req_ctx.GetDataIfExists<KernelFallbackCompatRequestState>();
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
      builder, device_manager, pflr, intra_op_threadpool, session_metadata,
      /*runner=*/nullptr);
}

StatusOr<RCReference<tfrt::RequestContext>> SetUpRequestContext(
    HostContext* host_ctx, tfrt::ResourceContext* resource_context,
    tfrt::RequestContext* src_req_ctx) {
  // Using the same logic as in the c'tor of FunctionLibraryRuntime::Options,
  // to avoid clash with any Session-generated step ID. DirectSession and
  // MasterSession generates non-negative step IDs.
  int64_t step_id = -std::abs(static_cast<int64_t>(random::New64()));

  tfrt::RequestContextBuilder request_context_builder(
      host_ctx, resource_context, step_id);

  TF_RETURN_IF_ERROR(SetUpKernelFallbackCompatRequestContextForBatch(
      &request_context_builder, *src_req_ctx));

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
      SetUpRequestContext(host_ctx_, resource_context_, exec_ctx.request_ctx());
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
  host_ctx_->Await(results);
  for (AsyncValue* arg : arguments) {
    arg->DropRef();
  }

  // The first result is a Chain.
  combined_outputs->reserve(results.size() - 1);
  llvm::SmallVector<const tfrt::DecodedDiagnostic*, 3> errors;
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
      final_status = tfrt::CreateTfErrorStatus(*errors[0]);
    } else {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      for (auto* error : errors) {
        os << *error << ";\n";
      }
      final_status = errors::Internal(std::move(os.str()));
    }
  }
  done(final_status);
}

REGISTER_KERNEL_BUILDER(Name("_BatchFunctionFallback").Device(DEVICE_CPU),
                        BatchFunctionFallbackKernel);

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
    // TFRT BEF function pointer.
    .Attr("tfrt_bef_func: int")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace
}  // namespace tfd
}  // namespace tensorflow
