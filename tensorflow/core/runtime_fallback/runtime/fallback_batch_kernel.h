/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_FALLBACK_BATCH_KERNEL_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_FALLBACK_BATCH_KERNEL_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/batch_kernels.h"
#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"
#include "tensorflow/core/kernels/batching_util/warmup.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/random.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

class BatchFunctionFallbackKernelBase : public AsyncOpKernel {
 public:
  explicit BatchFunctionFallbackKernelBase(OpKernelConstruction* c);

 protected:
  // Validates 'allowed_batch_sizes_'. The entries must increase monotonically,
  // and the last one must equal 'max_batch_size_'.
  absl::Status ValidateAllowedBatchSizes() const;

  // Initialize vars by reading from op-kernel-construction.
  // Vars
  // - enable_adaptive_batch_threads_
  //   true if value of attribute `kEnableAdaptiveSchedulerAttr` is true, or
  //   if `num_batch_threads` is not positive.
  // - adaptive_batch_scheduler_options_
  //   Read from corresponding attributes as long as they are set.
  void SetAdaptiveBatchSchedulerOptions(OpKernelConstruction* c,
                                        int32_t num_batch_threads);

  static int32 NumBatchThreadsFromEnvironmentWithDefault(
      int default_num_batch_threads);
  static thread::ThreadPool* GetOrCreateBatchThreadsPool();
  static constexpr int64_t kBatchThreadPoolSize = 128;

  std::string container_;
  std::string shared_name_;
  std::string batcher_queue_;
  int32_t num_batch_threads_;
  int32_t max_batch_size_;
  int32_t batch_timeout_micros_;
  int32_t max_enqueued_batches_;
  std::vector<int32_t> allowed_batch_sizes_;
  int32 low_priority_max_batch_size_;
  int32 low_priority_batch_timeout_micros_;
  int32 low_priority_max_enqueued_batches_;
  std::vector<int32> low_priority_allowed_batch_sizes_;
  std::string mixed_priority_policy_;
  bool enable_large_batch_splitting_;
  bool has_attribute_enable_large_batch_splitting_;
  bool disable_padding_;
  std::string batch_padding_policy_;

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
  std::optional<AdaptiveBatchSchedulerOptions>
      adaptive_batch_scheduler_options_ = std::nullopt;
};

// Legacy TF kernel which is a variant of tf.BatchFunction.
template <typename BatchResourceType>
class BatchFunctionFallbackKernel : public BatchFunctionFallbackKernelBase {
 public:
  using BatchFunctionType = typename BatchResourceType::BatchFunctionType;

  explicit BatchFunctionFallbackKernel(OpKernelConstruction* c)
      : BatchFunctionFallbackKernelBase(c) {
    int64_t handle;
    OP_REQUIRES_OK(c, c->GetAttr("opaque_function_handle", &handle));
    batch_function_ = BatchResourceType::CastHandleToFunction(handle);
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final;

 private:
  BatchFunctionType batch_function_;
};

template <typename BatchResourceType>
void BatchFunctionFallbackKernel<BatchResourceType>::ComputeAsync(
    OpKernelContext* c, DoneCallback done) {
  RecordBatchSplitUsage(has_attribute_enable_large_batch_splitting_
                            ? std::make_optional(enable_large_batch_splitting_)
                            : std::nullopt,
                        GetModelName(c));
  RecordBatchParamNumBatchThreads(num_batch_threads_, GetModelName(c));
  OP_REQUIRES_VALUE(tfrt::ResourceContext * client_graph_resource_context, c,
                    BatchResourceType::GetClientGraphResourceContext(c));
  OP_REQUIRES_ASYNC(
      c, client_graph_resource_context != nullptr,
      errors::FailedPrecondition("client graph resource context not found"),
      done);
  std::function<
      absl::StatusOr<tensorflow::core::RefCountPtr<BatchResourceType>>()>
      creator;
  if (adaptive_batch_scheduler_options_ != std::nullopt) {
    creator = [this, c]()
        -> absl::StatusOr<tensorflow::core::RefCountPtr<BatchResourceType>> {
      serving::AdaptiveSharedBatchScheduler<
          serving::BatchResourceBase::BatchTask>::Options
          adaptive_shared_batch_scheduler_options;
      adaptive_shared_batch_scheduler_options.thread_pool_name =
          "adaptive_batch_threads";
      adaptive_shared_batch_scheduler_options.num_batch_threads =
          adaptive_batch_scheduler_options_->max_in_flight_batches_limit;
      adaptive_shared_batch_scheduler_options.thread_pool =
          GetOrCreateBatchThreadsPool();

      // When we explicitly specify 'thread_pool', you'd think ASBS would ignore
      // 'num_batch_threads', but in fact ASBS still uses num_batch_threads as
      // the max number of in-flight batches.  It makes no sense to have more
      // in-flight batches than threads (it would result in strictly bad
      // batching decisions), so we cap this parameter (which otherwise comes
      // from the saved model) to the actual number of batch threads (which
      // comes from a process-wide environment variable).
      //
      // We have to apply the same capping to min_ and initial_
      // in_flight_batches_limit below to produce valid configurations.
      adaptive_shared_batch_scheduler_options.num_batch_threads = std::min(
          NumBatchThreadsFromEnvironmentWithDefault(kBatchThreadPoolSize),
          adaptive_batch_scheduler_options_->max_in_flight_batches_limit);

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
          std::min(
              NumBatchThreadsFromEnvironmentWithDefault(kBatchThreadPoolSize),
              adaptive_batch_scheduler_options_->min_in_flight_batches_limit);
      adaptive_shared_batch_scheduler_options
          .initial_in_flight_batches_limit = std::min(
          NumBatchThreadsFromEnvironmentWithDefault(kBatchThreadPoolSize),
          adaptive_batch_scheduler_options_->initial_in_flight_batches_limit);
      adaptive_shared_batch_scheduler_options.batches_to_average_over =
          adaptive_batch_scheduler_options_->batches_to_average_over;
      adaptive_shared_batch_scheduler_options.fifo_scheduling = true;

      std::unique_ptr<BatchResourceType> new_resource;
      auto status = BatchResourceType::Create(
          c, adaptive_shared_batch_scheduler_options, max_batch_size_,
          batch_timeout_micros_, max_enqueued_batches_, allowed_batch_sizes_,
          batch_function_, disable_padding_, &new_resource);
      if (!status.ok()) return status;
      if (c->session_metadata() != nullptr) {
        new_resource->set_session_metadata(*c->session_metadata());
      }
      return tensorflow::core::RefCountPtr<BatchResourceType>(
          new_resource.release());
    };
  } else {
    creator = [this, c]()
        -> absl::StatusOr<tensorflow::core::RefCountPtr<BatchResourceType>> {
      serving::BatchResourceOptions batch_resource_options;
      TF_ASSIGN_OR_RETURN(
          batch_resource_options.mixed_priority_batching_policy,
          serving::GetMixedPriorityBatchingPolicy(mixed_priority_policy_));
      batch_resource_options.num_batch_threads = num_batch_threads_;
      batch_resource_options.max_batch_size = max_batch_size_;
      batch_resource_options.batch_timeout_micros = batch_timeout_micros_;
      batch_resource_options.max_enqueued_batches = max_enqueued_batches_;
      batch_resource_options.allowed_batch_sizes = allowed_batch_sizes_;
      batch_resource_options.batch_padding_policy = batch_padding_policy_;
      batch_resource_options.low_priority_max_batch_size =
          low_priority_max_batch_size_;
      batch_resource_options.low_priority_batch_timeout_micros =
          low_priority_batch_timeout_micros_;
      batch_resource_options.low_priority_max_enqueued_batches =
          low_priority_max_enqueued_batches_;
      batch_resource_options.low_priority_allowed_batch_sizes =
          low_priority_allowed_batch_sizes_;

      serving::ModelBatchStats& model_batch_stats =
          serving::GlobalBatchStatsRegistry().model(
              /* model_name= */ std::string(GetModelName(c)),
              /* op_name= */ c->op_kernel().name());
      model_batch_stats.SetBatchTimeoutMicros(batch_timeout_micros_);
      model_batch_stats.SetNumBatchThreads(num_batch_threads_);

      std::unique_ptr<BatchResourceType> new_resource;
      auto status = BatchResourceType::Create(
          c, batch_resource_options, batch_function_,
          enable_large_batch_splitting_, disable_padding_, &new_resource);
      if (!status.ok()) return status;
      if (c->session_metadata() != nullptr) {
        new_resource->set_session_metadata(*c->session_metadata());
      }
      return tensorflow::core::RefCountPtr<BatchResourceType>(
          new_resource.release());
    };
  }

  auto br = client_graph_resource_context->GetOrCreateResource<
      tensorflow::core::RefCountPtr<BatchResourceType>>(shared_name_, creator);
  if (!br.ok()) OP_REQUIRES_OK_ASYNC(c, br.status(), done);
  auto expected_name = BatchResourceType::GetBatchFunctionName(batch_function_);
  auto received_name =
      BatchResourceType::GetBatchFunctionName((*br)->get()->batch_function());

  // TODO(b/187173237): When we can guarantee only 1 copy of BEF function is
  // generated for the batched function, we can assert the pointers are equal
  OP_REQUIRES_ASYNC(
      c, expected_name == received_name,
      errors::InvalidArgument(absl::StrCat(
          "Provided BEF function doesn't match with BatchResource. Expected:",
          expected_name, " Received:", received_name)),
      done);
  const uint64_t guid = random::New64();
  auto create_batch_task_fn = [c]() {
    return BatchResourceType::CreateBatchTask(c);
  };
  absl::Status status;
  if (serving::ShouldWarmupAllBatchSizes(c)) {
    status = (*br)->get()->RegisterWarmupInputs(guid, c, batcher_queue_,
                                                create_batch_task_fn, done);
  } else {
    status = (*br)->get()->RegisterInput(guid, c, batcher_queue_,
                                         create_batch_task_fn, done);
  }
  OP_REQUIRES_OK_ASYNC(c, status, done);
  // Assume br calls done, so nothing to do here.
}

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_FALLBACK_BATCH_KERNEL_H_
