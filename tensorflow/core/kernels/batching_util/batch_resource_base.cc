/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/cost_constants.h"
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/cost_util.h"
#include "tensorflow/core/common_runtime/request_cost.h"
#include "tensorflow/core/common_runtime/request_cost_accessor.h"
#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"
#include "tensorflow/core/kernels/batching_util/concat_split_util.h"
#include "tensorflow/core/kernels/batching_util/input_split_metadata.h"
#include "tensorflow/core/kernels/batching_util/threadsafe_status.h"
#include "tensorflow/core/kernels/batching_util/warmup.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/percentile_sampler.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/util/incremental_barrier.h"
#include "tsl/platform/criticality.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace serving {
namespace {

// TODO(b/181883417): Replace with RecordPaddingSizeV2.
void RecordPaddingSize(int32_t padding_size, const string& model_name,
                       int32_t execution_batch_size, const string& op_name) {
  static auto* cell = tensorflow::monitoring::PercentileSampler<3>::New(
      {"/tensorflow/serving/batching/padding_size",
       "Tracks the padding size distribution on batches by model_name (if "
       "available).",
       "model_name", "execution_batch_size", "op_name"},
      /*percentiles=*/{25.0, 50.0, 75.0, 90.0, 95.0, 99.0},
      /*max_samples=*/1024, tensorflow::monitoring::UnitOfMeasure::kNumber);
  cell->GetCell(model_name, absl::StrCat(execution_batch_size), op_name)
      ->Add(static_cast<double>(padding_size));
}

void RecordPaddingSizeV2(int32_t padding_size, const string& model_name,
                         int32_t execution_batch_size, const string& op_name) {
  // Bucket containing 0 has bounds [-2/3, 2/3).
  // Remaining buckets are centered at powers of 2 and have bounds:
  // [(2/3) * 2^i, (4/3) * 2^i) for i = 1, ..., 13.
  // Largest bucket has range: [(2/3) *  2^14, DBL_MAX]

  std::vector<double> bucket_limits;
  // populate bound for zero bucket
  bucket_limits.push_back(-2.0 / 3.0);
  // populate rest of bounds
  double bound = 2.0 / 3.0;
  double growth_factor = 2;
  for (int i = 0; i < 16; i++) {
    bucket_limits.push_back(bound);
    bound *= growth_factor;
  }

  static auto* cell = tensorflow::monitoring::Sampler<3>::New(
      {"/tensorflow/serving/batching/padding_size_v2",
       "Tracks the padding size distribution on batches by model_name (if "
       "available).",
       "model_name", "execution_batch_size", "op_name"},
      monitoring::Buckets::Explicit(bucket_limits));
  cell->GetCell(model_name, absl::StrCat(execution_batch_size), op_name)
      ->Add(static_cast<double>(padding_size));
}

// TODO(b/181883417): Replace with RecordInputBatchSizeV2.
void RecordInputBatchSize(int32_t batch_size, const string& model_name,
                          const string& op_name) {
  static auto* cell = tensorflow::monitoring::PercentileSampler<2>::New(
      {"/tensorflow/serving/batching/input_batch_size",
       "Tracks the batch size distribution on the inputs by model_name (if "
       "available).",
       "model_name", "op_name"},
      /*percentiles=*/{25.0, 50.0, 75.0, 90.0, 95.0, 99.0},
      /*max_samples=*/1024, tensorflow::monitoring::UnitOfMeasure::kNumber);
  cell->GetCell(model_name, op_name)->Add(static_cast<double>(batch_size));
}

void RecordInputBatchSizeV2(int32_t batch_size, const string& model_name,
                            const string& op_name) {
  static auto* cell = tensorflow::monitoring::Sampler<2>::New(
      {"/tensorflow/serving/batching/input_batch_size_v2",
       "Tracks the batch size distribution on the inputs by model_name (if "
       "available).",
       "model_name", "op_name"},
      // Buckets centered at powers of 2, and have bounds:
      // [(2/3) * 2^i, (4/3) * 2^i] for i = 0, ..., 13.
      // Largest bucket has range: [(2/3) *  2^14, DBL_MAX]
      monitoring::Buckets::Exponential(2.0 / 3.0, 2, 15));
  cell->GetCell(model_name, op_name)->Add(static_cast<double>(batch_size));
}

// Record the actual batch size without padding.
void RecordBatchSize(int32_t batch_size, const string& model_name,
                     const string& op_name) {
  static auto* cell = tensorflow::monitoring::Sampler<2>::New(
      {"/tensorflow/serving/batching/batch_size",
       "Tracks the batch size distribution on the batch result by model_name "
       "(if available).",
       "model_name", "op_name"},
      monitoring::Buckets::Exponential(1, 1.5, 20));
  cell->GetCell(model_name, op_name)->Add(static_cast<double>(batch_size));
}

void RecordProcessedBatchSize(int32_t batch_size, const string& model_name,
                              const string& op_name) {
  static auto* cell = tensorflow::monitoring::PercentileSampler<2>::New(
      {"/tensorflow/serving/batching/processed_batch_size",
       "Tracks the batch size distribution on processing by model_name (if "
       "available).",
       "model_name", "op_name"},
      /*percentiles=*/{25.0, 50.0, 75.0, 90.0, 95.0, 99.0},
      /*max_samples=*/1024, tensorflow::monitoring::UnitOfMeasure::kNumber);
  cell->GetCell(model_name, op_name)->Add(static_cast<double>(batch_size));
}

// Export the exact number instead of the distribution of processed batch size.
void RecordProcessedBatchSizeV2(int32_t batch_size, const string& model_name,
                                const string& op_name) {
  static auto* cell = monitoring::Counter<3>::New(
      "/tensorflow/serving/batching/processed_batch_size_v2",
      "Tracks the batch size on processing by model_name and op name (if "
      "available).",
      "model_name", "op_name", "batch_size");
  cell->GetCell(model_name, op_name, std::to_string(batch_size))
      ->IncrementBy(1);
}

// TODO(b/181883417): Replace with RecordBatchDelayUsV2.
void RecordBatchDelayUs(int64_t batch_delay_us, const string& model_name,
                        const string& op_name, int32_t batch_size) {
  static auto* cell = monitoring::PercentileSampler<3>::New(
      {"/tensorflow/serving/batching/batch_delay_us",
       "Tracks the batching delay (in microseconds) for inputs by model_name "
       "(if available).",
       "model_name", "op_name", "processed_batch_size"},
      /*percentiles=*/{25.0, 50.0, 75.0, 90.0, 95.0, 99.0},
      /*max_samples=*/1024, monitoring::UnitOfMeasure::kTime);
  cell->GetCell(model_name, op_name, std::to_string(batch_size))
      ->Add(static_cast<double>(batch_delay_us));
}

void RecordBatchDelayUsV2(int64_t batch_delay_us, const string& model_name,
                          const string& op_name, int32_t batch_size) {
  static auto* cell = tensorflow::monitoring::Sampler<3>::New(
      {"/tensorflow/serving/batching/batch_delay_us_v2",
       "Tracks the batching delay (in microseconds) for inputs by model_name "
       "(if available).",
       "model_name", "op_name", "processed_batch_size"},
      // It's 27 buckets with the last bucket being 2^26 to DBL_MAX;
      // so the limits are [1, 2, 4, 8, ..., 64 * 1024 * 1024, DBL_MAX].
      monitoring::Buckets::Exponential(1, 2, 27));
  cell->GetCell(model_name, op_name, std::to_string(batch_size))
      ->Add(static_cast<double>(batch_delay_us));
}

void RecordBatchTaskSizeSum(int32_t batch_task_size,
                            int32_t unbatched_task_size,
                            const string& model_name, const string& op_name) {
  static auto* cell = tensorflow::monitoring::Counter<3>::New(
      "/tensorflow/serving/batching/batch_task_size_sum",
      "Tracks the sum of the task sizes in a batch.", "model_name", "op_name",
      "is_batched");
  cell->GetCell(model_name, op_name, "true")->IncrementBy(batch_task_size);
  cell->GetCell(model_name, op_name, "false")->IncrementBy(unbatched_task_size);
}

void RecordBatchParamBatchTimeoutMicros(int64_t batch_timeout_micros,
                                        const string& model_name,
                                        const string& op_name) {
  static auto* cell = monitoring::Gauge<int64_t, 2>::New(
      "/tensorflow/serving/batching/batch_timeout_micros",
      "Tracks how long a request can wait before being processed by a batch.",
      "model_name", "op_name");
  cell->GetCell(model_name, op_name)->Set(batch_timeout_micros);
}

void RecordBatchParamMaxBatchSize(int64_t max_batch_size,
                                  const string& model_name,
                                  const string& op_name) {
  static auto* cell = monitoring::Gauge<int64_t, 2>::New(
      "/tensorflow/serving/batching/max_batch_size",
      "Tracks the maximum size of a batch.", "model_name", "op_name");
  cell->GetCell(model_name, op_name)->Set(max_batch_size);
}

void RecordBatchParamPaddingPolicy(const string& batch_padding_policy,
                                   const string& model_name,
                                   const string& op_name) {
  static auto* cell = monitoring::Gauge<string, 2>::New(
      "/tensorflow/serving/batching/configured_batch_padding_policy",
      "The value of BatchFunction.batch_padding_policy attribute.",
      "model_name", "op_name");
  cell->GetCell(model_name, op_name)->Set(batch_padding_policy);
}

void RecordBatchParamMaxEnqueuedBatches(int64_t max_enqueued_batches,
                                        const string& model_name,
                                        const string& op_name) {
  static auto* cell = monitoring::Gauge<int64_t, 2>::New(
      "/tensorflow/serving/batching/max_enqueued_batches",
      "Tracks the maximum number of enqueued batches.", "model_name",
      "op_name");
  cell->GetCell(model_name, op_name)->Set(max_enqueued_batches);
}

void RecordBatchParamAllowedBatchSizes(const string& allowed_batch_sizes,
                                       const string& model_name,
                                       const string& op_name) {
  static auto* cell = monitoring::Gauge<string, 2>::New(
      "/tensorflow/serving/batching/allowed_batch_sizes",
      "Tracks the sizes that are allowed to form a batch.", "model_name",
      "op_name");
  cell->GetCell(model_name, op_name)->Set(allowed_batch_sizes);
}

void RecordBatchCosts(const std::string& model_name,
                      const int64_t processed_size,
                      const absl::string_view cost_type,
                      const absl::Duration total_cost) {
  static auto* cell = tensorflow::monitoring::Sampler<3>::New(
      {"/tensorflow/serving/batching/costs",
       "Tracks the batch costs (in microseconds) by model name and processed "
       "size.",
       "model_name", "processed_size", "cost_type"},
      // It's 27 buckets with the last bucket being 2^26 to DBL_MAX;
      // so the limits are [1, 2, 4, 8, ..., 64 * 1024 * 1024 (~64s), DBL_MAX].
      monitoring::Buckets::Exponential(1, 2, 27));
  cell->GetCell(model_name, std::to_string(processed_size),
                std::string(cost_type))
      ->Add(absl::ToDoubleMicroseconds(total_cost));
}

const string& GetModelName(OpKernelContext* ctx) {
  static string* kModelNameUnset = new string("model_name_unset");
  if (!ctx->session_metadata()) return *kModelNameUnset;
  if (ctx->session_metadata()->name().empty()) return *kModelNameUnset;
  return ctx->session_metadata()->name();
}

// Returns the sum of the task sizes. The caller must guarantee that the
// unique_ptrs in the argument vectors are not null.
int GetTotalTaskSize(
    const std::vector<std::unique_ptr<BatchResourceBase::BatchTask>>& tasks) {
  int tasks_size = 0;
  for (const auto& task : tasks) {
    tasks_size += task->size();
  }
  return tasks_size;
}

}  // namespace

std::unique_ptr<BatchResourceBase::BatchTask>
BatchResourceBase::BatchTask::CreateSplitTask(
    int split_index, AsyncOpKernel::DoneCallback done_callback) {
  std::unique_ptr<BatchTask> task = CreateDerivedTask();

  task->guid = this->guid;
  task->propagated_context = Context(ContextKind::kThread);
  task->inputs.reserve(this->inputs.size());
  task->captured_inputs = this->captured_inputs;
  task->context = this->context;
  task->done_callback = done_callback;
  task->split_index = split_index;
  task->output = this->output;
  task->status = this->status;
  task->is_partial = true;
  task->start_time = this->start_time;
  task->request_cost = this->request_cost;
  task->forced_warmup_batch_size = this->forced_warmup_batch_size;

  return task;
}

using ::tensorflow::concat_split_util::Concat;
using ::tensorflow::concat_split_util::Split;
using TensorMatrix = std::vector<std::vector<Tensor>>;

string GetTensorNamesAndShapesString(const OpKernelContext* context,
                                     const OpInputList& tensors) {
  std::stringstream out;
  int i = 0;
  for (const Tensor& tensor : tensors) {
    out << " - " << context->op_kernel().requested_input(i++) << " has shape "
        << tensor.shape().DebugString() << "\n";
  }
  return out.str();
}

absl::Status BatchResourceBase::RegisterWarmupInputs(
    int64_t guid, OpKernelContext* context, const string& batcher_queue_name,
    const CreateBatchTaskFn& create_batch_task_fn,
    AsyncOpKernel::DoneCallback done) {
  auto shared_status = std::make_shared<ThreadSafeStatus>();
  auto create_batch_task_fn_share_status = [&create_batch_task_fn,
                                            &shared_status]() {
    auto batch_task = create_batch_task_fn();
    if (!batch_task.ok()) {
      return batch_task;
    }
    (*batch_task)->status = shared_status;
    return batch_task;
  };
  auto warmup_counter =
      std::make_shared<absl::BlockingCounter>(allowed_batch_sizes_.size());
  // Enqueue warmup batches.
  for (int i = 0; i < allowed_batch_sizes_.size(); ++i) {
    absl::Status status = RegisterInput(
        guid, context, batcher_queue_name, create_batch_task_fn_share_status,
        [warmup_counter = warmup_counter.get()]() {
          warmup_counter->DecrementCount();
        },
        allowed_batch_sizes_[i]);
    if (!status.ok()) return status;
  }
  // Enqueue real batch if the other batches were enqueued successfully.
  return RegisterInput(
      guid, context, batcher_queue_name, create_batch_task_fn_share_status,
      [warmup_counter, context, shared_status, done = std::move(done)]() {
        warmup_counter->Wait();
        context->SetStatus(shared_status->status());
        done();
      });
}

absl::Status BatchResourceBase::RegisterInput(
    int64_t guid, OpKernelContext* context, const string& batcher_queue_name,
    const CreateBatchTaskFn& create_batch_task_fn,
    AsyncOpKernel::DoneCallback done_callback, int forced_warmup_batch_size) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BatchTask> batch_components,
                      create_batch_task_fn());
  batch_components->start_time = EnvTime::NowNanos();
  batch_components->guid = guid;
  batch_components->propagated_context = Context(ContextKind::kThread);

  OpInputList tensors;
  TF_RETURN_IF_ERROR(context->input_list("in_tensors", &tensors));
  batch_components->inputs.reserve(tensors.size());
  for (const Tensor& tensor : tensors) {
    if (tensor.shape().dims() == 0) {
      return errors::InvalidArgument(
          "Batching input tensors must have at least one dimension.\nBelow are "
          "the input tensors: \n",
          GetTensorNamesAndShapesString(context, tensors));
    }
    if (tensors.size() >= 2 &&
        tensor.shape().dim_size(0) != tensors[0].shape().dim_size(0)) {
      return errors::InvalidArgument(
          "Batching input tensors supplied in a given op invocation must "
          "have equal 0th-dimension size.\nBelow are the input tensors: \n",
          GetTensorNamesAndShapesString(context, tensors));
    }
    batch_components->inputs.push_back(tensor);
  }
  RecordInputBatchSize(tensors[0].shape().dim_size(0), GetModelName(context),
                       context->op_kernel().name());
  RecordInputBatchSizeV2(tensors[0].shape().dim_size(0), GetModelName(context),
                         context->op_kernel().name());
  if (batcher_) {
    RecordBatchParamBatchTimeoutMicros(
        batcher_queue_options_.batch_timeout_micros, GetModelName(context),
        context->op_kernel().name());
    RecordBatchParamMaxBatchSize(
        batcher_queue_options_.max_execution_batch_size, GetModelName(context),
        context->op_kernel().name());
    RecordBatchParamMaxEnqueuedBatches(
        batcher_queue_options_.max_enqueued_batches, GetModelName(context),
        context->op_kernel().name());
    RecordBatchParamPaddingPolicy(
        this->batcher_queue_options_.batch_padding_policy,
        GetModelName(context), context->op_kernel().name());
  } else if (adaptive_batcher_) {
    RecordBatchParamBatchTimeoutMicros(
        adaptive_batcher_queue_options_.batch_timeout_micros,
        GetModelName(context), context->op_kernel().name());
    RecordBatchParamMaxBatchSize(adaptive_batcher_queue_options_.max_batch_size,
                                 GetModelName(context),
                                 context->op_kernel().name());
    RecordBatchParamMaxEnqueuedBatches(
        adaptive_batcher_queue_options_.max_enqueued_batches,
        GetModelName(context), context->op_kernel().name());
  } else {
    return errors::Internal("No batcher defined.");
  }
  RecordBatchParamAllowedBatchSizes(allowed_batch_sizes_str_,
                                    GetModelName(context),
                                    context->op_kernel().name());

  // Degenerate case where the input is empty. Just return an empty tensor.
  if (tensors[0].shape().dim_size(0) == 0) {
    for (int i = 0; i < context->num_outputs(); i++) {
      Tensor* empty_output;
      AllocatorAttributes cpu_alloc;
      cpu_alloc.set_on_host(true);
      TF_RETURN_IF_ERROR(context->allocate_output(i, TensorShape({0}),
                                                  &empty_output, cpu_alloc));
    }
    done_callback();
    return absl::OkStatus();
  }
  OpInputList captured_tensors;
  const auto captured_status =
      context->input_list("captured_tensors", &captured_tensors);
  if (captured_status.ok()) {
    batch_components->captured_inputs.reserve(captured_tensors.size());
    for (const Tensor& captured_tensor : captured_tensors) {
      batch_components->captured_inputs.push_back(captured_tensor);
    }
  }
  batch_components->context = context;
  batch_components->split_index = 0;
  batch_components->output = std::make_shared<TensorMatrix>();
  if (!batch_components->status) {
    // A shared status has already been injected if `RegisterWarmupInputs`
    // was called. If not, create the `ThreadSafeStatus` and tie the setting
    // of the kernel context's status to this shared status.
    batch_components->status = std::make_shared<ThreadSafeStatus>();
    batch_components->done_callback = [done_callback = std::move(done_callback),
                                       shared_status = batch_components->status,
                                       context = context]() {
      context->SetStatus(shared_status->status());
      done_callback();
    };
  } else {
    // Otherwise `RegisterWarmupInputs` was called and already setup the
    // `done_callback` and `status` correctly for this `BatchTask`.
    batch_components->done_callback = std::move(done_callback);
  }
  batch_components->forced_warmup_batch_size = forced_warmup_batch_size;

  std::unique_ptr<RequestCostAccessor> request_cost_accessor =
      CreateRequestCostAccessor();
  if (request_cost_accessor) {
    batch_components->request_cost = request_cost_accessor->GetRequestCost();
  }

  BatcherQueueT* batcher_queue;
  TF_RETURN_IF_ERROR(LookupOrCreateBatcherQueue(
      /* queue_name= */ batcher_queue_name,
      /* model_name= */ GetModelName(context),
      /* op_name= */ context->op_kernel().name(), /* queue= */ &batcher_queue));

  if (!session_metadata().name().empty()) {
    absl::MutexLock lock(&outstanding_batch_mu_);
    WarmupStateRegistry::Key key(session_metadata().name(),
                                 session_metadata().version());
    if (GetGlobalWarmupStateRegistry().Lookup(key)) {
      outstanding_batch_mu_.Await({+[](int* num_outstanding_batched_items) {
                                     return *num_outstanding_batched_items == 0;
                                   },
                                   &num_outstanding_batched_items_});
    }
    num_outstanding_batched_items_ += batch_components->size();
  }

  return batcher_queue->Schedule(&batch_components);
}

/*static*/ BatchResourceBase::BatcherT::QueueOptions
BatchResourceBase::GetBatcherQueueOptions(
    int32_t num_batch_threads, int32_t max_batch_size,
    int32_t batch_timeout_micros, int32_t max_enqueued_batches,
    const std::vector<int32>& allowed_batch_sizes,
    bool enable_large_batch_splitting, bool disable_padding) {
  return GetBatcherQueueOptions(
      num_batch_threads, max_batch_size, batch_timeout_micros,
      max_enqueued_batches, allowed_batch_sizes, enable_large_batch_splitting,
      disable_padding,
      /*batch_padding_policy=*/kPadUpPolicy,
      /*low_priority_max_batch_size=*/0,
      /*low_priority_batch_timeout_micros=*/0,
      /*low_priority_max_enqueued_batches=*/0,
      /*low_priority_allowed_batch_sizes=*/{},
      /*mixed_priority_batching_policy*/
      MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize);
}

/*static*/ BatchResourceBase::BatcherT::QueueOptions
BatchResourceBase::GetBatcherQueueOptions(
    int32_t num_batch_threads, int32_t max_batch_size,
    int32_t batch_timeout_micros, int32_t max_enqueued_batches,
    const std::vector<int32>& allowed_batch_sizes,
    bool enable_large_batch_splitting, bool disable_padding,
    absl::string_view batch_padding_policy, int32_t low_priority_max_batch_size,
    int32_t low_priority_batch_timeout_micros,
    int32_t low_priority_max_enqueued_batches,
    const std::vector<int32>& low_priority_allowed_batch_sizes,
    MixedPriorityBatchingPolicy mixed_priority_batching_policy) {
  BatcherT::QueueOptions batcher_queue_options;
  batcher_queue_options.input_batch_size_limit = max_batch_size;
  batcher_queue_options.max_enqueued_batches = max_enqueued_batches;
  batcher_queue_options.batch_timeout_micros = batch_timeout_micros;
  batcher_queue_options.batch_padding_policy =
      std::string(batch_padding_policy);
  if (low_priority_max_batch_size > 0) {
    batcher_queue_options.enable_priority_queue = true;
  }
  batcher_queue_options.high_priority_queue_options.input_batch_size_limit =
      max_batch_size;
  batcher_queue_options.high_priority_queue_options.max_enqueued_batches =
      max_enqueued_batches;
  batcher_queue_options.high_priority_queue_options.batch_timeout_micros =
      batch_timeout_micros;
  batcher_queue_options.low_priority_queue_options.input_batch_size_limit =
      low_priority_max_batch_size;
  batcher_queue_options.low_priority_queue_options.max_enqueued_batches =
      low_priority_max_enqueued_batches;
  batcher_queue_options.low_priority_queue_options.batch_timeout_micros =
      low_priority_batch_timeout_micros;
  if (low_priority_allowed_batch_sizes.empty()) {
    batcher_queue_options.low_priority_queue_options.max_execution_batch_size =
        low_priority_max_batch_size;
  } else {
    batcher_queue_options.low_priority_queue_options.max_execution_batch_size =
        *low_priority_allowed_batch_sizes.rbegin();
  }
  batcher_queue_options.low_priority_queue_options.allowed_batch_sizes =
      low_priority_allowed_batch_sizes;
  batcher_queue_options.mixed_priority_batching_policy =
      mixed_priority_batching_policy;
  batcher_queue_options.enable_large_batch_splitting =
      enable_large_batch_splitting;
  if (enable_large_batch_splitting) {
    batcher_queue_options.split_input_task_func =
        [](std::unique_ptr<BatchTask>* input_task,
           int open_batch_remaining_slot, int max_batch_size,
           std::vector<std::unique_ptr<BatchTask>>* output_tasks)
        -> absl::Status {
      return SplitInputTask(input_task, open_batch_remaining_slot,
                            max_batch_size, output_tasks);
    };

    if (allowed_batch_sizes.empty()) {
      batcher_queue_options.max_execution_batch_size = max_batch_size;
      batcher_queue_options.high_priority_queue_options
          .max_execution_batch_size = max_batch_size;
    } else {
      batcher_queue_options.max_execution_batch_size =
          *allowed_batch_sizes.rbegin();
      batcher_queue_options.high_priority_queue_options
          .max_execution_batch_size = *allowed_batch_sizes.rbegin();
      batcher_queue_options.allowed_batch_sizes = allowed_batch_sizes;
    }
  }
  batcher_queue_options.disable_padding = disable_padding;

  return batcher_queue_options;
}

/*static*/ BatchResourceBase::AdaptiveBatcherT::QueueOptions
BatchResourceBase::GetAdaptiveBatcherQueueOptions(
    int32_t max_batch_size, int32_t batch_timeout_micros,
    int32_t max_enqueued_batches, bool enable_large_batch_splitting,
    const std::vector<int32>& allowed_batch_sizes, bool disable_padding) {
  AdaptiveBatcherT::QueueOptions batcher_queue_options;
  batcher_queue_options.max_input_task_size =
      std::make_optional(max_batch_size);
  batcher_queue_options.max_enqueued_batches = max_enqueued_batches;
  batcher_queue_options.batch_timeout_micros = batch_timeout_micros;
  if (allowed_batch_sizes.empty()) {
    batcher_queue_options.max_batch_size = max_batch_size;
  } else {
    batcher_queue_options.max_batch_size = *allowed_batch_sizes.rbegin();
  }

  if (enable_large_batch_splitting) {
    batcher_queue_options.split_input_task_func =
        [](std::unique_ptr<BatchTask>* input_task,
           int open_batch_remaining_slot, int max_batch_size,
           std::vector<std::unique_ptr<BatchTask>>* output_tasks)
        -> absl::Status {
      return SplitInputTask(input_task, open_batch_remaining_slot,
                            max_batch_size, output_tasks);
    };
  }
  batcher_queue_options.disable_padding = disable_padding;

  return batcher_queue_options;
}

/*static*/ absl::Status BatchResourceBase::ValidateBatch(const BatchT& batch) {
  for (int task_idx = 0; task_idx < batch.num_tasks(); ++task_idx) {
    const BatchResourceBase::BatchTask& task = batch.task(task_idx);

    if (task.inputs.size() != batch.task(0).inputs.size()) {
      return errors::InvalidArgument(
          "Batching inputs must have equal number of edges");
    }
  }

  return absl::OkStatus();
}

bool BatchResourceBase::IsLowPriorityBatch(const BatchT& batch) const {
  if (!batcher_queue_options_.enable_priority_queue) return false;
  if (batch.empty()) return false;

  // TODO(b/316379576): Once the criticality and priority become configurable,
  // this should rely on the batch parameters instead of the hard coded value.
  return batch.task(0).criticality() ==
             tsl::criticality::Criticality::kSheddablePlus ||
         batch.task(0).criticality() ==
             tsl::criticality::Criticality::kSheddable;
}

// Returns the smallest entry in 'allowed_batch_sizes_' that is greater than
// or equal to 'batch_size'. If 'allowed_batch_sizes_' is empty, simply
// returns 'batch_size'.
int BatchResourceBase::RoundToLowestAllowedBatchSize(
    int batch_size, bool is_low_priority_batch) const {
  const std::vector<int32>& allowed_batch_sizes =
      is_low_priority_batch ? batcher_queue_options_.low_priority_queue_options
                                  .allowed_batch_sizes
                            : allowed_batch_sizes_;

  return GetNextAllowedBatchSize(batch_size, allowed_batch_sizes,
                                 batcher_queue_options_.disable_padding);
}

absl::Status BatchResourceBase::ConcatInputTensors(
    const BatchT& batch,
    const std::vector<std::unique_ptr<BatchTask>>& unbatched_tasks,
    OpKernelContext* context, std::vector<Tensor>* concatenated_tensors) const {
  if (batch.num_tasks() == 0) {
    return errors::InvalidArgument("Empty batch.");
  }

  int unbatched_tasks_size = GetTotalTaskSize(unbatched_tasks);
  const bool just_for_warmup = batch.task(0).forced_warmup_batch_size > 0;
  const int padded_batch_size =
      just_for_warmup
          ? batch.task(0).forced_warmup_batch_size
          : RoundToLowestAllowedBatchSize(batch.size() + unbatched_tasks_size,
                                          IsLowPriorityBatch(batch));
  const int padding_amount =
      just_for_warmup ? padded_batch_size
                      : padded_batch_size - batch.size() - unbatched_tasks_size;
  tsl::profiler::TraceMe trace_me(
      [padded_batch_size, padding_amount,
       disable_padding = batcher_queue_options_.disable_padding]() {
        return tsl::profiler::TraceMeEncode(
            "ConcatInputTensors",
            {{"batch_size_after_padding", padded_batch_size},
             {"padding_amount", padding_amount},
             {"disable_padding", disable_padding}});
      });
  RecordBatchTaskSizeSum(batch.size(), unbatched_tasks_size,
                         GetModelName(context), context->op_kernel().name());

  // TODO(b/316379576): Add metrics for the breakdown between the size of the
  // original batch size and the unbatched task size and update the batch size
  // to include the unbatched tasks.
  RecordPaddingSize(padding_amount, GetModelName(context), padded_batch_size,
                    context->op_kernel().name());
  RecordPaddingSizeV2(padding_amount, GetModelName(context), padded_batch_size,
                      context->op_kernel().name());
  RecordProcessedBatchSize(padded_batch_size, GetModelName(context),
                           context->op_kernel().name());
  RecordProcessedBatchSizeV2(padded_batch_size, GetModelName(context),
                             context->op_kernel().name());
  RecordBatchSize(batch.size(), GetModelName(context),
                  context->op_kernel().name());

  // All tasks should have the same number of input edges.
  const int num_inputs = batch.task(0).inputs.size();
  concatenated_tensors->reserve(num_inputs);

  // Process each input one at a time (the typical case has just one). When
  // `just_for_warmup` is true, the real data is not added. Otherwise, the real
  // data is added to the front of each `concatenated_tensor`.
  for (int i = 0; i < num_inputs; ++i) {
    // Concatenate the tasks ith input tensors into a big output tensor.
    std::vector<Tensor> to_concatenate;
    if (just_for_warmup) {
      to_concatenate.reserve(padding_amount);
    } else {
      to_concatenate.reserve(batch.num_tasks() + unbatched_tasks.size() +
                             padding_amount);
      for (int task_idx = 0; task_idx < batch.num_tasks(); ++task_idx) {
        to_concatenate.push_back(batch.task(task_idx).inputs.at(i));
      }
      for (int task_idx = 0; task_idx < unbatched_tasks.size(); ++task_idx) {
        to_concatenate.push_back(unbatched_tasks[task_idx]->inputs.at(i));
      }
    }

    // Add padding as needed if padding is allowed. Use the first row of the
    // first task's tensor as the data for padding.
    if (padding_amount != 0) {
      const Tensor& padding_source = batch.task(0).inputs.at(i);
      Tensor padding;
      if (padding_source.shape().dim_size(0) == 0) {
        return errors::InvalidArgument(
            "Cannot use an empty tensor with zero rows as padding when "
            "batching. (Input ",
            i, " got shape ", padding_source.shape().DebugString(), ".)");
      }
      if (padding_source.shape().dim_size(0) == 1) {
        padding = padding_source;
      } else {
        padding = padding_source.Slice(0, 1);
      }
      for (int i = 0; i < padding_amount; ++i) {
        to_concatenate.push_back(padding);
      }
    }

    Tensor concatenated_tensor;
    absl::Status concat_status =
        Concat(context, to_concatenate, &concatenated_tensor);
    TF_RETURN_IF_ERROR(concat_status);
    concatenated_tensors->push_back(concatenated_tensor);
  }
  return absl::OkStatus();
}

/*static*/ absl::Status BatchResourceBase::SplitInputTask(
    std::unique_ptr<BatchTask>* input_task_ptr, int open_batch_remaining_slot,
    int max_batch_size, std::vector<std::unique_ptr<BatchTask>>* output_tasks) {
  BatchTask& input_task = *(*input_task_ptr);
  const int64_t input_task_size = input_task.size();

  DCHECK_GT(input_task_size, 0);

  std::shared_ptr<ThreadSafeStatus> shared_status = input_task.status;

  // `split_task_done_callback` runs only after all splitted tasks are
  // complete.
  std::function<void()> split_task_done_callback =
      [done_callback = input_task.done_callback, output = input_task.output,
       forced_warmup_batch_size = input_task.forced_warmup_batch_size,
       op_kernel_context = input_task.context,
       status = shared_status]() mutable {
        const int num_output = op_kernel_context->num_outputs();
        for (int i = 0; i < num_output; ++i) {
          Tensor output_tensor;

          // Concat would memcpy each input tensor to one output tensor.
          // In this context, Concat can be further optimized to get rid of
          // some (probably all) memcpy when input tensors are slices of
          // another copy.
          std::vector<Tensor> to_concatenate;
          to_concatenate.reserve(output->size());
          for (int j = 0; j < output->size(); ++j) {
            to_concatenate.push_back(std::move((*output)[j][i]));
          }
          const auto concat_status =
              Concat(op_kernel_context, to_concatenate, &output_tensor);
          if (!concat_status.ok()) {
            status->Update(concat_status);
          }
          if (forced_warmup_batch_size == 0) {
            op_kernel_context->set_output(i, std::move(output_tensor));
          }
        }
        done_callback();
      };
  IncrementalBarrier barrier(split_task_done_callback);

  const internal::InputSplitMetadata input_split_metadata(
      input_task_size, open_batch_remaining_slot, max_batch_size);

  const absl::FixedArray<int>& task_sizes = input_split_metadata.task_sizes();
  const int num_batches = task_sizes.size();
  std::vector<int64_t> output_task_sizes;
  output_task_sizes.resize(num_batches);
  for (int i = 0; i < num_batches; i++) {
    output_task_sizes[i] = task_sizes[i];
  }

  input_task.output->resize(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    (*input_task.output)[i].resize(input_task.context->num_outputs());
  }

  output_tasks->reserve(num_batches);
  for (int i = 0; i < num_batches; i++) {
    output_tasks->push_back(input_task.CreateSplitTask(i, barrier.Inc()));
  }

  const int num_input_tensors = input_task.inputs.size();

  // Splits each input tensor according to `output_task_sizes`, and
  // initializes input of `output_tasks` with split results.
  for (int i = 0; i < num_input_tensors; ++i) {
    std::vector<Tensor> split_tensors;
    const Tensor& input_tensor = input_task.inputs[i];
    // TODO(b/154140947):
    // Figure out the optimal implementation of Split, by using
    // 'Tensor::Slice' and eliminating unnecessary memcpy as much as possible.
    const absl::Status split_status = Split(input_task.context, input_tensor,
                                            output_task_sizes, &split_tensors);
    if (!split_status.ok()) {
      return errors::Internal(
          "When splitting input, Tensor split operation failed: ",
          split_status.message());
    }
    if (split_tensors.size() != output_task_sizes.size()) {
      return errors::Internal(
          "When splitting input, tensor split operation did not work as "
          "expected; got ",
          split_tensors.size(), " splits; expected ", output_task_sizes.size());
    }
    for (int j = 0; j < output_tasks->size(); ++j) {
      BatchTask& output_task = *((*output_tasks)[j]);
      auto moved_tensor_iter = std::next(split_tensors.begin(), j);
      std::move(moved_tensor_iter, moved_tensor_iter + 1,
                std::back_inserter(output_task.inputs));
    }
  }
  return absl::OkStatus();
}

absl::Status BatchResourceBase::SplitOutputTensors(
    const std::vector<Tensor>& combined_outputs, BatchT* batch,
    std::vector<std::unique_ptr<BatchTask>>& unbatched_tasks) const {
  DCHECK_GE(batch->num_tasks(), 1);
  if (batch->num_tasks() < 1) {
    return errors::Internal("Batch size expected to be positive; was ",
                            batch->num_tasks());
  }

  std::vector<int64_t> task_sizes_plus_optional_padding;
  task_sizes_plus_optional_padding.reserve(batch->num_tasks() +
                                           unbatched_tasks.size());
  for (int i = 0; i < batch->num_tasks(); ++i) {
    task_sizes_plus_optional_padding.push_back(batch->task(i).size());
  }
  for (int i = 0; i < unbatched_tasks.size(); ++i) {
    task_sizes_plus_optional_padding.push_back(unbatched_tasks[i]->size());
  }
  int unbatched_tasks_size = GetTotalTaskSize(unbatched_tasks);
  const int padding_size =
      batcher_queue_options_.disable_padding
          ? 0
          : RoundToLowestAllowedBatchSize(batch->size() + unbatched_tasks_size,
                                          IsLowPriorityBatch(*batch)) -
                batch->size() - unbatched_tasks_size;
  if (padding_size > 0) {
    task_sizes_plus_optional_padding.push_back(padding_size);
  }

  DCHECK_EQ(batch->task(0).context->num_outputs(), combined_outputs.size());
  int combined_outputs_size = combined_outputs.size();
  if (combined_outputs_size != batch->task(0).context->num_outputs()) {
    return errors::Internal("Wrong number of batched output tensors");
  }

  // Split each element of `combined_outputs` according to task sizes
  // within the batch, and use this to populate context outputs.
  for (int i = 0, iter_limit = combined_outputs.size(); i < iter_limit; ++i) {
    const Tensor& output_tensor = combined_outputs[i];
    if (output_tensor.shape().dims() == 0) {
      return errors::FailedPrecondition(
          "Batched output tensor has 0 dimensions");
    }
    int64_t zeroth_dim_output_tensor_size = output_tensor.shape().dim_size(0);
    if (zeroth_dim_output_tensor_size !=
        static_cast<int64_t>(batch->size() + unbatched_tasks_size +
                             padding_size)) {
      return errors::FailedPrecondition(
          "Batched output tensor's 0th dimension does not equal the sum of "
          "the 0th dimension sizes of the input tensors. "
          "0th dimension size: ",
          zeroth_dim_output_tensor_size, "; batch size: ", batch->size(),
          "; unbatched tasks size: ", unbatched_tasks_size,
          "; padding size: ", padding_size);
    }

    std::vector<Tensor> split_tensor;
    const absl::Status split_status = tensor::Split(
        output_tensor, task_sizes_plus_optional_padding, &split_tensor);
    DCHECK(split_status.ok()) << split_status;
    if (!split_status.ok()) {
      return errors::Internal("Tensor split operation failed: ",
                              split_status.message());
    }
    DCHECK_EQ(split_tensor.size(), task_sizes_plus_optional_padding.size());
    if (split_tensor.size() != task_sizes_plus_optional_padding.size()) {
      return errors::Internal(
          "Tensor split operation did not work as expected; got ",
          split_tensor.size(), " splits; expected ",
          task_sizes_plus_optional_padding.size());
    }

    // Ignore a possible final split_tensors entry containing the padding.
    for (int j = 0; j < batch->num_tasks(); ++j) {
      BatchTask& task = *(batch->mutable_task(j));
      if (task.is_partial) {
        std::vector<Tensor>& tensor_vector = (*task.output)[task.split_index];
        tensor_vector[i] = std::move(split_tensor[j]);
      } else {
        task.context->set_output(i, split_tensor[j]);
      }
    }
    for (int j = 0; j < unbatched_tasks.size(); ++j) {
      // The unbatched tasks are not split, so no need to handle the partial
      // case separately.
      unbatched_tasks[j]->context->set_output(
          i, split_tensor[batch->num_tasks() + j]);
    }
  }

  return absl::OkStatus();
}

void BatchResourceBase::CleanUpFunctionHelper(
    BatchTask& task, const absl::Status& status) const {
  WithContext wc(task.propagated_context);
  if (!status.ok()) {
    if (!absl::StrContains(status.message(),
                           "Function was cancelled before it was started")) {
      task.status->Update(status);
    } else {
      // Do not propagate this error; Prefer a more helpful error message.
      LOG(ERROR) << "ERROR!!!! " << status.message();
    }
  }
  task.done_callback();
}

void BatchResourceBase::ProcessFuncBatch(
    std::unique_ptr<BatchT> batch,
    std::vector<std::unique_ptr<BatchTask>> unbatched_tasks) const {
  if (batch->empty()) {
    return;
  }

  // We use the 'propagated_context' from one of the threads which setup one
  // of the tasks. This will propagate any common context over all the threads
  // which are running this Session, of which this BatchOp is a part.
  WithContext wc(batch->task(batch->num_tasks() - 1).propagated_context);

  // TODO(b/185852990): Add a unit test to check the context is correctly set.
  // Creates the CostMeasurements within the same context that runs the Session.
  const CostMeasurement::Context batching_context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements =
      CreateCostMeasurements(batching_context);

  auto& last_task = batch->task(batch->num_tasks() - 1);
  OpKernelContext* last_task_context = last_task.context;
  const std::string& model_name = GetModelName(last_task_context);
  const std::string& op_name = last_task_context->op_kernel().name();

  // Regardless of the outcome, we need to propagate the status to the
  // individual tasks and signal that they are done. We use MakeCleanup() to
  // ensure that this happens no matter how we exit the method below.
  absl::Status status;
  bool cleanup_done = false;
  int64_t processed_size = batch->size();
  auto cleanup_fn = [&](const absl::Status& status) {
    if (cleanup_done) {
      return;
    }
    // TODO(b/316379576): Update this to take the unbatch task cost into
    // consideration when excluding the wasted cost and propagate cost to the
    // unbatched tasks.
    SplitBatchCostsAndRecordMetrics(
        /* model_name= */ model_name, /* op_name= */ op_name,
        batch_cost_measurements, processed_size, *batch);
    // Clear the measurements before unblocking the batch task, as measurements
    // are associated with the task's thread context.
    batch_cost_measurements.clear();
    for (int i = 0; i < batch->num_tasks(); ++i) {
      CleanUpFunctionHelper(*batch->mutable_task(i), status);
    }
    for (int i = 0; i < unbatched_tasks.size(); ++i) {
      CleanUpFunctionHelper(*unbatched_tasks[i], status);
    }
    cleanup_done = true;
  };

  auto finally =
      gtl::MakeCleanup([&cleanup_fn, &status] { cleanup_fn(status); });

  status = ValidateBatch(*batch);
  if (!status.ok()) {
    return;
  }

  std::vector<Tensor> concatenated_tensors;
  status = ConcatInputTensors(*batch, unbatched_tasks, last_task_context,
                              &concatenated_tensors);
  processed_size = RoundToLowestAllowedBatchSize(batch->size());
  if (!status.ok()) {
    return;
  }

  std::vector<Tensor> combined_outputs;
  std::vector<Tensor> args(concatenated_tensors.begin(),
                           concatenated_tensors.end());
  const auto& captured_inputs =
      batch->task(batch->num_tasks() - 1).captured_inputs;
  args.insert(args.end(), captured_inputs.begin(), captured_inputs.end());

  uint64 current_time = EnvTime::NowNanos();
  for (int i = 0; i < batch->num_tasks(); ++i) {
    RecordBatchDelayUs((current_time - batch->task(i).start_time) * 1e-3,
                       model_name, last_task_context->op_kernel().name(),
                       processed_size);
    RecordBatchDelayUsV2((current_time - batch->task(i).start_time) * 1e-3,
                         model_name, last_task_context->op_kernel().name(),
                         processed_size);
  }
  // Releases the cleanup method here, because the callback of the function
  // library runtime will handle it now.
  finally.release();
  ProcessFuncBatchImpl(
      last_task, args, &combined_outputs, [&](const absl::Status& run_status) {
        absl::Status final_status;
        auto run_finally = gtl::MakeCleanup([&]() {
          // We do the cleanup here as an optimization, so that
          // it runs in the underlying TF inter-op threadpool.
          // Running it in the threadpool, let's the ensuing
          // ops be scheduled faster, because the executor will
          // add them to the front of the threadpool's task
          // queue rather than the end.
          cleanup_fn(final_status);
        });
        final_status = run_status;
        if (!final_status.ok()) {
          return;
        }
        if (last_task.forced_warmup_batch_size == 0) {
          final_status = SplitOutputTensors(combined_outputs, batch.get(),
                                            unbatched_tasks);
        }
      });
}

// Processes a batch of one or more BatchTask entries.
void BatchResourceBase::ProcessBatch(std::unique_ptr<BatchT> batch) const {
  if (batch->empty()) {
    return;
  }

  // We use the 'propagated_context' from one of the threads which setup one
  // of the tasks. This will propagate any common context over all the threads
  // which are running this Session, of which this BatchOp is a part.
  WithContext wc(batch->task(batch->num_tasks() - 1).propagated_context);

  // TODO(b/185852990): Add a unit test to check the context is correctly set.
  // Creates the CostMeasurement within the same context that runs the Session.
  const CostMeasurement::Context batching_context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements =
      CreateCostMeasurements(batching_context);

  int64_t processed_size = batch->size();

  OpKernelContext* last_task_context =
      batch->task(batch->num_tasks() - 1).context;
  AsyncOpKernel::DoneCallback last_task_callback =
      batch->task(batch->num_tasks() - 1).done_callback;
  const std::string& model_name = GetModelName(last_task_context);
  const std::string& op_name = last_task_context->op_kernel().name();

  auto batch_cost_cleanup = gtl::MakeCleanup([&] {
    SplitBatchCostsAndRecordMetrics(
        /* model_name= */ model_name, /* op_name= */ op_name,
        batch_cost_measurements, processed_size, *batch);
  });

  OP_REQUIRES_OK_ASYNC(last_task_context, ValidateBatch(*batch),
                       last_task_callback);

  // All tasks should have the same number of input edges.
  const int num_input_edges = batch->task(0).inputs.size();
  std::vector<Tensor> concatenated_tensors;
  const absl::Status concat_status =
      ConcatInputTensors(*batch, {}, last_task_context, &concatenated_tensors);
  processed_size = RoundToLowestAllowedBatchSize(batch->size());
  OP_REQUIRES_OK_ASYNC(last_task_context, concat_status, last_task_callback);

  // Process each input edge one at a time (the typical case has just one).
  for (int i = 0; i < num_input_edges; ++i) {
    last_task_context->set_output(i, concatenated_tensors[i]);

    // Emit batch->num_tasks() - 1 empty output tensors.
    for (int task_idx = 0; task_idx < batch->num_tasks() - 1; ++task_idx) {
      const BatchTask& task = batch->task(task_idx);
      TensorShape output_shape(task.inputs[i].shape());
      output_shape.set_dim(0, 0);
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          task.context, task.context->allocate_output(i, output_shape, &output),
          task.done_callback);
    }
  }
  // Emit batch->num_tasks() - 1 empty index tensors.
  for (int task_idx = 0; task_idx < batch->num_tasks() - 1; ++task_idx) {
    const BatchTask& task = batch->task(task_idx);
    TensorShape index_shape({0, 3});
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        task.context,
        task.context->allocate_output(num_input_edges, index_shape, &output),
        task.done_callback);
  }
  // Emit all ID tensors.
  for (int task_idx = 0; task_idx < batch->num_tasks(); ++task_idx) {
    const BatchTask& task = batch->task(task_idx);
    Tensor* id;
    OP_REQUIRES_OK_ASYNC(task.context,
                         task.context->allocate_output(num_input_edges + 1,
                                                       TensorShape({}), &id),
                         task.done_callback);
    id->scalar<int64_t>()() = task.guid;
  }
  OP_REQUIRES_OK_ASYNC(
      last_task_context,
      EmitIndexTensor(last_task_context, *batch, num_input_edges),
      last_task_callback);

  // Signal done for each element of the batch. (At this point, the contexts
  // are no longer guaranteed to remain live.)
  for (int task_idx = 0; task_idx < batch->num_tasks(); ++task_idx) {
    batch->mutable_task(task_idx)->done_callback();
  }
}

/*static*/ absl::Status BatchResourceBase::EmitIndexTensor(
    OpKernelContext* context, const BatchT& batch, int output_index) {
  const TensorShape index_shape({batch.num_tasks(), 3});
  Tensor* index = nullptr;
  TF_RETURN_IF_ERROR(
      context->allocate_output(output_index, index_shape, &index));
  auto index_flat = index->shaped<int64_t, 2>({batch.num_tasks(), 3});
  size_t offset = 0;
  for (int task_idx = 0; task_idx < batch.num_tasks(); ++task_idx) {
    const BatchTask& task = batch.task(task_idx);
    index_flat(task_idx, 0) = task.guid;
    index_flat(task_idx, 1) = offset;
    index_flat(task_idx, 2) = offset + task.size();
    offset += task.size();
  }
  return absl::OkStatus();
}

void BatchResourceBase::ProcessBatchCallBack(
    std::unique_ptr<Batch<BatchTask>> batch,
    std::vector<std::unique_ptr<BatchTask>> unbatched_tasks) {
  if (!session_metadata().name().empty()) {
    absl::MutexLock lock(&outstanding_batch_mu_);
    num_outstanding_batched_items_ -= batch->size();
  }
  if (!has_process_batch_function_) {
    ProcessBatch(std::move(batch));
  } else {
    ProcessFuncBatch(std::move(batch), std::move(unbatched_tasks));
  }
}

absl::Status BatchResourceBase::LookupOrCreateBatcherQueue(
    const string& queue_name, const string& model_name, const string& op_name,
    BatcherQueueT** queue) {
  mutex_lock l(batcher_queues_mu_);

  auto it = batcher_queues_.find(queue_name);
  if (it != batcher_queues_.end()) {
    *queue = it->second.get();
    return absl::OkStatus();
  }

  std::unique_ptr<BatcherQueueT> new_queue;
  if (batcher_) {
    BatcherT::QueueOptions batcher_queue_options = batcher_queue_options_;
    batcher_queue_options.model_batch_stats = &GlobalBatchStatsRegistry().model(
        /* model_name= */ model_name, /* op_name= */ op_name);

    TF_RETURN_IF_ERROR(batcher_->AddQueue(
        batcher_queue_options,
        absl::bind_front(&BatchResourceBase::ProcessBatchCallBack, this),
        &new_queue));
  } else if (adaptive_batcher_) {
    std::function<void(std::unique_ptr<Batch<BatchTask>>)>
        reduced_process_batch_callback = [this](std::unique_ptr<BatchT> batch) {
          ProcessBatchCallBack(std::move(batch), {});
        };
    TF_RETURN_IF_ERROR(adaptive_batcher_->AddQueue(
        adaptive_batcher_queue_options_, reduced_process_batch_callback,
        &new_queue));
  } else {
    return errors::Internal("No batcher defined.");
  }
  *queue = new_queue.get();
  batcher_queues_[queue_name] = std::move(new_queue);
  return absl::OkStatus();
}

void BatchResourceBase::SplitBatchCostsAndRecordMetrics(
    const std::string& model_name, const std::string& op_name,
    const std::vector<std::unique_ptr<CostMeasurement>>&
        batch_cost_measurements,
    const int64_t processed_size, BatchT& batch) {
  absl::flat_hash_map<std::string, absl::Duration> batch_costs;
  // 1. Split the batch costs to each task.
  for (const auto& batch_cost_measurement : batch_cost_measurements) {
    if (batch_cost_measurement->GetTotalCost() <= absl::ZeroDuration()) {
      continue;
    }
    if (batch.size() == 0) {  // NOLINT: empty() checks the batch contains 0
                              // tasks. size() gets the sum of task sizes.
      LOG_EVERY_N_SEC(ERROR, 60)
          << "Non-zero cost collected but the batch size is 0.";
      return;
    }
    if (processed_size == 0) {
      LOG_EVERY_N_SEC(ERROR, 60)
          << "Non-zero cost collected but the processed size is 0.";
      return;
    }
    const absl::string_view cost_type = batch_cost_measurement->GetCostType();
    const absl::Duration total_cost = batch_cost_measurement->GetTotalCost();
    batch_costs[cost_type] = total_cost;

    // Smeared batch cost: cost for processing this batch.
    RecordBatchCosts(model_name, processed_size,
                     absl::StrCat(cost_type, kWithSmearSuffix), total_cost);
    // Non-smeared batch cost: cost for processing inputs in this batch, i.e.
    // cost for processing paddings is excluded.
    RecordBatchCosts(model_name, processed_size,
                     absl::StrCat(cost_type, kNoSmearSuffix),
                     total_cost / processed_size * batch.size());

    // Register batch stats for in-process use.
    if (cost_type == kTpuCostName) {
      ModelBatchStats& model_stats = GlobalBatchStatsRegistry().model(
          /* model_name= */ model_name, /* op_name= */ op_name);
      model_stats.batch_size(processed_size).tpu_cost().Register(total_cost);
      // batch.size() is the size of the original batch before padding.
      model_stats.RegisterProcessedSize(batch.size());
    }

    for (int i = 0; i < batch.num_tasks(); i++) {
      RequestCost* request_cost = batch.task(i).request_cost;
      // Skip recording the cost if the request_cost is null.
      if (!request_cost) continue;

      // Smeared cost: cost of paddings are assigned to each task.
      const auto cost_with_smear =
          total_cost / batch.size() * batch.task(i).size();

      // Non-smeared cost: cost of paddings are not assigned to any tasks.
      const auto cost_no_smear =
          total_cost / processed_size * batch.task(i).size();

      request_cost->RecordCost(
          {{absl::StrCat(cost_type, kWithSmearSuffix), cost_with_smear},
           {absl::StrCat(cost_type, kNoSmearSuffix), cost_no_smear}});
    }
  }

  // 2. Records the batch metrics in each task.
  const int64_t padding_size = processed_size - batch.size();
  for (int i = 0; i < batch.num_tasks(); i++) {
    RequestCost* request_cost = batch.task(i).request_cost;
    // Skip recording the metrics if the request_cost is null.
    if (!request_cost) continue;

    request_cost->RecordBatchMetrics(RequestCost::BatchMetrics{
        processed_size, static_cast<int64_t>(batch.task(i).size()),
        padding_size, batch_costs});
  }
}

}  // namespace serving
}  // namespace tensorflow
