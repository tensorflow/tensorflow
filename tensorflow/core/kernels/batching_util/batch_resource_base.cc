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

#include <sstream>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/cost_constants.h"
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/cost_util.h"
#include "tensorflow/core/common_runtime/request_cost_accessor.h"
#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/batching_util/concat_split_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/percentile_sampler.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/util/incremental_barrier.h"

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
  static auto* cell = tensorflow::monitoring::Sampler<3>::New(
      {"/tensorflow/serving/batching/padding_size_v2",
       "Tracks the padding size distribution on batches by model_name (if "
       "available).",
       "model_name", "execution_batch_size", "op_name"},
      // It's 14 buckets with the last bucket being 2^13 to DBL_MAX;
      // so the limits are [1, 2, 4, 8, ..., 8 * 1024, DBL_MAX].
      monitoring::Buckets::Exponential(1, 2, 14));
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
      // It's 14 buckets with the last bucket being 2^13 to DBL_MAX;
      // so the limits are [1, 2, 4, 8, ..., 8 * 1024, DBL_MAX].
      monitoring::Buckets::Exponential(1, 2, 14));
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

const string& GetModelName(OpKernelContext* ctx) {
  static string* kModelNameUnset = new string("model_name_unset");
  if (!ctx->session_metadata()) return *kModelNameUnset;
  if (ctx->session_metadata()->name().empty()) return *kModelNameUnset;
  return ctx->session_metadata()->name();
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

Status BatchResourceBase::RegisterInput(
    int64_t guid, OpKernelContext* context, const string& batcher_queue_name,
    AsyncOpKernel::DoneCallback done_callback) {
  std::unique_ptr<BatchTask> batch_components;
  TF_RETURN_IF_ERROR(CreateBatchTask(context, &batch_components));
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
    return OkStatus();
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
  batch_components->done_callback = std::move(done_callback);
  batch_components->split_index = 0;
  batch_components->output = std::make_shared<TensorMatrix>();
  batch_components->status = std::make_shared<ThreadSafeStatus>();

  std::unique_ptr<RequestCostAccessor> request_cost_accessor =
      CreateRequestCostAccessor();
  if (request_cost_accessor) {
    batch_components->request_cost = request_cost_accessor->GetRequestCost();
  }

  BatcherQueueT* batcher_queue;
  TF_RETURN_IF_ERROR(
      LookupOrCreateBatcherQueue(batcher_queue_name, &batcher_queue));
  return batcher_queue->Schedule(&batch_components);
}

/*static*/ BatchResourceBase::BatcherT::QueueOptions
BatchResourceBase::GetBatcherQueueOptions(
    int32_t num_batch_threads, int32_t max_batch_size,
    int32_t batch_timeout_micros, int32_t max_enqueued_batches,
    const std::vector<int32>& allowed_batch_sizes,
    bool enable_large_batch_splitting) {
  BatcherT::QueueOptions batcher_queue_options;
  batcher_queue_options.input_batch_size_limit = max_batch_size;
  batcher_queue_options.max_enqueued_batches = max_enqueued_batches;
  batcher_queue_options.batch_timeout_micros = batch_timeout_micros;
  batcher_queue_options.enable_large_batch_splitting =
      enable_large_batch_splitting;
  if (enable_large_batch_splitting) {
    batcher_queue_options.split_input_task_func =
        [](std::unique_ptr<BatchTask>* input_task,
           int open_batch_remaining_slot, int max_batch_size,
           std::vector<std::unique_ptr<BatchTask>>* output_tasks) -> Status {
      return SplitInputTask(input_task, open_batch_remaining_slot,
                            max_batch_size, output_tasks);
    };

    if (allowed_batch_sizes.empty()) {
      batcher_queue_options.max_execution_batch_size = max_batch_size;
    } else {
      batcher_queue_options.max_execution_batch_size =
          *allowed_batch_sizes.rbegin();
    }
  }

  return batcher_queue_options;
}

/*static*/ BatchResourceBase::AdaptiveBatcherT::QueueOptions
BatchResourceBase::GetAdaptiveBatcherQueueOptions(
    int32_t max_batch_size, int32_t batch_timeout_micros,
    int32_t max_enqueued_batches, bool enable_large_batch_splitting,
    const std::vector<int32>& allowed_batch_sizes) {
  AdaptiveBatcherT::QueueOptions batcher_queue_options;
  batcher_queue_options.max_input_task_size =
      absl::make_optional(max_batch_size);
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
           std::vector<std::unique_ptr<BatchTask>>* output_tasks) -> Status {
      return SplitInputTask(input_task, open_batch_remaining_slot,
                            max_batch_size, output_tasks);
    };
  }

  return batcher_queue_options;
}

/*static*/ Status BatchResourceBase::ValidateBatch(const BatchT& batch) {
  for (int task_idx = 0; task_idx < batch.num_tasks(); ++task_idx) {
    const BatchResourceBase::BatchTask& task = batch.task(task_idx);

    if (task.inputs.size() != batch.task(0).inputs.size()) {
      return errors::InvalidArgument(
          "Batching inputs must have equal number of edges");
    }
  }

  return OkStatus();
}

// Returns the smallest entry in 'allowed_batch_sizes_' that is greater than
// or equal to 'batch_size'. If 'allowed_batch_sizes_' is empty, simply
// returns 'batch_size'.
int BatchResourceBase::RoundToLowestAllowedBatchSize(int batch_size) const {
  if (allowed_batch_sizes_.empty()) {
    return batch_size;
  }
  for (int allowed_size : allowed_batch_sizes_) {
    if (allowed_size >= batch_size) {
      return allowed_size;
    }
  }
  LOG(ERROR) << "Batch size " << batch_size
             << " is greater than largest allowed size; "
                "ignoring allowed sizes constraint.";
  return batch_size;
}

Status BatchResourceBase::ConcatInputTensors(
    const BatchT& batch, OpKernelContext* context,
    std::vector<Tensor>* concatenated_tensors) const {
  if (batch.num_tasks() == 0) {
    return errors::InvalidArgument("Empty batch.");
  }

  const int padded_batch_size = RoundToLowestAllowedBatchSize(batch.size());
  const int padding_amount = padded_batch_size - batch.size();
  profiler::TraceMe trace_me([padded_batch_size, padding_amount]() {
    return profiler::TraceMeEncode(
        "ConcatInputTensors", {{"batch_size_after_padding", padded_batch_size},
                               {"padding_amount", padding_amount}});
  });
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

  // Process each input one at a time (the typical case has just one).
  for (int i = 0; i < num_inputs; ++i) {
    // Concatenate the tasks ith input tensors into a big output tensor.
    std::vector<Tensor> to_concatenate;
    to_concatenate.reserve(batch.num_tasks());
    for (int task_idx = 0; task_idx < batch.num_tasks(); ++task_idx) {
      to_concatenate.push_back(batch.task(task_idx).inputs.at(i));
    }

    // Add padding as needed. Use the first row of the first task's tensor as
    // the data for padding.
    if (padding_amount > 0) {
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
    Status concat_status =
        Concat(context, to_concatenate, &concatenated_tensor);
    TF_RETURN_IF_ERROR(concat_status);
    concatenated_tensors->push_back(concatenated_tensor);
  }
  return OkStatus();
}

/*static*/ Status BatchResourceBase::SplitInputTask(
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
       op_kernel_context = input_task.context, status = shared_status]() {
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

          op_kernel_context->set_output(i, std::move(output_tensor));
        }
        op_kernel_context->SetStatus(status->status());
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
    const Status split_status = Split(input_task.context, input_tensor,
                                      output_task_sizes, &split_tensors);
    if (!split_status.ok()) {
      return errors::Internal(
          "When splitting input, Tensor split operation failed: ",
          split_status.error_message());
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
  return OkStatus();
}

Status BatchResourceBase::SplitOutputTensors(
    const std::vector<Tensor>& combined_outputs, BatchT* batch) const {
  DCHECK_GE(batch->num_tasks(), 1);
  if (batch->num_tasks() < 1) {
    return errors::Internal("Batch size expected to be positive; was ",
                            batch->num_tasks());
  }

  std::vector<int64_t> task_sizes_plus_optional_padding;
  task_sizes_plus_optional_padding.reserve(batch->num_tasks());
  for (int i = 0; i < batch->num_tasks(); ++i) {
    task_sizes_plus_optional_padding.push_back(batch->task(i).size());
  }
  const int padding_size =
      RoundToLowestAllowedBatchSize(batch->size()) - batch->size();
  if (padding_size > 0) {
    task_sizes_plus_optional_padding.push_back(padding_size);
  }

  // For each output tensor name, a divided-up tensor with one entry per task.
  std::map<string, std::vector<Tensor>> split_tensors;

  DCHECK_EQ(batch->task(0).context->num_outputs(), combined_outputs.size());
  int combined_outputs_size = combined_outputs.size();
  if (combined_outputs_size != batch->task(0).context->num_outputs()) {
    return errors::Internal("Wrong number of batched output tensors");
  }

  // Generate 'split_tensors' and populate the context outputs.
  for (int i = 0, iter_limit = combined_outputs.size(); i < iter_limit; ++i) {
    const Tensor& output_tensor = combined_outputs[i];
    if (output_tensor.shape().dims() == 0) {
      return errors::FailedPrecondition(
          "Batched output tensor has 0 dimensions");
    }
    if (output_tensor.shape().dim_size(0) !=
        static_cast<int64_t>(batch->size() + padding_size)) {
      return errors::FailedPrecondition(
          "Batched output tensor's 0th dimension does not equal the sum of "
          "the 0th dimension sizes of the input tensors");
    }

    std::vector<Tensor> split_tensor;
    const Status split_status = tensor::Split(
        output_tensor, task_sizes_plus_optional_padding, &split_tensor);
    DCHECK(split_status.ok()) << split_status.ToString();
    if (!split_status.ok()) {
      return errors::Internal("Tensor split operation failed: ",
                              split_status.error_message());
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
  }

  return OkStatus();
}

void BatchResourceBase::ProcessFuncBatch(std::unique_ptr<BatchT> batch) const {
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

  // Regardless of the outcome, we need to propagate the status to the
  // individual tasks and signal that they are done. We use MakeCleanup() to
  // ensure that this happens no matter how we exit the method below.
  Status status;
  bool cleanup_done = false;
  int64_t processed_size = batch->size();
  auto cleanup_fn = [&cleanup_done, &batch, &processed_size,
                     &batch_cost_measurements](const Status& status) {
    if (cleanup_done) {
      return;
    }
    SplitBatchCosts(batch_cost_measurements, processed_size, *batch);
    // Clear the measurements before unblocking the batch task, as measurements
    // are associated with the task's thread context.
    batch_cost_measurements.clear();
    for (int i = 0; i < batch->num_tasks(); ++i) {
      WithContext wc(batch->task(i).propagated_context);
      if (batch->task(i).is_partial) {
        batch->mutable_task(i)->status->Update(status);
      } else {
        batch->mutable_task(i)->context->SetStatus(status);
      }
      batch->mutable_task(i)->done_callback();
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
  status = ConcatInputTensors(*batch, last_task_context, &concatenated_tensors);
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
  const string& model_name = GetModelName(last_task_context);
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
      last_task, args, &combined_outputs, [&](const Status& run_status) {
        Status final_status;
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
        final_status = SplitOutputTensors(combined_outputs, batch.get());
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
  auto batch_cost_split_cleanup = gtl::MakeCleanup([&] {
    SplitBatchCosts(batch_cost_measurements, processed_size, *batch);
  });

  OpKernelContext* last_task_context =
      batch->task(batch->num_tasks() - 1).context;
  AsyncOpKernel::DoneCallback last_task_callback =
      batch->task(batch->num_tasks() - 1).done_callback;

  OP_REQUIRES_OK_ASYNC(last_task_context, ValidateBatch(*batch),
                       last_task_callback);

  // All tasks should have the same number of input edges.
  const int num_input_edges = batch->task(0).inputs.size();
  std::vector<Tensor> concatenated_tensors;
  const Status concat_status =
      ConcatInputTensors(*batch, last_task_context, &concatenated_tensors);
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

/*static*/ Status BatchResourceBase::EmitIndexTensor(OpKernelContext* context,
                                                     const BatchT& batch,
                                                     int output_index) {
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
  return OkStatus();
}

// Looks up the batcher queue for 'queue_name'. If it did't previously exist,
// creates it.
Status BatchResourceBase::LookupOrCreateBatcherQueue(const string& queue_name,
                                                     BatcherQueueT** queue) {
  mutex_lock l(batcher_queues_mu_);

  auto it = batcher_queues_.find(queue_name);
  if (it != batcher_queues_.end()) {
    *queue = it->second.get();
    return OkStatus();
  }

  std::unique_ptr<BatcherQueueT> new_queue;
  auto process_batch_callback = [this](std::unique_ptr<BatchT> batch) {
    if (!has_process_batch_function_) {
      ProcessBatch(std::move(batch));
    } else {
      ProcessFuncBatch(std::move(batch));
    }
  };
  if (batcher_) {
    TF_RETURN_IF_ERROR(batcher_->AddQueue(batcher_queue_options_,
                                          process_batch_callback, &new_queue));
  } else if (adaptive_batcher_) {
    TF_RETURN_IF_ERROR(adaptive_batcher_->AddQueue(
        adaptive_batcher_queue_options_, process_batch_callback, &new_queue));
  } else {
    return errors::Internal("No batcher defined.");
  }
  *queue = new_queue.get();
  batcher_queues_[queue_name] = std::move(new_queue);
  return OkStatus();
}

Status BatchResourceBase::CreateBatchTask(
    OpKernelContext* context,
    std::unique_ptr<BatchResourceBase::BatchTask>* output) const {
  *output = absl::make_unique<BatchResourceBase::BatchTask>();
  return OkStatus();
}

void BatchResourceBase::SplitBatchCosts(
    std::vector<std::unique_ptr<CostMeasurement>>& batch_cost_measurements,
    const int64_t processed_size, BatchT& batch) {
  for (auto& batch_cost_measurement : batch_cost_measurements) {
    if (batch_cost_measurement->GetTotalCost() <= absl::ZeroDuration()) {
      return;
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
}

}  // namespace serving
}  // namespace tensorflow
