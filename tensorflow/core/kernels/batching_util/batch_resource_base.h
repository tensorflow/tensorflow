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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_RESOURCE_BASE_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_RESOURCE_BASE_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/synchronization/blocking_counter.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/request_cost.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/threadsafe_status.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tsl/platform/criticality.h"

namespace tensorflow {
namespace serving {

// Base class for resource that encapsulating the state and logic for batching
// tensors.
class BatchResourceBase : public ResourceBase {
 public:
  // Given a BatchTask (from one op invocation) with 'num_outputs'== M and
  // splitted into N sub tasks, TensorMatrix is a N X M matrix.
  // Namely, TensorMatrix[i][j] indicates the i-th split tensor of j-th output;
  // concatenating tensors along the 2nd dimension gives a output tensor.
  typedef std::vector<std::vector<Tensor>> TensorMatrix;

  // One task to be batched, corresponds to a `slice` of input from one batch-op
  // invocation.
  //
  // Given input from one batch-op invocation, a `slice` of this input is:
  // 1) Split each Tensor in `BatchTask::inputs` along the 0th dimension.
  // 2) 'split_index' is calculated along the 0-th dimension.
  //
  // Note input from one batch-op invocation is valid and considered a
  // specialized `slice`.
  struct BatchTask : public tensorflow::serving::BatchTask {
    // A unique ID to identify this invocation of Batch.
    int64_t guid;

    Context propagated_context;

    std::vector<Tensor> inputs;
    std::vector<Tensor> captured_inputs;
    OpKernelContext* context;
    AsyncOpKernel::DoneCallback done_callback;

    // The index of this split, along the 0-th dimension of input from op
    // invocation.
    int split_index = 0;

    // Two-dimensional tensor matrix, ownership shared by:
    // 1) each split of task (to fill one row in this matrix)
    // and
    // 2) callback that runs to merge output of individual splits for an op
    // invocation, after all splits complete.
    std::shared_ptr<TensorMatrix> output;

    // 'status' records error (could be from any split) if at least one split
    // returns error, OK otherwise.
    // Ownership is shared by individual splits and callback.
    std::shared_ptr<ThreadSafeStatus> status;

    bool is_partial = false;

    uint64 start_time;

    size_t size() const override { return inputs[0].shape().dim_size(0); }

    // Create a split task from this one. The caller needs to setup the inputs
    // of the new task
    std::unique_ptr<BatchTask> CreateSplitTask(
        int split_index, AsyncOpKernel::DoneCallback done_callback);

    // RequestCost is for collecting the cost and must outlive the batching
    // processing.
    //
    // For example, to collect cost in rpc processing, `request_cost` is owned
    // by rpc handler and points to the RequestCost of an rpc which provides
    // the inputs to this BatchTask.
    //
    // After the batch processing, the request cost will be incremented with
    // this task's processing costs.
    RequestCost* request_cost = nullptr;

    tsl::criticality::Criticality criticality;

    // If nonzero, make a batch of this size entirely out of padding. This
    // batch is processed, but is not propagated to the kernel outputs.
    int forced_warmup_batch_size = 0;

   protected:
    virtual std::unique_ptr<BatchTask> CreateDerivedTask() {
      return std::make_unique<BatchTask>();
    }
  };

  // Appending a T suffix to make the type alias different to those in
  // tensorflow::serving namespace, because some versions of compiler complain
  // about changing meaning of the symbols.
  using BatcherT = SharedBatchScheduler<BatchResourceBase::BatchTask>;
  using AdaptiveBatcherT =
      AdaptiveSharedBatchScheduler<BatchResourceBase::BatchTask>;
  using BatcherQueueT = BatchScheduler<BatchResourceBase::BatchTask>;
  using BatchT = Batch<BatchResourceBase::BatchTask>;

  BatchResourceBase(bool has_process_batch_function,
                    std::shared_ptr<BatcherT> batcher,
                    const BatcherT::QueueOptions& batcher_queue_options,
                    std::vector<int32> allowed_batch_sizes)
      : has_process_batch_function_(has_process_batch_function),
        batcher_(std::move(batcher)),
        batcher_queue_options_(batcher_queue_options),
        allowed_batch_sizes_(std::move(allowed_batch_sizes)),
        allowed_batch_sizes_str_(absl::StrJoin(allowed_batch_sizes_, ",")) {}

  BatchResourceBase(bool has_process_batch_function,
                    std::shared_ptr<AdaptiveBatcherT> batcher,
                    const AdaptiveBatcherT::QueueOptions& batcher_queue_options,
                    std::vector<int32> allowed_batch_sizes)
      : has_process_batch_function_(has_process_batch_function),
        adaptive_batcher_(std::move(batcher)),
        adaptive_batcher_queue_options_(batcher_queue_options),
        allowed_batch_sizes_(std::move(allowed_batch_sizes)),
        allowed_batch_sizes_str_(absl::StrJoin(allowed_batch_sizes_, ",")) {}

  void set_session_metadata(tensorflow::SessionMetadata session_metadata) {
    session_metadata_ = std::move(session_metadata);
  }

  const SessionMetadata& session_metadata() const { return session_metadata_; }

  using CreateBatchTaskFn =
      std::function<StatusOr<std::unique_ptr<BatchTask>>()>;

  // Like `RegisterInput`, but extra "dummy" batches are processed for each
  // batch size. Only the real request's outputs are propagated to the caller.
  Status RegisterWarmupInputs(int64_t guid, OpKernelContext* context,
                              const string& batcher_queue_name,
                              const CreateBatchTaskFn& create_batch_task_fn,
                              AsyncOpKernel::DoneCallback done);
  // Ingests data from one invocation of the batch op. The data is enqueued to
  // be combined with others into a batch, asynchronously.
  Status RegisterInput(int64_t guid, OpKernelContext* context,
                       const string& batcher_queue_name,
                       const CreateBatchTaskFn& create_batch_task_fn,
                       AsyncOpKernel::DoneCallback done_callback,
                       int forced_warmup_batch_size = 0);

  static BatcherT::QueueOptions GetBatcherQueueOptions(
      int32_t num_batch_threads, int32_t max_batch_size,
      int32_t batch_timeout_micros, int32_t max_enqueued_batches,
      const std::vector<int32>& allowed_batch_sizes,
      bool enable_large_batch_splitting, bool disable_padding);

  static BatcherT::QueueOptions GetBatcherQueueOptions(
      int32_t num_batch_threads, int32_t max_batch_size,
      int32_t batch_timeout_micros, int32_t max_enqueued_batches,
      const std::vector<int32>& allowed_batch_sizes,
      bool enable_large_batch_splitting, bool disable_padding,
      int32_t low_priority_max_batch_size,
      int32_t low_priority_batch_timeout_micros,
      int32_t low_priority_max_enqueued_batches,
      const std::vector<int32>& low_priority_allowed_batch_sizes);

  static AdaptiveBatcherT::QueueOptions GetAdaptiveBatcherQueueOptions(
      int32_t max_batch_size, int32_t batch_timeout_micros,
      int32_t max_enqueued_batches, bool enable_large_batch_splitting,
      const std::vector<int32>& allowed_batch_sizes, bool disable_padding);

  // Split 'input' of 'input_task_ptr' along 0th dimension, into a list of
  // 'output_tasks'.
  // Task sizes are determined by
  // 1) open_batch_remaining_slot
  // 2) max_batch_size
  // 3) size-of-input-task
  // in a way that
  // 1) Task sizes add up to `size-of-input-task`.
  // 2) Task sizes from left to right are like
  //    [open_batch_remaining_slot, max_batch_size, max_batch_size, ...,
  //    `size-of-input-task` - `sum-of-previous-elements`].
  //
  // REQUIRES:
  // Caller should make sure size-of-input-task is greater than
  // open_batch_remaining_slot.
  static Status SplitInputTask(
      std::unique_ptr<BatchTask>* input_task_ptr, int open_batch_remaining_slot,
      int max_batch_size,
      std::vector<std::unique_ptr<BatchTask>>* output_tasks);

  // Splits the batch costs to each task.
  //
  // Inputs:
  // 1) batch_cost_measurements, which provides the total cost of each type;
  // 2) processed_size, it's the batch size plus the padding amount;
  // 3) batch, provides the batch size and input sizes.
  //
  // Outputs:
  // The request_cost in each batch task will be updated.
  // - This function will use two approaches to split the batch cost (if it's
  //   non-zero), thus two costs will be output.
  //   1) smeared cost: batch cost is split proportionally to each task's size,
  //      and paddings do not share any cost;
  //   2) non-smeared cost: batch cost is split proportionally to each task or
  //      padding's size. Here padding's cost is not assigned to any tasks.
  // - This function will also record the metrics of this batch in each task,
  //   including:
  //   1) the batch size;
  //   2) the input size from this task;
  //   3) the padding amount.
  static void SplitBatchCostsAndRecordMetrics(
      std::vector<std::unique_ptr<CostMeasurement>>& batch_cost_measurements,
      int64_t processed_size, BatchT& batch);

 private:
  // Implementation of calling the process batch function.
  virtual void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& last_task,
      absl::Span<const Tensor> inputs, std::vector<Tensor>* combined_outputs,
      std::function<void(const Status&)> done) const = 0;

  // Validates that it's legal to combine the tasks in 'batch' into a batch.
  // Assumes the batch is non-empty.
  static Status ValidateBatch(const BatchT& batch);

  // Returns the smallest entry in 'allowed_batch_sizes_' that is greater than
  // or equal to 'batch_size'. If 'allowed_batch_sizes_' is empty, simply
  // returns 'batch_size'.
  int RoundToLowestAllowedBatchSize(int batch_size) const;

  Status ConcatInputTensors(const BatchT& batch, OpKernelContext* context,
                            std::vector<Tensor>* concatenated_tensors) const;

  Status SplitOutputTensors(const std::vector<Tensor>& combined_outputs,
                            BatchT* batch) const;

  void ProcessFuncBatch(std::unique_ptr<BatchT> batch) const;

  // Processes a batch of one or more BatchTask entries.
  void ProcessBatch(std::unique_ptr<BatchT> batch) const;

  // Emits an index tensor, which the Unbatch op will use to un-concatenate
  // the tensor and attribute the pieces to the right batch keys. The index
  // tensor contains, for each input: [batch_key, start_offset, end_offset]
  // where start_offset and end_offset represent the range of entries in the
  // concatenated tensors that belong to that input.
  //
  // Emits the result to the output at 'output_index' using 'context'.
  static Status EmitIndexTensor(OpKernelContext* context, const BatchT& batch,
                                int output_index);

  // Looks up the batcher queue for 'queue_name'. If it did't previously exist,
  // creates it.
  Status LookupOrCreateBatcherQueue(const string& queue_name,
                                    BatcherQueueT** queue);

  SessionMetadata session_metadata_;

  absl::Mutex outstanding_batch_mu_;
  int num_outstanding_batched_items_ TF_GUARDED_BY(outstanding_batch_mu_) = 0;

  // True if user specified a batch processing function for this resource.
  const bool has_process_batch_function_;
  // A batch scheduler, and options for creating queues.
  std::shared_ptr<BatcherT> batcher_;
  BatcherT::QueueOptions batcher_queue_options_;

  // A batch scheduler, and options for creating queues.
  std::shared_ptr<AdaptiveBatcherT> adaptive_batcher_;
  AdaptiveBatcherT::QueueOptions adaptive_batcher_queue_options_;

  // A collection of batcher queues, keyed on queue name.
  // TODO(olston): Garbage-collect unused queues (perhaps simply remove empty
  // ones (with a time delay?); it's okay if they get recreated later).
  mutable mutex batcher_queues_mu_;
  std::map<string, std::unique_ptr<BatcherQueueT>> batcher_queues_
      TF_GUARDED_BY(batcher_queues_mu_);

  std::vector<int32> allowed_batch_sizes_;
  // A concatenated string of <allowed_batch_sizes_>, separated by ",". This is
  // used to record batching parameter.
  string allowed_batch_sizes_str_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_RESOURCE_BASE_H_
