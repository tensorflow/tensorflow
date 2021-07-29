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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_INPUT_TASK_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_INPUT_TASK_H_

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/concat_split_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/incremental_barrier.h"

namespace tensorflow {
namespace serving {

template <typename TaskType>
class BatchInputTask;

// A RAII-style object that holds a ref-counted batch-input-task, and
// represents a slice of batch-input-task.

// To be handed out to callers of `BatchInputTask::GetNextTaskHandle` quickly
// (i.e. not necessarily waiting for input split)
//
// GetSplitTask evaluates to the slice of task.
template <typename TaskType>
class BatchInputTaskHandle {
 public:
  BatchInputTaskHandle(
      std::shared_ptr<BatchInputTask<TaskType>> batch_input_task, int split_id);

  // Should be called once. Returns nullptr on subsequent calls.
  std::unique_ptr<TaskType> GetSplitTask();

  int split_id() const { return split_id_; }

 private:
  std::shared_ptr<BatchInputTask<TaskType>> batch_input_task_;

  // The handle evaluates to the N-th slice of original task, and
  // N is `split_id_`.
  const int split_id_;

  std::atomic<bool> once_{false};
};

// BatchInputTask encapsulates a input (`input_task`) to be batched and the
// information to get task splits after it's enqueued, so as to support lazy
// split of a task.
//
// Input split could reduce excessive padding for efficiency; lazy split
// moves task-split out of the critical path of enqueue and dequeue and reduces
// contention.
//
// BatchInputTask is thread safe.
//
// Usage:
//
// ... a deque with frequent enqueue and dequeue operations ...
// std::deque<shared_ptr<BatchInputTask<BatchTask>>> deque_ TF_GUARDED_BY(mu_);
//
// ... input_task provided by batch scheduling  ...
// std::unique_ptr<TaskType> input_task;
//
// ... Enqueue path ...
//
// {
//   mutex_lock l(mu_);
//   ...... construct `batch_input_task` quickly without split ......
//   std::shared_ptr<BatchInputTask<BatchTask>> batch_input_task;
//   deque_.push_back(batch_input_task);
// }
//
// ... Dequeue path ...
// vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>> handles;
// {
//    mutex_lock l(mu_);
//    auto handle = deque_.front()->GetNextTaskHandle();
//    handles.push_back(std::move(handle));
//    ... call `GetNextTaskHandle` until we accumuate enough to form a batch ...
// }
// ...... `mu_` is released ......
// Caller calls `BatchInputTaskHandle::GetSplitTask` to lazily evaluate each
// task in the batch.
template <typename TaskType>
class BatchInputTask
    : public std::enable_shared_from_this<BatchInputTask<TaskType>> {
 public:
  using BatchSplitFunc = std::function<Status(
      std::unique_ptr<TaskType>* input_task, int first_output_task_size,
      int input_batch_size_limit,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)>;

  // TODO(b/194294263):
  // Add a SplitMetadataFunc in constructor, so users of this class specify
  // both how to split, and how to compute split metadata in a consistent way.
  BatchInputTask(std::unique_ptr<TaskType> input_task,
                 int open_batch_remaining_slot, int batch_size_limit,
                 BatchSplitFunc split_func);

  // A stateful method to hand out the next task to be processed.
  // Returns nullptr if all batches are given out.
  std::unique_ptr<BatchInputTaskHandle<TaskType>> GetNextTaskHandle();

  // Following method exposes split metadata of this task.
  // Metadata are used to determine batch construction so needed before split
  // happens.
  //
  // Task size of `input_task`
  size_t size() const;

  // The number of batches the input spans.
  int num_batches() const;

  // The number of new batches this input adds.
  int num_new_batches() const;

  // The task size of the last batch.
  int tail_batch_task_size() const;

 private:
  friend class BatchInputTaskHandle<TaskType>;

  std::unique_ptr<TaskType> GetSplitTask(int split_id);

  Status SplitBatches(std::vector<std::unique_ptr<TaskType>>* output_tasks);

  std::unique_ptr<TaskType> input_task_;

  const int input_task_size_ = 0;
  const int open_batch_remaining_slot_;

  const int batch_size_limit_;

  const BatchSplitFunc split_func_;

  // The number of batches that this input appends to.
  // Should be either zero or one.
  const int num_batches_reused_ = 0;

  // The number of batches this input spans over.
  int num_batches_ = 0;

  // The task size of the last batch.
  int tail_batch_task_size_;

  mutable absl::once_flag once_;

  std::vector<std::unique_ptr<TaskType>> task_splits_;
  Status split_status_;

  mutable mutex mu_;
  int next_task_id_ TF_GUARDED_BY(mu_) = 0;
};

//
// Implementation details. API readers may skip.
//

template <typename TaskType>
BatchInputTaskHandle<TaskType>::BatchInputTaskHandle(
    std::shared_ptr<BatchInputTask<TaskType>> batch_input_task, int split_id)
    : batch_input_task_(batch_input_task), split_id_(split_id) {}

template <typename TaskType>
std::unique_ptr<TaskType> BatchInputTaskHandle<TaskType>::GetSplitTask() {
  if (once_.load(std::memory_order_acquire)) {
    return nullptr;
  }
  once_.store(true, std::memory_order_release);
  return batch_input_task_->GetSplitTask(split_id_);
}

template <typename TaskType>
BatchInputTask<TaskType>::BatchInputTask(
    std::unique_ptr<TaskType> input_task, int open_batch_remaining_slot,
    int batch_size_limit,
    std::function<Status(std::unique_ptr<TaskType>* input_task,
                         int first_output_task_size, int input_batch_size_limit,
                         std::vector<std::unique_ptr<TaskType>>* output_tasks)>
        split_func)
    : input_task_(std::move(input_task)),
      input_task_size_(input_task_->size()),
      open_batch_remaining_slot_(open_batch_remaining_slot),
      batch_size_limit_(batch_size_limit),
      split_func_(split_func),
      num_batches_reused_((open_batch_remaining_slot_ > 0) ? 1 : 0) {
  // The total task size starting from current open batch, after this task is
  // enqueued.
  const int task_size_from_open_batch =
      (open_batch_remaining_slot_ > 0)
          ? (input_task_size_ + batch_size_limit_ - open_batch_remaining_slot_)
          : input_task_size_;

  num_batches_ =
      (task_size_from_open_batch + batch_size_limit_ - 1) / batch_size_limit_;

  tail_batch_task_size_ = task_size_from_open_batch % batch_size_limit_;
  if (tail_batch_task_size_ == 0) {
    tail_batch_task_size_ = batch_size_limit_;
  }
}

template <typename TaskType>
size_t BatchInputTask<TaskType>::size() const {
  return input_task_size_;
}

template <typename TaskType>
int BatchInputTask<TaskType>::num_batches() const {
  return num_batches_;
}

template <typename TaskType>
int BatchInputTask<TaskType>::tail_batch_task_size() const {
  return tail_batch_task_size_;
}

template <typename TaskType>
int BatchInputTask<TaskType>::num_new_batches() const {
  return num_batches_ - num_batches_reused_;
}

template <typename TaskType>
std::unique_ptr<BatchInputTaskHandle<TaskType>>
BatchInputTask<TaskType>::GetNextTaskHandle() {
  mutex_lock l(mu_);
  if (next_task_id_ < num_batches_) {
    auto handle = std::make_unique<BatchInputTaskHandle<TaskType>>(
        this->shared_from_this(), next_task_id_);
    next_task_id_++;
    return handle;
  }
  return nullptr;
}

template <typename TaskType>
std::unique_ptr<TaskType> BatchInputTask<TaskType>::GetSplitTask(int split_id) {
  absl::call_once(once_,
                  [this]() { split_status_ = SplitBatches(&task_splits_); });
  if (!split_status_.ok()) {
    return nullptr;
  }
  if (split_id >= 0 && split_id < task_splits_.size()) {
    return std::move(task_splits_[split_id]);
  }
  return nullptr;
}

template <typename TaskType>
Status BatchInputTask<TaskType>::SplitBatches(
    std::vector<std::unique_ptr<TaskType>>* output_tasks) {
  return split_func_(&input_task_, open_batch_remaining_slot_,
                     batch_size_limit_, output_tasks);
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_INPUT_TASK_H_
