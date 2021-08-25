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
#include "absl/container/fixed_array.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/concat_split_util.h"
#include "tensorflow/core/kernels/batching_util/input_split_metadata.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/incremental_barrier.h"

namespace tensorflow {
namespace serving {

namespace internal {
template <typename TaskType>
class BatchInputTaskHandleTestAccess;

template <typename TaskType>
class BatchInputTaskTestAccess;

template <typename TaskType>
class BatchInputTask;

// A RAII-style object that holds a ref-counted batch-input-task, and
// represents a slice of batch-input-task.

// To be handed out to callers of `BatchInputTask::ToTaskHandles` quickly
// (i.e. not necessarily waiting for input split)
//
// `BatchInputTaskHandle::GetSplitTask` evaluates to the slice of task.
template <typename TaskType>
class BatchInputTaskHandle : public BatchTask {
 public:
  BatchInputTaskHandle(
      std::shared_ptr<BatchInputTask<TaskType>> batch_input_task, int split_id,
      size_t task_size);

  // Should be called once. Returns nullptr on subsequent calls.
  std::unique_ptr<TaskType> GetSplitTask();

  // Returns the size of this task.
  size_t size() const override { return task_size_; }

 private:
  template <typename T>
  friend class internal::BatchInputTaskHandleTestAccess;

  int split_id() const { return split_id_; }

  std::shared_ptr<BatchInputTask<TaskType>> batch_input_task_;

  // The handle evaluates to the N-th slice of original task, and
  // N is `split_id_`.
  const int split_id_;

  const size_t task_size_;

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
// Usage
//
// ... a deque with frequent enqueue and dequeue operations ...
// ... Note, a deque of Batch of BatchInputTaskHandle is used to form batches
//     at enqueue time (split is lazy at deque time);
// ... For use cases to form batches at dequeue time, we can use a deque of
//     BatchInputTaskHandle directly, and "peek" metadata to form a batch by
//     then.
// std::deque<std::unique_ptr<Batch<BatchInputTaskHandle<TaskType>>>> deque_
//     TF_GUARDED_BY(mu_);
//
// std::unique_ptr<TaskType> input_task;
//
// ... Enqueue path ...
//
// {
//   mutex_lock l(mu_);
//   std::shared_ptr<BatchInputTask<TaskType>> batch_input_task =
//       ConstructLazyBatchWithoutSplit(input_task);
//
//   std::vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>> task_handles;
//   input_batch->ToTaskHandles(&task_handles);
//   for (int i = 0; i < task_handles.size(); ++i) {
//     EnqueueTaskHandleIntoDeque(deque_);
//   }
//
// ... Dequeue path ...
// std::unique_ptr<Batch<BatchInputTaskHandle<TaskType>>> handles_to_schedule;
// {
//    mutex_lock l(mu_);
//    ... HasBatchToSchedule could be customized or specialized
//    ... (e.g., readiness depending on enqueue time)
//    if (HasBatchToSchedule(deque_)) {
//      handles_to_schedule = std::move(deque_.front());
//      deque_.pop_front();
//    }
// }
// ...... `mu_` is released ......
//
// std::vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>> tasks_in_batch =
//     RemoveAllTasksFromBatch(handles_to_schedule);
//
// std::unique_ptr<Batch<TaskType>> batch_to_schedule;
// for (int i = 0; i < tasks_in_batch.size(); i++) {
//   batch_to_schedule->AddTask(std::move(tasks_in_batch[i]->GetSplitTask()));
// }
// batch_to_schedule->Close();
//
// `batch_to_schedule` is ready for schedule.
template <typename TaskType>
class BatchInputTask
    : public std::enable_shared_from_this<BatchInputTask<TaskType>> {
 public:
  using SplitInputFunc = std::function<Status(
      std::unique_ptr<TaskType>* input_task, int first_output_task_size,
      int input_batch_size_limit,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)>;

  BatchInputTask(std::unique_ptr<TaskType> input_task,
                 int open_batch_remaining_slot, int batch_size_limit,
                 SplitInputFunc split_input_func);

  // Outputs the task handles for the input task.
  // Each task handle represents a slice of task after input task is split, and
  // could evaluate to that slice.
  //
  // NOTE:
  // Each task handle in `output_task_handles` takes ownership of a reference of
  // this BatchInputTask.
  void ToTaskHandles(
      std::vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>>*
          output_task_handles);

 private:
  friend class BatchInputTaskHandle<TaskType>;
  template <typename T>
  friend class internal::BatchInputTaskTestAccess;

  std::unique_ptr<TaskType> GetSplitTask(int split_id);

  Status SplitBatches(std::vector<std::unique_ptr<TaskType>>* output_tasks);

  std::unique_ptr<TaskType> input_task_;

  const int input_task_size_ = 0;
  const int open_batch_remaining_slot_;

  const int batch_size_limit_;
  const SplitInputFunc split_func_;

  const InputSplitMetadata input_split_metadata_;

  mutable absl::once_flag once_;

  std::vector<std::unique_ptr<TaskType>> task_splits_;
  Status split_status_;
};

//
// Implementation details. API readers may skip.
//

template <typename TaskType>
BatchInputTaskHandle<TaskType>::BatchInputTaskHandle(
    std::shared_ptr<BatchInputTask<TaskType>> batch_input_task, int split_id,
    size_t task_size)
    : batch_input_task_(batch_input_task),
      split_id_(split_id),
      task_size_(task_size) {}

template <typename TaskType>
std::unique_ptr<TaskType> BatchInputTaskHandle<TaskType>::GetSplitTask() {
  if (once_.load(std::memory_order_acquire)) {
    return nullptr;
  }
  once_.store(true, std::memory_order_release);
  return batch_input_task_->GetSplitTask(split_id_);
}

template <typename TaskType>
BatchInputTask<TaskType>::BatchInputTask(std::unique_ptr<TaskType> input_task,
                                         int open_batch_remaining_slot,
                                         int batch_size_limit,
                                         SplitInputFunc split_input_func)
    : input_task_(std::move(input_task)),
      input_task_size_(input_task_->size()),
      open_batch_remaining_slot_(open_batch_remaining_slot),
      batch_size_limit_(batch_size_limit),
      split_func_(split_input_func),
      input_split_metadata_(input_task_size_, open_batch_remaining_slot,
                            batch_size_limit) {}

template <typename TaskType>
void BatchInputTask<TaskType>::ToTaskHandles(
    std::vector<std::unique_ptr<BatchInputTaskHandle<TaskType>>>*
        task_handles) {
  const absl::FixedArray<int>& task_sizes = input_split_metadata_.task_sizes();
  task_handles->resize(task_sizes.size());
  for (int i = 0; i < task_handles->size(); i++) {
    (*task_handles)[i] = std::make_unique<BatchInputTaskHandle<TaskType>>(
        this->shared_from_this(), i, task_sizes[i]);
  }
}

template <typename TaskType>
std::unique_ptr<TaskType> BatchInputTask<TaskType>::GetSplitTask(int split_id) {
  absl::call_once(once_,
                  [this]() { split_status_ = SplitBatches(&task_splits_); });
  if (!split_status_.ok()) {
    LOG_EVERY_N_SEC(WARNING, 60 /* seconds */)
        << "Split task with error: " << split_status_ << " split metadata is "
        << input_split_metadata_.DebugString();
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

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_INPUT_TASK_H_
