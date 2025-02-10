/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Abstractions for processing small tasks in a batched fashion, to reduce
// processing times and costs that can be amortized across multiple tasks.
//
// The core class is BatchScheduler, which groups tasks into batches.
//
// BatchScheduler encapsulates logic for aggregating multiple tasks into a
// batch, and kicking off processing of a batch on a thread pool it manages.
//
// This file defines an abstract BatchScheduler class.

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_H_

#include <stddef.h>
#include <sys/types.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/criticality.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace serving {

const absl::string_view kLowPriorityPaddingWithMaxBatchSizeAttrValue =
    "low_priority_padding_with_max_batch_size";
const absl::string_view kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue =
    "low_priority_padding_with_next_allowed_batch_size";
const absl::string_view kPriorityIsolationAttrValue = "priority_isolation";
const absl::string_view kPriorityMergeAttrValue = "priority_merge";

enum class MixedPriorityBatchingPolicy {
  kLowPriorityPaddingWithMaxBatchSize,
  kLowPriorityPaddingWithNextAllowedBatchSize,
  kPriorityIsolation,
  kPriorityMerge,
};

absl::StatusOr<MixedPriorityBatchingPolicy> GetMixedPriorityBatchingPolicy(
    absl::string_view attr_value);

// The abstract superclass for a unit of work to be done as part of a batch.
//
// An implementing subclass typically contains (or points to):
//  (a) input data;
//  (b) a thread-safe completion signal (e.g. a Notification);
//  (c) a place to store the outcome (success, or some error), upon completion;
//  (d) a place to store the output data, upon success.
//
// Items (b), (c) and (d) are typically non-owned pointers to data homed
// elsewhere, because a task's ownership gets transferred to a BatchScheduler
// (see below) and it may be deleted as soon as it is done executing.
class BatchTask {
 public:
  virtual ~BatchTask() = default;

  // Returns the size of the task, in terms of how much it contributes to the
  // size of a batch. (A batch's size is the sum of its task sizes.)
  virtual size_t size() const = 0;

  // Returns the criticality of associated with the task. It defaults to
  // kCritical.
  virtual tsl::criticality::Criticality criticality() const {
    return tsl::criticality::Criticality::kCritical;
  }
};

// A thread-safe collection of BatchTasks. Tasks can be either added or removed
// from the TaskQueue. It is mainly used to hold the registered tasks without
// forming batches, so that the batches can be formed more flexibly right before
// they get scheduled for execution.
//
// Type parameter TaskType must be a subclass of BatchTask.
template <typename TaskType>
class TaskQueue {
 public:
  TaskQueue() = default;

  struct TaskWrapper {
    std::unique_ptr<TaskType> task;
    uint64 start_time_micros;

    TaskWrapper(std::unique_ptr<TaskType> task, uint64 start_time_micros)
        : task(std::move(task)), start_time_micros(start_time_micros) {}
  };

  // Appends a task to the end of the queue with the given start time.
  void AddTask(std::unique_ptr<TaskType> task, uint64 start_time_micros);

  // Adds a task to the front of the queue with the given start time.
  void PrependTask(std::unique_ptr<TaskType> task, uint64 start_time_micros);

  // Removes a task from the front of the queue, i.e., the oldest task in the
  // queue.
  std::unique_ptr<TaskType> RemoveTask();

  // Removes tasks from the front of the queue as many as possible as long as
  // the sum of sizes of the removed tasks don't exceed the 'size' given as the
  // argument.
  std::vector<std::unique_ptr<TaskType>> RemoveTask(int size);

  // Returns the start time of the earliest task in the queue. If the queue is
  // empty, return the null value.
  std::optional<uint64> EarliestTaskStartTime() const;

  // Returns true iff the queue contains 0 tasks.
  bool empty() const;

  // Returns the number of tasks in the queue.
  int num_tasks() const;

  // Returns the sum of the task sizes.
  int size() const;

 private:
  mutable mutex mu_;

  // Tasks in the queue.
  std::deque<TaskWrapper> tasks_ TF_GUARDED_BY(mu_);

  // The sum of the sizes of the tasks in 'tasks_'.
  int size_ TF_GUARDED_BY(mu_) = 0;

  // Whether the queue is empty.
  std::atomic<bool> empty_ TF_GUARDED_BY(mu_){true};

  // The copy constructor and the assign op are deleted.
  TaskQueue(const TaskQueue&) = delete;
  void operator=(const TaskQueue&) = delete;
};

template <typename TaskType>
void TaskQueue<TaskType>::AddTask(std::unique_ptr<TaskType> task,
                                  uint64 start_time_micros) {
  {
    mutex_lock l(mu_);
    size_ += task->size();
    tasks_.emplace_back(std::move(task), start_time_micros);
    empty_.store(false);
  }
}

template <typename TaskType>
void TaskQueue<TaskType>::PrependTask(std::unique_ptr<TaskType> task,
                                      uint64 start_time_micros) {
  {
    mutex_lock l(mu_);
    size_ += task->size();
    tasks_.emplace_front(std::move(task), start_time_micros);
    empty_.store(false);
  }
}

template <typename TaskType>
std::unique_ptr<TaskType> TaskQueue<TaskType>::RemoveTask() {
  {
    mutex_lock l(mu_);
    if (tasks_.empty()) {
      return nullptr;
    }
    std::unique_ptr<TaskType> task = std::move(tasks_.front().task);
    size_ -= task->size();
    tasks_.pop_front();
    if (tasks_.empty()) {
      empty_.store(true);
    }
    return task;
  }
}

template <typename TaskType>
std::vector<std::unique_ptr<TaskType>> TaskQueue<TaskType>::RemoveTask(
    int size) {
  {
    mutex_lock l(mu_);
    if (tasks_.empty()) {
      return {};
    }

    int size_lower_bound = size_ - size;
    std::vector<std::unique_ptr<TaskType>> remove_tasks;
    while (!tasks_.empty() &&
           size_ - static_cast<int>(tasks_.front().task->size()) >=
               size_lower_bound) {
      size_ -= static_cast<int>(tasks_.front().task->size());
      remove_tasks.push_back(std::move(tasks_.front().task));
      tasks_.pop_front();
      if (tasks_.empty()) {
        empty_.store(true);
      }
    }
    return remove_tasks;
  }
}

template <typename TaskType>
bool TaskQueue<TaskType>::empty() const {
  {
    mutex_lock l(mu_);
    return empty_.load();
  }
}

template <typename TaskType>
std::optional<uint64> TaskQueue<TaskType>::EarliestTaskStartTime() const {
  {
    mutex_lock l(mu_);

    if (tasks_.empty()) {
      return std::nullopt;
    }

    return tasks_.front().start_time_micros;
  }
}

template <typename TaskType>
int TaskQueue<TaskType>::num_tasks() const {
  {
    mutex_lock l(mu_);
    return tasks_.size();
  }
}

template <typename TaskType>
int TaskQueue<TaskType>::size() const {
  {
    mutex_lock l(mu_);
    return size_;
  }
}

// A thread-safe collection of BatchTasks, to be executed together in some
// fashion.
//
// At a given time, a batch is either "open" or "closed": an open batch can
// accept new tasks; a closed one cannot. A batch is monotonic: initially it is
// open and tasks can be added to it; then it is closed and its set of tasks
// remains fixed for the remainder of its life. A closed batch cannot be re-
// opened.
//
// Type parameter TaskType must be a subclass of BatchTask.
template <typename TaskType>
class Batch {
 public:
  Batch();
  explicit Batch(uint64 traceme_context_id);
  virtual ~Batch();  // Blocks until the batch is closed.

  // Appends 'task' to the batch. After calling AddTask(), the newly-added task
  // can be accessed via task(num_tasks()-1) or mutable_task(num_tasks()-1).
  // Dies if the batch is closed.
  void AddTask(std::unique_ptr<TaskType> task, uint64 start_time_micros = 0);

  // Removes the most recently added task. Returns nullptr if the batch is
  // empty.
  std::unique_ptr<TaskType> RemoveTask();

  // Caller takes ownership of returned tasks.
  // Must be called after a batch is closed.
  std::vector<std::unique_ptr<TaskType>> RemoveAllTasks();

  // Returns the number of tasks in the batch.
  int num_tasks() const;

  // Returns true iff the batch contains 0 tasks.
  bool empty() const;

  // Returns a reference to the ith task (in terms of insertion order).
  const TaskType& task(int i) const;

  // Returns a pointer to the ith task (in terms of insertion order).
  //
  // Caller doesn't take ownership.
  TaskType* mutable_task(int i);

  // Returns the sum of the task sizes.
  size_t size() const;

  // Returns true iff the batch is currently closed.
  bool IsClosed() const;

  // Blocks until the batch is closed.
  void WaitUntilClosed() const;

  // Marks the batch as closed. Dies if called more than once.
  void Close();

  // Returns the TraceMe context id of this batch.
  uint64 traceme_context_id() const;

  // Attempts to trim this batch to a new, smaller size (not to be confused with
  // the number of tasks in the batch). On success, the trimmed tasks go into
  // 'out_trimmed_tasks' in the same order the tasks were in this batch.
  //
  // The method might not succeed if it needs to split a large task to hit the
  // correct size.
  void TryTrimToNewSize(
      int new_size, std::vector<std::unique_ptr<TaskType>>& out_trimmed_tasks);

  // Returns the start time of the earliest task in the queue. If the queue is
  // empty, return the null value.
  std::optional<uint64> EarliestTaskStartTime() const;

 private:
  mutable mutex mu_;

  // The tasks in the batch.
  std::vector<std::unique_ptr<TaskType>> tasks_ TF_GUARDED_BY(mu_);

  // The sum of the sizes of the tasks in 'tasks_'.
  size_t size_ TF_GUARDED_BY(mu_) = 0;

  std::atomic<bool> empty_ TF_GUARDED_BY(mu_){true};

  // Whether the batch has been closed.
  Notification closed_;

  // The TracMe context id.
  const uint64 traceme_context_id_;

  // The minimum start time of all tasks in the batch.
  // If the batch is empty, the value is undefined.
  uint64 earliest_task_start_time_micros_ TF_GUARDED_BY(mu_);

  Batch(const Batch&) = delete;
  void operator=(const Batch&) = delete;
};

// An abstract batch scheduler class. Collects individual tasks into batches,
// and processes each batch on a pool of "batch threads" that it manages. The
// actual logic for processing a batch is accomplished via a callback.
//
// Type parameter TaskType must be a subclass of BatchTask.
template <typename TaskType>
class BatchScheduler {
 public:
  virtual ~BatchScheduler() = default;

  // Submits a task to be processed as part of a batch.
  //
  // Ownership of '*task' is transferred to the callee iff the method returns
  // Status::OK. In that case, '*task' is left as nullptr. Otherwise, '*task' is
  // left as-is.
  //
  // If no batch processing capacity is available to process this task at the
  // present time, and any task queue maintained by the implementing subclass is
  // full, this method returns an UNAVAILABLE error code. The client may retry
  // later.
  //
  // Other problems, such as the task size being larger than the maximum batch
  // size, yield other, permanent error types.
  //
  // In all cases, this method returns "quickly" without blocking for any
  // substantial amount of time. If the method returns Status::OK, the task is
  // processed asynchronously, and any errors that occur during the processing
  // of the batch that includes the task can be reported to 'task'.
  virtual absl::Status Schedule(std::unique_ptr<TaskType>* task) = 0;

  // Returns the number of tasks that have been scheduled (i.e. accepted by
  // Schedule()), but have yet to be handed to a thread for execution as part of
  // a batch. Note that this returns the number of tasks, not the aggregate task
  // size (so if there is one task of size 3 and one task of size 5, this method
  // returns 2 rather than 8).
  virtual size_t NumEnqueuedTasks() const = 0;

  // Returns a guaranteed number of size 1 tasks that can be Schedule()d without
  // getting an UNAVAILABLE error. In a typical implementation, returns the
  // available space on a queue.
  //
  // There are two important caveats:
  //  1. The guarantee does not extend to varying-size tasks due to possible
  //     internal fragmentation of batches.
  //  2. The guarantee only holds in a single-thread environment or critical
  //     section, i.e. if an intervening thread cannot call Schedule().
  //
  // This method is useful for monitoring, or for guaranteeing a future slot in
  // the schedule (but being mindful about the caveats listed above).
  virtual size_t SchedulingCapacity() const = 0;

  // Returns the maximum allowed size of tasks submitted to the scheduler. (This
  // is typically equal to a configured maximum batch size.)
  virtual size_t max_task_size() const = 0;
};

//////////
// Implementation details follow. API users need not read.

template <typename TaskType>
Batch<TaskType>::Batch() : Batch(0) {}

template <typename TaskType>
Batch<TaskType>::Batch(uint64 traceme_context_id)
    : traceme_context_id_(traceme_context_id) {}

template <typename TaskType>
Batch<TaskType>::~Batch() {
  WaitUntilClosed();
}

template <typename TaskType>
void Batch<TaskType>::AddTask(std::unique_ptr<TaskType> task,
                              uint64 start_time_micros) {
  DCHECK(!IsClosed());
  {
    mutex_lock l(mu_);
    size_ += task->size();
    tasks_.push_back(std::move(task));
    empty_.store(false);
    if (tasks_.size() == 1) {
      earliest_task_start_time_micros_ = start_time_micros;
    } else {
      earliest_task_start_time_micros_ =
          std::min(earliest_task_start_time_micros_, start_time_micros);
    }
  }
}

template <typename TaskType>
std::optional<uint64> Batch<TaskType>::EarliestTaskStartTime() const {
  {
    mutex_lock l(mu_);
    if (tasks_.empty()) {
      return std::nullopt;
    }
    return earliest_task_start_time_micros_;
  }
}

template <typename TaskType>
std::vector<std::unique_ptr<TaskType>> Batch<TaskType>::RemoveAllTasks() {
  DCHECK(IsClosed());
  {
    mutex_lock l(mu_);
    size_ = 0;
    empty_.store(true);
    std::vector<std::unique_ptr<TaskType>> tasks_to_return;

    // Swapping vector takes constant time.
    tasks_to_return.swap(tasks_);
    return std::move(tasks_to_return);
  }
}

template <typename TaskType>
std::unique_ptr<TaskType> Batch<TaskType>::RemoveTask() {
  {
    mutex_lock l(mu_);
    if (tasks_.empty()) {
      return nullptr;
    }
    std::unique_ptr<TaskType> task = std::move(tasks_.back());
    size_ -= task->size();
    tasks_.pop_back();
    if (tasks_.empty()) {
      empty_.store(true);
    }
    return task;
  }
}

template <typename TaskType>
int Batch<TaskType>::num_tasks() const {
  {
    mutex_lock l(mu_);
    return tasks_.size();
  }
}

template <typename TaskType>
bool Batch<TaskType>::empty() const TF_NO_THREAD_SAFETY_ANALYSIS {
  // tracer is added to zoom in about this method.
  // TODO(b/160249203): Remove tracer after evaluating a change to reduce
  // lock contention and cpu usage (which is observed in profiler and
  // very data-driven).
  tsl::profiler::TraceMe tracer("BatchTask::empty");
  return empty_.load();
}

template <typename TaskType>
const TaskType& Batch<TaskType>::task(int i) const {
  DCHECK_GE(i, 0);
  {
    mutex_lock l(mu_);
    DCHECK_LT(i, tasks_.size());
    return *tasks_[i].get();
  }
}

template <typename TaskType>
TaskType* Batch<TaskType>::mutable_task(int i) {
  DCHECK_GE(i, 0);
  {
    mutex_lock l(mu_);
    DCHECK_LT(i, tasks_.size());
    return tasks_[i].get();
  }
}

template <typename TaskType>
size_t Batch<TaskType>::size() const {
  {
    mutex_lock l(mu_);
    return size_;
  }
}

template <typename TaskType>
bool Batch<TaskType>::IsClosed() const {
  return const_cast<Notification*>(&closed_)->HasBeenNotified();
}

template <typename TaskType>
void Batch<TaskType>::WaitUntilClosed() const {
  const_cast<Notification*>(&closed_)->WaitForNotification();
}

template <typename TaskType>
void Batch<TaskType>::Close() {
  closed_.Notify();
}

template <typename TaskType>
uint64 Batch<TaskType>::traceme_context_id() const {
  return traceme_context_id_;
}

template <typename TaskType>
void Batch<TaskType>::TryTrimToNewSize(
    int new_size, std::vector<std::unique_ptr<TaskType>>& out_trimmed_tasks) {
  mutex_lock l(mu_);
  DCHECK_GT(new_size, 0);
  DCHECK_LT(new_size, size_);
  DCHECK(out_trimmed_tasks.empty());

  // Index of the first task to trim away. It is possible that it is the index
  // of a task of size larger than 1 that will have to be split in order to get
  // to the target new_size.
  int32 first_task_to_move = 0;
  // The sum of sizes of tasks i, where i < first_task_to_move.
  int32 size_of_previous_tasks = 0;
  while (size_of_previous_tasks + tasks_[first_task_to_move]->size() <=
         new_size) {
    size_of_previous_tasks += tasks_[first_task_to_move]->size();
    first_task_to_move++;
    // The loop must always stop before this check is tripped because new_size
    // must never be larger than the size of the batch.
    DCHECK_LT(first_task_to_move, tasks_.size());
  }

  // Check whether task 'first_task_to_move' will have to be split.
  if (size_of_previous_tasks < new_size) {
    // TODO: b/325954758 - Consider supporting splitting large tasks and then
    // drop 'Try' from the method name.
    return;
  }
  DCHECK_EQ(size_of_previous_tasks, new_size);

  // Actually trim.
  out_trimmed_tasks.reserve(tasks_.size() - first_task_to_move);
  std::move(tasks_.begin() + first_task_to_move, tasks_.end(),
            std::back_inserter(out_trimmed_tasks));
  tasks_.resize(first_task_to_move);
  size_ = new_size;
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_SCHEDULER_H_
