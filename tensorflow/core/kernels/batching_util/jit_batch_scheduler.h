/* Copyright 2024 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_JIT_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_JIT_BATCH_SCHEDULER_H_

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace serving {

// A BatchScheduler implementation that uses a Just-in-Time (JIT) batching
// strategy.
//
// Unlike BasicBatchScheduler, which forms batches as soon as requests arrive
// (up to max_batch_size), JitBatchScheduler queues incoming requests in a
// priority queue and only forms a batch when the execution device (e.g., TPU)
// is ready to process it. This minimizes the time requests spend waiting in a
// formed batch and allows higher-priority requests to "jump the queue".
//
// The scheduler maintains:
// 1. A priority queue of incoming tasks.
// 2. A small queue of ready-to-execute batches.
// 3. A background thread that assembles batches from the priority queue
//    and pushes them to the ready queue.
//
// High-priority tasks (e.g., CriticalPlus) are processed before lower-priority
// tasks. If the priority queue is full, lower-priority tasks may be preempted.
template <typename TaskType>
class JitBatchScheduler : public BatchScheduler<TaskType> {
 public:
  struct Options {
    // The name to use for the pool of batch threads.
    std::string thread_pool_name = {"jit_batch_threads"};

    // The number of threads to use to process batches.
    int num_batch_threads = port::MaxParallelism();

    // The maximum size of each batch.
    int max_batch_size = 1000;

    // The maximum number of tasks to hold in the priority queue.
    int max_queue_size = 1000;

    // The maximum number of formed batches to keep ready.
    // Keeping this small (e.g. 1) ensures JIT behavior.
    int max_ready_batches = 1;

    // The environment to use.
    Env* env = Env::Default();
  };

  static absl::Status Create(
      const Options& options,
      std::function<void(std::unique_ptr<Batch<TaskType>>)>
          process_batch_callback,
      std::unique_ptr<BatchScheduler<TaskType>>* scheduler);

  ~JitBatchScheduler() override;

  absl::Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;
  size_t max_task_size() const override { return options_.max_batch_size; }

 private:
  explicit JitBatchScheduler(
      const Options& options,
      std::function<void(std::unique_ptr<Batch<TaskType>>)>
          process_batch_callback);

  // The main loop for the background thread that assembles batches.
  void BatchAssemblyLoop();

  // Helper to find the lowest priority task in the heap.
  // Returns iterator to the lowest priority task.
  typename std::vector<std::unique_ptr<TaskType>>::iterator
  FindLowestPriorityTask() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Comparator for the max-heap. Returns true if 'a' has lower priority than
  // 'b' (so 'b' goes to top).
  static bool CompareTasks(const std::unique_ptr<TaskType>& a,
                           const std::unique_ptr<TaskType>& b) {
    // Criticality enum: Sheddable(0) < ... < CriticalPlus(3).
    // We want max heap, so largest value at top.
    return a->criticality() < b->criticality();
  }

  const Options options_;
  std::function<void(std::unique_ptr<Batch<TaskType>>)> process_batch_callback_;

  std::unique_ptr<thread::ThreadPool> batch_thread_pool_;
  std::unique_ptr<Thread> assembly_thread_;

  mutable mutex mu_;
  condition_variable cv_;
  bool stop_ TF_GUARDED_BY(mu_) = false;

  // Max-heap of pending tasks.
  std::vector<std::unique_ptr<TaskType>> task_queue_ TF_GUARDED_BY(mu_);

  // Queue of fully assembled batches ready for processing.
  std::deque<std::unique_ptr<Batch<TaskType>>> ready_batches_
      TF_GUARDED_BY(mu_);

  // Tracks the number of batches currently being processed by the thread pool.
  // We use this to decide when to trigger assembly of the next batch.
  int num_batches_in_flight_ TF_GUARDED_BY(mu_) = 0;
};

// Implementation details

template <typename TaskType>
absl::Status JitBatchScheduler<TaskType>::Create(
    const Options& options,
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* scheduler) {
  if (options.num_batch_threads < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "num_batch_threads must be positive; was ", options.num_batch_threads));
  }
  if (options.max_batch_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "max_batch_size must be positive; was ", options.max_batch_size));
  }
  if (options.max_queue_size <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "max_queue_size must be positive; was ", options.max_queue_size));
  }

  scheduler->reset(
      new JitBatchScheduler<TaskType>(options, process_batch_callback));
  return absl::OkStatus();
}

template <typename TaskType>
JitBatchScheduler<TaskType>::JitBatchScheduler(
    const Options& options,
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback)
    : options_(options), process_batch_callback_(process_batch_callback) {
  batch_thread_pool_ = std::make_unique<thread::ThreadPool>(
      options.env, options.thread_pool_name, options.num_batch_threads);

  assembly_thread_ = std::unique_ptr<Thread>(options.env->StartThread(
      {}, "JitBatchAssembly", [this]() { BatchAssemblyLoop(); }));
}

template <typename TaskType>
JitBatchScheduler<TaskType>::~JitBatchScheduler() {
  {
    mutex_lock l(mu_);
    stop_ = true;
    cv_.notify_all();
  }
  // The thread will exit its loop and be cleaned up when the unique_ptr is
  // destroyed.
}

template <typename TaskType>
absl::Status JitBatchScheduler<TaskType>::Schedule(
    std::unique_ptr<TaskType>* task) {
  size_t size = (*task)->size();
  if (size > options_.max_batch_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task size ", size, " is larger than maximum batch size ",
                     options_.max_batch_size));
  }

  mutex_lock l(mu_);

  if (task_queue_.size() >= options_.max_queue_size) {
    // Queue is full. Try to preempt lowest priority task.
    auto lowest_it = FindLowestPriorityTask();
    if (lowest_it != task_queue_.end()) {
      // Check if new task has higher priority than the lowest task.
      if (CompareTasks(*lowest_it, *task)) {  // lowest < new
        // Preempt lowest.
        std::unique_ptr<TaskType> preempted_task = std::move(*lowest_it);
        task_queue_.erase(lowest_it);
        std::make_heap(task_queue_.begin(), task_queue_.end(), CompareTasks);

        // Notify the preempted task.
        preempted_task->FinishTask(absl::UnavailableError(
            "Task preempted from JitBatchScheduler queue "
            "due to higher priority task."));
      } else {
        return absl::UnavailableError(
            "JitBatchScheduler queue is full and new task priority is not "
            "higher than lowest.");
      }
    } else {
      return absl::UnavailableError("JitBatchScheduler queue is full.");
    }
  }

  task_queue_.push_back(std::move(*task));
  std::push_heap(task_queue_.begin(), task_queue_.end(), CompareTasks);
  cv_.notify_one();  // Wake up assembly thread

  return absl::OkStatus();
}

template <typename TaskType>
size_t JitBatchScheduler<TaskType>::NumEnqueuedTasks() const {
  mutex_lock l(mu_);
  return task_queue_.size();  // Approximate, excludes ready batches
}

template <typename TaskType>
size_t JitBatchScheduler<TaskType>::SchedulingCapacity() const {
  mutex_lock l(mu_);
  return options_.max_queue_size - task_queue_.size();
}

template <typename TaskType>
void JitBatchScheduler<TaskType>::BatchAssemblyLoop() {
  while (true) {
    std::unique_ptr<Batch<TaskType>> new_batch;
    {
      mutex_lock l(mu_);
      while (!stop_) {
        bool can_assemble = !task_queue_.empty() &&
                            ready_batches_.size() < options_.max_ready_batches;
        bool can_dispatch = !ready_batches_.empty() &&
                            num_batches_in_flight_ < options_.num_batch_threads;

        if (can_assemble || can_dispatch) break;
        cv_.wait(l);
      }
      if (stop_) return;

      // Assemble
      if (!task_queue_.empty() &&
          ready_batches_.size() < options_.max_ready_batches) {
        // Create a new batch
        // We need to generate a traceme id.
        static uint64_t batch_id_counter = 0;
        new_batch = std::make_unique<Batch<TaskType>>(++batch_id_counter);

        while (!task_queue_.empty() &&
               new_batch->size() + task_queue_.front()->size() <=
                   options_.max_batch_size) {
          // Pop highest priority
          std::pop_heap(task_queue_.begin(), task_queue_.end(), CompareTasks);
          new_batch->AddTask(std::move(task_queue_.back()));
          task_queue_.pop_back();
        }

        new_batch->Close();
        ready_batches_.push_back(std::move(new_batch));
      }

      // Dispatch
      while (!ready_batches_.empty() &&
             num_batches_in_flight_ < options_.num_batch_threads) {
        auto batch = std::move(ready_batches_.front());
        ready_batches_.pop_front();
        num_batches_in_flight_++;

        batch_thread_pool_->Schedule([this, batch_ptr = batch.release()]() {
          std::unique_ptr<Batch<TaskType>> batch(batch_ptr);
          process_batch_callback_(std::move(batch));

          {
            mutex_lock l(mu_);
            num_batches_in_flight_--;
            cv_.notify_one();  // Trigger assembly of next batch if needed
          }
        });
      }
    }
  }
}

template <typename TaskType>
typename std::vector<std::unique_ptr<TaskType>>::iterator
JitBatchScheduler<TaskType>::FindLowestPriorityTask() {
  if (task_queue_.empty()) return task_queue_.end();

  // In a max-heap, the minimum is one of the leaves.
  // Leaves are at indices [size/2 ... size-1].
  // But we also need to check the whole array if we are not sure about
  // implementation details, though property guarantees it. For
  // safety/simplicity in this initial version, linear scan is fine.

  auto min_it = task_queue_.begin();
  for (auto it = task_queue_.begin(); it != task_queue_.end(); ++it) {
    // CompareTasks(a, b) returns true if a < b (priority).
    // So if CompareTasks(*it, *min_it) is true, it means *it has lower priority
    // than *min_it.
    if (CompareTasks(*it, *min_it)) {
      min_it = it;
    }
  }
  return min_it;
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_JIT_BATCH_SCHEDULER_H_
