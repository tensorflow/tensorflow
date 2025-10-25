/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/ynnpack/slinky_threadpool.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "slinky/base/function_ref.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"
#include "xla/backends/cpu/runtime/work_queue.h"
#include "tsl/profiler/lib/traceme.h"

#define EIGEN_USE_THREADS
#include "Eigen/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

//===----------------------------------------------------------------------===//
// Task
//===----------------------------------------------------------------------===//

namespace {

// Running a task can result in three states:
//
//   kPending:  The task is still being processed by the worker threads.
//   kComplete: The caller thread is the one who completed the task.
//   kDone:     The task is done and all work items have been processed, however
//              the caller thread did't process any work items.
//
// We need this state to signal the waiter thread just once, from a thread that
// completed the task.S
enum class TaskState { kPending, kComplete, kDone };

class Task final : public SlinkyThreadPool::task {
 public:
  Task(SlinkyThreadPool::task_body body, size_t num_work_items,
       size_t num_partitions);

  // Runs this task by processing work items in the current thread.
  TaskState Run();

  // Returns true if the work queue is empty. It doesn't mean that the task is
  // complete, as some threads might still be working on this task.
  bool IsEmptyWorkQueue() const;

  // Returns the number of workers that are currently working on this task.
  int64_t num_workers() const;

  // Returns true if the task is done.
  bool done() const final;

 private:
  SlinkyThreadPool::task_body body_;
  WorkQueue work_queue_;

  ABSL_CACHELINE_ALIGNED std::atomic<size_t> worker_index_;
  ABSL_CACHELINE_ALIGNED std::atomic<size_t> pending_work_items_;
};
}  // namespace

Task::Task(SlinkyThreadPool::task_body body, size_t num_work_items,
           size_t num_partitions)
    : body_(std::move(body)),
      work_queue_(num_work_items, num_partitions),
      worker_index_(0),
      pending_work_items_(num_work_items) {}

TaskState Task::Run() {
  // If we have more workers joining the task than the number of partitions,
  // then we have to wrap around to the first partition.
  size_t worker_index = worker_index_.fetch_add(1, std::memory_order_relaxed);
  if (ABSL_PREDICT_FALSE(worker_index >= work_queue_.num_partitions())) {
    worker_index %= work_queue_.num_partitions();
  }

  // Each worker processes the body using its own copy of the task.
  Worker w(worker_index, &work_queue_);
  size_t num_processed_work_items = 0;

  if (std::optional<size_t> item = w.Pop(/*notify_work_stealing=*/false)) {
    SlinkyThreadPool::task_body body = body_;

    do {
      body(*item);
      ++num_processed_work_items;
    } while ((item = w.Pop(/*notify_work_stealing=*/false)).has_value());
  }

  // The number of pending work items should never go below zero.
  size_t previous_work_items = pending_work_items_.fetch_sub(
      num_processed_work_items, std::memory_order_acq_rel);
  DCHECK_GE(previous_work_items, num_processed_work_items);

  // Task is done if we have no more work items to process. Task is complete if
  // we are the one who processed the last work item.
  bool is_done = previous_work_items == num_processed_work_items;
  bool is_complete = is_done && num_processed_work_items > 0;

  return is_complete ? TaskState::kComplete
         : is_done   ? TaskState::kDone
                     : TaskState::kPending;
}

int64_t Task::num_workers() const {
  return worker_index_.load(std::memory_order_relaxed);
}

bool Task::IsEmptyWorkQueue() const { return work_queue_.IsEmpty(); }

bool Task::done() const {
  return pending_work_items_.load(std::memory_order_acquire) == 0;
}

//===----------------------------------------------------------------------===//
// SlinkyThreadPool::Impl
//===----------------------------------------------------------------------===//

// We keep a stack of tasks that are currently being processed by current
// thread, to avoid recursive calls.
static thread_local std::vector<const Task*> task_stack;  // NOLINT

class SlinkyThreadPool::Impl : public slinky::ref_counted<Impl> {
 public:
  explicit Impl(Eigen::ThreadPoolInterface* threadpool);

  // Enqueues a new task into the queue and returns a reference to it.
  slinky::ref_count<Task> Enqueue(SlinkyThreadPool::task_body body,
                                  size_t num_work_items, size_t num_partitions);

  // Work on the single task and return the state of the task.
  TaskState WorkOnTask(Task* task);

  // Work on all tasks in the queue. Returns when Run out of tasks to process.
  void WorkOnTasks(const absl::Condition& condition);

  void Await(const absl::Condition& condition);
  void AtomicCall(slinky::function_ref<void()> t);

  // Returns true if we can schedule more workers into the underlying scheduler.
  bool CanScheduleWorkers() const;

  // Schedules the given number of workers for the given task. Worker scheduling
  // uses recursive work splitting and early exit if the task does not need any
  // more workers, of if we reached the maximum number of scheduled workers.
  void ScheduleWorkers(int64_t num_workers, slinky::ref_count<Task> task);

  size_t thread_count() const { return thread_count_; }

 private:
  friend class slinky::ref_counted<Impl>;
  static void destroy(Impl* ptr) { delete ptr; }

  // A state of the work scheduling for a given task.
  struct ScheduleState : public slinky::ref_counted<ScheduleState> {
    ScheduleState(int64_t remaining_workers, slinky::ref_count<Task> task,
                  slinky::ref_count<Impl> impl)
        : remaining_workers(remaining_workers),
          task(std::move(task)),
          impl(std::move(impl)) {}

    static void destroy(ScheduleState* ptr) { delete ptr; }

    std::atomic<int64_t> remaining_workers;
    slinky::ref_count<Task> task;
    slinky::ref_count<Impl> impl;
  };

  // Worker scheduling function for the underlying scheduler.
  template <bool release_impl_ref>
  static void ScheduleWorkers(ScheduleState* context);

  // Dequeues a pending task from the queue.
  slinky::ref_count<Task> Dequeue();

  // Signals all waiter threads waiting on the waiter mutex.
  void SignalWaiters();

  Eigen::ThreadPoolInterface* threadpool_;
  size_t thread_count_;

  std::deque<slinky::ref_count<Task>> tasks_ ABSL_GUARDED_BY(tasks_mutex_);

  // A mutex for guarding mutable state accessed concurrently.
  ABSL_CACHELINE_ALIGNED absl::Mutex tasks_mutex_;

  // A mutex for signalling threads waiting on the tasks or conditions.
  ABSL_CACHELINE_ALIGNED absl::Mutex waiter_mutex_;
};

SlinkyThreadPool::Impl::Impl(Eigen::ThreadPoolInterface* threadpool)
    : threadpool_(threadpool),
      thread_count_(threadpool_ ? threadpool_->NumThreads() : 0) {}

slinky::ref_count<Task> SlinkyThreadPool::Impl::Enqueue(
    SlinkyThreadPool::task_body body, size_t num_work_items,
    size_t num_partitions) {
  slinky::ref_count<Task> task(
      new Task(std::move(body), num_work_items, num_partitions));

  absl::MutexLock lock(tasks_mutex_);
  return tasks_.emplace_back(std::move(task));
}

slinky::ref_count<Task> SlinkyThreadPool::Impl::Dequeue() {
  absl::MutexLock lock(tasks_mutex_);

  for (auto i = tasks_.begin(); i != tasks_.end();) {
    slinky::ref_count<Task>& task = *i;

    // Task doesn't have any more work items to process.
    if (ABSL_PREDICT_FALSE(task->IsEmptyWorkQueue())) {
      i = tasks_.erase(i);
      continue;
    }

    // Don't Run the same task multiple times on the same thread.
    if (ABSL_PREDICT_FALSE(absl::c_contains(task_stack, &*task))) {
      ++i;
      continue;
    }

    return task;
  }

  return nullptr;
}

TaskState SlinkyThreadPool::Impl::WorkOnTask(Task* task) {
  DCHECK(absl::c_find(task_stack, task) == task_stack.end());

  task_stack.push_back(task);
  TaskState state = task->Run();
  task_stack.pop_back();

  // If we are the one who completed the task, we signal the waiters to wake upS
  // any threads that are waiting for the task completion. If the task was
  // completed by another worker, we do nothing to avoid the cost of waking up
  // the same thread multiple times.
  if (ABSL_PREDICT_FALSE(state == TaskState::kComplete)) {
    SignalWaiters();
  }

  return state;
}

void SlinkyThreadPool::Impl::WorkOnTasks(const absl::Condition& condition) {
  while (slinky::ref_count<Task> task = Dequeue()) {
    WorkOnTask(&*task);

    if (ABSL_PREDICT_TRUE(condition.Eval())) {
      return;
    }
  }
}

void SlinkyThreadPool::Impl::Await(const absl::Condition& condition) {
  if (ABSL_PREDICT_FALSE(!condition.Eval())) {
    tsl::profiler::TraceMe trace("SlinkyThreadPool::Await");
    absl::MutexLock lock(waiter_mutex_);
    waiter_mutex_.Await(condition);
  }
}

void SlinkyThreadPool::Impl::SignalWaiters() {
  absl::MutexLock lock(waiter_mutex_);
}

void SlinkyThreadPool::Impl::AtomicCall(slinky::function_ref<void()> t) {
  absl::MutexLock lock(waiter_mutex_);
  t();
}

bool SlinkyThreadPool::Impl::CanScheduleWorkers() const {
  // One reference is owned by the parent SlinkyThreadPool, every other
  // reference is owned by a worker scheduled into the underlying scheduler.
  return ref_count() < 1 + thread_count();
}

void SlinkyThreadPool::Impl::ScheduleWorkers(int64_t num_workers,
                                             slinky::ref_count<Task> task) {
  if (ABSL_PREDICT_TRUE(num_workers > 0 && CanScheduleWorkers())) {
    slinky::ref_count<ScheduleState> state(
        new ScheduleState(num_workers - 1, std::move(task), {this}));
    threadpool_->Schedule([state = state.take()] {
      ScheduleWorkers</*release_impl_ref=*/false>(state);
    });
  }
}

template <bool release_impl_ref>
void SlinkyThreadPool::Impl::ScheduleWorkers(ScheduleState* context) {
  auto state = slinky::ref_count<ScheduleState>::assume(context);

  // We recursively keep scheduling workers into the underlying scheduler.
  // This is more efficient than scheduling them sequentially from a single
  // thread, because workers can start processing the task sooner and we
  // distribute thread wake-ups evenly across underlying threads.
  static constexpr int32_t kNumRecursiveWorkers = 2;

  for (size_t i = 0; i < kNumRecursiveWorkers; ++i) {
    bool schedule_worker =
        state->impl->CanScheduleWorkers() && !state->task->IsEmptyWorkQueue() &&
        state->remaining_workers.fetch_sub(1, std::memory_order_relaxed) > 0;

    if (ABSL_PREDICT_TRUE(!schedule_worker)) {
      break;
    }

    // Add +1 reference to account for the scheduled worker, as we use `impl`
    // reference count to track the number of active workers.
    state->impl->add_ref();
    state->impl->threadpool_->Schedule(
        [state = slinky::ref_count<ScheduleState>(state).take()] {
          SlinkyThreadPool::Impl::ScheduleWorkers</*release_impl_ref=*/true>(
              state);
        });
  }

  // Keep processing tasks from the queue until we are out of tasks.
  static constexpr bool kFalse = false;
  state->impl->WorkOnTasks(absl::Condition(&kFalse));

  // One `impl` reference implicitly owned by the `state`, every additional
  // reference is added and released explicitly by the worker task.
  if constexpr (release_impl_ref) {
    state->impl->release();
  }
}

//===----------------------------------------------------------------------===//
// SlinkyThreadPool
//===----------------------------------------------------------------------===//

SlinkyThreadPool::SlinkyThreadPool(Eigen::ThreadPoolDevice* device)
    : impl_(new Impl(device ? device->getPool() : nullptr)) {}

SlinkyThreadPool::SlinkyThreadPool(Eigen::ThreadPoolInterface* threadpool)
    : impl_(new Impl(threadpool)) {}

SlinkyThreadPool::SlinkyThreadPool(SlinkyThreadPool&&) = default;
SlinkyThreadPool& SlinkyThreadPool::operator=(SlinkyThreadPool&&) = default;

SlinkyThreadPool::~SlinkyThreadPool() = default;

slinky::ref_count<SlinkyThreadPool::task> SlinkyThreadPool::enqueue(
    size_t n, task_body t, int32_t max_workers) {
  CHECK_GE(max_workers, n);

  // Don't create more partitions than the number of threads. Also make sure
  // that we have at least one partition (if we don't have a scheduler).
  size_t num_partitions = std::min<size_t>(n, thread_count());
  num_partitions = std::max<size_t>(1, num_partitions);

  auto task = impl_->Enqueue(std::move(t), n, num_partitions);

  // If we don't have any worker threads, we return a task to the caller, and
  // assume that the caller will wait on it.
  if (ABSL_PREDICT_FALSE(impl_->thread_count() == 0)) {
    return task;
  }

  // We assume that the caller will immediately start working on the task, so we
  // need to schedule workers only for the remaining number of partitions.
  impl_->ScheduleWorkers(/*num_workers=*/num_partitions - 1, task);

  return task;
}

void SlinkyThreadPool::wait_for(task* t) {
  Task* task = static_cast<Task*>(t);
  TaskState state = impl_->WorkOnTask(task);

  // If the task is complete or done, we are immediately done with waiting.
  if (ABSL_PREDICT_TRUE(state == TaskState::kComplete ||
                        state == TaskState::kDone)) {
    return;
  }

  // Switch to the work stealing mode and work on other tasks in the queue
  // until the given task is done.
  impl_->WorkOnTasks(absl::Condition(task, &Task::done));
  impl_->Await(absl::Condition(task, &Task::done));
}

void SlinkyThreadPool::wait_for(predicate_ref condition) {
  impl_->WorkOnTasks(absl::Condition(&condition));
  impl_->Await(absl::Condition(&condition));
}

void SlinkyThreadPool::atomic_call(slinky::function_ref<void()> t) {
  impl_->AtomicCall(t);
}

int SlinkyThreadPool::thread_count() const { return impl_->thread_count(); }

}  // namespace xla::cpu
