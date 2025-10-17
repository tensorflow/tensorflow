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

#include "xla/backends/cpu/ynn_threadpool.h"

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
#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "slinky/base/function_ref.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"

#define EIGEN_USE_THREADS
#include "Eigen/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

//===----------------------------------------------------------------------===//
// work_queue
//===----------------------------------------------------------------------===//

namespace {

// Forward declare.
class worker;

// A work queue that partitions `num_work_items` work items into
// `num_partitions` partitions processed by parallel workers.
class work_queue {
 public:
  work_queue(size_t num_work_items, size_t num_partitions);

  // Returns the next work item in the given partition. Returns std::nullopt
  // if the partition is complete.
  std::optional<size_t> pop_work_item(size_t partition_index);

  // Return the partition [begin, end) work item range.
  std::pair<size_t, size_t> partition_range(size_t partition_index) const;

  size_t num_partitions() const { return partitions_.size(); }

  // If work queue is empty, it means that all work items are being processed by
  // the workers, and the task will be done once all workers complete.
  bool is_empty() const { return empty_.load(std::memory_order_relaxed); }

 private:
  friend class worker;

  // Work items partition tracking the next work item to process.
  struct partition {
    void initialize(size_t begin, size_t end);

    // Tracks index of the next work item in the assigned partition.
    ABSL_CACHELINE_ALIGNED std::atomic<size_t> index;
    size_t begin;
    size_t end;
  };

  void set_empty() { empty_.store(true, std::memory_order_relaxed); }

  absl::FixedArray<partition, 32> partitions_;
  ABSL_CACHELINE_ALIGNED std::atomic<bool> empty_;
};

}  // namespace

void work_queue::partition::initialize(size_t begin, size_t end) {
  index.store(begin, std::memory_order_relaxed);
  this->begin = begin;
  this->end = end;
}

work_queue::work_queue(size_t num_work_items, size_t num_partitions)
    : partitions_(num_partitions), empty_(num_work_items == 0) {
  size_t partition_size = num_work_items / num_partitions;
  size_t rem_work = num_work_items % num_partitions;
  for (size_t i = 0, begin = 0, end = 0; i < num_partitions; ++i, begin = end) {
    end = begin + partition_size + ((i < rem_work) ? 1 : 0);
    partitions_[i].initialize(begin, end);
  }
}

std::optional<size_t> work_queue::pop_work_item(size_t partition_index) {
  DCHECK(partition_index < partitions_.size()) << "Invalid partition index";
  partition& partition = partitions_.data()[partition_index];

  // Check if partition is already empty.
  if (size_t index = partition.index.load(std::memory_order_relaxed);
      ABSL_PREDICT_FALSE(index >= partition.end)) {
    return std::nullopt;
  }

  // Try to acquire the next work item in the partition.
  size_t index = partition.index.fetch_add(1, std::memory_order_relaxed);
  return ABSL_PREDICT_FALSE(index >= partition.end) ? std::nullopt
                                                    : std::make_optional(index);
}

std::pair<size_t, size_t> work_queue::partition_range(
    size_t partition_index) const {
  DCHECK(partition_index < partitions_.size()) << "Invalid partition index";
  return {partitions_[partition_index].begin, partitions_[partition_index].end};
}

//===----------------------------------------------------------------------===//
// worker
//===----------------------------------------------------------------------===//

namespace {

// Worker processes work items from the work queue starting from the assigned
// work partition. Once the assigned partition is complete it tries to pop
// the work item from the next partition. Once the work queue is empty (the
// worker wraps around to the initial partition) it returns and empty work item.
class worker {
 public:
  worker(size_t partition_index, work_queue* queue);

  std::optional<size_t> pop_work_item();

 private:
  size_t initial_partition_index_;
  size_t partition_index_;
  work_queue* queue_;
};

}  // namespace

worker::worker(size_t partition_index, work_queue* queue)
    : initial_partition_index_(partition_index),
      partition_index_(partition_index),
      queue_(queue) {}

std::optional<size_t> worker::pop_work_item() {
  std::optional<size_t> work = queue_->pop_work_item(partition_index_);
  if (ABSL_PREDICT_TRUE(work)) {
    return work;
  }

  while (!work.has_value() && !queue_->is_empty()) {
    // Wrap around to the first partition.
    if (ABSL_PREDICT_FALSE(++partition_index_ >= queue_->num_partitions())) {
      partition_index_ = 0;
    }

    // We checked all partitions and got back to the partition we started from.
    if (ABSL_PREDICT_FALSE(partition_index_ == initial_partition_index_)) {
      queue_->set_empty();
      break;
    }

    work = queue_->pop_work_item(partition_index_);
  }

  return work;
}

//===----------------------------------------------------------------------===//
// task_impl
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
// completed the task.
enum class task_state { kPending, kComplete, kDone };

class task_impl final : public YnnThreadpool::task {
 public:
  task_impl(YnnThreadpool::task_body body, size_t num_work_items,
            size_t num_partitions);

  // Runs this task by process work items in the current thread.
  task_state run();

  int64_t num_workers() const;
  bool is_empty_work_queue() const;
  bool done() const final;

 private:
  YnnThreadpool::task_body body_;
  work_queue work_queue_;

  ABSL_CACHELINE_ALIGNED std::atomic<size_t> worker_index_;
  ABSL_CACHELINE_ALIGNED std::atomic<size_t> pending_work_items_;
};

}  // namespace

task_impl::task_impl(YnnThreadpool::task_body body, size_t num_work_items,
                     size_t num_partitions)
    : body_(std::move(body)),
      work_queue_(num_work_items, num_partitions),
      worker_index_(0),
      pending_work_items_(num_work_items) {}

task_state task_impl::run() {
  // If we have more workers joining the task than the number of partitions,
  // then we have to wrap around to the first partition.
  size_t worker_index = worker_index_.fetch_add(1, std::memory_order_relaxed);
  if (ABSL_PREDICT_FALSE(worker_index >= work_queue_.num_partitions())) {
    worker_index %= work_queue_.num_partitions();
  }

  // Each worker processes the body using its own copy of the task.
  worker w(worker_index, &work_queue_);
  size_t num_processed_work_items = 0;

  if (std::optional<size_t> item = w.pop_work_item()) {
    YnnThreadpool::task_body body = body_;

    do {
      body(*item);
      ++num_processed_work_items;
    } while ((item = w.pop_work_item()).has_value());
  }

  // The number of pending work items should never go below zero.
  size_t previous_work_items = pending_work_items_.fetch_sub(
      num_processed_work_items, std::memory_order_acq_rel);
  DCHECK_GE(previous_work_items, num_processed_work_items);

  // Task is done if we have no more work items to process. Task is complete if
  // we are the one who processed the last work item.
  bool is_done = previous_work_items == num_processed_work_items;
  bool is_complete = is_done && num_processed_work_items > 0;

  return is_complete ? task_state::kComplete
         : is_done   ? task_state::kDone
                     : task_state::kPending;
}

int64_t task_impl::num_workers() const {
  return worker_index_.load(std::memory_order_relaxed);
}

bool task_impl::is_empty_work_queue() const { return work_queue_.is_empty(); }

bool task_impl::done() const {
  return pending_work_items_.load(std::memory_order_acquire) == 0;
}

//===----------------------------------------------------------------------===//
// YnnThreadpool::impl
//===----------------------------------------------------------------------===//

// We keep a stack of tasks that are currently being processed by current
// thread, to avoid recursive calls.
static thread_local std::vector<const task_impl*> task_stack;  // NOLINT

class YnnThreadpool::impl : public slinky::ref_counted<impl> {
 public:
  explicit impl(Eigen::ThreadPoolInterface* threadpool);

  // Work on the single task and return the state of the task.
  task_state work_on_task(task_impl* task);

  // Work on all tasks in the queue. Returns when run out of tasks to process.
  void work_on_tasks(const absl::Condition& condition);

  // Enqueues a new task into the queue and returns a reference to it.
  slinky::ref_count<task_impl> enqueue(YnnThreadpool::task_body body,
                                       size_t num_work_items,
                                       size_t num_partitions);

  void await(const absl::Condition& condition);

  void atomic_call(slinky::function_ref<void()> t);

  // Returns true if we can schedule more workers into the underlying scheduler.
  bool can_schedule_workers() const;

  // Schedules the given number of workers for the given task. Worker scheduling
  // uses recursive work splitting and early exit if the task does not need any
  // more workers, of if we reached the maximum number of scheduled workers.
  void schedule_workers(int64_t num_workers, slinky::ref_count<task_impl> task);

  size_t thread_count() const { return thread_count_; }

 private:
  friend class slinky::ref_counted<impl>;
  static void destroy(impl* ptr) { delete ptr; }

  // A state of the work scheduling for a given task.
  struct schedule_state : public slinky::ref_counted<schedule_state> {
    schedule_state(int64_t remaining_workers, slinky::ref_count<task_impl> task,
                   slinky::ref_count<impl> impl)
        : remaining_workers(remaining_workers),
          task(std::move(task)),
          impl(std::move(impl)) {}

    static void destroy(schedule_state* ptr) { delete ptr; }

    std::atomic<int64_t> remaining_workers;
    slinky::ref_count<task_impl> task;
    slinky::ref_count<impl> impl;
  };

  // Worker scheduling function for the underlying scheduler.
  template <bool release_impl_ref>
  static void schedule_workers(schedule_state* context);

  // Dequeues a pending task from the queue.
  slinky::ref_count<task_impl> dequeue();

  // Signals all waiter threads waiting on the waiter mutex.
  void signal_waiters();

  Eigen::ThreadPoolInterface* threadpool_;
  size_t thread_count_;

  std::deque<slinky::ref_count<task_impl>> tasks_ ABSL_GUARDED_BY(tasks_mutex_);

  // A mutex for guarding mutable state accessed concurrently.
  ABSL_CACHELINE_ALIGNED absl::Mutex tasks_mutex_;

  // A mutex for signalling threads waiting on the tasks or conditions.
  ABSL_CACHELINE_ALIGNED absl::Mutex waiter_mutex_;
};

YnnThreadpool::impl::impl(Eigen::ThreadPoolInterface* threadpool)
    : threadpool_(threadpool),
      thread_count_(threadpool_ ? threadpool_->NumThreads() : 0) {}

slinky::ref_count<task_impl> YnnThreadpool::impl::enqueue(
    YnnThreadpool::task_body body, size_t num_work_items,
    size_t num_partitions) {
  slinky::ref_count<task_impl> task(
      new task_impl(std::move(body), num_work_items, num_partitions));

  absl::MutexLock lock(tasks_mutex_);
  return tasks_.emplace_back(std::move(task));
}

slinky::ref_count<task_impl> YnnThreadpool::impl::dequeue() {
  absl::MutexLock lock(tasks_mutex_);

  for (auto i = tasks_.begin(); i != tasks_.end();) {
    slinky::ref_count<task_impl>& task = *i;

    // Task doesn't have any more work items to process.
    if (ABSL_PREDICT_FALSE(task->is_empty_work_queue())) {
      i = tasks_.erase(i);
      continue;
    }

    // Don't run the same task multiple times on the same thread.
    if (ABSL_PREDICT_FALSE(absl::c_contains(task_stack, &*task))) {
      ++i;
      continue;
    }

    return task;
  }

  return nullptr;
}

task_state YnnThreadpool::impl::work_on_task(task_impl* task) {
  DCHECK(absl::c_find(task_stack, task) == task_stack.end());

  task_stack.push_back(task);
  task_state state = task->run();
  task_stack.pop_back();

  // If we are the one who completed the task, we signal the waiters to wake upS
  // any threads that are waiting for the task completion. If the task was
  // completed by another worker, we do nothing to avoid the cost of waking up
  // the same thread multiple times.
  if (ABSL_PREDICT_FALSE(state == task_state::kComplete)) {
    signal_waiters();
  }

  return state;
}

void YnnThreadpool::impl::work_on_tasks(const absl::Condition& condition) {
  while (slinky::ref_count<task_impl> task = dequeue()) {
    work_on_task(&*task);

    if (ABSL_PREDICT_TRUE(condition.Eval())) {
      return;
    }
  }
}

void YnnThreadpool::impl::await(const absl::Condition& condition) {
  if (ABSL_PREDICT_FALSE(!condition.Eval())) {
    absl::MutexLock lock(waiter_mutex_);
    waiter_mutex_.Await(condition);
  }
}

void YnnThreadpool::impl::signal_waiters() {
  absl::MutexLock lock(waiter_mutex_);
}

void YnnThreadpool::impl::atomic_call(slinky::function_ref<void()> t) {
  absl::MutexLock lock(waiter_mutex_);
  t();
}

bool YnnThreadpool::impl::can_schedule_workers() const {
  // One reference is owned by the parent YnnThreadpool, every other
  // reference is owned by a worker scheduled into the underlying scheduler.
  return ref_count() < 1 + thread_count();
}

void YnnThreadpool::impl::schedule_workers(int64_t num_workers,
                                           slinky::ref_count<task_impl> task) {
  if (ABSL_PREDICT_TRUE(num_workers > 0 && can_schedule_workers())) {
    slinky::ref_count<schedule_state> state(
        new schedule_state(num_workers - 1, std::move(task), {this}));
    threadpool_->Schedule([state = state.take()]() {
      schedule_workers</*release_impl_ref=*/false>(state);
    });
  }
}

template <bool release_impl_ref>
void YnnThreadpool::impl::schedule_workers(schedule_state* context) {
  auto state = slinky::ref_count<schedule_state>::assume(context);

  // We recursively keep scheduling workers into the underlying scheduler.
  // This is more efficient than scheduling them sequentially from a single
  // thread, because workers can start processing the task sooner and we
  // distribute thread wake-ups evenly across underlying threads.
  static constexpr int32_t kNumRecursiveWorkers = 2;

  for (size_t i = 0; i < kNumRecursiveWorkers; ++i) {
    bool schedule_worker =
        state->impl->can_schedule_workers() &&
        !state->task->is_empty_work_queue() &&
        state->remaining_workers.fetch_sub(1, std::memory_order_relaxed) > 0;

    if (ABSL_PREDICT_TRUE(!schedule_worker)) {
      break;
    }

    // Add +1 reference to account for the scheduled worker, as we use `impl`
    // reference count to track the number of active workers.
    state->impl->add_ref();
    state->impl->threadpool_->Schedule(
        [state = slinky::ref_count<schedule_state>(state).take()]() {
          YnnThreadpool::impl::schedule_workers</*release_impl_ref=*/true>(
              state);
        });
  }

  // Keep processing tasks from the queue until we are out of tasks.
  static constexpr bool kFalse = false;
  state->impl->work_on_tasks(absl::Condition(&kFalse));

  // One `impl` reference implicitly owned by the `state`, every additional
  // reference is added and released explicitly by the worker task.
  if constexpr (release_impl_ref) {
    state->impl->release();
  }
}

//===----------------------------------------------------------------------===//
// YnnThreadpool
//===----------------------------------------------------------------------===//

YnnThreadpool::YnnThreadpool(Eigen::ThreadPoolDevice* device)
    : impl_(new impl(device ? device->getPool() : nullptr)) {}

YnnThreadpool::YnnThreadpool(Eigen::ThreadPoolInterface* threadpool)
    : impl_(new impl(threadpool)) {}

YnnThreadpool::~YnnThreadpool() = default;

slinky::ref_count<YnnThreadpool::task> YnnThreadpool::enqueue(
    size_t n, task_body t, int32_t max_workers) {
  CHECK_GE(max_workers, n);

  // Don't create more partitions than the number of threads. Also make sure
  // that we have at least one partition (if we don't have a scheduler).
  size_t num_partitions = std::min<size_t>(n, thread_count());
  num_partitions = std::max<size_t>(1, num_partitions);

  auto task = impl_->enqueue(std::move(t), n, num_partitions);

  // If we don't have any worker threads, we return a task to the caller, and
  // assume that the caller will wait on it.
  if (ABSL_PREDICT_FALSE(impl_->thread_count() == 0)) {
    return task;
  }

  // We assume that the caller will immediately start working on the task, so we
  // need to schedule workers only for the remaining number of partitions.
  impl_->schedule_workers(/*num_workers=*/num_partitions - 1, task);

  return task;
}

void YnnThreadpool::wait_for(task* t) {
  task_impl* task = static_cast<task_impl*>(t);
  task_state state = impl_->work_on_task(task);

  // If the task is complete or done, we are immediately done with waiting.
  if (ABSL_PREDICT_TRUE(state == task_state::kComplete ||
                        state == task_state::kDone)) {
    return;
  }

  // Switch to the work stealing mode and work on other tasks in the queue
  // until the given task is done.
  impl_->work_on_tasks(absl::Condition(task, &task_impl::done));
  impl_->await(absl::Condition(task, &task_impl::done));
}

void YnnThreadpool::wait_for(predicate_ref condition) {
  impl_->work_on_tasks(absl::Condition(&condition));
  impl_->await(absl::Condition(&condition));
}

void YnnThreadpool::atomic_call(slinky::function_ref<void()> t) {
  impl_->atomic_call(t);
}

int YnnThreadpool::thread_count() const { return impl_->thread_count(); }

}  // namespace xla::cpu
