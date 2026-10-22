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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/match.h"
#include "tensorflow/core/platform/mutex.h"
#define EIGEN_USE_THREADS

#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h"

#define EIGEN_USE_THREADS

#include "absl/synchronization/barrier.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tfrt/host_context/task_function.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

TEST(RunHandlerUtilTest, TestBasicScheduling) {
  int num_threads = 2;
  int num_handlers = 10;

  RunHandlerPool::Options options;
  options.num_intra_op_threads = num_threads;
  options.num_inter_op_threads = num_threads;
  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(options));

  // RunHandler should always be able to run num_threads inter closures
  absl::Barrier barrier(num_threads);

  tensorflow::BlockingCounter counter(2 * num_handlers * num_threads);

  tensorflow::thread::ThreadPool test_pool(tensorflow::Env::Default(), "test",
                                           num_handlers);
  for (int i = 0; i < num_handlers; ++i) {
    test_pool.Schedule([&counter, &barrier, &pool, i, num_threads]() {
      auto handler = pool->Get(i);
      tensorflow::BlockingCounter local_counter(2 * num_threads);
      for (int j = 0; j < num_threads; ++j) {
        handler->ScheduleInterOpClosure(
            TaskFunction([&local_counter, &counter, &barrier, i]() {
              if (i == 2) {
                barrier.Block();
              }
              counter.DecrementCount();
              local_counter.DecrementCount();
            }));
        handler->ScheduleIntraOpClosure(
            TaskFunction([&local_counter, &counter]() {
              counter.DecrementCount();
              local_counter.DecrementCount();
            }));
      }
      local_counter.Wait();
    });
  }
  counter.Wait();
}

TEST(RunHandlerUtilTest, PrioritySchedulingTest) {
  int num_threads = 2;
  RunHandlerPool::Options pool_options;
  pool_options.num_intra_op_threads = num_threads;
  pool_options.num_inter_op_threads = num_threads;
  pool_options.num_threads_in_sub_thread_pool = {2};
  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(pool_options));

  RunHandlerOptions options = RunHandlerOptions();
  options.priority = 2;
  auto handler1 = pool->Get(/*step_id=*/1, /*timeout_in_ms=*/0, options);
  options.priority = 1;
  auto handler2 = pool->Get(/*step_id=*/2, /*timeout_in_ms=*/0, options);
  options.priority = 3;
  auto handler3 = pool->Get(/*step_id=*/3, /*timeout_in_ms=*/0, options);

  // The active requests should be ordered by priorites.
  std::vector<int64_t> sorted_active_list =
      pool->GetActiveHandlerPrioritiesForTesting();
  EXPECT_EQ(sorted_active_list.size(), 3);
  EXPECT_EQ(sorted_active_list[0], 3);
  EXPECT_EQ(sorted_active_list[1], 2);
  EXPECT_EQ(sorted_active_list[2], 1);

  handler1.reset();
  options.priority = 5;
  auto handler4 = pool->Get(/*step_id=*/4, /*timeout_in_ms=*/0, options);
  options.priority = 4;
  auto handler5 = pool->Get(/*step_id=*/5, /*timeout_in_ms=*/0, options);
  sorted_active_list = pool->GetActiveHandlerPrioritiesForTesting();
  EXPECT_EQ(sorted_active_list.size(), 4);
  EXPECT_EQ(sorted_active_list[0], 5);
  EXPECT_EQ(sorted_active_list[1], 4);
  EXPECT_EQ(sorted_active_list[2], 3);
  EXPECT_EQ(sorted_active_list[3], 1);
}

TEST(RunHandlerUtilTest, PriorityExecutionTest) {
  // Set up a pool with 1 intra-op thread and 1 inter-op thread. The frist
  // request will be scheduled immediately and be blocked until the rest of the
  // two requests are enqueued. The execution order of the latter two requests
  // should follow the priority.
  int num_threads = 1;
  RunHandlerPool::Options pool_options;
  pool_options.num_intra_op_threads = num_threads;
  pool_options.num_inter_op_threads = num_threads;
  pool_options.num_threads_in_sub_thread_pool = {1};
  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(pool_options));

  RunHandlerOptions options;

  options.priority = 2;
  auto handler_med = pool->Get(2, 0, options);

  options.priority = 3;
  auto handler_high = pool->Get(1, 0, options);

  options.priority = 1;
  auto handler_low = pool->Get(3, 0, options);

  std::vector<int64_t> sorted_active_list =
      pool->GetActiveHandlerPrioritiesForTesting();
  EXPECT_EQ(sorted_active_list.size(), 3);
  EXPECT_EQ(sorted_active_list[0], 3);
  EXPECT_EQ(sorted_active_list[1], 2);
  EXPECT_EQ(sorted_active_list[2], 1);

  absl::Notification start_block;
  absl::Notification continue_block;

  // Block the worker on the low priority handler.
  handler_low->ScheduleInterOpClosure(TaskFunction([&]() {
    start_block.Notify();
    continue_block.WaitForNotification();
  }));

  start_block.WaitForNotification();

  std::vector<int> execution_order;
  tensorflow::mutex mu;
  tensorflow::BlockingCounter counter(2);

  // Schedule task on medium priority handler.
  handler_med->ScheduleInterOpClosure(TaskFunction([&]() {
    {
      tensorflow::mutex_lock l(mu);
      execution_order.push_back(2);
    }
    counter.DecrementCount();
  }));

  // Schedule task on high priority handler.
  handler_high->ScheduleInterOpClosure(TaskFunction([&]() {
    {
      tensorflow::mutex_lock l(mu);
      execution_order.push_back(3);
    }
    counter.DecrementCount();
  }));

  continue_block.Notify();
  counter.Wait();

  // The worker should pick tasks from high priority handlers first because
  // after finishing the task on the low priority handler, it iterates from the
  // beginning of the sorted list (high, med, low).
  EXPECT_EQ(execution_order.size(), 2);
  EXPECT_EQ(execution_order[0], 3);
  EXPECT_EQ(execution_order[1], 2);
}

TEST(RunHandlerUtilTest, IntraOpThreadPool) {
  int num_threads = 2;
  RunHandlerPool::Options pool_options;
  pool_options.num_intra_op_threads = num_threads;
  pool_options.num_inter_op_threads = num_threads;
  pool_options.num_threads_in_sub_thread_pool = {2};
  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(pool_options));

  RunHandlerOptions options = RunHandlerOptions();
  auto handler = pool->Get(/*step_id=*/1, /*timeout_in_ms=*/0, options);
  auto* intra_pool = handler->AsIntraThreadPoolInterface();

  absl::Notification notification;
  intra_pool->Schedule([&notification]() { notification.Notify(); });
  notification.WaitForNotification();
}

// Verifies that ScheduleIntraOpClosure enqueues work to the non-blocking
// (intra-op) queue.
TEST(RunHandlerUtilTest, ScheduleIntraOpClosureRoutesToNonBlockingQueue) {
  RunHandlerPool::Options options;
  options.num_inter_op_threads = 1;  // 1 blocking thread
  options.num_intra_op_threads = 1;  // 1 non-blocking thread
  options.num_threads_in_sub_thread_pool = {2};

  // Notifications must be declared before the pool and handler to avoid race
  // conditions in destruction.
  absl::Notification blocker_started;
  absl::Notification blocker_release;
  absl::Notification intra_done;

  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(options));
  auto handler = pool->Get(/*step_id=*/1, /*timeout_in_ms=*/0);

  // Block the sole inter-op (blocking) thread.
  handler->ScheduleInterOpClosure(TaskFunction([&]() {
    blocker_started.Notify();
    blocker_release.WaitForNotification();
  }));
  blocker_started.WaitForNotification();

  // Schedule an intra-op closure. With the correct implementation this goes
  // to the non-blocking queue and the intra-op thread picks it up.
  handler->ScheduleIntraOpClosure(TaskFunction([&]() { intra_done.Notify(); }));

  // If ScheduleIntraOpClosure incorrectly enqueued as blocking work, the
  // intra-op thread cannot pick it up and this would hang.
  EXPECT_TRUE(intra_done.WaitForNotificationWithTimeout(absl::Seconds(10)));

  // Unblock the inter-op thread so the pool can shut down cleanly.
  blocker_release.Notify();
}

// Copyable RAII token for std::function closures that invokes `on_destroy`
// when the last copy of the closure stored in TaskImpl is destroyed.
class ClosureLifetimeToken {
 public:
  explicit ClosureLifetimeToken(std::function<void()> on_destroy)
      : guard_(nullptr,
               [on_destroy = std::move(on_destroy)](void*) { on_destroy(); }) {}

 private:
  // Shared guard whose deleter runs once all copies of the closure are
  // destroyed.
  std::shared_ptr<void> guard_;
};

// Verifies that destroying a RunHandler while intra-op or inter-op closures
// remain unpopped in its queue drains and executes those closures (including
// cross-queue closures enqueued during draining) instead of crashing on
// non-empty TaskQueueSize.
TEST(RunHandlerUtilTest, ReleaseHandlerDrainsUnpoppedClosures) {
  constexpr int64_t kGetTimeoutMs = 5000;
  constexpr absl::Duration kWaitTimeout = absl::Seconds(10);

  RunHandlerPool::Options options;
  options.num_inter_op_threads = 1;
  options.num_intra_op_threads = 1;
  options.max_concurrent_handler = 2;

  absl::Notification inter_blocker_started;
  absl::Notification intra_blocker_started;
  absl::Notification release_blockers;

  absl::Mutex mu;
  int completed_intra ABSL_GUARDED_BY(mu) = 0;
  int completed_nested_inter ABSL_GUARDED_BY(mu) = 0;
  int completed_inter ABSL_GUARDED_BY(mu) = 0;

  auto pool = std::make_unique<RunHandlerPool>(options);
  auto blocker_handler = pool->Get(/*step_id=*/1, kGetTimeoutMs);
  ASSERT_NE(blocker_handler, nullptr);
  auto unblock_workers = absl::MakeCleanup([&]() {
    if (!release_blockers.HasBeenNotified()) {
      release_blockers.Notify();
    }
  });

  // Deterministically occupy both worker threads in the pool.
  blocker_handler->ScheduleInterOpClosure(TaskFunction([&]() {
    inter_blocker_started.Notify();
    release_blockers.WaitForNotification();
  }));
  blocker_handler->ScheduleIntraOpClosure(TaskFunction([&]() {
    intra_blocker_started.Notify();
    release_blockers.WaitForNotification();
  }));
  ASSERT_TRUE(
      inter_blocker_started.WaitForNotificationWithTimeout(kWaitTimeout));
  ASSERT_TRUE(
      intra_blocker_started.WaitForNotificationWithTimeout(kWaitTimeout));

  // Schedule intra-op closures via AsIntraThreadPoolInterface() and have the
  // first unpopped intra-op closure enqueue a nested inter-op closure while
  // draining so QuiesceWorkSource must loop back to PopBlockingTask().
  auto intra_handler = pool->Get(/*step_id=*/2, kGetTimeoutMs);
  ASSERT_NE(intra_handler, nullptr);
  RunHandler* raw_intra_handler = intra_handler.get();
  constexpr int kNumIntraClosures = 4;
  for (int i = 0; i < kNumIntraClosures; ++i) {
    intra_handler->AsIntraThreadPoolInterface()->Schedule(
        [&, i, raw_intra_handler]() {
          if (i == 0) {
            raw_intra_handler->ScheduleInterOpClosure(TaskFunction([&]() {
              absl::MutexLock lock(mu);
              ++completed_nested_inter;
            }));
          }
          absl::MutexLock lock(mu);
          ++completed_intra;
        });
  }
  intra_handler.reset();
  {
    absl::MutexLock lock(mu);
    EXPECT_EQ(completed_intra, kNumIntraClosures);
    EXPECT_EQ(completed_nested_inter, 1);
  }

  // Reuse the second handler slot and verify unpopped inter-op closures are
  // also drained cleanly on destruction.
  auto inter_handler = pool->Get(/*step_id=*/3, kGetTimeoutMs);
  ASSERT_NE(inter_handler, nullptr);
  EXPECT_EQ(inter_handler->step_id(), 3);
  constexpr int kNumInterClosures = 2;
  for (int i = 0; i < kNumInterClosures; ++i) {
    inter_handler->ScheduleInterOpClosure(TaskFunction([&]() {
      absl::MutexLock lock(mu);
      ++completed_inter;
    }));
  }
  inter_handler.reset();
  {
    absl::MutexLock lock(mu);
    EXPECT_EQ(completed_inter, kNumInterClosures);
  }

  release_blockers.Notify();
  blocker_handler.reset();
  pool->Quiesce();
}

// Verifies that destroying a RunHandler waits for an already-popped, in-flight
// closure, any nested child closures it schedules, and closure capture
// destruction before recycling the handler back to free_handlers_.
TEST(RunHandlerUtilTest,
     ReleaseHandlerWaitsForInFlightClosureSchedulingNestedWork) {
  constexpr int64_t kGetTimeoutMs = 5000;
  constexpr absl::Duration kWaitTimeout = absl::Seconds(10);

  // Use a single blocking worker thread (which steals both inter-op and
  // intra-op tasks) so keeping `parent` alive pins the sole pool worker and
  // guarantees `child` can only run once `releaser` enters QuiesceWorkSource().
  RunHandlerPool::Options options;
  options.num_inter_op_threads = 1;
  options.num_intra_op_threads = 0;
  options.max_concurrent_handler = 1;

  absl::Notification parent_started;
  absl::Notification allow_parent_to_schedule_child;
  absl::Notification child_started;
  absl::Notification allow_child_to_finish;
  absl::Notification child_done;
  absl::Notification allow_parent_to_finish;
  absl::Notification reset_done;

  absl::Mutex mu;
  bool parent_finished ABSL_GUARDED_BY(mu) = false;
  bool parent_capture_destroyed ABSL_GUARDED_BY(mu) = false;
  bool child_finished ABSL_GUARDED_BY(mu) = false;
  bool child_capture_destroyed ABSL_GUARDED_BY(mu) = false;

  auto pool = std::make_unique<RunHandlerPool>(options);
  auto handler = pool->Get(/*step_id=*/1, kGetTimeoutMs);
  ASSERT_NE(handler, nullptr);
  auto* intra_pool = handler->AsIntraThreadPoolInterface();
  std::unique_ptr<tensorflow::Thread> releaser;
  auto unblock_all = absl::MakeCleanup([&]() {
    if (!allow_parent_to_schedule_child.HasBeenNotified()) {
      allow_parent_to_schedule_child.Notify();
    }
    if (!allow_child_to_finish.HasBeenNotified()) {
      allow_child_to_finish.Notify();
    }
    if (!allow_parent_to_finish.HasBeenNotified()) {
      allow_parent_to_finish.Notify();
    }
  });

  intra_pool->Schedule([&, parent_token = ClosureLifetimeToken([&]() {
                             absl::MutexLock lock(mu);
                             parent_capture_destroyed = true;
                           })]() {
    (void)parent_token;
    parent_started.Notify();
    allow_parent_to_schedule_child.WaitForNotification();
    intra_pool->Schedule([&, child_token = ClosureLifetimeToken([&]() {
                               absl::MutexLock lock(mu);
                               child_capture_destroyed = true;
                             })]() {
      (void)child_token;
      child_started.Notify();
      allow_child_to_finish.WaitForNotification();
      {
        absl::MutexLock lock(mu);
        child_finished = true;
      }
      child_done.Notify();
    });
    allow_parent_to_finish.WaitForNotification();
    absl::MutexLock lock(mu);
    parent_finished = true;
  });

  // Once parent_started is notified, the parent closure has been popped from
  // the queue (TaskQueueSize == 0) and pins the sole worker in the pool.
  ASSERT_TRUE(parent_started.WaitForNotificationWithTimeout(kWaitTimeout));

  releaser.reset(tensorflow::Env::Default()->StartThread(
      tensorflow::ThreadOptions(), "releaser", [&]() {
        handler.reset();
        {
          absl::MutexLock lock(mu);
          EXPECT_TRUE(parent_finished);
          EXPECT_TRUE(parent_capture_destroyed);
          EXPECT_TRUE(child_finished);
          EXPECT_TRUE(child_capture_destroyed);
        }
        reset_done.Notify();
      }));

  allow_parent_to_schedule_child.Notify();
  // Because the sole pool worker is blocked inside `parent`, `child_started`
  // proves `releaser` is actively inside handler.reset() ->
  // QuiesceWorkSource().
  ASSERT_TRUE(child_started.WaitForNotificationWithTimeout(kWaitTimeout));
  EXPECT_FALSE(reset_done.HasBeenNotified());

  // Finish `child` while `parent` remains in-flight on the pool worker so
  // `releaser` must stay blocked in WaitForWorkOrCompletion().
  allow_child_to_finish.Notify();
  ASSERT_TRUE(child_done.WaitForNotificationWithTimeout(kWaitTimeout));
  EXPECT_FALSE(reset_done.HasBeenNotified());

  allow_parent_to_finish.Notify();
  ASSERT_TRUE(reset_done.WaitForNotificationWithTimeout(kWaitTimeout));
  releaser.reset();

  // The single handler in the pool should now be cleanly reusable.
  auto reused_handler = pool->Get(/*step_id=*/2, kGetTimeoutMs);
  ASSERT_NE(reused_handler, nullptr);
  EXPECT_EQ(reused_handler->step_id(), 2);
}

// Models xla::cpu::Worker::Parallelize under contention and early parallel
// branch failure: recursive binary-split intra-op worker closures steal from a
// shared work-item pool, and the request destroys its RunHandler either as soon
// as all work items finish or mid-split before all work items are claimed.
TEST(RunHandlerUtilTest, ConcurrentWorkStealingEarlyRelease) {
  // Shared per-request state modeling xla::cpu::ParallelizeContext. Inherits
  // enable_shared_from_this so scheduled worker closures retain state after
  // `all_items_done` unblocks the caller (avoiding test-side UAF even if
  // draining regressed).
  struct WorkStealingState
      : public std::enable_shared_from_this<WorkStealingState> {
    WorkStealingState(tensorflow::thread::ThreadPoolInterface* pool,
                      int total_items)
        : intra_pool(pool), num_items(total_items) {}

    bool HasRemainingItems() {
      absl::MutexLock lock(mu);
      return next_item < num_items;
    }

    bool TryStealItem() {
      absl::MutexLock lock(mu);
      if (next_item >= num_items) {
        return false;
      }
      ++next_item;
      return true;
    }

    void CompleteItems(int count) {
      if (count == 0) {
        return;
      }
      absl::MutexLock lock(mu);
      completed_items += count;
      if (completed_items == num_items) {
        all_items_done.Notify();
      }
    }

    void Parallelize(int start_worker, int end_worker) {
      while (end_worker - start_worker > 1) {
        if (!HasRemainingItems()) {
          return;
        }
        int mid_worker = (start_worker + end_worker) / 2;
        auto self = shared_from_this();
        {
          absl::MutexLock lock(mu);
          ++scheduled_closures;
        }
        intra_pool->Schedule([self, mid_worker, end_worker,
                              token = ClosureLifetimeToken([self]() {
                                absl::MutexLock lock(self->mu);
                                ++self->destroyed_captures;
                              })]() {
          (void)token;
          self->Parallelize(mid_worker, end_worker);
          absl::MutexLock lock(self->mu);
          ++self->completed_closures;
        });
        end_worker = mid_worker;
      }
      int local_completed = 0;
      while (TryStealItem()) {
        ++local_completed;
      }
      CompleteItems(local_completed);
    }

    void ScheduleRootOnInterOp(RunHandler* handler, int num_workers) {
      auto self = shared_from_this();
      {
        absl::MutexLock lock(mu);
        ++scheduled_closures;
      }
      handler->ScheduleInterOpClosure(TaskFunction(
          [self, num_workers, token = ClosureLifetimeToken([self]() {
                                absl::MutexLock lock(self->mu);
                                ++self->destroyed_captures;
                              })]() {
            (void)token;
            self->Parallelize(/*start_worker=*/0, /*end_worker=*/num_workers);
            absl::MutexLock lock(self->mu);
            ++self->completed_closures;
          }));
    }

    // Intra-op thread pool interface owned by the request's RunHandler.
    tensorflow::thread::ThreadPoolInterface* const intra_pool;
    // Total work items in the parallel loop (distinct from worker count).
    const int num_items;
    // Guards work-item claiming and closure completion/destruction counters.
    absl::Mutex mu;
    // Index of the next unclaimed work item; consulted by HasRemainingItems()
    // before execution so binary splitting stops once all items are claimed.
    int next_item ABSL_GUARDED_BY(mu) = 0;
    // Count of stolen work items that have finished execution; triggers
    // `all_items_done` when it reaches `num_items`.
    int completed_items ABSL_GUARDED_BY(mu) = 0;
    // Total closures scheduled onto `handler` for this request.
    int scheduled_closures ABSL_GUARDED_BY(mu) = 0;
    // Total scheduled closures whose bodies finished running.
    int completed_closures ABSL_GUARDED_BY(mu) = 0;
    // Total scheduled closures whose TaskImpl captures were destroyed.
    int destroyed_captures ABSL_GUARDED_BY(mu) = 0;
    // Notified once all `num_items` stolen work items finish execution.
    absl::Notification all_items_done;
  };

  constexpr int64_t kGetTimeoutMs = 5000;
  constexpr absl::Duration kWaitTimeout = absl::Seconds(10);

  RunHandlerPool::Options options;
  options.num_inter_op_threads = 2;
  options.num_intra_op_threads = 2;
  options.max_concurrent_handler = 4;
  auto pool = std::make_unique<RunHandlerPool>(options);

  constexpr int kNumCallers = 6;
  constexpr int kIterationsPerCaller = 30;
  constexpr int kWorkItemsPerRequest = 8;
  constexpr int kWorkersPerRequest = 16;
  tensorflow::BlockingCounter callers_done(kNumCallers);
  tensorflow::thread::ThreadPool caller_pool(tensorflow::Env::Default(),
                                             "callers", kNumCallers);

  for (int c = 0; c < kNumCallers; ++c) {
    caller_pool.Schedule([&, c]() {
      for (int iter = 0; iter < kIterationsPerCaller; ++iter) {
        const int64_t expected_step_id = c * 1000 + iter;
        auto handler = pool->Get(expected_step_id, kGetTimeoutMs);
        EXPECT_NE(handler, nullptr);
        if (handler == nullptr) {
          break;
        }
        auto state = std::make_shared<WorkStealingState>(
            handler->AsIntraThreadPoolInterface(), kWorkItemsPerRequest);
        if (iter % 2 == 0) {
          // Even iterations: caller steals work items and releases the handler
          // as soon as all work items finish while excess worker closures may
          // remain queued or in-flight.
          state->Parallelize(/*start_worker=*/0,
                             /*end_worker=*/kWorkersPerRequest);
          const bool waited =
              state->all_items_done.WaitForNotificationWithTimeout(
                  kWaitTimeout);
          EXPECT_TRUE(waited);
          if (!waited) {
            break;
          }
        } else {
          // Odd iterations: release the handler immediately while binary
          // splitting is still in progress on another branch (modeling early
          // error unwinding on a parallel graph branch).
          state->ScheduleRootOnInterOp(handler.get(), kWorkersPerRequest);
        }
        EXPECT_EQ(handler->step_id(), expected_step_id);
        handler.reset();

        EXPECT_TRUE(state->all_items_done.HasBeenNotified());
        absl::MutexLock lock(state->mu);
        EXPECT_EQ(state->completed_items, kWorkItemsPerRequest);
        EXPECT_EQ(state->completed_closures, state->scheduled_closures);
        EXPECT_EQ(state->destroyed_captures, state->scheduled_closures);
      }
      callers_done.DecrementCount();
    });
  }
  callers_done.Wait();
  pool->Quiesce();
}

class RunHandlerThreadPoolTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  bool adaptive_waiting_time() { return std::get<0>(GetParam()); }

  bool wait_if_no_active_request() { return std::get<1>(GetParam()); }
};

TEST_P(RunHandlerThreadPoolTest, EnqueueTask) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      internal::RunHandlerThreadPool::Options(
          /*num_blocking_threads=*/0, /*num_non_blocking_threads=*/0,
          /*wait_if_no_active_request=*/true,
          /*non_blocking_threads_sleep_time_micro_sec=*/250,
          /*blocking_threads_max_sleep_time_micro_sec=*/250,
          /*use_adaptive_waiting_time=*/true, /*enable_wake_up=*/true,
          /*max_concurrent_handler=*/128,
          /*num_threads_in_sub_thread_pool=*/{0},
          /*sub_thread_request_percentage=*/{1}),
      tensorflow::Env::Default(), tensorflow::ThreadOptions(),
      "tf_run_handler_pool", &waiters_mu, &waiters);
  internal::ThreadWorkSource tws;
  tws.SetWaiter(1, &waiters[0], &waiters_mu[0]);

  int result = 0;
  std::function<void()> fn = [&result] { result = 1; };
  std::function<void()> fn2 = [&result] { result = 2; };
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/true,
                                         TaskFunction(fn));
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/true), 1);
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/true,
                                         TaskFunction(fn2));
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/true), 2);
  tws.PopBlockingTask().f->f();
  EXPECT_EQ(result, 1);
  tws.PopBlockingTask().f->f();
  EXPECT_EQ(result, 2);

  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/false,
                                         TaskFunction(fn));
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/false), 1);
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/false,
                                         TaskFunction(fn2));
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/false), 2);
  tws.PopNonBlockingTask(0, true).f->f();
  EXPECT_EQ(result, 1);
  tws.PopNonBlockingTask(0, true).f->f();
  EXPECT_EQ(result, 2);
  EXPECT_TRUE(absl::StrContains(tws.ToString(), "traceme_id = 0"));
}

template <typename Work, unsigned kSize>
constexpr int RunQueueCapacity(const Eigen::RunQueue<Work, kSize>*) {
  return static_cast<int>(kSize);
}

class ThreadWorkSourceOverflowTest : public testing::TestWithParam<bool> {};

// Verifies that overflowing a full Eigen::RunQueue executes the overflowed
// task inline via ExecuteTask() with proper inflight/pending accounting (and
// no transient zero pending count) for both blocking and non-blocking queues.
TEST_P(ThreadWorkSourceOverflowTest,
       OverflowTaskExecutesInlineWithoutLeakingPendingTaskCount) {
  const bool is_blocking = GetParam();
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(1);
  waiters_mu.resize(1);
  Eigen::MaxSizeVector<internal::Waiter> waiters(1);
  waiters.resize(1);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      internal::RunHandlerThreadPool::Options(
          /*num_blocking_threads=*/0, /*num_non_blocking_threads=*/0,
          /*wait_if_no_active_request=*/true,
          /*non_blocking_threads_sleep_time_micro_sec=*/250,
          /*blocking_threads_max_sleep_time_micro_sec=*/250,
          /*use_adaptive_waiting_time=*/true,
          /*enable_wake_up=*/true,
          /*max_concurrent_handler=*/128,
          /*num_threads_in_sub_thread_pool=*/{0},
          /*sub_thread_request_percentage=*/{1}),
      tensorflow::Env::Default(), tensorflow::ThreadOptions(),
      "tf_run_handler_pool", &waiters_mu, &waiters);
  internal::ThreadWorkSource tws;
  tws.SetWaiter(1, &waiters[0], &waiters_mu[0]);

  constexpr int kSingleQueueCapacity =
      RunQueueCapacity(static_cast<const internal::Queue*>(nullptr));
  const int queue_capacity =
      is_blocking ? kSingleQueueCapacity
                  : kSingleQueueCapacity *
                        static_cast<int>(tws.NonBlockingWorkShardingFactor());
  const int total_tasks = queue_capacity + 1;

  absl::Mutex mu;
  int completed ABSL_GUARDED_BY(mu) = 0;
  int destroyed_captures ABSL_GUARDED_BY(mu) = 0;
  int64_t overflow_pending_during_exec ABSL_GUARDED_BY(mu) = -1;
  int64_t overflow_inflight_during_exec ABSL_GUARDED_BY(mu) = -1;
  int64_t overflow_other_inflight_during_exec ABSL_GUARDED_BY(mu) = -1;

  for (int i = 0; i < total_tasks; ++i) {
    run_handler_thread_pool.AddWorkToQueue(
        &tws, is_blocking,
        TaskFunction([&, i, token = ClosureLifetimeToken([&]() {
                              absl::MutexLock lock(mu);
                              ++destroyed_captures;
                            })]() {
          (void)token;
          const int64_t pending_now = tws.GetPendingTaskCount();
          const int64_t inflight_now = tws.GetInflightTaskCount(is_blocking);
          const int64_t other_inflight_now =
              tws.GetInflightTaskCount(!is_blocking);
          absl::MutexLock lock(mu);
          if (i == queue_capacity) {
            overflow_pending_during_exec = pending_now;
            overflow_inflight_during_exec = inflight_now;
            overflow_other_inflight_during_exec = other_inflight_now;
          }
          ++completed;
        }));
  }

  {
    absl::MutexLock lock(mu);
    EXPECT_EQ(completed, 1);
    EXPECT_EQ(destroyed_captures, 1);
    EXPECT_EQ(overflow_pending_during_exec, queue_capacity + 1);
    EXPECT_EQ(overflow_inflight_during_exec, 1);
    EXPECT_EQ(overflow_other_inflight_during_exec, 0);
  }
  EXPECT_EQ(tws.TaskQueueSize(is_blocking), queue_capacity);
  EXPECT_EQ(tws.GetPendingTaskCount(), queue_capacity);
  EXPECT_EQ(tws.GetInflightTaskCount(is_blocking), 0);
  EXPECT_EQ(tws.GetEnqueueCount(), static_cast<uint64_t>(queue_capacity));

  run_handler_thread_pool.QuiesceWorkSource(&tws);

  {
    absl::MutexLock lock(mu);
    EXPECT_EQ(completed, total_tasks);
    EXPECT_EQ(destroyed_captures, total_tasks);
  }
  EXPECT_EQ(tws.TaskQueueSize(is_blocking), 0);
  EXPECT_EQ(tws.GetPendingTaskCount(), 0);
  EXPECT_EQ(tws.GetInflightTaskCount(is_blocking), 0);
}

// Verifies with 0 pool workers that QuiesceWorkSource() detects enqueues via
// both the pre-park enqueue_count_ check and the parked pending_cv_.SignalAll()
// wakeup in ThreadWorkSource::EnqueueTask().
TEST(RunHandlerUtilTest, QuiesceWorkSourceWakesOnConcurrentEnqueue) {
  constexpr absl::Duration kWaitTimeout = absl::Seconds(10);

  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(1);
  waiters_mu.resize(1);
  Eigen::MaxSizeVector<internal::Waiter> waiters(1);
  waiters.resize(1);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      internal::RunHandlerThreadPool::Options(
          /*num_blocking_threads=*/0, /*num_non_blocking_threads=*/0,
          /*wait_if_no_active_request=*/true,
          /*non_blocking_threads_sleep_time_micro_sec=*/250,
          /*blocking_threads_max_sleep_time_micro_sec=*/250,
          /*use_adaptive_waiting_time=*/true,
          /*enable_wake_up=*/true,
          /*max_concurrent_handler=*/128,
          /*num_threads_in_sub_thread_pool=*/{0},
          /*sub_thread_request_percentage=*/{1}),
      tensorflow::Env::Default(), tensorflow::ThreadOptions(),
      "tf_run_handler_pool", &waiters_mu, &waiters);
  internal::ThreadWorkSource tws;
  tws.SetWaiter(1, &waiters[0], &waiters_mu[0]);

  absl::Notification first_nested_done;
  absl::Notification second_nested_done;
  absl::Notification quiesce_done;

  // Simulate an already-popped in-flight parent task holding pending_tasks_ > 0
  // while the queues are empty.
  const uint64_t initial_enqueue_count = tws.GetEnqueueCount();
  tws.IncrementPendingTaskCount();
  bool simulated_parent_active = true;

  // Verify the pre-park race guard: when enqueue_count_ advances after
  // `initial_enqueue_count` was sampled, WaitForWorkOrCompletion returns false
  // immediately without parking.
  run_handler_thread_pool.AddWorkToQueue(
      &tws, /*is_blocking=*/false,
      TaskFunction([&]() { first_nested_done.Notify(); }));
  EXPECT_EQ(tws.GetEnqueueCount(), initial_enqueue_count + 1);
  EXPECT_FALSE(tws.WaitForWorkOrCompletion(initial_enqueue_count));

  std::unique_ptr<tensorflow::Thread> drainer;
  auto release_simulated_parent = absl::MakeCleanup([&]() {
    if (simulated_parent_active) {
      tws.DecrementPendingTaskCount();
      simulated_parent_active = false;
    }
  });

  drainer.reset(tensorflow::Env::Default()->StartThread(
      tensorflow::ThreadOptions(), "drainer", [&]() {
        run_handler_thread_pool.QuiesceWorkSource(&tws);
        quiesce_done.Notify();
      }));

  // `drainer` drains `first_nested` and then parks in WaitForWorkOrCompletion()
  // because `simulated_parent_active` still holds pending_tasks_ == 1.
  ASSERT_TRUE(first_nested_done.WaitForNotificationWithTimeout(kWaitTimeout));
  const absl::Time park_deadline = absl::Now() + kWaitTimeout;
  while (tws.GetDrainingWaiterCount() < 1 && absl::Now() < park_deadline) {
    tensorflow::Env::Default()->SleepForMicroseconds(1000);
  }
  ASSERT_EQ(tws.GetDrainingWaiterCount(), 1);
  EXPECT_FALSE(quiesce_done.HasBeenNotified());

  // With 0 pool worker threads and `drainer` parked on pending_cv_,
  // `second_nested` can only execute if EnqueueTask() signals pending_cv_.
  run_handler_thread_pool.AddWorkToQueue(
      &tws, /*is_blocking=*/false,
      TaskFunction([&]() { second_nested_done.Notify(); }));
  ASSERT_TRUE(second_nested_done.WaitForNotificationWithTimeout(kWaitTimeout));

  const absl::Time repark_deadline = absl::Now() + kWaitTimeout;
  while (tws.GetDrainingWaiterCount() < 1 && absl::Now() < repark_deadline) {
    tensorflow::Env::Default()->SleepForMicroseconds(1000);
  }
  ASSERT_EQ(tws.GetDrainingWaiterCount(), 1);
  EXPECT_FALSE(quiesce_done.HasBeenNotified());

  // Complete the simulated in-flight parent and verify QuiesceWorkSource exits.
  tws.DecrementPendingTaskCount();
  simulated_parent_active = false;
  ASSERT_TRUE(quiesce_done.WaitForNotificationWithTimeout(kWaitTimeout));
  drainer.reset();
  EXPECT_EQ(tws.GetPendingTaskCount(), 0);
}

// Verifies that ScopedActiveWorkSource allows nested draining of a *different*
// ThreadWorkSource while failing fast with a CHECK on re-entrant draining of
// the *same* ThreadWorkSource.
TEST(RunHandlerDeathTest, ReentrantQuiesceWorkSourceCheckFails) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(1);
  waiters_mu.resize(1);
  Eigen::MaxSizeVector<internal::Waiter> waiters(1);
  waiters.resize(1);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      internal::RunHandlerThreadPool::Options(
          /*num_blocking_threads=*/0, /*num_non_blocking_threads=*/0,
          /*wait_if_no_active_request=*/true,
          /*non_blocking_threads_sleep_time_micro_sec=*/250,
          /*blocking_threads_max_sleep_time_micro_sec=*/250,
          /*use_adaptive_waiting_time=*/true,
          /*enable_wake_up=*/true,
          /*max_concurrent_handler=*/128,
          /*num_threads_in_sub_thread_pool=*/{0},
          /*sub_thread_request_percentage=*/{1}),
      tensorflow::Env::Default(), tensorflow::ThreadOptions(),
      "tf_run_handler_pool", &waiters_mu, &waiters);
  internal::ThreadWorkSource outer_tws;
  outer_tws.SetWaiter(1, &waiters[0], &waiters_mu[0]);
  outer_tws.SetTracemeId(42);
  internal::ThreadWorkSource inner_tws;
  inner_tws.SetWaiter(1, &waiters[0], &waiters_mu[0]);
  inner_tws.SetTracemeId(99);

  bool inner_drained = false;
  run_handler_thread_pool.AddWorkToQueue(
      &inner_tws, /*is_blocking=*/true,
      TaskFunction([&]() { inner_drained = true; }));
  run_handler_thread_pool.AddWorkToQueue(
      &outer_tws, /*is_blocking=*/true, TaskFunction([&]() {
        run_handler_thread_pool.QuiesceWorkSource(&inner_tws);
      }));
  run_handler_thread_pool.QuiesceWorkSource(&outer_tws);
  EXPECT_TRUE(inner_drained);
  EXPECT_EQ(inner_tws.GetPendingTaskCount(), 0);
  EXPECT_EQ(outer_tws.GetPendingTaskCount(), 0);

  EXPECT_DEATH(
      {
        run_handler_thread_pool.AddWorkToQueue(
            &outer_tws, /*is_blocking=*/true, TaskFunction([&]() {
              run_handler_thread_pool.QuiesceWorkSource(&outer_tws);
            }));
        run_handler_thread_pool.QuiesceWorkSource(&outer_tws);
      },
      "Cannot quiesce ThreadWorkSource \\(traceme_id=42\\)");
}

TEST_P(RunHandlerThreadPoolTest, FindTask) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      internal::RunHandlerThreadPool::Options(
          /*num_blocking_threads=*/1, /*num_non_blocking_threads=*/0,
          /*wait_if_no_active_request=*/true,
          /*non_blocking_threads_sleep_time_micro_sec=*/250,
          /*blocking_threads_max_sleep_time_micro_sec=*/250,
          /*use_adaptive_waiting_time=*/true, /*enable_wake_up=*/true,
          /*max_concurrent_handler=*/128,
          /*num_threads_in_sub_thread_pool=*/{1},
          /*sub_thread_request_percentage=*/{1}),
      tensorflow::Env::Default(), tensorflow::ThreadOptions(),
      "tf_run_handler_pool", &waiters_mu, &waiters);

  EXPECT_EQ(run_handler_thread_pool.NumBlockingThreads(), 1);
  EXPECT_EQ(run_handler_thread_pool.NumNonBlockingThreads(), 0);

  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(5);
  thread_work_sources.resize(5);
  for (int i = 0; i < 5; ++i) {
    thread_work_sources[i] = new internal::ThreadWorkSource();
    thread_work_sources[i]->SetWaiter(1, &waiters[0], &waiters_mu[0]);
  }

  {
    // The thread should search the task following round robin fashion.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));

    const auto find_blocking_task_from_all_handlers =
        [&](bool* task_from_blocking_queue, internal::Task* t) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              /*searching_range_start=*/0, /*searching_range_end=*/5,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };
    bool task_from_blocking_queue;
    internal::Task t;
    find_blocking_task_from_all_handlers(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    find_blocking_task_from_all_handlers(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 3);

    find_blocking_task_from_all_handlers(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    find_blocking_task_from_all_handlers(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 3);
  }

  {
    // Task out of searching range cannot be found.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));

    const auto find_blocking_task_from_range =
        [&](bool* task_from_blocking_queue, internal::Task* t, int range_start,
            int range_end) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              range_start, range_end,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };

    bool task_from_blocking_queue;
    internal::Task t;
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 3);
    EXPECT_EQ(t.f, nullptr);

    // Clean up the queue.
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 5);
  }

  {
    // The thread should search from start range if the current index is
    // smaller.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));

    const auto find_blocking_task_from_range =
        [&](bool* task_from_blocking_queue, internal::Task* t, int range_start,
            int range_end) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              range_start, range_end,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };
    bool task_from_blocking_queue;
    internal::Task t;
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 3, 5);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 3);

    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 5);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);
  }

  {
    // The thread should search within the range even if the current index
    // is larger than searching_range_end;
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));

    const auto find_blocking_task_from_range =
        [&](bool* task_from_blocking_queue, internal::Task* t, int range_start,
            int range_end) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              range_start, range_end,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };
    bool task_from_blocking_queue;
    // Make the current index to be 3.
    internal::Task t;
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 5);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    // Search in a smaller range.
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 3);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    // Clean up the queue.
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 5);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 3);
  }

  {
    // We prefer blocking task for blocking threads.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/false, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    const auto blocking_thread_find_task_from_all_handler =
        [&](bool* task_from_blocking_queue, internal::Task* t) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              /*searching_range_start=*/0, /*searching_range_end=*/5,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };
    bool task_from_blocking_queue;
    internal::Task t;
    blocking_thread_find_task_from_all_handler(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    blocking_thread_find_task_from_all_handler(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, false);
    t.f->f();
    EXPECT_EQ(result, 2);
  }

  {
    // Nonblocking threads can only pick up non-blocking task.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/false, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));

    const auto find_task_from_all_handler = [&](bool* task_from_blocking_queue,
                                                internal::Task* t,
                                                bool is_blocking_thread) {
      internal::ThreadWorkSource* tws;
      *t = run_handler_thread_pool.FindTask(
          /*searching_range_start=*/0, /*searching_range_end=*/5,
          /*thread_id=*/0,
          /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
          is_blocking_thread, thread_work_sources, task_from_blocking_queue,
          &tws);
    };
    bool task_from_blocking_queue;
    internal::Task t;
    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/false);
    EXPECT_EQ(task_from_blocking_queue, false);
    t.f->f();
    EXPECT_EQ(result, 2);

    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/false);
    EXPECT_EQ(t.f, nullptr);

    // Clean up the queue.
    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/true);
  }

  {
    // There is a limit for max_blocking_inflight requests.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));

    const auto find_task_from_all_handler = [&](bool* task_from_blocking_queue,
                                                internal::Task* t,
                                                bool is_blocking_thread) {
      internal::ThreadWorkSource* tws;
      *t = run_handler_thread_pool.FindTask(
          /*searching_range_start=*/0, /*searching_range_end=*/5,
          /*thread_id=*/0,
          /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
          is_blocking_thread, thread_work_sources, task_from_blocking_queue,
          &tws);
    };

    bool task_from_blocking_queue;
    internal::Task t;
    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/false);
    EXPECT_EQ(task_from_blocking_queue, false);
    EXPECT_EQ(t.f, nullptr);

    // Clean up the queue.
    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/true);
  }

  for (int i = 0; i < 5; ++i) {
    delete thread_work_sources[i];
  }
}

TEST_P(RunHandlerThreadPoolTest, RoundRobinExecution) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(1);
  waiters_mu.resize(1);
  Eigen::MaxSizeVector<internal::Waiter> waiters(1);
  waiters.resize(1);
  internal::RunHandlerThreadPool* run_handler_thread_pool =
      new internal::RunHandlerThreadPool(
          internal::RunHandlerThreadPool::Options(
              /*num_blocking_threads=*/1, /*num_non_blocking_threads=*/0,
              /*wait_if_no_active_request=*/true,
              /*non_blocking_threads_sleep_time_micro_sec=*/250,
              /*blocking_threads_max_sleep_time_micro_sec=*/250,
              /*use_adaptive_waiting_time=*/true, /*enable_wake_up=*/true,
              /*max_concurrent_handler=*/128,
              /*num_threads_in_sub_thread_pool=*/{1},
              /*sub_thread_request_percentage=*/{1}),
          tensorflow::Env::Default(), tensorflow::ThreadOptions(),
          "tf_run_handler_pool", &waiters_mu, &waiters);
  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(3);
  thread_work_sources.resize(3);
  internal::ThreadWorkSource tws[3];
  for (int i = 0; i < 3; ++i) {
    tws[i].SetWaiter(1, &waiters[0], &waiters_mu[0]);
    thread_work_sources[i] = &tws[i];
  }

  int result = 0;
  tensorflow::mutex mu;
  bool ok_to_execute = false;
  bool ok_to_validate = false;
  tensorflow::condition_variable function_start;
  tensorflow::condition_variable function_end;
  std::vector<std::function<void()>> fns;
  for (int i = 0; i < 3; ++i) {
    fns.push_back([&result, &mu, &function_start, &function_end, &ok_to_execute,
                   &ok_to_validate, i] {
      tensorflow::mutex_lock l(mu);
      while (!ok_to_execute) {
        function_start.wait(l);
      }
      result = i;
      ok_to_execute = false;
      ok_to_validate = true;
      function_end.notify_one();
    });
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            TaskFunction(fns[i]));
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            TaskFunction(fns[i]));
  }
  run_handler_thread_pool->Start();
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/0, /*version=*/1, thread_work_sources);

  // Validate the execution should be roundrobin.
  tensorflow::mutex_lock l(mu);
  for (int round = 0; round < 2; ++round) {
    for (int i = 0; i < 3; ++i) {
      ok_to_execute = true;
      function_start.notify_one();
      while (!ok_to_validate) {
        function_end.wait(l);
      }
      ok_to_validate = false;
      EXPECT_EQ(result, i);
    }
  }

  delete run_handler_thread_pool;
}

TEST_P(RunHandlerThreadPoolTest, MultipleSubThreadPool) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool* run_handler_thread_pool =
      new internal::RunHandlerThreadPool(
          internal::RunHandlerThreadPool::Options(
              /*num_blocking_threads=*/2, /*num_non_blocking_threads=*/0,
              /*wait_if_no_active_request=*/true,
              /*non_blocking_threads_sleep_time_micro_sec=*/250,
              /*blocking_threads_max_sleep_time_micro_sec=*/250,
              /*use_adaptive_waiting_time=*/true, /*enable_wake_up=*/true,
              /*max_concurrent_handler=*/128,
              /*num_threads_in_sub_thread_pool=*/{1, 1},
              /*sub_thread_request_percentage=*/{0.5, 1}),
          tensorflow::Env::Default(), tensorflow::ThreadOptions(),
          "tf_run_handler_pool", &waiters_mu, &waiters);
  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(4);
  thread_work_sources.resize(4);
  internal::ThreadWorkSource tws[4];
  for (int i = 0; i < 4; ++i) {
    tws[i].SetWaiter(1, &waiters[i / 2], &waiters_mu[i / 2]);
    thread_work_sources[i] = &tws[i];
  }

  int result = 0;
  tensorflow::mutex mu;
  bool ok_to_execute = false;
  bool ok_to_validate = false;
  tensorflow::condition_variable function_start;
  tensorflow::condition_variable function_end;

  std::vector<std::function<void()>> fns;
  for (int i = 0; i < 4; ++i) {
    fns.push_back([&result, &mu, &function_start, &function_end, &ok_to_execute,
                   &ok_to_validate, i] {
      tensorflow::mutex_lock l(mu);
      while (!ok_to_execute) {
        function_start.wait(l);
      }
      result = i;
      ok_to_execute = false;
      ok_to_validate = true;
      function_end.notify_one();
    });
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            TaskFunction(fns[i]));
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            TaskFunction(fns[i]));
  }
  run_handler_thread_pool->StartOneThreadForTesting();
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/0, /*version=*/1, thread_work_sources);
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/1, /*version=*/1, thread_work_sources);

  // Pick task from the given sub thread pool requests in a round robin fashion.
  tensorflow::mutex_lock l(mu);
  for (int round = 0; round < 2; ++round) {
    for (int i = 0; i < 2; ++i) {
      ok_to_execute = true;
      function_start.notify_one();
      while (!ok_to_validate) {
        function_end.wait(l);
      }
      ok_to_validate = false;
      EXPECT_EQ(result, i);
    }
  }

  // Pick task from any task if there is no tasks from the requests in the sub
  // thread pool.
  for (int i = 0; i < 2; ++i) {
    for (int round = 0; round < 2; ++round) {
      ok_to_execute = true;
      function_start.notify_one();
      while (!ok_to_validate) {
        function_end.wait(l);
      }
      ok_to_validate = false;
      EXPECT_EQ(result, i + 2);
    }
  }

  delete run_handler_thread_pool;
}

INSTANTIATE_TEST_SUITE_P(BlockingAndNonBlocking, ThreadWorkSourceOverflowTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "BlockingInterOp"
                                             : "NonBlockingIntraOp";
                         });

INSTANTIATE_TEST_SUITE_P(Parameter, RunHandlerThreadPoolTest,
                         testing::Combine(::testing::Bool(),
                                          ::testing::Bool()));

}  // namespace
}  // namespace tf
}  // namespace tfrt
