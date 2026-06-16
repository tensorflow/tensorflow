/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/worker_thread.h"

#include <atomic>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace {

// ---------------------------------------------------------------------------
// WorkerThread::Drain() tests
//
// Drain() works by scheduling a sentinel closure after all currently-queued
// work and blocking until the sentinel executes.  This guarantees that every
// closure enqueued *before* the Drain() call has completed when Drain()
// returns.

// Calling Drain() on an idle (empty-queue) thread must return immediately
// without deadlocking.
TEST(WorkerThreadTest, DrainOnEmptyQueue) {
  WorkerThread thread(tsl::Env::Default(), "test");
  thread.Drain();  // Must not block or deadlock.
}

// All closures scheduled before Drain() must have completed when Drain()
// returns.
TEST(WorkerThreadTest, DrainWaitsForAllScheduledClosures) {
  WorkerThread thread(tsl::Env::Default(), "test");

  constexpr int kNumClosures = 200;
  std::atomic<int> counter{0};

  for (int i = 0; i < kNumClosures; ++i) {
    thread.Schedule(
        [&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
  }

  thread.Drain();

  // Every closure enqueued before Drain() must have run.
  EXPECT_EQ(counter.load(std::memory_order_relaxed), kNumClosures);
}

// Closures execute in FIFO order, and all must complete before Drain() returns.
TEST(WorkerThreadTest, DrainPreservesExecutionOrder) {
  WorkerThread thread(tsl::Env::Default(), "test");

  constexpr int kNumClosures = 10;
  std::vector<int> order;
  absl::Mutex mu;

  for (int i = 0; i < kNumClosures; ++i) {
    thread.Schedule([i, &order, &mu]() {
      absl::MutexLock lock(mu);
      order.push_back(i);
    });
  }

  thread.Drain();

  absl::MutexLock lock(mu);
  ASSERT_EQ(static_cast<int>(order.size()), kNumClosures);
  for (int i = 0; i < kNumClosures; ++i) {
    EXPECT_EQ(order[i], i) << "closure " << i << " executed out of order";
  }
}

// Multiple sequential Drain() calls each wait only for the work enqueued
// before that call.
TEST(WorkerThreadTest, MultipleDrainsAreIndependent) {
  WorkerThread thread(tsl::Env::Default(), "test");

  std::atomic<int> counter{0};

  for (int round = 0; round < 5; ++round) {
    thread.Schedule(
        [&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    thread.Drain();
    // The closure for this round must have completed.
    EXPECT_EQ(counter.load(std::memory_order_relaxed), round + 1);
  }
}

// Drain() on an already-drained thread (second consecutive call with no
// intervening Schedule()) returns immediately.
TEST(WorkerThreadTest, ConsecutiveDrainsOnIdleThread) {
  WorkerThread thread(tsl::Env::Default(), "test");

  thread.Drain();  // First drain on empty queue — must not deadlock.
  thread.Drain();  // Second drain on still-empty queue — must not deadlock.
}
}  // namespace
}  // namespace xla
