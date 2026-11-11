/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/semaphore.h"

#include <atomic>
#include <cstdint>
#include <utility>

#include "absl/synchronization/notification.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

TEST(SemaphoreTest, UnthreadedTests) {
  Semaphore semaphore(2);
  EXPECT_EQ(semaphore.capacity(), 2);
  EXPECT_FALSE(semaphore.TryAcquire(semaphore.capacity() + 1));
  EXPECT_TRUE(semaphore.TryAcquire(semaphore.capacity()));
  semaphore.Release(semaphore.capacity());
  semaphore.Acquire(1);
  semaphore.Release(1);

  semaphore.Acquire(2);
  semaphore.Release(2);

  semaphore.Acquire(1);
  semaphore.Acquire(1);
  semaphore.Release(1);
  semaphore.Acquire(1);
  semaphore.Release(1);
  semaphore.Acquire(1);
  semaphore.Release(2);

  {
    auto a = semaphore.ScopedAcquire(1);
    EXPECT_EQ(a.amount(), 1);
    { auto b = semaphore.ScopedAcquire(1); }
    { auto c = semaphore.ScopedAcquire(1); }
  }
  {
    auto d = semaphore.ScopedAcquire(2);
    EXPECT_EQ(d.amount(), 2);
  }
}

TEST(SemaphoreTest, ConcurrentTest) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 2);
  Semaphore semaphore(2);
  semaphore.Acquire(1);

  absl::Notification a_done;
  pool.Schedule([&]() {
    semaphore.Acquire(2);
    semaphore.Release(2);
    a_done.Notify();
  });

  absl::Notification b_done;
  pool.Schedule([&]() {
    semaphore.Acquire(1);
    semaphore.Release(1);
    b_done.Notify();
  });
  b_done.WaitForNotification();
  EXPECT_FALSE(a_done.HasBeenNotified());
  semaphore.Release(1);
  a_done.WaitForNotification();
}

TEST(SemaphoreTest, ScopedReservationMoveAssignment) {
  Semaphore sem(10);
  EXPECT_EQ(sem.value(), 10);
  {
    auto r1 = sem.ScopedAcquire(5);
    EXPECT_EQ(sem.value(), 5);
    {
      auto r2 = sem.ScopedAcquire(3);
      EXPECT_EQ(sem.value(), 2);

      // Moving r2 into r1 should release r1's previous reservation (5 units).
      r1 = std::move(r2);
      // Immediately after move assignment, r1 has 3 units, so value should be
      // 2 + 5 = 7.
      EXPECT_EQ(sem.value(), 7);
      EXPECT_EQ(r1.amount(), 3);
    }
    // r2 was destroyed above. If r2 erroneously retained its reservation,
    // destroying it would have released tokens and increased sem.value().
    // Because r2 was properly disarmed by the move, sem.value() remains 7.
    EXPECT_EQ(sem.value(), 7);
  }
  // After r1 goes out of scope, sem should be back to full capacity (10).
  EXPECT_EQ(sem.value(), 10);

  // Test self move-assignment
  {
    auto r3 = sem.ScopedAcquire(4);
    EXPECT_EQ(sem.value(), 6);
    auto& r3_ref = r3;
    r3 = std::move(r3_ref);
    EXPECT_EQ(sem.value(), 6);
    EXPECT_EQ(r3.amount(), 4);
  }
  EXPECT_EQ(sem.value(), 10);
}

TEST(SemaphoreTest, TryAcquireDoesNotBargeAheadOfWaiter) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 1);
  Semaphore semaphore(2);
  semaphore.Acquire(2);

  absl::Notification waiter_acquired;
  absl::Notification allow_waiter_release;
  absl::Notification waiter_done;

  pool.Schedule([&]() {
    semaphore.Acquire(2);
    waiter_acquired.Notify();
    allow_waiter_release.WaitForNotification();
    semaphore.Release(2);
    waiter_done.Notify();
  });

  // Deterministically wait until the waiter is queued at head_.
  while (semaphore.TryAcquire(0)) {
  }

  // Call semaphore.Release(1) so semaphore.value() == 1 while the Acquire(2)
  // waiter remains queued at head_.
  semaphore.Release(1);
  EXPECT_EQ(semaphore.value(), 1);

  // Verify TryAcquire must not barge ahead of a queued waiter.
  EXPECT_FALSE(semaphore.TryAcquire(1));

  // Verify semaphore.Acquire(1) and semaphore.Release(1) succeed (a smaller
  // Acquire can bypass an unsatisfiable larger waiter).
  semaphore.Acquire(1);
  semaphore.Release(1);

  // Call semaphore.Release(1) to hand off all 2 units directly to the queued
  // Acquire(2) waiter.
  semaphore.Release(1);

  waiter_acquired.WaitForNotification();
  // Before allowing the waiter to release, verify value is 0 and TryAcquire(1)
  // returns false.
  EXPECT_EQ(semaphore.value(), 0);
  EXPECT_FALSE(semaphore.TryAcquire(1));

  // Notify the waiter to call semaphore.Release(2) and finish.
  allow_waiter_release.Notify();
  waiter_done.WaitForNotification();
  EXPECT_EQ(semaphore.value(), 2);
}

TEST(SemaphoreTest, ConcurrentAcquireTryAcquireReleaseStress) {
  constexpr int kNumThreads = 8;
  constexpr int kItersPerThread = 200;
  Semaphore semaphore(2);
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "stress", kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      pool.Schedule([&, i]() {
        const int64_t amount = (i % 4 == 0) ? 2 : 1;
        for (int iter = 0; iter < kItersPerThread; ++iter) {
          if (!semaphore.TryAcquire(amount)) {
            semaphore.Acquire(amount);
          }
          semaphore.Release(amount);
        }
      });
    }
  }
  EXPECT_EQ(semaphore.value(), 2);
}

void BM_ContendedAcquireTryAcquire(benchmark::State& state) {
  static std::atomic<uint64_t> max_wait_ns{0};
  if (state.thread_index() == 0) {
    max_wait_ns.store(0, std::memory_order_relaxed);
  }
  static Semaphore* semaphore = new Semaphore(4);
  uint64_t local_max_wait_ns = 0;
  for (auto _ : state) {
    const uint64_t start_ns = tsl::Env::Default()->NowNanos();
    if (!semaphore->TryAcquire(1)) {
      semaphore->Acquire(1);
    }
    const uint64_t wait_ns = tsl::Env::Default()->NowNanos() - start_ns;
    if (wait_ns > local_max_wait_ns) {
      local_max_wait_ns = wait_ns;
      uint64_t cur = max_wait_ns.load(std::memory_order_relaxed);
      while (wait_ns > cur && !max_wait_ns.compare_exchange_weak(
                                  cur, wait_ns, std::memory_order_relaxed)) {
      }
    }
    semaphore->Release(1);
  }
  state.counters["max_wait_us"] = benchmark::Counter(
      static_cast<double>(max_wait_ns.load(std::memory_order_relaxed)) / 1e3,
      benchmark::Counter::kAvgThreads);
}
BENCHMARK(BM_ContendedAcquireTryAcquire)->ThreadRange(1, 32);

}  // namespace
}  // namespace xla
