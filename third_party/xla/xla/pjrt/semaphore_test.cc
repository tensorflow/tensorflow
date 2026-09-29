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

#include <utility>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "xla/hlo/testlib/test.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"

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

}  // namespace
}  // namespace xla
