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

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "xla/test.h"
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

}  // namespace
}  // namespace xla
