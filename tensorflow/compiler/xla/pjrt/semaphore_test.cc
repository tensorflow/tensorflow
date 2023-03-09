/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/semaphore.h"

#include "absl/synchronization/notification.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/threadpool.h"

namespace xla {
namespace {

TEST(SemaphoreTest, UnthreadedTests) {
  Semaphore semaphore(2);
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
    { auto b = semaphore.ScopedAcquire(1); }
    { auto c = semaphore.ScopedAcquire(1); }
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
