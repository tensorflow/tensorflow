/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/test.h"

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

TEST(NotificationTest, TestSingleNotification) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", 1);

  int counter = 0;
  Notification start;
  Notification proceed;
  thread_pool->Schedule([&start, &proceed, &counter] {
    start.Notify();
    proceed.WaitForNotification();
    ++counter;
  });

  // Wait for the thread to start
  start.WaitForNotification();

  // The thread should be waiting for the 'proceed' notification.
  EXPECT_EQ(0, counter);

  // Unblock the thread
  proceed.Notify();

  delete thread_pool;  // Wait for closure to finish.

  // Verify the counter has been incremented
  EXPECT_EQ(1, counter);
}

TEST(NotificationTest, TestMultipleThreadsWaitingOnNotification) {
  const int num_closures = 4;
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", num_closures);

  mutex lock;
  int counter = 0;
  Notification n;

  for (int i = 0; i < num_closures; ++i) {
    thread_pool->Schedule([&n, &lock, &counter] {
      n.WaitForNotification();
      mutex_lock l(lock);
      ++counter;
    });
  }

  // Sleep 1 second.
  Env::Default()->SleepForMicroseconds(1 * 1000 * 1000);

  EXPECT_EQ(0, counter);

  n.Notify();
  delete thread_pool;  // Wait for all closures to finish.
  EXPECT_EQ(4, counter);
}

TEST(NotificationTest, TestWaitWithTimeoutOnNotifiedNotification) {
  Notification n;
  n.Notify();
  EXPECT_TRUE(WaitForNotificationWithTimeout(&n, 1000));
}

}  // namespace
}  // namespace tensorflow
