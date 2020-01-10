#include <gtest/gtest.h>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/port.h"

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
  sleep(1);

  EXPECT_EQ(0, counter);

  n.Notify();
  delete thread_pool;  // Wait for all closures to finish.
  EXPECT_EQ(4, counter);
}

}  // namespace
}  // namespace tensorflow
