#include "tensorflow/core/framework/cancellation.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

TEST(Cancellation, SimpleNoCancel) {
  bool is_cancelled = false;
  CancellationManager* manager = new CancellationManager();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(
      token, [&is_cancelled]() { is_cancelled = true; });
  EXPECT_TRUE(registered);
  bool deregistered = manager->DeregisterCallback(token);
  EXPECT_TRUE(deregistered);
  delete manager;
  EXPECT_FALSE(is_cancelled);
}

TEST(Cancellation, SimpleCancel) {
  bool is_cancelled = false;
  CancellationManager* manager = new CancellationManager();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(
      token, [&is_cancelled]() { is_cancelled = true; });
  EXPECT_TRUE(registered);
  manager->StartCancel();
  EXPECT_TRUE(is_cancelled);
  delete manager;
}

TEST(Cancellation, CancelBeforeRegister) {
  CancellationManager* manager = new CancellationManager();
  auto token = manager->get_cancellation_token();
  manager->StartCancel();
  bool registered = manager->RegisterCallback(token, nullptr);
  EXPECT_FALSE(registered);
  delete manager;
}

TEST(Cancellation, DeregisterAfterCancel) {
  bool is_cancelled = false;
  CancellationManager* manager = new CancellationManager();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(
      token, [&is_cancelled]() { is_cancelled = true; });
  EXPECT_TRUE(registered);
  manager->StartCancel();
  EXPECT_TRUE(is_cancelled);
  bool deregistered = manager->DeregisterCallback(token);
  EXPECT_FALSE(deregistered);
  delete manager;
}

TEST(Cancellation, CancelMultiple) {
  bool is_cancelled_1 = false, is_cancelled_2 = false, is_cancelled_3 = false;
  CancellationManager* manager = new CancellationManager();
  auto token_1 = manager->get_cancellation_token();
  bool registered_1 = manager->RegisterCallback(
      token_1, [&is_cancelled_1]() { is_cancelled_1 = true; });
  EXPECT_TRUE(registered_1);
  auto token_2 = manager->get_cancellation_token();
  bool registered_2 = manager->RegisterCallback(
      token_2, [&is_cancelled_2]() { is_cancelled_2 = true; });
  EXPECT_TRUE(registered_2);
  EXPECT_FALSE(is_cancelled_1);
  EXPECT_FALSE(is_cancelled_2);
  manager->StartCancel();
  EXPECT_TRUE(is_cancelled_1);
  EXPECT_TRUE(is_cancelled_2);
  EXPECT_FALSE(is_cancelled_3);
  auto token_3 = manager->get_cancellation_token();
  bool registered_3 = manager->RegisterCallback(
      token_3, [&is_cancelled_3]() { is_cancelled_3 = true; });
  EXPECT_FALSE(registered_3);
  EXPECT_FALSE(is_cancelled_3);
  delete manager;
}

TEST(Cancellation, IsCancelled) {
  CancellationManager* cm = new CancellationManager();
  thread::ThreadPool w(Env::Default(), "test", 4);
  std::vector<Notification> done(8);
  for (size_t i = 0; i < done.size(); ++i) {
    Notification* n = &done[i];
    w.Schedule([n, cm]() {
      while (!cm->IsCancelled()) {
      }
      n->Notify();
    });
  }
  Env::Default()->SleepForMicroseconds(1000000 /* 1 second */);
  cm->StartCancel();
  for (size_t i = 0; i < done.size(); ++i) {
    done[i].WaitForNotification();
  }
  delete cm;
}

}  // namespace tensorflow
