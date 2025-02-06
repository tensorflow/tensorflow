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

#include "xla/tsl/framework/cancellation.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/notification.h"

namespace tsl {

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

TEST(Cancellation, StartCancelTriggersAllCallbacks) {
  bool is_cancelled_1 = false;
  bool is_cancelled_2 = false;
  auto manager = std::make_unique<CancellationManager>();
  auto token_1 = manager->get_cancellation_token();
  EXPECT_TRUE(manager->RegisterCallbackWithErrorLogging(
      token_1, [&is_cancelled_1]() { is_cancelled_1 = true; }, "TestCallback"));
  auto token_2 = manager->get_cancellation_token();
  EXPECT_TRUE(manager->RegisterCallback(
      token_2, [&is_cancelled_2]() { is_cancelled_2 = true; }));
  manager->StartCancel();
  EXPECT_TRUE(is_cancelled_1);
  EXPECT_TRUE(is_cancelled_2);
}

TEST(Cancellation, StartCancelWithStatusTriggersAllCallbacks) {
  bool is_cancelled_1 = false;
  bool is_cancelled_2 = false;
  auto manager = std::make_unique<CancellationManager>();
  auto token_1 = manager->get_cancellation_token();
  EXPECT_TRUE(manager->RegisterCallbackWithErrorLogging(
      token_1, [&is_cancelled_1]() { is_cancelled_1 = true; }, "TestCallback"));
  auto token_2 = manager->get_cancellation_token();
  EXPECT_TRUE(manager->RegisterCallback(
      token_2, [&is_cancelled_2]() { is_cancelled_2 = true; }));
  manager->StartCancelWithStatus(absl::OkStatus());
  EXPECT_TRUE(is_cancelled_1);
  EXPECT_TRUE(is_cancelled_2);
}

TEST(Cancellation, CancelBeforeRegister) {
  auto manager = std::make_unique<CancellationManager>();
  auto token = manager->get_cancellation_token();
  manager->StartCancel();
  bool registered = manager->RegisterCallback(token, nullptr);
  EXPECT_FALSE(registered);
}

TEST(Cancellation, DeregisterAfterCancel) {
  bool is_cancelled = false;
  auto manager = std::make_unique<CancellationManager>();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(
      token, [&is_cancelled]() { is_cancelled = true; });
  EXPECT_TRUE(registered);
  manager->StartCancel();
  EXPECT_TRUE(is_cancelled);
  bool deregistered = manager->DeregisterCallback(token);
  EXPECT_FALSE(deregistered);
}

TEST(Cancellation, CancelMultiple) {
  bool is_cancelled_1 = false, is_cancelled_2 = false, is_cancelled_3 = false;
  auto manager = std::make_unique<CancellationManager>();
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
}

TEST(Cancellation, IsCancelled) {
  auto cm = std::make_unique<CancellationManager>();
  thread::ThreadPool w(Env::Default(), "test", 4);
  std::vector<Notification> done(8);
  for (size_t i = 0; i < done.size(); ++i) {
    Notification* n = &done[i];
    w.Schedule([n, &cm]() {
      while (!cm->IsCancelled()) {
      }
      ASSERT_FALSE(cm->IsCancelling());
      n->Notify();
    });
  }
  Env::Default()->SleepForMicroseconds(1000000 /* 1 second */);
  cm->StartCancel();
  for (size_t i = 0; i < done.size(); ++i) {
    done[i].WaitForNotification();
  }
}

TEST(Cancellation, IsCancelling) {
  CancellationManager cm;
  Notification started_cancelling;
  Notification can_finish_cancel;
  Notification cancel_done;
  thread::ThreadPool w(Env::Default(), "test", 1);
  auto token = cm.get_cancellation_token();
  ASSERT_TRUE(
      cm.RegisterCallback(token, [&started_cancelling, &can_finish_cancel]() {
        started_cancelling.Notify();
        can_finish_cancel.WaitForNotification();
      }));
  w.Schedule([&cm, &cancel_done]() {
    cm.StartCancel();
    cancel_done.Notify();
  });
  started_cancelling.WaitForNotification();
  ASSERT_TRUE(cm.IsCancelling());
  can_finish_cancel.Notify();
  cancel_done.WaitForNotification();
  ASSERT_FALSE(cm.IsCancelling());
  ASSERT_TRUE(cm.IsCancelled());
}

TEST(Cancellation, TryDeregisterWithoutCancel) {
  bool is_cancelled = false;
  auto manager = std::make_unique<CancellationManager>();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(
      token, [&is_cancelled]() { is_cancelled = true; });
  EXPECT_TRUE(registered);
  bool deregistered = manager->TryDeregisterCallback(token);
  EXPECT_TRUE(deregistered);
  EXPECT_FALSE(is_cancelled);
}

TEST(Cancellation, TryDeregisterAfterCancel) {
  bool is_cancelled = false;
  auto manager = std::make_unique<CancellationManager>();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(
      token, [&is_cancelled]() { is_cancelled = true; });
  EXPECT_TRUE(registered);
  manager->StartCancel();
  EXPECT_TRUE(is_cancelled);
  bool deregistered = manager->TryDeregisterCallback(token);
  EXPECT_FALSE(deregistered);
}

TEST(Cancellation, TryDeregisterDuringCancel) {
  Notification cancel_started, finish_callback, cancel_complete;
  auto manager = std::make_unique<CancellationManager>();
  auto token = manager->get_cancellation_token();
  bool registered = manager->RegisterCallback(token, [&]() {
    cancel_started.Notify();
    finish_callback.WaitForNotification();
  });
  EXPECT_TRUE(registered);

  thread::ThreadPool w(Env::Default(), "test", 1);
  w.Schedule([&]() {
    manager->StartCancel();
    cancel_complete.Notify();
  });
  cancel_started.WaitForNotification();

  bool deregistered = manager->TryDeregisterCallback(token);
  EXPECT_FALSE(deregistered);

  finish_callback.Notify();
  cancel_complete.WaitForNotification();
}

TEST(Cancellation, Parent_CancelManyChildren) {
  CancellationManager parent;
  std::vector<std::unique_ptr<CancellationManager>> children;
  for (size_t i = 0; i < 5; ++i) {
    children.push_back(absl::make_unique<CancellationManager>(&parent));
    EXPECT_FALSE(children.back()->IsCancelled());
  }
  parent.StartCancel();
  for (auto& child : children) {
    EXPECT_TRUE(child->IsCancelled());
  }
}

TEST(Cancellation, Parent_NotCancelled) {
  CancellationManager parent;
  {
    CancellationManager child(&parent);
    child.StartCancel();
    EXPECT_TRUE(child.IsCancelled());
  }
  EXPECT_FALSE(parent.IsCancelled());
}

TEST(Cancellation, Parent_AlreadyCancelled) {
  CancellationManager parent;
  parent.StartCancel();
  EXPECT_TRUE(parent.IsCancelled());

  CancellationManager child(&parent);
  EXPECT_TRUE(child.IsCancelled());
}

TEST(Cancellation, Parent_RandomDestructionOrder) {
  CancellationManager parent;
  std::random_device rd;
  std::mt19937 g(rd());

  // To cover the linked-list codepaths, perform multiple randomized rounds of
  // registering and deregistering children with `parent`.
  for (int rounds = 0; rounds < 100; ++rounds) {
    std::vector<std::unique_ptr<CancellationManager>> children;

    // 1. Register a random number of children with the parent.
    std::uniform_int_distribution<int> dist(1, 9);
    const size_t round_size = dist(rd);
    for (size_t i = 0; i < round_size; ++i) {
      children.push_back(absl::make_unique<CancellationManager>(&parent));
      EXPECT_FALSE(children.back()->IsCancelled());
    }

    // 2. Deregister the children in a random order.
    std::vector<size_t> destruction_order(round_size);
    std::iota(destruction_order.begin(), destruction_order.end(), 0);
    std::shuffle(destruction_order.begin(), destruction_order.end(), g);
    for (size_t index : destruction_order) {
      children[index].reset();
    }
  }
}

}  // namespace tsl
