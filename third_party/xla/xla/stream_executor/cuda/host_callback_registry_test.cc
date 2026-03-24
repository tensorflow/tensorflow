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

#include "xla/stream_executor/cuda/host_callback_registry.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

namespace stream_executor::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

struct BackgroundEnqueue {
  std::unique_ptr<std::atomic<bool>> cancelled;
  HostCallbackRegistry::EnqueueCb enqueue_cb;

  void signal_stop() { cancelled->store(true, std::memory_order_release); }
};

// Helper to pass an enqueue functor by reference.
HostCallbackRegistry::EnqueueCb AsReference(
    HostCallbackRegistry::EnqueueCb& enqueue_cb) {
  return [&enqueue_cb](HostCallbackRegistry::DeviceCb cb,
                       void* data) -> absl::Status {
    return enqueue_cb(cb, data);
  };
}

// Single threaded enqueue functor that does not support cancellation.
HostCallbackRegistry::EnqueueCb CreateBackgroundEnqueueCb() {
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "EnqueueFunctor", 1);
  return [thread_pool = std::move(thread_pool)](
             HostCallbackRegistry::DeviceCb cb, void* data) -> absl::Status {
    thread_pool->Schedule([cb, data] { cb(data); });
    return absl::OkStatus();
  };
}

// Single threaded enqueue functor that will not invoke callbacks if cancelled.
BackgroundEnqueue CreateBackgroundEnqueueCbWithCancellation() {
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "EnqueueFunctor", 1);
  auto cancellation_ptr = std::make_unique<std::atomic<bool>>(false);
  auto enqueue_cb = [thread_pool = std::move(thread_pool),
                     cancelled_ptr = cancellation_ptr.get()](
                        HostCallbackRegistry::DeviceCb cb,
                        void* data) -> absl::Status {
    thread_pool->Schedule([cb, data, cancelled_ptr] {
      if (cancelled_ptr->load(std::memory_order_acquire)) {
        return;
      }
      cb(data);
    });
    return absl::OkStatus();
  };
  return BackgroundEnqueue{std::move(cancellation_ptr), std::move(enqueue_cb)};
}

struct RegistryArgs {
  int device_ordinal{0};
  HostCallbackRegistry::StatusCb synchronization_callback = []() {
    return absl::OkStatus();
  };
  HostCallbackRegistry::StatusCb status_callback = []() {
    return absl::OkStatus();
  };
  absl::Duration poll_interval = absl::Milliseconds(10);
};

HostCallbackRegistry CreateRegistry(RegistryArgs args = {}) {
  return HostCallbackRegistry(
      args.device_ordinal, std::move(args.synchronization_callback),
      std::move(args.status_callback), args.poll_interval);
}

class HostCallbackRegistryTest : public ::testing::Test {};

TEST_F(HostCallbackRegistryTest, AddAndExecuteCallback) {
  absl::Notification done;
  bool called = false;
  HostCallbackRegistry registry = CreateRegistry();
  auto status = registry.AddCallback(
      [&]() {
        called = true;
        done.Notify();
        return absl::OkStatus();
      },
      /*error_cb=*/nullptr,
      /*enqueue_cb=*/
      [](HostCallbackRegistry::DeviceCb cb, void* data) {
        cb(data);
        return absl::OkStatus();
      });

  EXPECT_THAT(status, IsOk());
  done.WaitForNotification();
  EXPECT_TRUE(called);
}

TEST_F(HostCallbackRegistryTest, MultipleCallbacks) {
  int count = 0;
  HostCallbackRegistry registry = CreateRegistry();
  auto status1 = registry.AddCallback(
      [&]() {
        count++;
        return absl::OkStatus();
      },
      /*error_cb=*/nullptr,
      /*enqueue_cb=*/
      [](HostCallbackRegistry::DeviceCb cb, void* data) {
        cb(data);
        return absl::OkStatus();
      });

  auto status2 = registry.AddCallback(
      [&]() {
        count++;
        return absl::OkStatus();
      },
      /*error_cb=*/nullptr,
      /*enqueue_cb=*/
      [](HostCallbackRegistry::DeviceCb cb, void* data) {
        cb(data);
        return absl::OkStatus();
      });

  EXPECT_THAT(status1, IsOk());
  EXPECT_THAT(status2, IsOk());
  EXPECT_EQ(count, 2);
}

TEST_F(HostCallbackRegistryTest, CallbackReturnsError) {
  absl::Notification error_notified;
  absl::Status callback_status;

  HostCallbackRegistry registry = CreateRegistry();
  auto status =
      registry.AddCallback([]() { return absl::InternalError("test error"); },
                           [&](absl::Status s) {
                             callback_status = s;
                             error_notified.Notify();
                           },
                           [](HostCallbackRegistry::DeviceCb cb, void* data) {
                             cb(data);
                             return absl::OkStatus();
                           });

  EXPECT_THAT(status, IsOk());
  error_notified.WaitForNotification();
  EXPECT_THAT(callback_status,
              StatusIs(absl::StatusCode::kInternal, "test error"));
}

TEST_F(HostCallbackRegistryTest, EnqueueFails) {
  bool error_notified = false;
  HostCallbackRegistry registry = CreateRegistry();
  auto status =
      registry.AddCallback([]() { return absl::OkStatus(); },
                           [&](absl::Status s) { error_notified = true; },
                           [](HostCallbackRegistry::DeviceCb cb, void* data) {
                             return absl::InternalError("enqueue failed");
                           });

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal, "enqueue failed"));
  EXPECT_TRUE(error_notified);
}

TEST_F(HostCallbackRegistryTest, BackgroundEnqueueCallbackFails) {
  absl::Notification done;
  HostCallbackRegistry registry = CreateRegistry();
  HostCallbackRegistry::EnqueueCb enqueue_cb = CreateBackgroundEnqueueCb();
  std::atomic<int32_t> total_count = 0;
  std::atomic<int32_t> success_count = 0;
  std::atomic<int32_t> error_count = 0;
  for (int i = 0; i < 10; ++i) {
    auto status = registry.AddCallback(
        [&]() {
          int32_t prev = total_count++;
          if (prev == 5) {
            return absl::InternalError("test error");
          }
          success_count++;
          if (total_count >= 11) {
            done.Notify();
          }
          return absl::OkStatus();
        },
        [&](absl::Status s) {
          total_count++;
          error_count++;
          if (total_count >= 11) {
            done.Notify();
          }
        },
        AsReference(enqueue_cb));
    EXPECT_THAT(status, IsOk());
  }
  done.WaitForNotification();
  // 9 + 1 is called both for success and error.
  EXPECT_EQ(total_count, 11);
  EXPECT_EQ(success_count, 9);
  ASSERT_EQ(error_count, 1);
}

TEST_F(HostCallbackRegistryTest, BackgroundEnqueueFailure) {
  absl::Notification success_done;
  absl::Notification error_done;
  absl::Notification error_notified;
  absl::Notification monitor_failed;
  RegistryArgs args;
  BackgroundEnqueue background_enqueue =
      CreateBackgroundEnqueueCbWithCancellation();
  args.status_callback = [&] {
    if (error_notified.HasBeenNotified()) {
      return absl::InternalError("Injected stream refresh error");
    }
    return absl::OkStatus();
  };
  args.synchronization_callback = [&, first_call = true]() mutable {
    // Cancel the enqueue functor.
    if (first_call) {
      background_enqueue.signal_stop();
      monitor_failed.Notify();
      first_call = false;
    }
    return absl::OkStatus();
  };
  HostCallbackRegistry registry = CreateRegistry(std::move(args));
  std::atomic<int32_t> success_count = 0;
  std::atomic<int32_t> error_count = 0;
  for (int i = 0; i < 10; ++i) {
    absl::Status status = registry.AddCallback(
        [&]() {
          int32_t current = ++success_count;
          if (current == 5) {
            error_notified.Notify();
            // Wait for the monitor to wake up and read our cancellation.
            monitor_failed.WaitForNotification();
            success_done.Notify();
          }
          return absl::OkStatus();
        },
        [&](absl::Status s) {
          error_count++;
          if (error_count == 5) {
            error_done.Notify();
          }
        },
        AsReference(background_enqueue.enqueue_cb));
    ASSERT_THAT(status, IsOk());
  }
  success_done.WaitForNotification();
  error_done.WaitForNotification();
  EXPECT_EQ(success_count + error_count, 10);
  EXPECT_EQ(success_count, 5);
  ASSERT_EQ(error_count, 5);
  background_enqueue.enqueue_cb = nullptr;  // Delete the "stream".
}

TEST_F(HostCallbackRegistryTest, BackgroundEnqueueNoSyncOnFailure) {
  absl::Notification error_notified;
  auto background_enqueue = CreateBackgroundEnqueueCbWithCancellation();
  RegistryArgs args;
  args.status_callback = [&] {
    if (error_notified.HasBeenNotified()) {
      return absl::InternalError("Injected stream refresh error");
    }
    return absl::OkStatus();
  };
  args.synchronization_callback = [&, first_call = true]() mutable {
    if (first_call) {
      background_enqueue.signal_stop();
      first_call = false;
    }
    return absl::OkStatus();
  };
  std::atomic<int32_t> success_count = 0;
  std::atomic<int32_t> error_count = 0;
  {
    HostCallbackRegistry registry = CreateRegistry(std::move(args));
    for (int i = 0; i < 10; ++i) {
      absl::Status status = registry.AddCallback(
          [&]() {
            int32_t current = ++success_count;
            if (current == 0) {
              error_notified.Notify();
            }
            absl::SleepFor(absl::Milliseconds(1));  // Simulate some work.
            return absl::OkStatus();
          },
          [&](absl::Status s) { error_count++; },
          AsReference(background_enqueue.enqueue_cb));
      ASSERT_THAT(status, IsOk());
    }
    background_enqueue.enqueue_cb = nullptr;  // Delete the "stream".
  }
  EXPECT_EQ(success_count + error_count, 10);  // All callbacks are called.
  EXPECT_GE(success_count, 1);                 // At least. Maybe more.
}

TEST_F(HostCallbackRegistryTest, MultipleProducers) {
  std::atomic<int32_t> count = 0;
  std::atomic<int32_t> error_count = 0;
  constexpr int kNumCallbacks = 100;
  {
    HostCallbackRegistry registry = CreateRegistry({});
    auto background_enqueue = CreateBackgroundEnqueueCbWithCancellation();
    tsl::thread::ThreadPool producer_pool(tsl::Env::Default(), "test", 5);
    for (int i = 0; i < kNumCallbacks; ++i) {
      producer_pool.Schedule([&] {
        absl::Status status = registry.AddCallback(
            [&]() {
              ++count;
              absl::SleepFor(absl::Milliseconds(1));  // Simulate some work.
              return absl::OkStatus();
            },
            [&](absl::Status s) { error_count++; },
            AsReference(background_enqueue.enqueue_cb));
        ASSERT_THAT(status, IsOk());
      });
    }
  }
  EXPECT_EQ(count + error_count, kNumCallbacks);  // All callbacks are called.
}

void BM_AddCallback(benchmark::State& state) {
  HostCallbackRegistry registry = CreateRegistry();

  for (auto _ : state) {
    absl::Status status = registry.AddCallback(
        []() { return absl::OkStatus(); },
        /*error_cb=*/nullptr, [](...) { return absl::OkStatus(); });
    benchmark::DoNotOptimize(status);
  }
}

BENCHMARK(BM_AddCallback);

}  // namespace
}  // namespace stream_executor::gpu
