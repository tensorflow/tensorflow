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
#include <vector>

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
using RegistryHandle =
    ::stream_executor::gpu::HostCallbackRegistry::RegistryHandle;

struct BackgroundEnqueue {
  std::unique_ptr<std::atomic<bool>> cancelled;
  RegistryHandle::EnqueueCb enqueue_cb;

  void signal_stop() { cancelled->store(true, std::memory_order_release); }
};

// Helper to pass an enqueue functor by reference.
RegistryHandle::EnqueueCb AsReference(RegistryHandle::EnqueueCb& enqueue_cb) {
  return
      [&enqueue_cb](RegistryHandle::DeviceCb cb, void* data) -> absl::Status {
        return enqueue_cb(cb, data);
      };
}

// Single threaded enqueue functor that does not support cancellation.
RegistryHandle::EnqueueCb CreateBackgroundEnqueueCb() {
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "EnqueueFunctor", 1);
  return [thread_pool = std::move(thread_pool)](RegistryHandle::DeviceCb cb,
                                                void* data) -> absl::Status {
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
                        RegistryHandle::DeviceCb cb,
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
  RegistryHandle::StatusCb synchronization_callback = []() {
    return absl::OkStatus();
  };
  RegistryHandle::StatusCb status_callback = []() { return absl::OkStatus(); };
  absl::Duration poll_interval = absl::Milliseconds(10);
};

struct RegistryTuple {
  std::unique_ptr<HostCallbackRegistry> host_callback_registry;
  std::unique_ptr<RegistryHandle> handle;
};

RegistryTuple CreateRegistry(RegistryArgs args = {}) {
  auto host_callback_registry =
      std::make_unique<HostCallbackRegistry>(0, args.poll_interval);
  HostCallbackRegistry* registry_ptr = host_callback_registry.get();
  return {std::move(host_callback_registry),
          registry_ptr->CreateHandle(std::move(args.synchronization_callback),
                                     std::move(args.status_callback))};
}

class RegistryHandleTest : public ::testing::Test {};

TEST_F(RegistryHandleTest, AddAndExecuteCallback) {
  absl::Notification done;
  bool called = false;
  RegistryTuple registry_tuple = CreateRegistry();
  auto status = registry_tuple.handle->AddCallback(
      [&]() {
        called = true;
        done.Notify();
        return absl::OkStatus();
      },
      /*error_cb=*/nullptr,
      /*enqueue_cb=*/
      [](RegistryHandle::DeviceCb cb, void* data) {
        cb(data);
        return absl::OkStatus();
      });

  EXPECT_THAT(status, IsOk());
  done.WaitForNotification();
  EXPECT_TRUE(called);
}

TEST_F(RegistryHandleTest, MultipleCallbacks) {
  int count = 0;
  RegistryTuple registry_tuple = CreateRegistry();
  RegistryHandle& handle = *registry_tuple.handle;
  auto status1 = handle.AddCallback(
      [&]() {
        count++;
        return absl::OkStatus();
      },
      /*error_cb=*/nullptr,
      /*enqueue_cb=*/
      [](RegistryHandle::DeviceCb cb, void* data) {
        cb(data);
        return absl::OkStatus();
      });

  auto status2 = handle.AddCallback(
      [&]() {
        count++;
        return absl::OkStatus();
      },
      /*error_cb=*/nullptr,
      /*enqueue_cb=*/
      [](RegistryHandle::DeviceCb cb, void* data) {
        cb(data);
        return absl::OkStatus();
      });

  EXPECT_THAT(status1, IsOk());
  EXPECT_THAT(status2, IsOk());
  EXPECT_EQ(count, 2);
}

TEST_F(RegistryHandleTest, CallbackReturnsError) {
  absl::Notification error_notified;
  absl::Status callback_status;

  RegistryTuple registry_tuple = CreateRegistry();
  auto& handle = *registry_tuple.handle;
  auto status =
      handle.AddCallback([]() { return absl::InternalError("test error"); },
                         [&](absl::Status s) {
                           callback_status = s;
                           error_notified.Notify();
                         },
                         [](RegistryHandle::DeviceCb cb, void* data) {
                           cb(data);
                           return absl::OkStatus();
                         });

  EXPECT_THAT(status, IsOk());
  error_notified.WaitForNotification();
  EXPECT_THAT(callback_status,
              StatusIs(absl::StatusCode::kInternal, "test error"));
}

TEST_F(RegistryHandleTest, EnqueueFails) {
  bool error_notified = false;
  RegistryTuple registry_tuple = CreateRegistry();
  auto status = registry_tuple.handle->AddCallback(
      []() { return absl::OkStatus(); },
      [&](absl::Status s) { error_notified = true; },
      [](RegistryHandle::DeviceCb cb, void* data) {
        return absl::InternalError("enqueue failed");
      });

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal, "enqueue failed"));
  EXPECT_TRUE(error_notified);
}

TEST_F(RegistryHandleTest, BackgroundEnqueueCallbackFails) {
  absl::Notification done;
  RegistryTuple registry_tuple = CreateRegistry();
  RegistryHandle::EnqueueCb enqueue_cb = CreateBackgroundEnqueueCb();
  std::atomic<int32_t> total_count = 0;
  std::atomic<int32_t> success_count = 0;
  std::atomic<int32_t> error_count = 0;
  for (int i = 0; i < 10; ++i) {
    auto status = registry_tuple.handle->AddCallback(
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

TEST_F(RegistryHandleTest, BackgroundEnqueueFailure) {
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
  RegistryTuple registry_tuple = CreateRegistry(std::move(args));
  std::atomic<int32_t> success_count = 0;
  std::atomic<int32_t> error_count = 0;
  for (int i = 0; i < 10; ++i) {
    absl::Status status = registry_tuple.handle->AddCallback(
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

TEST_F(RegistryHandleTest, BackgroundEnqueueNoSyncOnFailure) {
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
    RegistryTuple registry_tuple = CreateRegistry(std::move(args));
    for (int i = 0; i < 10; ++i) {
      absl::Status status = registry_tuple.handle->AddCallback(
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

TEST_F(RegistryHandleTest, MultipleProducers) {
  std::atomic<int32_t> count = 0;
  std::atomic<int32_t> error_count = 0;
  constexpr int kNumCallbacks = 100;
  {
    RegistryTuple registry_tuple = CreateRegistry({});
    auto background_enqueue = CreateBackgroundEnqueueCbWithCancellation();
    tsl::thread::ThreadPool producer_pool(tsl::Env::Default(), "test", 5);
    for (int i = 0; i < kNumCallbacks; ++i) {
      producer_pool.Schedule([&] {
        absl::Status status = registry_tuple.handle->AddCallback(
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

TEST_F(RegistryHandleTest, MultipleStreamsRegistration) {
  constexpr int kNumStreams = 10;
  std::atomic<int> completed_count{0};
  absl::Notification all_done;

  auto host_callback_registry =
      std::make_unique<HostCallbackRegistry>(0, absl::Milliseconds(10));
  std::vector<std::unique_ptr<RegistryHandle>> handles;

  for (int i = 0; i < kNumStreams; ++i) {
    handles.push_back(host_callback_registry->CreateHandle(
        []() { return absl::OkStatus(); }, []() { return absl::OkStatus(); }));
  }

  for (int i = 0; i < kNumStreams; ++i) {
    auto status = handles[i]->AddCallback(
        [&]() {
          if (++completed_count == kNumStreams) {
            all_done.Notify();
          }
          return absl::OkStatus();
        },
        nullptr,
        [](StreamCallbackRegistry::DeviceCb cb, void* data) {
          cb(data);
          return absl::OkStatus();
        });
    EXPECT_THAT(status, IsOk());
  }

  all_done.WaitForNotification();
  EXPECT_EQ(completed_count.load(), kNumStreams);
}

TEST_F(RegistryHandleTest, ConcurrentDeregistration) {
  constexpr int kNumStreams = 50;
  auto host_callback_registry =
      std::make_unique<HostCallbackRegistry>(0, absl::Milliseconds(1));

  // Shared because thread pool wants copyable types.
  std::vector<std::shared_ptr<RegistryHandle>> handles;
  for (int i = 0; i < kNumStreams; ++i) {
    auto unique_handle = host_callback_registry->CreateHandle(
        []() { return absl::OkStatus(); }, []() { return absl::OkStatus(); });
    handles.push_back(
        std::shared_ptr<RegistryHandle>(std::move(unique_handle)));
  }

  // Destroy handles from multiple threads to simulate concurrent stream
  // destruction. This tests the Copy-On-Write logic.
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "destructor_pool",
                                      10);
  for (int i = 0; i < kNumStreams; ++i) {
    thread_pool.Schedule([h = std::move(handles[i])]() {
      // RAII destructor calls DeregisterHandle
    });
  }
}

TEST_F(RegistryHandleTest, DeregistrationHappensFast) {
  constexpr int kNumStreams = 50;
  // Make sure that polling is not a factor.
  auto host_callback_registry =
      std::make_unique<HostCallbackRegistry>(0, absl::InfiniteDuration());
  // Shared because thread pool wants copyable types.
  std::vector<std::shared_ptr<RegistryHandle>> handles;
  for (int i = 0; i < kNumStreams; ++i) {
    auto unique_handle = host_callback_registry->CreateHandle(
        []() { return absl::OkStatus(); }, []() { return absl::OkStatus(); });
    handles.push_back(
        std::shared_ptr<RegistryHandle>(std::move(unique_handle)));
  }
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "destructor_pool",
                                      10);
  for (int i = 0; i < kNumStreams; ++i) {
    thread_pool.Schedule([h = std::move(handles[i])]() {
      // RAII destructor calls DeregisterHandle
    });
  }
  // Test shouldn't time out.
}

void BM_AddCallback(benchmark::State& state) {
  RegistryTuple registry = CreateRegistry();

  for (auto _ : state) {
    absl::Status status = registry.handle->AddCallback(
        []() { return absl::OkStatus(); },
        /*error_cb=*/nullptr, [](...) { return absl::OkStatus(); });
    benchmark::DoNotOptimize(status);
  }
}

BENCHMARK(BM_AddCallback);

}  // namespace
}  // namespace stream_executor::gpu
