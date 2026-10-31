/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_async_execution_limiter.h"

#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/python/ifrt/device.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::xla::ifrt::DeviceId;

TEST(IfrtAsyncExecutionLimiterTest, TryAcquireAndReleaseSingleDevice) {
  IfrtAsyncExecutionLimiter limiter(/*max_inflight_per_device=*/2);
  EXPECT_EQ(limiter.max_inflight_per_device(), 2);

  const DeviceId d0(0);
  const DeviceId d1(1);

  std::optional<IfrtAsyncExecutionLimiter::Permit> p1 =
      limiter.TryAcquire({d0});
  std::optional<IfrtAsyncExecutionLimiter::Permit> p2 =
      limiter.TryAcquire({d0});
  ASSERT_TRUE(p1.has_value());
  ASSERT_TRUE(p2.has_value());
  EXPECT_EQ(limiter.InflightForDevice(d0), 2);

  // Third permit on d0 fails without blocking, while d1 remains independent.
  EXPECT_THAT(limiter.TryAcquire({d0}), Eq(std::nullopt));
  std::optional<IfrtAsyncExecutionLimiter::Permit> p_d1 =
      limiter.TryAcquire({d1});
  ASSERT_TRUE(p_d1.has_value());
  EXPECT_EQ(limiter.InflightForDevice(d1), 1);

  // Move semantics transfer ownership cleanly without double-release.
  IfrtAsyncExecutionLimiter::Permit moved_p1 = *std::move(p1);
  EXPECT_FALSE(static_cast<bool>(*p1));
  EXPECT_EQ(limiter.InflightForDevice(d0), 2);

  moved_p1.Release();
  EXPECT_EQ(limiter.InflightForDevice(d0), 1);

  std::optional<IfrtAsyncExecutionLimiter::Permit> p3 =
      limiter.TryAcquire({d0});
  ASSERT_TRUE(p3.has_value());
  EXPECT_EQ(limiter.InflightForDevice(d0), 2);
}

TEST(IfrtAsyncExecutionLimiterTest,
     RunWhenPermittedEnforcesCapAndStrictFifoWithoutBlockingCaller) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "limiter_test",
                               /*num_threads=*/8);
  IfrtAsyncExecutionLimiter limiter(/*max_inflight_per_device=*/2);
  const DeviceId d0(0);

  // Hold both slots initially via TryAcquire.
  std::optional<IfrtAsyncExecutionLimiter::Permit> initial_1 =
      limiter.TryAcquire({d0});
  std::optional<IfrtAsyncExecutionLimiter::Permit> initial_2 =
      limiter.TryAcquire({d0});
  ASSERT_TRUE(initial_1.has_value());
  ASSERT_TRUE(initial_2.has_value());

  constexpr int kNumQueued = 5;
  absl::Mutex order_mu;
  std::vector<int> grant_order;
  std::vector<IfrtAsyncExecutionLimiter::Permit> held_permits(kNumQueued);
  std::vector<absl::Notification> granted(kNumQueued);

  // All 5 RunWhenPermitted calls must return immediately without blocking.
  for (int i = 0; i < kNumQueued; ++i) {
    limiter.RunWhenPermitted({d0}, pool,
                             [&, i](IfrtAsyncExecutionLimiter::Permit permit) {
                               {
                                 absl::MutexLock lock(order_mu);
                                 grant_order.push_back(i);
                                 held_permits[i] = std::move(permit);
                               }
                               granted[i].Notify();
                             });
  }

  EXPECT_EQ(limiter.QueueDepthForDevice(d0), kNumQueued);
  EXPECT_EQ(limiter.InflightForDevice(d0), 2);
  EXPECT_FALSE(granted[0].HasBeenNotified());

  // While waiters are queued, a fresh TryAcquire must NOT barge in ahead of
  // FIFO waiters even as permits are released.
  initial_1->Release();
  granted[0].WaitForNotification();
  EXPECT_THAT(limiter.TryAcquire({d0}), Eq(std::nullopt));

  initial_2->Release();
  granted[1].WaitForNotification();
  EXPECT_EQ(limiter.InflightForDevice(d0), 2);
  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 3);

  for (int i = 0; i < kNumQueued; ++i) {
    granted[i].WaitForNotification();
    {
      absl::MutexLock lock(order_mu);
      held_permits[i].Release();
    }
  }

  absl::MutexLock lock(order_mu);
  EXPECT_THAT(grant_order, ElementsAre(0, 1, 2, 3, 4));
  EXPECT_EQ(limiter.InflightForDevice(d0), 0);
  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 0);
}

TEST(IfrtAsyncExecutionLimiterTest,
     MultiDeviceAcquisitionCanonicalizesOrderWithoutDeadlock) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "spmd_limiter_test",
                               /*num_threads=*/8);
  tsl::thread::ThreadPool callers(tsl::Env::Default(), "spmd_callers",
                                  /*num_threads=*/8);
  IfrtAsyncExecutionLimiter limiter(/*max_inflight_per_device=*/2);

  const DeviceId d0(0);
  const DeviceId d1(1);

  constexpr int kNumRequests = 40;
  std::atomic<int> active_d0{0};
  std::atomic<int> active_d1{0};
  std::atomic<int> peak_d0{0};
  std::atomic<int> peak_d1{0};
  absl::BlockingCounter done(kNumRequests);

  for (int i = 0; i < kNumRequests; ++i) {
    callers.Schedule([&, i]() {
      // Half the callers pass {d0, d1}, half pass {d1, d0, d1} (unsorted +
      // duplicate); canonicalization must prevent AB-BA deadlocks and
      // double-counting.
      std::vector<DeviceId> req_devices =
          (i % 2 == 0) ? std::vector<DeviceId>{d0, d1}
                       : std::vector<DeviceId>{d1, d0, d1};
      limiter.RunWhenPermitted(
          req_devices, pool, [&](IfrtAsyncExecutionLimiter::Permit permit) {
            const int c0 = active_d0.fetch_add(1) + 1;
            const int c1 = active_d1.fetch_add(1) + 1;
            int prev0 = peak_d0.load();
            while (c0 > prev0 && !peak_d0.compare_exchange_weak(prev0, c0)) {
            }
            int prev1 = peak_d1.load();
            while (c1 > prev1 && !peak_d1.compare_exchange_weak(prev1, c1)) {
            }
            tsl::Env::Default()->SleepForMicroseconds(500);
            active_d0.fetch_sub(1);
            active_d1.fetch_sub(1);
            permit.Release();
            done.DecrementCount();
          });
    });
  }

  done.Wait();
  EXPECT_LE(peak_d0.load(), 2);
  EXPECT_LE(peak_d1.load(), 2);
  EXPECT_EQ(limiter.InflightForDevice(d0), 0);
  EXPECT_EQ(limiter.InflightForDevice(d1), 0);
}

TEST(IfrtAsyncExecutionLimiterTest, EmptyDeviceIdsExecutesInlineImmediately) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "empty_test",
                               /*num_threads=*/2);
  IfrtAsyncExecutionLimiter limiter(/*max_inflight_per_device=*/1);

  std::optional<IfrtAsyncExecutionLimiter::Permit> p = limiter.TryAcquire({});
  ASSERT_TRUE(p.has_value());
  EXPECT_FALSE(static_cast<bool>(*p));

  bool ran_inline = false;
  limiter.RunWhenPermitted({}, pool,
                           [&](IfrtAsyncExecutionLimiter::Permit permit) {
                             EXPECT_FALSE(static_cast<bool>(permit));
                             ran_inline = true;
                           });
  EXPECT_TRUE(ran_inline);
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
