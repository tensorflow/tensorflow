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
#include <cstdint>
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

TEST(IfrtAsyncExecutionLimiterTest,
     TryAcquireAndReleaseSingleDeviceByteBudget) {
  IfrtAsyncExecutionLimiter limiter(/*max_bytes_per_device=*/100);
  EXPECT_EQ(limiter.max_bytes_per_device(), 100);

  const DeviceId d0(0);
  const DeviceId d1(1);

  std::optional<IfrtAsyncExecutionLimiter::Permit> p1 =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/40);
  std::optional<IfrtAsyncExecutionLimiter::Permit> p2 =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/50);
  ASSERT_TRUE(p1.has_value());
  ASSERT_TRUE(p2.has_value());
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 2);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 90);

  // Next 20-byte request exceeds 100 on d0 without blocking, while a 10-byte
  // request fits and d1 remains completely independent.
  EXPECT_THAT(limiter.TryAcquire({d0}, /*bytes_per_device=*/20),
              Eq(std::nullopt));
  std::optional<IfrtAsyncExecutionLimiter::Permit> p_d1 =
      limiter.TryAcquire({d1}, /*bytes_per_device=*/80);
  ASSERT_TRUE(p_d1.has_value());
  EXPECT_EQ(limiter.InflightRequestsForDevice(d1), 1);
  EXPECT_EQ(limiter.InflightBytesForDevice(d1), 80);

  // Move semantics transfer ownership cleanly without double-release.
  IfrtAsyncExecutionLimiter::Permit moved_p1 = *std::move(p1);
  EXPECT_FALSE(static_cast<bool>(*p1));
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 2);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 90);

  moved_p1.Release();
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 1);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 50);

  std::optional<IfrtAsyncExecutionLimiter::Permit> p3 =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/50);
  ASSERT_TRUE(p3.has_value());
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 2);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 100);
}

TEST(IfrtAsyncExecutionLimiterTest,
     OversizedSingleRequestAdmittedWhenIdleAndNeverStarved) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "oversized_test",
                               /*num_threads=*/4);
  IfrtAsyncExecutionLimiter limiter(/*max_bytes_per_device=*/100);
  const DeviceId d0(0);

  // 1. An oversized request (5000 > 100) on an idle device is admitted
  // immediately so giant models never stall.
  std::optional<IfrtAsyncExecutionLimiter::Permit> giant =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/5000);
  ASSERT_TRUE(giant.has_value());
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 1);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 5000);

  // While the oversized request is active (`inflight_bytes > max`), even a
  // 0-byte request cannot enter concurrently.
  EXPECT_THAT(limiter.TryAcquire({d0}, /*bytes_per_device=*/0),
              Eq(std::nullopt));
  giant->Release();
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 0);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 0);

  // 2. When 2 small requests (40 bytes each) are active, a queued oversized
  // request (5000 bytes) blocks subsequent small requests from barging ahead of
  // it, runs solo once both 40-byte requests finish, and then wakes the
  // trailing small waiter.
  std::optional<IfrtAsyncExecutionLimiter::Permit> s1 =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/40);
  std::optional<IfrtAsyncExecutionLimiter::Permit> s2 =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/40);
  ASSERT_TRUE(s1.has_value());
  ASSERT_TRUE(s2.has_value());

  absl::Mutex mu;
  std::vector<int> grant_order;
  IfrtAsyncExecutionLimiter::Permit held_giant;
  IfrtAsyncExecutionLimiter::Permit held_small;
  absl::Notification giant_granted;
  absl::Notification trailing_small_granted;

  limiter.RunWhenPermitted({d0}, /*bytes_per_device=*/5000, pool,
                           [&](IfrtAsyncExecutionLimiter::Permit permit) {
                             {
                               absl::MutexLock lock(mu);
                               grant_order.push_back(0);
                               held_giant = std::move(permit);
                             }
                             giant_granted.Notify();
                           });
  limiter.RunWhenPermitted({d0}, /*bytes_per_device=*/10, pool,
                           [&](IfrtAsyncExecutionLimiter::Permit permit) {
                             {
                               absl::MutexLock lock(mu);
                               grant_order.push_back(1);
                               held_small = std::move(permit);
                             }
                             trailing_small_granted.Notify();
                           });

  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 2);
  // Even though 10 bytes would fit in the remaining 20-byte budget (80 + 10 <=
  // 100), TryAcquire must NOT barge ahead of the queued 5000-byte waiter!
  EXPECT_THAT(limiter.TryAcquire({d0}, /*bytes_per_device=*/10),
              Eq(std::nullopt));

  // Releasing only s1 leaves s2 active (`inflight_requests == 1`), so the
  // 5000-byte waiter must still wait until the device is completely idle.
  s1->Release();
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 1);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 40);
  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 2);
  EXPECT_FALSE(giant_granted.HasBeenNotified());

  // Releasing s2 brings `inflight_requests` to 0, admitting the 5000-byte
  // waiter solo while keeping the trailing 10-byte waiter queued.
  s2->Release();
  giant_granted.WaitForNotification();
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 1);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 5000);
  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 1);
  EXPECT_FALSE(trailing_small_granted.HasBeenNotified());

  {
    absl::MutexLock lock(mu);
    held_giant.Release();
  }
  trailing_small_granted.WaitForNotification();
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 1);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 10);
  {
    absl::MutexLock lock(mu);
    held_small.Release();
    EXPECT_THAT(grant_order, ElementsAre(0, 1));
  }
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 0);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 0);
}

TEST(IfrtAsyncExecutionLimiterTest,
     ReleaseLargePermitWakesMultipleSmallWaitersAtOnce) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "multi_wake_test",
                               /*num_threads=*/8);
  IfrtAsyncExecutionLimiter limiter(/*max_bytes_per_device=*/1000);
  const DeviceId d0(0);

  std::optional<IfrtAsyncExecutionLimiter::Permit> large =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/1500);
  ASSERT_TRUE(large.has_value());

  constexpr int kNumSmall = 4;
  absl::Mutex mu;
  std::vector<IfrtAsyncExecutionLimiter::Permit> permits(kNumSmall);
  std::vector<absl::Notification> granted(kNumSmall);
  for (int i = 0; i < kNumSmall; ++i) {
    limiter.RunWhenPermitted({d0}, /*bytes_per_device=*/300, pool,
                             [&, i](IfrtAsyncExecutionLimiter::Permit permit) {
                               {
                                 absl::MutexLock lock(mu);
                                 permits[i] = std::move(permit);
                               }
                               granted[i].Notify();
                             });
  }
  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 4);

  // Releasing the single 1500-byte permit frees all 1000 bytes of budget,
  // waking the first 3 x 300-byte waiters (900 bytes <= 1000) in one release
  // while leaving the 4th 300-byte waiter queued!
  large->Release();
  granted[0].WaitForNotification();
  granted[1].WaitForNotification();
  granted[2].WaitForNotification();

  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 3);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 900);
  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 1);
  EXPECT_FALSE(granted[3].HasBeenNotified());

  {
    absl::MutexLock lock(mu);
    permits[0].Release();
  }
  granted[3].WaitForNotification();
  {
    absl::MutexLock lock(mu);
    for (int i = 1; i < kNumSmall; ++i) {
      permits[i].Release();
    }
  }
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 0);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 0);
  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 0);
}

TEST(IfrtAsyncExecutionLimiterTest,
     RunWhenPermittedEnforcesCapAndStrictFifoWithoutBlockingCaller) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "limiter_test",
                               /*num_threads=*/8);
  IfrtAsyncExecutionLimiter limiter(/*max_bytes_per_device=*/100);
  const DeviceId d0(0);

  // Hold two 50-byte permits initially via TryAcquire (100 / 100 bytes).
  std::optional<IfrtAsyncExecutionLimiter::Permit> initial_1 =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/50);
  std::optional<IfrtAsyncExecutionLimiter::Permit> initial_2 =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/50);
  ASSERT_TRUE(initial_1.has_value());
  ASSERT_TRUE(initial_2.has_value());

  constexpr int kNumQueued = 5;
  absl::Mutex order_mu;
  std::vector<int> grant_order;
  std::vector<IfrtAsyncExecutionLimiter::Permit> held_permits(kNumQueued);
  std::vector<absl::Notification> granted(kNumQueued);

  // All 5 RunWhenPermitted calls must return immediately without blocking.
  for (int i = 0; i < kNumQueued; ++i) {
    limiter.RunWhenPermitted({d0}, /*bytes_per_device=*/50, pool,
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
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 2);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 100);
  EXPECT_FALSE(granted[0].HasBeenNotified());

  // While waiters are queued, a fresh TryAcquire must NOT barge in ahead of
  // FIFO waiters even as permits are released.
  initial_1->Release();
  granted[0].WaitForNotification();
  EXPECT_THAT(limiter.TryAcquire({d0}, /*bytes_per_device=*/10),
              Eq(std::nullopt));

  initial_2->Release();
  granted[1].WaitForNotification();
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 2);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 100);
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
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 0);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 0);
  EXPECT_EQ(limiter.QueueDepthForDevice(d0), 0);
}

TEST(IfrtAsyncExecutionLimiterTest,
     MultiDeviceAcquisitionCanonicalizesOrderWithoutDeadlock) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "spmd_limiter_test",
                               /*num_threads=*/8);
  tsl::thread::ThreadPool callers(tsl::Env::Default(), "spmd_callers",
                                  /*num_threads=*/8);
  IfrtAsyncExecutionLimiter limiter(/*max_bytes_per_device=*/100);

  const DeviceId d0(0);
  const DeviceId d1(1);

  constexpr int kNumRequests = 40;
  std::atomic<int> active_reqs{0};
  std::atomic<int64_t> active_bytes{0};
  std::atomic<int> max_reqs_when_oversized{0};
  std::atomic<int64_t> peak_fitting_bytes{0};
  absl::BlockingCounter done(kNumRequests);

  for (int i = 0; i < kNumRequests; ++i) {
    callers.Schedule([&, i]() {
      // Mix small (40B), medium (70B), and oversized (250B > 100B) requests
      // across {d0, d1} and {d1, d0, d1}; canonicalization must prevent AB-BA
      // deadlocks and double-counting.
      std::vector<DeviceId> req_devices =
          (i % 2 == 0) ? std::vector<DeviceId>{d0, d1}
                       : std::vector<DeviceId>{d1, d0, d1};
      const int64_t req_bytes = (i % 7 == 0) ? 250 : ((i % 3 == 0) ? 70 : 40);
      limiter.RunWhenPermitted(
          req_devices, req_bytes, pool,
          [&, req_bytes](IfrtAsyncExecutionLimiter::Permit permit) {
            const int cur_reqs = active_reqs.fetch_add(1) + 1;
            const int64_t cur_bytes =
                active_bytes.fetch_add(req_bytes) + req_bytes;
            if (req_bytes > 100) {
              int prev = max_reqs_when_oversized.load();
              while (cur_reqs > prev &&
                     !max_reqs_when_oversized.compare_exchange_weak(prev,
                                                                    cur_reqs)) {
              }
            } else {
              int64_t prev_b = peak_fitting_bytes.load();
              while (cur_bytes > prev_b &&
                     !peak_fitting_bytes.compare_exchange_weak(prev_b,
                                                               cur_bytes)) {
              }
            }
            tsl::Env::Default()->SleepForMicroseconds(500);
            active_bytes.fetch_sub(req_bytes);
            active_reqs.fetch_sub(1);
            permit.Release();
            done.DecrementCount();
          });
    });
  }

  done.Wait();
  EXPECT_LE(max_reqs_when_oversized.load(), 1);
  EXPECT_LE(peak_fitting_bytes.load(), 100);
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 0);
  EXPECT_EQ(limiter.InflightRequestsForDevice(d1), 0);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 0);
  EXPECT_EQ(limiter.InflightBytesForDevice(d1), 0);
}

TEST(IfrtAsyncExecutionLimiterTest,
     EmptyDevicesAndNegativeBytesHandledCleanly) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "empty_test",
                               /*num_threads=*/2);
  IfrtAsyncExecutionLimiter limiter(/*max_bytes_per_device=*/10);
  const DeviceId d0(0);

  std::optional<IfrtAsyncExecutionLimiter::Permit> p =
      limiter.TryAcquire({}, /*bytes_per_device=*/50);
  ASSERT_TRUE(p.has_value());
  EXPECT_FALSE(static_cast<bool>(*p));

  bool ran_inline = false;
  limiter.RunWhenPermitted({}, /*bytes_per_device=*/50, pool,
                           [&](IfrtAsyncExecutionLimiter::Permit permit) {
                             EXPECT_FALSE(static_cast<bool>(permit));
                             ran_inline = true;
                           });
  EXPECT_TRUE(ran_inline);

  // Negative byte estimates are safely clamped to 0.
  std::optional<IfrtAsyncExecutionLimiter::Permit> neg =
      limiter.TryAcquire({d0}, /*bytes_per_device=*/-42);
  ASSERT_TRUE(neg.has_value());
  EXPECT_EQ(limiter.InflightRequestsForDevice(d0), 1);
  EXPECT_EQ(limiter.InflightBytesForDevice(d0), 0);
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
