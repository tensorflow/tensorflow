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

#include "xla/runtime/hang_watchdog.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

TEST(HangWatchdogTest, Cancelled) {
  HangWatchdog watchdog(tsl::Env::Default(), "watchdog");

  absl::Notification notification;

  // Hold the guard to simulate an action that is still in progress.
  auto guard = watchdog.Watch("test", absl::Milliseconds(1),
                              [&] { notification.Notify(); });

  notification.WaitForNotification();
}

TEST(HangWatchdogTest, Completed) {
  HangWatchdog watchdog(tsl::Env::Default(), "watchdog");

  absl::Notification notification;

  {  // Immediately destroy the guard to signal the action completed.
    auto guard = watchdog.Watch("test", absl::Milliseconds(100),
                                [&] { notification.Notify(); });
  }

  // Wait longer than the timeout to verify cancel was NOT called.
  EXPECT_FALSE(notification.WaitForNotificationWithTimeout(absl::Seconds(1)));
}

// Stress test HangWatchdog under concurrent execution to detect data races.
class HangWatchdogStressTest : public ::testing::TestWithParam<int32_t> {};

TEST_P(HangWatchdogStressTest, StressTest) {
  std::optional<tsl::thread::ThreadPool> pool;
  pool.emplace(tsl::Env::Default(), "stress", 8);

  std::optional<HangWatchdog> watchdog;
  watchdog.emplace(tsl::Env::Default(), "watchdog", GetParam());

  static constexpr size_t n = 1000;
  std::vector<std::shared_ptr<HangWatchdog::Guard>> guards(n);

  absl::BlockingCounter counter(n);
  for (size_t i = 0; i < n; ++i) {
    pool->Schedule([&, i] {
      guards[i] = watchdog->Watch("test", absl::Milliseconds(1),
                                  [&] { counter.DecrementCount(); });
    });
  }

  // Wait for all tasks to finish.
  (pool.reset(), watchdog.reset());
  counter.Wait();
}

INSTANTIATE_TEST_SUITE_P(HangWatchdogStress, HangWatchdogStressTest,
                         ::testing::ValuesIn({1, 2, 4, 8}));

}  // namespace
}  // namespace xla
