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

TEST(HangWatchdogTest, StressTest) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "stress", 8);
  HangWatchdog watchdog(tsl::Env::Default(), "watchdog");

  for (size_t i = 0; i < 1000; ++i) {
    watchdog.Watch("test", absl::Milliseconds(1), [] {});
  }
}

}  // namespace
}  // namespace xla
