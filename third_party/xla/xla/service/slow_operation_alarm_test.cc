/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/slow_operation_alarm.h"

#include <thread>  // NOLINT

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(SlowOperationAlarmTest, MsgFnIsCalledOnlyOnce) {
  absl::Notification alarm_fired;

  SlowOperationAlarm alarm(absl::ZeroDuration(), [&] {
    CHECK(!alarm_fired.HasBeenNotified());
    alarm_fired.Notify();
    return "msg";
  });
  alarm_fired.WaitForNotification();
  // The destructor of the alarm should not call the message function.
}

TEST(SlowOperationAlarmTest, CancelIsNotRacy) {
  for (int i = 0; i < 10000; ++i) {
    absl::Notification alarm_fired;
    absl::Notification alarm_may_continue;

    std::thread unblock_alarm_thread([&] {
      alarm_fired.WaitForNotification();
      alarm_may_continue.Notify();
    });

    bool canceled = false;
    {
      SlowOperationAlarm alarm(absl::ZeroDuration(), [&] {
        alarm_fired.Notify();
        alarm_may_continue.WaitForNotification();
        canceled = true;
        return "msg";
      });

      alarm_fired.WaitForNotification();
      alarm.cancel();
      EXPECT_TRUE(alarm.fired());
    }

    EXPECT_TRUE(canceled);
    unblock_alarm_thread.join();
  }
}

}  // namespace
}  // namespace xla
