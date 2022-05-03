/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/distributed_runtime/preemption/preemption_notifier.h"

#include <csignal>
#include <functional>

#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
TEST(PreemptNotifierTest, WillBePreemptedAt) {
  auto env = Env::Default();
  std::unique_ptr<PreemptionNotifier> preempt_notifier =
      CreateSigtermNotifier(env);
  absl::Time start_time = absl::Now();
  env->SchedClosureAfter(/*micros=*/absl::ToInt64Microseconds(absl::Seconds(1)),
                         []() { std::raise(SIGTERM); });

  // Preempt time should be current timestamp.
  absl::Time preempt_time = preempt_notifier->WillBePreemptedAt();

  // Make sure that preempt time is approximately correct.
  absl::Duration time_diff = preempt_time - start_time;
  // Signal was raised 1 second after start time.
  EXPECT_GT(time_diff, absl::Seconds(1));
  // Listen to signal once per second, so we should catch within 2 seconds.
  EXPECT_LT(time_diff, absl::Seconds(3));
}

TEST(PreemptNotifierTest,
     WillBePreemptedAt_AlreadyPreempted_ReturnsImmediately) {
  auto env = Env::Default();
  std::unique_ptr<PreemptionNotifier> preempt_notifier =
      CreateSigtermNotifier(env);
  absl::Time start_time = absl::Now();
  std::raise(SIGTERM);
  // Note: sleep for a while to ensure that (a) SIGTERM is fully handled and (b)
  // the correct death time is returned (time when signal is caught instead of
  // when WillBePreemptedAt() is called).
  env->SleepForMicroseconds(absl::ToInt64Microseconds(absl::Seconds(2)));

  // Preempt time should be current timestamp.
  absl::Time preempt_time = preempt_notifier->WillBePreemptedAt();

  // Make sure that preempt time is approximately correct.
  absl::Duration time_diff = preempt_time - start_time;
  // Signal was raised immediately after start time.
  EXPECT_GT(time_diff, absl::ZeroDuration());
  // Verify that preempt time is not set at the time of call (2 seconds after
  // signal was raised).
  EXPECT_LT(time_diff, absl::Seconds(2));
}

TEST(PreemptNotifierTest, WillBePreemptedAtAsync_SameResultForAllCallbacks) {
  auto env = Env::Default();
  std::unique_ptr<PreemptionNotifier> preempt_notifier =
      CreateSigtermNotifier(env);
  env->SchedClosureAfter(/*micros=*/absl::ToInt64Microseconds(absl::Seconds(1)),
                         []() { std::raise(SIGTERM); });

  // Preempt time should be current timestamp.
  absl::Time preempt_time;
  absl::Time preempt_time_2;
  absl::Notification n;
  absl::Notification n_2;
  preempt_notifier->WillBePreemptedAtAsync(
      [&preempt_time, &n](absl::Time time) {
        preempt_time = time;
        n.Notify();
      });
  preempt_notifier->WillBePreemptedAtAsync(
      [&preempt_time_2, &n_2](absl::Time time) {
        preempt_time_2 = time;
        n_2.Notify();
      });
  n.WaitForNotification();
  n_2.WaitForNotification();

  // Make sure that the same preempt time is returned for both calls.
  EXPECT_EQ(preempt_time, preempt_time_2);
}

TEST(PreemptNotifierTest, Reset_TwoDifferentPreemptTimesRecorded) {
  auto env = Env::Default();
  std::unique_ptr<PreemptionNotifier> preempt_notifier =
      CreateSigtermNotifier(env);

  // Raise first signal.
  std::raise(SIGTERM);
  absl::Time preempt_time = preempt_notifier->WillBePreemptedAt();

  preempt_notifier->Reset();

  // Raise second signal.
  std::raise(SIGTERM);
  absl::Time preempt_time_2 = preempt_notifier->WillBePreemptedAt();

  // Verify that two different preempt times are recorded.
  EXPECT_NE(preempt_time, preempt_time_2);
}

}  // namespace
}  // namespace tensorflow
