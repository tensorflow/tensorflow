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

#include "xla/pjrt/device_event_utils.h"

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/pjrt/device_event.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace {

TEST(DeviceEventUtilsTest, GetErrors) {
  auto [promise1, future1] = tsl::MakePromise<bool>();
  auto error_av =
      tsl::MakeErrorAsyncValueRef(absl::InternalError("test error"));

  PjRtDeviceEventRef ev1 =
      PjRtDeviceEventPtr::FromAsyncValue(future1.async_value()).CopyRef();
  PjRtDeviceEventRef ev2 =
      PjRtDeviceEventPtr::FromAsyncValue(error_av.get()).CopyRef();

  std::vector<PjRtDeviceEventRef> events;
  events.push_back(std::move(ev1));
  events.push_back(std::move(ev2));

  promise1.Set(true);

  absl::Status status = xla::GetErrors(events);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "test error");
}

TEST(DeviceEventUtilsTest, RunWhenReady) {
  auto av1 = tsl::MakeUnconstructedAsyncValueRef<bool>();
  auto av2 = tsl::MakeUnconstructedAsyncValueRef<bool>();

  PjRtDeviceEventRef ev1 =
      PjRtDeviceEventPtr::FromAsyncValue(av1.GetAsyncValue()).CopyRef();
  PjRtDeviceEventRef ev2 =
      PjRtDeviceEventPtr::FromAsyncValue(av2.GetAsyncValue()).CopyRef();

  std::vector<PjRtDeviceEventRef> events;
  events.push_back(std::move(ev1));
  events.push_back(std::move(ev2));

  bool run = false;
  RunWhenReady(events, [&run]() { run = true; });

  EXPECT_FALSE(run);

  av1.emplace(true);
  EXPECT_FALSE(run);
  av2.emplace(true);
  EXPECT_TRUE(run);
}

TEST(DeviceEventUtilsTest, ExecuteWhenReady) {
  auto av1 = tsl::MakeUnconstructedAsyncValueRef<bool>();
  PjRtDeviceEventRef ev1 =
      PjRtDeviceEventPtr::FromAsyncValue(av1.GetAsyncValue()).CopyRef();
  std::vector<PjRtDeviceEventRef> events;
  events.push_back(std::move(ev1));

  auto& executor = tsl::InlineExecutor::Instance();
  bool run = false;
  ExecuteWhenReady(events, &executor, [&run]() { run = true; });

  EXPECT_FALSE(run);
  av1.emplace(true);
  EXPECT_TRUE(run);
}

}  // namespace
}  // namespace xla
