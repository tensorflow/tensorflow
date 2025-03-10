/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/tfrt/gpu_event.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace {

using ::tsl::testing::StatusIs;

TEST(GpuEventTest, AfterAllEmpty) { EXPECT_TRUE(AfterAll({}).IsAvailable()); }

TEST(GpuEventTest, AfterAllSingle) {
  auto event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto after_all = AfterAll({event});
  EXPECT_FALSE(after_all.IsAvailable());
  event.SetStateConcrete();
  EXPECT_TRUE(after_all.IsAvailable());
  EXPECT_EQ(after_all.GetAsyncValue(), event.GetAsyncValue());
}

TEST(GpuEventTest, AfterAllMultiple) {
  auto event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto event2 = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto event3 = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto after_all = AfterAll({event, event2, event3});
  EXPECT_FALSE(after_all.IsAvailable());
  event.SetStateConcrete();
  EXPECT_FALSE(after_all.IsAvailable());
  event2.SetStateConcrete();
  EXPECT_FALSE(after_all.IsAvailable());
  event3.SetStateConcrete();
  EXPECT_TRUE(after_all.IsAvailable());
}

TEST(GpuEventTest, AfterAllError) {
  auto event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto event2 = tsl::MakeErrorAsyncValueRef(absl::InternalError("error"));
  auto event3 = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto after_all = AfterAll({event, event2, event3});
  EXPECT_FALSE(after_all.IsAvailable());
  event.SetStateConcrete();
  EXPECT_FALSE(after_all.IsAvailable());
  event3.SetStateConcrete();
  EXPECT_TRUE(after_all.IsAvailable());
  EXPECT_THAT(after_all.GetError(),
              StatusIs(absl::StatusCode::kInternal, "error"));
}

TEST(TfrtEventSetTest, AfterAllEmpty) {
  TfrtEventSet event_set;
  auto after_all = event_set.AfterAll();
  EXPECT_TRUE(after_all.IsAvailable());
}

TEST(TfrtEventSetTest, AfterAllSingle) {
  TfrtEventSet event_set;
  auto event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  event_set.Add(event);
  auto after_all = event_set.AfterAll();
  EXPECT_FALSE(after_all.IsAvailable());
  event.SetStateConcrete();
  EXPECT_TRUE(after_all.IsAvailable());
  EXPECT_EQ(after_all.GetAsyncValue(), event.GetAsyncValue());
}

TEST(TfrtEventSetTest, AfterAllMultiple) {
  TfrtEventSet event_set;
  auto event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto event2 = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  event_set.Add(event);
  EXPECT_EQ(event_set.size(), 1);
  auto after_all = event_set.AfterAll();
  EXPECT_FALSE(after_all.IsAvailable());
  event.SetStateConcrete();
  EXPECT_TRUE(after_all.IsAvailable());
  event_set.Add(event2);
  EXPECT_EQ(event_set.size(), 2);
  auto after_all2 = event_set.AfterAll();
  EXPECT_FALSE(after_all2.IsAvailable());
  event2.SetStateConcrete();
  EXPECT_TRUE(after_all2.IsAvailable());
  event_set.Clear();
}

TEST(TfrtEventSetTest, ClearEvents) {
  TfrtEventSet event_set;
  auto event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  EXPECT_EQ(event_set.size(), 0);
  event_set.Add(event);
  EXPECT_EQ(event_set.size(), 1);
  event_set.Clear();
  EXPECT_EQ(event_set.size(), 0);
}

}  // namespace
}  // namespace xla
