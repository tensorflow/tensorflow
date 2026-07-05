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

#include "xla/pjrt/stream_executor_platform_id_mapping.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/stream_executor/platform_id.h"

namespace xla {
namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;

PLATFORM_DEFINE_ID(kTestPlatformId, test_platform);
constexpr PjRtPlatformId kTestPjRtPlatformId = 123;

PLATFORM_DEFINE_ID(kTestPlatformId2, test_platform2);
constexpr PjRtPlatformId kTestPjRtPlatformId2 = 456;

TEST(StreamExecutorPlatformIdMappingTest, SingleRegisteredMapping) {
  StreamExecutorPlatformIdMapping mapping;
  EXPECT_OK(mapping.AddMapping(kTestPlatformId, kTestPjRtPlatformId));

  EXPECT_THAT(mapping.GetPjRtPlatformId(kTestPlatformId),
              IsOkAndHolds(kTestPjRtPlatformId));
  EXPECT_THAT(mapping.GetStreamExecutorPlatformId(kTestPjRtPlatformId),
              IsOkAndHolds(kTestPlatformId));

  EXPECT_THAT(mapping.GetPjRtPlatformId(kTestPlatformId2),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(mapping.GetStreamExecutorPlatformId(kTestPjRtPlatformId2),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(StreamExecutorPlatformIdMappingTest, MultipleRegisteredMappings) {
  StreamExecutorPlatformIdMapping mapping;
  EXPECT_OK(mapping.AddMapping(kTestPlatformId, kTestPjRtPlatformId));
  EXPECT_OK(mapping.AddMapping(kTestPlatformId2, kTestPjRtPlatformId2));

  EXPECT_THAT(mapping.GetPjRtPlatformId(kTestPlatformId),
              IsOkAndHolds(kTestPjRtPlatformId));
  EXPECT_THAT(mapping.GetStreamExecutorPlatformId(kTestPjRtPlatformId),
              IsOkAndHolds(kTestPlatformId));
  EXPECT_THAT(mapping.GetPjRtPlatformId(kTestPlatformId2),
              IsOkAndHolds(kTestPjRtPlatformId2));
  EXPECT_THAT(mapping.GetStreamExecutorPlatformId(kTestPjRtPlatformId2),
              IsOkAndHolds(kTestPlatformId2));
}

}  // namespace
}  // namespace xla
