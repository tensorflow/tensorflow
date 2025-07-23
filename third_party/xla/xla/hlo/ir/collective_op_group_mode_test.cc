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

#include "xla/hlo/ir/collective_op_group_mode.h"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

TEST(CollectiveOpGroupModeTest, ToString) {
  EXPECT_EQ(CollectiveOpGroupModeToString(CollectiveOpGroupMode::kCrossReplica),
            "cross_replica");
  EXPECT_EQ(
      CollectiveOpGroupModeToString(CollectiveOpGroupMode::kCrossPartition),
      "cross_partition");
  EXPECT_EQ(CollectiveOpGroupModeToString(
                CollectiveOpGroupMode::kCrossReplicaAndPartition),
            "cross_replica_and_partition");
  EXPECT_EQ(CollectiveOpGroupModeToString(CollectiveOpGroupMode::kFlattenedID),
            "flattened_id");
}

TEST(CollectiveOpGroupModeTest, FromString) {
  EXPECT_EQ(StringToCollectiveOpGroupMode("cross_replica").value(),
            CollectiveOpGroupMode::kCrossReplica);
  EXPECT_EQ(StringToCollectiveOpGroupMode("cross_partition").value(),
            CollectiveOpGroupMode::kCrossPartition);
  EXPECT_EQ(
      StringToCollectiveOpGroupMode("cross_replica_and_partition").value(),
      CollectiveOpGroupMode::kCrossReplicaAndPartition);
  EXPECT_EQ(StringToCollectiveOpGroupMode("flattened_id").value(),
            CollectiveOpGroupMode::kFlattenedID);
}

// Tests for GetCollectOpGroupMode
namespace GetCollectiveOpGroupModeTest {
struct TestCase {
  bool has_channel_id;
  std::optional<bool> use_global_device_ids;
  std::optional<xla::CollectiveOpGroupMode> expected;

  std::string ToString() const {
    std::ostringstream s;
    s << (has_channel_id ? "chnl" : "nochnl");
    s << "_"
      << (use_global_device_ids
              ? (*use_global_device_ids ? "ugdi_true" : "ugdi_false")
              : "nougdi");
    return s.str();
  }
};

std::vector<TestCase> GetTestCases() {
  const std::vector<TestCase> test_cases = {
      // clang-format off
      // has_channel_id, use_global_device_ids, expected mode
      {false, std::nullopt,  CollectiveOpGroupMode::kCrossReplica},
      {false, false,         CollectiveOpGroupMode::kCrossReplica},
      {false, true,          std::nullopt},
      {true,  std::nullopt,  CollectiveOpGroupMode::kCrossPartition},
      {true,  false,         CollectiveOpGroupMode::kCrossReplicaAndPartition},
      {true,  true,          CollectiveOpGroupMode::kFlattenedID},
      // clang-format on
  };
  return test_cases;
}

class GetCollectOpGroupModeTest : public testing::TestWithParam<TestCase> {};

TEST_P(GetCollectOpGroupModeTest, Test) {
  const TestCase &tc = GetParam();
  absl::StatusOr<CollectiveOpGroupMode> actual =
      GetCollectiveOpGroupMode(tc.has_channel_id, tc.use_global_device_ids);
  if (tc.expected) {
    TF_ASSERT_OK(actual.status());
    EXPECT_EQ(*actual, *tc.expected);
  } else {
    EXPECT_FALSE(actual.ok());
  }
}

INSTANTIATE_TEST_SUITE_P(GetCollectOpGroupMode, GetCollectOpGroupModeTest,
                         testing::ValuesIn(GetTestCases()));
}  // namespace GetCollectiveOpGroupModeTest
}  // namespace
}  // namespace xla
