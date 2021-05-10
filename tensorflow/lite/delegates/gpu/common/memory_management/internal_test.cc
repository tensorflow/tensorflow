/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"

#include <cstddef>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"

namespace tflite {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(TaskProfileTest, EmptyRecords) {
  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles({});
  EXPECT_TRUE(task_profiles.empty());
  std::vector<size_t> positional_max = CalculatePositionalMaximums({});
  EXPECT_TRUE(positional_max.empty());
}

TEST(TaskProfileTest, OneRecord) {
  std::vector<TensorUsageRecord<size_t>> usage_records{
      {/*size=*/16, /*first=*/0, /*last=*/1}};
  const std::vector<std::vector<size_t>> correct_idx = {{0}, {0}};
  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles(usage_records);
  ASSERT_EQ(task_profiles.size(), correct_idx.size());
  for (size_t i = 0; i < task_profiles.size(); ++i) {
    ASSERT_EQ(task_profiles[i].size(), correct_idx[i].size());
    for (size_t j = 0; j < task_profiles[i].size(); ++j) {
      ASSERT_EQ(task_profiles[i][j].usage_record,
                &usage_records[correct_idx[i][j]]);
      ASSERT_EQ(task_profiles[i][j].idx, correct_idx[i][j]);
    }
  }
  std::vector<size_t> positional_max =
      CalculatePositionalMaximums(usage_records);
  EXPECT_THAT(positional_max, ElementsAre(16));
}

TEST(TaskProfileTest, ChainRecords) {
  std::vector<TensorUsageRecord<size_t>> usage_records{
      {/*size=*/16, /*first=*/0, /*last=*/1},
      {/*size=*/8, /*first=*/1, /*last=*/2},
      {/*size=*/64, /*first=*/2, /*last=*/3},
      {/*size=*/32, /*first=*/3, /*last=*/4},
      {/*size=*/8, /*first=*/4, /*last=*/5},
  };
  const std::vector<std::vector<size_t>> correct_idx = {{0},    {0, 1}, {2, 1},
                                                        {2, 3}, {3, 4}, {4}};
  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles(usage_records);
  ASSERT_EQ(task_profiles.size(), correct_idx.size());
  for (size_t i = 0; i < task_profiles.size(); ++i) {
    ASSERT_EQ(task_profiles[i].size(), correct_idx[i].size());
    for (size_t j = 0; j < task_profiles[i].size(); ++j) {
      ASSERT_EQ(task_profiles[i][j].usage_record,
                &usage_records[correct_idx[i][j]]);
      ASSERT_EQ(task_profiles[i][j].idx, correct_idx[i][j]);
    }
  }
  std::vector<size_t> positional_max =
      CalculatePositionalMaximums(usage_records);
  EXPECT_THAT(positional_max, ElementsAre(64, 32));
}

TEST(TaskProfileTest, ComplexRecords) {
  std::vector<TensorUsageRecord<size_t>> usage_records{
      {/*size=*/32, /*first=*/0, /*last=*/1},
      {/*size=*/32, /*first=*/1, /*last=*/4},
      {/*size=*/8, /*first=*/2, /*last=*/5},
      {/*size=*/16, /*first=*/3, /*last=*/5},
      {/*size=*/8, /*first=*/4, /*last=*/5},
      {/*size=*/64, /*first=*/5, /*last=*/7},
      {/*size=*/8, /*first=*/6, /*last=*/8},
      {/*size=*/8, /*first=*/7, /*last=*/8},
      {/*size=*/16, /*first=*/8, /*last=*/9}};
  const std::vector<std::vector<size_t>> correct_idx = {
      {0},          {0, 1}, {1, 2},    {1, 3, 2}, {1, 3, 2, 4},
      {5, 3, 2, 4}, {5, 6}, {5, 6, 7}, {8, 6, 7}, {8}};
  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles(usage_records);
  ASSERT_EQ(task_profiles.size(), correct_idx.size());
  for (size_t i = 0; i < task_profiles.size(); ++i) {
    ASSERT_EQ(task_profiles[i].size(), correct_idx[i].size());
    for (size_t j = 0; j < task_profiles[i].size(); ++j) {
      ASSERT_EQ(task_profiles[i][j].usage_record,
                &usage_records[correct_idx[i][j]]);
      ASSERT_EQ(task_profiles[i][j].idx, correct_idx[i][j]);
    }
  }
  std::vector<size_t> positional_max =
      CalculatePositionalMaximums(usage_records);
  EXPECT_THAT(positional_max, ElementsAre(64, 32, 8, 8));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
