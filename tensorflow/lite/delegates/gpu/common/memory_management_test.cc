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

#include "tensorflow/lite/delegates/gpu/common/memory_management.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(Model, EmptyRecords) {
  ObjectsAssignment<size_t> assignment;
  ASSERT_TRUE(
      AssignObjectsToTensors({}, MemoryStrategy::NAIVE, &assignment).ok());
  EXPECT_TRUE(assignment.object_ids.empty());
  EXPECT_TRUE(assignment.object_sizes.empty());

  ASSERT_TRUE(
      AssignObjectsToTensors({}, MemoryStrategy::EQUALITY, &assignment).ok());
  EXPECT_TRUE(assignment.object_ids.empty());
  EXPECT_TRUE(assignment.object_sizes.empty());

  ASSERT_TRUE(
      AssignObjectsToTensors({}, MemoryStrategy::GREEDY, &assignment).ok());
  EXPECT_TRUE(assignment.object_ids.empty());
  EXPECT_TRUE(assignment.object_sizes.empty());

  ASSERT_TRUE(
      AssignObjectsToTensors({}, MemoryStrategy::MINCOSTFLOW, &assignment)
          .ok());
  EXPECT_TRUE(assignment.object_ids.empty());
  EXPECT_TRUE(assignment.object_sizes.empty());
}

TEST(Model, OneRecord) {
  std::vector<TensorUsageRecord<size_t>> usage_records{
      {/*size=*/16, /*first=*/0, /*last=*/1}};
  ObjectsAssignment<size_t> assignment;
  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::NAIVE, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::EQUALITY,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16));

  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::GREEDY, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::MINCOSTFLOW,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16));
}

TEST(Model, ChainRecords) {
  std::vector<TensorUsageRecord<size_t>> usage_records{
      {/*size=*/16, /*first=*/0, /*last=*/1},
      {/*size=*/8, /*first=*/1, /*last=*/2},
      {/*size=*/64, /*first=*/2, /*last=*/3},
      {/*size=*/32, /*first=*/3, /*last=*/4},
      {/*size=*/8, /*first=*/4, /*last=*/5},
  };
  ObjectsAssignment<size_t> assignment;
  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::NAIVE, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16, 8, 64, 32, 8));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::EQUALITY,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 3, 1));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16, 8, 64, 32));

  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::GREEDY, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 1, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(64, 32));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::MINCOSTFLOW,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 1, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(64, 32));
}

TEST(Model, ComplexRecords) {
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
  ObjectsAssignment<size_t> assignment;
  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::NAIVE, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT(assignment.object_sizes,
              ElementsAre(32, 32, 8, 16, 8, 64, 8, 8, 16));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::EQUALITY,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 3, 4, 5, 4, 2, 3));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(32, 32, 8, 16, 8, 64));

  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::GREEDY, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 2, 3, 1, 3, 2, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(32, 64, 16, 8));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::MINCOSTFLOW,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 0, 3, 1, 3, 2, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(32, 64, 8, 8));
}

TEST(Model, BHWCRecords) {
  std::vector<TensorUsageRecord<BHWC>> usage_records{
      {/*size=*/BHWC(1, 1, 2, 8), /*first=*/0, /*last=*/1},
      {/*size=*/BHWC(1, 1, 2, 8), /*first=*/1, /*last=*/2},
      {/*size=*/BHWC(1, 1, 1, 16), /*first=*/2, /*last=*/4},
      {/*size=*/BHWC(1, 1, 2, 8), /*first=*/3, /*last=*/5},
      {/*size=*/BHWC(1, 1, 8, 2), /*first=*/4, /*last=*/5},
      {/*size=*/BHWC(1, 1, 2, 8), /*first=*/5, /*last=*/7},
      {/*size=*/BHWC(1, 16, 1, 1), /*first=*/6, /*last=*/8},
      {/*size=*/BHWC(16, 1, 1, 1), /*first=*/7, /*last=*/8},
      {/*size=*/BHWC(1, 1, 1, 16), /*first=*/8, /*last=*/9}};

  ObjectsAssignment<BHWC> assignment;
  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::NAIVE, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT(
      assignment.object_sizes,
      ElementsAre(BHWC(1, 1, 2, 8), BHWC(1, 1, 2, 8), BHWC(1, 1, 1, 16),
                  BHWC(1, 1, 2, 8), BHWC(1, 1, 8, 2), BHWC(1, 1, 2, 8),
                  BHWC(1, 16, 1, 1), BHWC(16, 1, 1, 1), BHWC(1, 1, 1, 16)));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::EQUALITY,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 1, 3, 0, 4, 5, 2));
  EXPECT_THAT(
      assignment.object_sizes,
      ElementsAre(BHWC(1, 1, 2, 8), BHWC(1, 1, 2, 8), BHWC(1, 1, 1, 16),
                  BHWC(1, 1, 8, 2), BHWC(1, 16, 1, 1), BHWC(16, 1, 1, 1)));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
