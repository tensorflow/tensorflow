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

#include <cstddef>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(Model, EmptyAssignment) {
  ObjectsAssignment<size_t> objects_assignment;
  OffsetsAssignment result = ObjectsToOffsets(objects_assignment);
  EXPECT_TRUE(result.offsets.empty());
  EXPECT_EQ(result.total_size, 0);
}

TEST(Model, OneObjectAssignment) {
  ObjectsAssignment<size_t> objects_assignment;
  objects_assignment.object_sizes = {16};
  objects_assignment.object_ids = {0};
  OffsetsAssignment result = ObjectsToOffsets(objects_assignment);
  EXPECT_EQ(result.total_size, 16);
  EXPECT_THAT(result.offsets, ElementsAre(0));

  objects_assignment.object_ids = {0, 0, 0};
  result = ObjectsToOffsets(objects_assignment);
  EXPECT_EQ(result.total_size, 16);
  EXPECT_THAT(result.offsets, ElementsAre(0, 0, 0));
}

TEST(Model, ManyObjectsAssignment) {
  ObjectsAssignment<size_t> objects_assignment;
  objects_assignment.object_sizes = {16, 8, 32, 32, 4, 16};
  objects_assignment.object_ids = {2, 0, 2, 1, 3, 3, 1, 5};
  OffsetsAssignment result = ObjectsToOffsets(objects_assignment);
  EXPECT_THAT(result.offsets, ElementsAre(24, 0, 24, 16, 56, 56, 16, 92));
}

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
      AssignObjectsToTensors({}, MemoryStrategy::GREEDY_IN_ORDER, &assignment)
          .ok());
  EXPECT_TRUE(assignment.object_ids.empty());
  EXPECT_TRUE(assignment.object_sizes.empty());

  ASSERT_TRUE(
      AssignObjectsToTensors({}, MemoryStrategy::MINCOSTFLOW, &assignment)
          .ok());
  EXPECT_TRUE(assignment.object_ids.empty());
  EXPECT_TRUE(assignment.object_sizes.empty());

  ASSERT_TRUE(
      AssignObjectsToTensors({}, MemoryStrategy::GREEDY_BY_BREADTH, &assignment)
          .ok());
  EXPECT_TRUE(assignment.object_ids.empty());
  EXPECT_TRUE(assignment.object_sizes.empty());

  ASSERT_TRUE(
      AssignObjectsToTensors({}, MemoryStrategy::GREEDY_BY_SIZE, &assignment)
          .ok());
  EXPECT_TRUE(assignment.object_ids.empty());
  EXPECT_TRUE(assignment.object_sizes.empty());

  OffsetsAssignment offsets_assignment;
  ASSERT_TRUE(AssignOffsetsToTensors({}, MemoryStrategy::GREEDY_BY_SIZE,
                                     &offsets_assignment)
                  .ok());
  EXPECT_TRUE(offsets_assignment.offsets.empty());
  EXPECT_EQ(offsets_assignment.total_size, 0);
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

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_IN_ORDER, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::MINCOSTFLOW,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_BY_BREADTH, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_BY_SIZE, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(16));

  OffsetsAssignment offsets_assignment;
  ASSERT_TRUE(AssignOffsetsToTensors(usage_records,
                                     MemoryStrategy::GREEDY_BY_SIZE,
                                     &offsets_assignment)
                  .ok());
  EXPECT_THAT(offsets_assignment.offsets, ElementsAre(0));
  EXPECT_EQ(offsets_assignment.total_size, 16);
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

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::MINCOSTFLOW,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 1, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(64, 32));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_IN_ORDER, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 1, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(64, 32));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_BY_BREADTH, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 1, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(64, 32));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_BY_SIZE, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 1, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(64, 32));

  OffsetsAssignment offsets_assignment;
  ASSERT_TRUE(AssignOffsetsToTensors(usage_records,
                                     MemoryStrategy::GREEDY_BY_SIZE,
                                     &offsets_assignment)
                  .ok());
  EXPECT_THAT(offsets_assignment.offsets, ElementsAre(0, 64, 0, 64, 0));
  EXPECT_EQ(offsets_assignment.total_size, 96);
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

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::MINCOSTFLOW,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 0, 3, 1, 3, 2, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(32, 64, 8, 8));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_IN_ORDER, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 2, 3, 1, 3, 2, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(32, 64, 16, 8));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_BY_BREADTH, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 4, 2, 1, 3, 0, 2, 3, 1));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(64, 16, 8, 8, 32));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_BY_SIZE, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(1, 0, 2, 1, 3, 0, 1, 2, 0));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(64, 32, 8, 8));

  OffsetsAssignment offsets_assignment;
  ASSERT_TRUE(AssignOffsetsToTensors(usage_records,
                                     MemoryStrategy::GREEDY_BY_SIZE,
                                     &offsets_assignment)
                  .ok());
  EXPECT_THAT(offsets_assignment.offsets,
              ElementsAre(0, 32, 80, 64, 88, 0, 64, 72, 0));
  EXPECT_EQ(offsets_assignment.total_size, 96);
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

TEST(Model, UInt2Records) {
  std::vector<TensorUsageRecord<uint2>> usage_records{
      {/*size=*/uint2(2, 8), /*first=*/0, /*last=*/1},
      {/*size=*/uint2(2, 8), /*first=*/1, /*last=*/2},
      {/*size=*/uint2(1, 12), /*first=*/2, /*last=*/4},
      {/*size=*/uint2(2, 8), /*first=*/3, /*last=*/5},
      {/*size=*/uint2(8, 2), /*first=*/4, /*last=*/5},
      {/*size=*/uint2(2, 8), /*first=*/5, /*last=*/7},
      {/*size=*/uint2(1, 8), /*first=*/6, /*last=*/8},
      {/*size=*/uint2(2, 8), /*first=*/7, /*last=*/8},
      {/*size=*/uint2(4, 1), /*first=*/8, /*last=*/9}};

  ObjectsAssignment<uint2> assignment;
  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::NAIVE, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT(assignment.object_sizes,
              ElementsAre(uint2(2, 8), uint2(2, 8), uint2(1, 12), uint2(2, 8),
                          uint2(8, 2), uint2(2, 8), uint2(1, 8), uint2(2, 8),
                          uint2(4, 1)));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::EQUALITY,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 0, 3, 1, 4, 0, 5));
  EXPECT_THAT(assignment.object_sizes,
              ElementsAre(uint2(2, 8), uint2(2, 8), uint2(1, 12), uint2(8, 2),
                          uint2(1, 8), uint2(4, 1)));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_IN_ORDER, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 0, 3, 1, 2, 0, 3));
  EXPECT_THAT(assignment.object_sizes,
              ElementsAre(uint2(2, 8), uint2(2, 8), uint2(1, 12), uint2(8, 2)));
}

TEST(Model, UInt3Records) {
  std::vector<TensorUsageRecord<uint3>> usage_records{
      {/*size=*/uint3(1, 2, 8), /*first=*/0, /*last=*/1},
      {/*size=*/uint3(4, 3, 2), /*first=*/1, /*last=*/2},
      {/*size=*/uint3(1, 1, 1), /*first=*/2, /*last=*/4},
      {/*size=*/uint3(2, 4, 1), /*first=*/3, /*last=*/5},
      {/*size=*/uint3(2, 2, 2), /*first=*/4, /*last=*/5},
      {/*size=*/uint3(8, 1, 2), /*first=*/5, /*last=*/7},
      {/*size=*/uint3(1, 2, 1), /*first=*/6, /*last=*/8},
      {/*size=*/uint3(1, 1, 1), /*first=*/7, /*last=*/8},
      {/*size=*/uint3(2, 2, 2), /*first=*/8, /*last=*/9}};

  ObjectsAssignment<uint3> assignment;
  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::NAIVE, &assignment)
          .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT(assignment.object_sizes,
              ElementsAre(uint3(1, 2, 8), uint3(4, 3, 2), uint3(1, 1, 1),
                          uint3(2, 4, 1), uint3(2, 2, 2), uint3(8, 1, 2),
                          uint3(1, 2, 1), uint3(1, 1, 1), uint3(2, 2, 2)));

  ASSERT_TRUE(AssignObjectsToTensors(usage_records, MemoryStrategy::EQUALITY,
                                     &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 3, 4, 5, 6, 2, 4));
  EXPECT_THAT(assignment.object_sizes,
              ElementsAre(uint3(1, 2, 8), uint3(4, 3, 2), uint3(1, 1, 1),
                          uint3(2, 4, 1), uint3(2, 2, 2), uint3(8, 1, 2),
                          uint3(1, 2, 1)));

  ASSERT_TRUE(AssignObjectsToTensors(
                  usage_records, MemoryStrategy::GREEDY_IN_ORDER, &assignment)
                  .ok());
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 0, 2, 1, 3, 2, 0, 1));
  EXPECT_THAT(assignment.object_sizes,
              ElementsAre(uint3(1, 2, 8), uint3(4, 3, 2), uint3(2, 4, 1),
                          uint3(8, 1, 2)));
}

TEST(Model, OffsetAssignmentWithAlignment) {
  std::vector<TensorUsageRecord<size_t>> usage_records{
      {/*size=*/16, /*first=*/0, /*last=*/1},
      {/*size=*/8, /*first=*/1, /*last=*/2},
      {/*size=*/64, /*first=*/2, /*last=*/3},
      {/*size=*/32, /*first=*/3, /*last=*/4},
      {/*size=*/8, /*first=*/4, /*last=*/5},
  };

  OffsetsAssignment offsets_assignment;
  ASSERT_TRUE(AssignOffsetsToTensors(usage_records,
                                     MemoryStrategy::GREEDY_BY_SIZE,
                                     &offsets_assignment,
                                     /*base_addr_align_bytes=*/128)
                  .ok());
  EXPECT_THAT(offsets_assignment.offsets, ElementsAre(0, 128, 0, 128, 0));
  EXPECT_EQ(offsets_assignment.total_size, 160);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
