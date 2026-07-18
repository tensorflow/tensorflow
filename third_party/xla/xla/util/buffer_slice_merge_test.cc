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

#include "xla/util/buffer_slice_merge.h"

#include <vector>

#include <gtest/gtest.h>
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace {

TEST(BufferSliceMergeTest, EmptySlices) {
  EXPECT_TRUE(MergeOverlappingSlices({}).empty());
}

TEST(BufferSliceMergeTest, SingleSlice) {
  BufferAllocation alloc(/*index=*/0, /*size=*/100, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/10, /*size=*/20);

  std::vector<BufferAllocation::Slice> merged =
      MergeOverlappingSlices({slice1});

  ASSERT_EQ(merged.size(), 1);
  EXPECT_EQ(merged[0], slice1);
}

TEST(BufferSliceMergeTest, MultipleDisjointSlices) {
  BufferAllocation alloc(/*index=*/0, /*size=*/100, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/10, /*size=*/20);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/40, /*size=*/10);
  BufferAllocation::Slice slice3(&alloc, /*offset=*/60, /*size=*/15);

  std::vector<BufferAllocation::Slice> merged =
      MergeOverlappingSlices({slice1, slice2, slice3});

  ASSERT_EQ(merged.size(), 3);
  EXPECT_EQ(merged[0], slice1);
  EXPECT_EQ(merged[1], slice2);
  EXPECT_EQ(merged[2], slice3);
}

TEST(BufferSliceMergeTest, OverlappingSlices) {
  BufferAllocation alloc(/*index=*/0, /*size=*/100, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/10, /*size=*/20);
  BufferAllocation::Slice slice_overlap1(&alloc, /*offset=*/15, /*size=*/20);

  std::vector<BufferAllocation::Slice> merged =
      MergeOverlappingSlices({slice1, slice_overlap1});

  ASSERT_EQ(merged.size(), 1);
  EXPECT_EQ(merged[0],
            BufferAllocation::Slice(&alloc, /*offset=*/10, /*size=*/25));
}

TEST(BufferSliceMergeTest, CompletelyNestedSlices) {
  BufferAllocation alloc(/*index=*/0, /*size=*/100, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/10, /*size=*/20);
  BufferAllocation::Slice slice_nested(&alloc, /*offset=*/12, /*size=*/5);

  std::vector<BufferAllocation::Slice> merged =
      MergeOverlappingSlices({slice1, slice_nested});

  ASSERT_EQ(merged.size(), 1);
  EXPECT_EQ(merged[0], slice1);
}

TEST(BufferSliceMergeTest, AdjacentSlices) {
  BufferAllocation alloc(/*index=*/0, /*size=*/100, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/10, /*size=*/20);
  BufferAllocation::Slice slice_adjacent(&alloc, /*offset=*/30, /*size=*/10);

  std::vector<BufferAllocation::Slice> merged =
      MergeOverlappingSlices({slice1, slice_adjacent});

  ASSERT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0], slice1);
  EXPECT_EQ(merged[1], slice_adjacent);
}

TEST(BufferSliceMergeTest, MultipleOverlappingAndDisjointSlices) {
  BufferAllocation alloc(/*index=*/0, /*size=*/100, /*color=*/0);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/10, /*size=*/20);
  BufferAllocation::Slice slice_overlap1(&alloc, /*offset=*/15, /*size=*/20);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/40, /*size=*/10);
  BufferAllocation::Slice slice_overlap2(&alloc, /*offset=*/45, /*size=*/15);

  std::vector<BufferAllocation::Slice> merged =
      MergeOverlappingSlices({slice1, slice_overlap1, slice2, slice_overlap2});

  ASSERT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0],
            BufferAllocation::Slice(&alloc, /*offset=*/10, /*size=*/25));
  EXPECT_EQ(merged[1],
            BufferAllocation::Slice(&alloc, /*offset=*/40, /*size=*/20));
}

}  // namespace
}  // namespace xla
