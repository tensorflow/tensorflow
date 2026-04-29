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

#include "xla/backends/gpu/libraries/cub/cub_sort_utils.h"

#include <gtest/gtest.h>

namespace xla::gpu {
namespace {

TEST(CubSortUtilsTest, NonSegmentedDoesntChangeScratchSize) {
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/100,
                                                 /*batch_size=*/1),
            100);
}

TEST(CubSortUtilsTest, SegmentedSortAddsSpace) {
  // For batch_size > 1, the scratch size is updated as follows:
  // 1. Aligned to sizeof(int32_t) = 4 boundary.
  // TODO(b/502873525): This alignment logic adds padding even when already
  // aligned. Fix in a follow-up.
  // 2. Offsets space added: (batch_size + 1) * sizeof(int32_t).

  // scratch_size = 0: aligned to 4 -> 4. Offsets: (2+1)*4 = 12. Total = 16.
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/0,
                                                 /*batch_size=*/2),
            16);
  // scratch_size = 1: aligned to 4 -> 4. Offsets: 12. Total = 16.
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/1,
                                                 /*batch_size=*/2),
            16);
  // scratch_size = 3: aligned to 4 -> 4. Offsets: 12. Total = 16.
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/3,
                                                 /*batch_size=*/2),
            16);
  // scratch_size = 4: aligned to 4 -> 8 (always adds padding if %4==0).
  // Offsets: 12. Total = 20.
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/4,
                                                 /*batch_size=*/2),
            20);
}

}  // namespace
}  // namespace xla::gpu
