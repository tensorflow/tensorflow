/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/block_map.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ruy/cpu_cache_size.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"

namespace ruy {
namespace {

#if RUY_PLATFORM(NEON_64)

// Unless otherwise specified, these tests have been tuned on ARM Cortex-A55.
void MakeBlockMapTuningTest(int rows, int cols, int depth, int kernel_rows,
                            int kernel_cols, int lhs_scalar_size,
                            int rhs_scalar_size, int tentative_thread_count,
                            Path path, int expected_num_blocks_base_log2,
                            int expected_rectangularness_log2) {
  BlockMap block_map;
  MakeBlockMap(rows, cols, depth, kernel_rows, kernel_cols, lhs_scalar_size,
               rhs_scalar_size, tentative_thread_count, path,
               LocalDataCacheSize(path), SharedDataCacheSize(path), &block_map);
  EXPECT_EQ(block_map.num_blocks_base_log2, expected_num_blocks_base_log2);
  EXPECT_EQ(std::min(block_map.rectangularness_log2[Side::kLhs],
                     block_map.rectangularness_log2[Side::kRhs]),
            0);
  EXPECT_EQ(std::max(block_map.rectangularness_log2[Side::kLhs],
                     block_map.rectangularness_log2[Side::kRhs]),
            expected_rectangularness_log2);
}

TEST(BlockMapTest, MakeBlockMapTuningTest8bitCubicShapesOneThreadNeonDotprod) {
  MakeBlockMapTuningTest(32, 32, 32, 8, 8, 1, 1, /* tentative_thread_count */ 1,
                         Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(48, 48, 48, 8, 8, 1, 1, /* tentative_thread_count */ 1,
                         Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(64, 64, 64, 8, 8, 1, 1, /* tentative_thread_count */ 1,
                         Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(96, 96, 96, 8, 8, 1, 1, /* tentative_thread_count */ 1,
                         Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(128, 128, 128, 8, 8, 1, 1,
                         /* tentative_thread_count */ 1, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(192, 192, 192, 8, 8, 1, 1,
                         /* tentative_thread_count */ 1, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(256, 256, 256, 8, 8, 1, 1,
                         /* tentative_thread_count */ 1, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 1,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(384, 384, 384, 8, 8, 1, 1,
                         /* tentative_thread_count */ 1, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 1,
                         /* expected_rectangularness_log2 */ 0);
}

TEST(BlockMapTest,
     MakeBlockMapTuningTest8bitCubicShapesFourThreadsNeonDotprod) {
  MakeBlockMapTuningTest(32, 32, 32, 8, 8, 1, 1, /* tentative_thread_count */ 4,
                         Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 1,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(48, 48, 48, 8, 8, 1, 1, /* tentative_thread_count */ 4,
                         Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 1,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(64, 64, 64, 8, 8, 1, 1, /* tentative_thread_count */ 4,
                         Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 1,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(96, 96, 96, 8, 8, 1, 1, /* tentative_thread_count */ 4,
                         Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 1,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(128, 128, 128, 8, 8, 1, 1,
                         /* tentative_thread_count */ 4, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 1,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(192, 192, 192, 8, 8, 1, 1,
                         /* tentative_thread_count */ 4, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 1,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(256, 256, 256, 8, 8, 1, 1,
                         /* tentative_thread_count */ 4, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 2,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(384, 384, 384, 8, 8, 1, 1,
                         /* tentative_thread_count */ 4, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 2,
                         /* expected_rectangularness_log2 */ 0);
}

TEST(BlockMapTest, MakeBlockMapTuningTest32bit) {
  MakeBlockMapTuningTest(256, 256, 256, 8, 8, 4, 4,
                         /* tentative_thread_count */ 4, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 3,
                         /* expected_rectangularness_log2 */ 0);
  MakeBlockMapTuningTest(4096, 4096, 4096, 8, 8, 4, 4,
                         /* tentative_thread_count */ 4, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 7,
                         /* expected_rectangularness_log2 */ 0);
}

TEST(BlockMapTest, MakeBlockMapTuningTestRectangular) {
  MakeBlockMapTuningTest(256, 16, 256, 8, 8, 1, 1,
                         /* tentative_thread_count */ 1, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 3);
  MakeBlockMapTuningTest(24, 2400, 256, 8, 8, 1, 1,
                         /* tentative_thread_count */ 1, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 6);
}

#endif

int L1Distance(const SidePair<int>& a, const SidePair<int>& b) {
  return std::abs(a[Side::kLhs] - b[Side::kLhs]) +
         std::abs(a[Side::kRhs] - b[Side::kRhs]);
}

void GetBlockByIndexSquareTest(int num_blocks_base_log2,
                               BlockMapTraversalOrder traversal_order) {
  // Arbitrary, does not affect this test. 3 is just a typical value.
  constexpr int kKernelSizeLog2 = 3;

  const int size_log2 = num_blocks_base_log2 + kKernelSizeLog2;
  BlockMap block_map;
  block_map.thread_count = 1;
  block_map.traversal_order = traversal_order;
  block_map.num_blocks_base_log2 = num_blocks_base_log2;
  for (Side side : {Side::kLhs, Side::kRhs}) {
    block_map.dims[side] = 1 << size_log2;
    block_map.rectangularness_log2[side] = 0;
    block_map.kernel_dims[side] = 1 << kKernelSizeLog2;
    block_map.small_block_dims[side] = block_map.kernel_dims[side];
    block_map.large_blocks[side] = 0;
  }

  const int num_blocks_per_side = 1 << num_blocks_base_log2;
  const int num_blocks = num_blocks_per_side * num_blocks_per_side;
  EXPECT_EQ(num_blocks, NumBlocks(block_map));

  // Perform a full traversal of all blocks, as if computing a whole matrix
  // multiplication.
  //
  // Used to record how many times each block was hit by the traversal.
  std::vector<int> block_hit_counts(num_blocks);
  // Here we guard an assumption that all traversal orders start at (0, 0).
  SidePair<int> previous_block_coords(0, 0);
  // Sum of L1 norm of the coordinate change at every step of the traversal.
  std::int64_t total_l1_distance = 0;
  // Number of jumps i.e. traversal steps with a L1 norm greater than 1.
  int discontinuity_count = 0;
  for (int block_index = 0; block_index < num_blocks; block_index++) {
    SidePair<int> block_coords;
    GetBlockByIndex(block_map, block_index, &block_coords);
    ++block_hit_counts[block_coords[Side::kLhs] +
                       num_blocks_per_side * block_coords[Side::kRhs]];
    int distance = L1Distance(block_coords, previous_block_coords);
    total_l1_distance += distance;
    discontinuity_count += (distance > 1);
    previous_block_coords = block_coords;
  }

  // Verify that each block was traversed exactly once.
  for (int l = 0; l < num_blocks_per_side; l++) {
    for (int r = 0; r < num_blocks_per_side; r++) {
      EXPECT_EQ(block_hit_counts[l + num_blocks_per_side * r], 1);
    }
  }

  // Verify that the discontinuity_count and total_l1_distance are as expected
  // for the given traversal_order.
  switch (traversal_order) {
    case BlockMapTraversalOrder::kFractalHilbert:
      // No discontinuity at all with this space-filling continuous curve!
      EXPECT_EQ(discontinuity_count, 0);
      // Therefore, total_l1_distance has to be the number of blocks minus one.
      EXPECT_EQ(total_l1_distance, num_blocks - 1);
      break;
    case BlockMapTraversalOrder::kLinear:
      EXPECT_EQ(discontinuity_count, num_blocks_per_side - 1);
      EXPECT_EQ(total_l1_distance,
                2 * num_blocks_per_side * (num_blocks_per_side - 1));
      break;
    case BlockMapTraversalOrder::kFractalZ:
      EXPECT_EQ(discontinuity_count, num_blocks > 1 ? (num_blocks / 2 - 1) : 0);
      EXPECT_EQ(total_l1_distance,
                2 * num_blocks_per_side * (num_blocks_per_side - 1));
      break;
    case BlockMapTraversalOrder::kFractalU: {
      if (num_blocks_base_log2 == 0) {
        EXPECT_EQ(discontinuity_count, 0);
        EXPECT_EQ(total_l1_distance, 0);
      } else {
        int expected_discontinuity_count = 0;
        int expected_total_l1_distance = 3;
        for (int i = 2; i <= num_blocks_base_log2; i++) {
          expected_discontinuity_count = 4 * expected_discontinuity_count + 2;
          expected_total_l1_distance =
              4 * expected_total_l1_distance + (1 << (i + 1)) - 1;
        }
        EXPECT_EQ(discontinuity_count, expected_discontinuity_count);
        EXPECT_EQ(total_l1_distance, expected_total_l1_distance);
      }
      break;
    }
    default:
      abort();
  }
}

TEST(BlockMapTest, GetBlockByIndexSquare) {
  for (int num_blocks_base_log2 = 0; num_blocks_base_log2 <= 10;
       num_blocks_base_log2++) {
    for (BlockMapTraversalOrder traversal_order :
         {BlockMapTraversalOrder::kLinear, BlockMapTraversalOrder::kFractalZ,
          BlockMapTraversalOrder::kFractalU,
          BlockMapTraversalOrder::kFractalHilbert}) {
      GetBlockByIndexSquareTest(num_blocks_base_log2, traversal_order);
    }
  }
}

}  // namespace
}  // namespace ruy

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
