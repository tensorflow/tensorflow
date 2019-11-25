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
#include <limits>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ruy/path.h"

namespace ruy {
namespace {

#if RUY_PLATFORM(NEON_64)
void MakeBlockMapTuningTest(int rows, int cols, int depth, int kernel_rows,
                            int kernel_cols, int lhs_scalar_size,
                            int rhs_scalar_size, int tentative_thread_count,
                            Path path, int expected_num_blocks_base_log2,
                            int expected_rectangularness_log2) {
  BlockMap block_map;
  MakeBlockMap(rows, cols, depth, kernel_rows, kernel_cols, lhs_scalar_size,
               rhs_scalar_size, tentative_thread_count, path,
               /* cache_friendly_traversal_threshold */ 32768, &block_map);
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
}

TEST(BlockMapTest, MakeBlockMapTuningTestRectangular) {
  MakeBlockMapTuningTest(256, 16, 256, 8, 8, 1, 1,
                         /* tentative_thread_count */ 1, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 4);
  MakeBlockMapTuningTest(24, 2400, 256, 8, 8, 1, 1,
                         /* tentative_thread_count */ 1, Path::kNeonDotprod,
                         /* expected_num_blocks_base_log2 */ 0,
                         /* expected_rectangularness_log2 */ 6);
}
#endif

}  // namespace
}  // namespace ruy

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
