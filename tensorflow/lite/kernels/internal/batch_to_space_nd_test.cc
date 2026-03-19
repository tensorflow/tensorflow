/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

#include <gtest/gtest.h>

namespace tflite {
namespace {

// A light wrapper of GetIndexRange which returns a pair of start / end
// indices.
std::pair<int, int> GetIndexRange(int spatial_index_dim, int block_shape_dim,
                                  int input_dim, int output_dim) {
  int index_start = 0;
  int index_end = 0;
  optimized_ops::GetIndexRange(spatial_index_dim, block_shape_dim, input_dim,
                               output_dim, &index_start, &index_end);
  return {index_start, index_end};
}

TEST(BatchToSpaceNDTest, TestIndexRange) {
  // Simple test case, no cropping.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/3, /*block_shape_dim=*/6,
                          /*input_dim=*/1, /*output_dim=*/6),
            std::make_pair(0, 1));

  // No cropping and input_dim > 1.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/2, /*block_shape_dim=*/6,
                          /*input_dim=*/5, /*output_dim=*/30),
            std::make_pair(0, 5));

  // With small cropping values (can be either at the beginning or at the end).
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/0, /*block_shape_dim=*/2,
                          /*input_dim=*/3, /*output_dim=*/4),
            std::make_pair(0, 2));

  // With positive cropping values at the beginning.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/-2, /*block_shape_dim=*/2,
                          /*input_dim=*/3, /*output_dim=*/4),
            std::make_pair(1, 3));

  // Large crop at the beginning.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/-30, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/5),
            std::make_pair(6, 7));

  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/-26, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/5),
            std::make_pair(6, 7));

  // Large crop at the end.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/0, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/5),
            std::make_pair(0, 1));

  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/4, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/5),
            std::make_pair(0, 1));

  // Rounding up incorrectly will fail this test.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/3, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/5),
            std::make_pair(0, 1));

  // Extreme cropping with output of a single spatial location.
  // Valid position 1, when large crop at the end.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/0, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/1),
            std::make_pair(0, 1));

  // Valid position 2, when large crop at the beginning.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/-30, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/1),
            std::make_pair(6, 7));

  // Invalid positions.
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/1, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/1),
            std::make_pair(0, 0));
  EXPECT_EQ(GetIndexRange(/*spatial_index_dim=*/-29, /*block_shape_dim=*/5,
                          /*input_dim=*/7, /*output_dim=*/1),
            std::make_pair(6, 6));
}

}  // namespace
}  // namespace tflite
