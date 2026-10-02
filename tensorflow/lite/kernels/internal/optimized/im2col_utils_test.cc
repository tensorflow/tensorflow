/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {
namespace {

using ::testing::ElementsAreArray;

// Regression test for b/512611805.
//
// DilatedIm2col3D must size the per-pixel memcpy/memset by the channel
// dimension (input_shape.Dims(4)), not the spatial depth dimension
// (input_shape.Dims(1)). When the spatial depth D exceeds the channel count C,
// the previous code copied/zeroed D elements per pixel into im2col slots sized
// for C elements, overrunning the im2col buffer (and over-reading the input) by
// (D - C) elements per pixel.
//
// Both the input and the im2col buffers below are sized exactly, so the
// out-of-bounds access is detected by ASan. The buffer contents are also
// checked against the expected im2col matrix to guard the functional result.
TEST(DilatedIm2col3DTest, DepthGreaterThanChannelsDoesNotOverflow) {
  // NDHWC input with spatial depth D=8 and a single channel C=1.
  const RuntimeShape input_shape({1, 8, 1, 1, 1});
  const std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8};

  Conv3DParams params;
  params.padding_values = Padding3DValues{};  // No padding.
  params.stride_width = 1;
  params.stride_height = 1;
  params.stride_depth = 1;
  params.dilation_width = 1;
  params.dilation_height = 1;
  // dilation_depth != 1 selects the DilatedIm2col3D path in Conv3D.
  params.dilation_depth = 2;

  const int filter_depth = 2;
  const int filter_height = 1;
  const int filter_width = 1;

  // VALID output depth = (D - ((filter_depth - 1) * dilation_depth + 1)) + 1
  //                    = (8 - 3) + 1 = 6.
  const int output_depth = 6;
  const RuntimeShape im2col_shape({1, output_depth, 1, 1, filter_depth});
  // Exact im2col size: (batches * out_d * out_h * out_w) rows, each of length
  // (filter_depth * filter_height * filter_width * input_channels) = 2.
  std::vector<float> im2col_data(output_depth * filter_depth);

  DilatedIm2col3D<float>(params, filter_depth, filter_height, filter_width,
                         /*zero_byte=*/0, input_shape, input_data.data(),
                         im2col_shape, im2col_data.data());

  // Column for output depth d holds the single channel of input depths d and
  // d + dilation_depth, i.e. {input[d], input[d + 2]}.
  EXPECT_THAT(im2col_data,
              ElementsAreArray({1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8}));
}

}  // namespace
}  // namespace optimized_ops
}  // namespace tflite
