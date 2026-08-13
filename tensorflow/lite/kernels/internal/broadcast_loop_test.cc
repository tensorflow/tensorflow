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

#include "tensorflow/lite/kernels/internal/reference/broadcast_loop.h"

#include <cstddef>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

namespace tflite {
namespace reference_ops {
namespace {

TEST(BroadcastLoopTest, SupportsRankGreaterThanInlineRank) {
  const RuntimeShape input1_shape({2, 1, 2, 1, 2, 1, 2, 1, 2});
  const RuntimeShape input2_shape({1, 2, 1, 2, 1, 2, 1, 2, 1});
  const RuntimeShape output_shape({2, 2, 2, 2, 2, 2, 2, 2, 2});

  std::vector<int> input1(input1_shape.FlatSize());
  std::vector<int> input2(input2_shape.FlatSize());
  std::vector<int> output(output_shape.FlatSize());
  std::iota(input1.begin(), input1.end(), 0);
  std::iota(input2.begin(), input2.end(), 100);

  BroadcastBinaryOpSimple(input1_shape, input1.data(), input2_shape,
                          input2.data(), output_shape, output.data(),
                          [](int a, int b) { return a + b; });

  for (size_t output_index = 0; output_index < output.size(); ++output_index) {
    size_t remaining_index = output_index;
    size_t input1_index = 0;
    size_t input2_index = 0;
    size_t input1_stride = 1;
    size_t input2_stride = 1;
    for (int dim = output_shape.DimensionsCount() - 1; dim >= 0; --dim) {
      const size_t output_dim = static_cast<size_t>(output_shape.Dims(dim));
      const size_t coordinate = remaining_index % output_dim;
      remaining_index /= output_dim;
      if (input1_shape.Dims(dim) != 1) {
        input1_index += coordinate * input1_stride;
      }
      if (input2_shape.Dims(dim) != 1) {
        input2_index += coordinate * input2_stride;
      }
      input1_stride *= static_cast<size_t>(input1_shape.Dims(dim));
      input2_stride *= static_cast<size_t>(input2_shape.Dims(dim));
    }
    EXPECT_EQ(output[output_index], input1[input1_index] + input2[input2_index])
        << "output index " << output_index;
  }
}

}  // namespace
}  // namespace reference_ops
}  // namespace tflite
