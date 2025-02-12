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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/test_util.h"

namespace tflite {
namespace {

// Runs the reference and optimized MaxPool functions and asserts the values
// are the same.
void RunOneMaxPoolTest(const PoolParams& params,
                       const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& output_shape) {
  const int buffer_size = output_shape.FlatSize();
  std::vector<int8_t> optimized_maxpool_output(buffer_size);
  std::vector<int8_t> reference_maxpool_output(buffer_size);

  reference_integer_ops::MaxPool(params, input_shape, input_data, output_shape,
                                 reference_maxpool_output.data());
  optimized_integer_ops::MaxPool(params, input_shape, input_data, output_shape,
                                 optimized_maxpool_output.data());

  for (int i = 0; i < buffer_size; i++) {
    ASSERT_TRUE(reference_maxpool_output[i] == optimized_maxpool_output[i]);
  }
}

// Creates random input shape (batch, height, width, depth), then computes
// output shape based on value of `padding_same`:
// `padding_same` == true, calculate output with padding == "SAME"
// `padding_same` == false, calculate output with padding == "VALID"
// With input/output shapes computed, fills the input data and calls the
// test function.
void CreateDataAndRunMaxPool(bool padding_same) {
  const int batch = UniformRandomInt(1, 2);
  const int input_depth = UniformRandomInt(1, 700);
  const int output_depth = input_depth;
  const int input_width_offset = UniformRandomInt(1, 30);
  const int input_height_offset = UniformRandomInt(1, 30);
  const int stride_width = UniformRandomInt(1, 10);
  const int stride_height = UniformRandomInt(1, 10);
  const int filter_width = UniformRandomInt(1, 10);
  const int filter_height = UniformRandomInt(1, 10);
  const int input_width = input_width_offset + filter_width;
  const int input_height = input_height_offset + filter_height;
  const int output_width =
      padding_same ? (input_width + stride_width - 1) / stride_width
                   : (input_width - filter_width + stride_width) / stride_width;
  const int output_height =
      padding_same
          ? (input_height + stride_height - 1) / stride_height
          : (input_height - filter_height + stride_height) / stride_height;

  auto input_shape =
      RuntimeShape({batch, input_height, input_width, input_depth});
  auto output_shape =
      RuntimeShape({batch, output_height, output_width, output_depth});
  const int buffer_size = input_shape.FlatSize();
  std::vector<int8_t> input_data(buffer_size);
  FillRandom(&input_data);

  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.quantized_activation_min =
      static_cast<int8_t>(std::numeric_limits<int8_t>::lowest());
  params.quantized_activation_max =
      static_cast<int8_t>(std::numeric_limits<int8_t>::max());
  auto compute_padding = [](int stride, int in_size, int filter_size,
                            int out_size) {
    int padding = ((out_size - 1) * stride + filter_size - in_size) / 2;
    return padding > 0 ? padding : 0;
  };
  params.padding_values.width =
      compute_padding(stride_width, input_width, filter_width, output_width);
  params.padding_values.height = compute_padding(stride_height, input_height,
                                                 filter_height, output_height);
  RunOneMaxPoolTest(params, input_shape, input_data.data(), output_shape);
}

TEST(TestMaxPool, SymmetricQuantMaxPool) {
  const int kTestsToRun = 10;
  for (int i = 0; i < kTestsToRun; i++) {
    CreateDataAndRunMaxPool(/*padding_same=*/true);
    CreateDataAndRunMaxPool(/*padding_same=*/false);
  }
}

}  // namespace
}  // namespace tflite
