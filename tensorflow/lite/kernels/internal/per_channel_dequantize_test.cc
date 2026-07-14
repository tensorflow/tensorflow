/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

TEST(PerChannelDequantize, TestInt8ToFloat_2D) {
  const std::vector<float> scales = {0.5, 0.25};
  const std::vector<int> zero_points = {-1, -1};
  const int quantized_dimension = 0;

  const RuntimeShape shape({2, 5});

  const std::vector<int8_t> input = {-128, -127, -126, -125, -124,
                                     123,  124,  125,  126,  127};
  std::vector<float> output(10, -1);

  PerChannelDequantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelDequantize(op_params, shape, input.data(), shape,
                                      output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({-63.5, -63, -62.5, -62, -61.5,
                                               31, 31.25, 31.5, 31.75, 32})));
}

TEST(PerChannelDequantize, TestInt8ToFloat_3D) {
  const std::vector<float> scales = {0.5, 0.25, 0.5, 0.25, 1.0};
  const std::vector<int> zero_points = {-1, 1, -1, 1, 0};
  const int quantized_dimension = 2;

  const RuntimeShape shape({1, 2, 5});

  const std::vector<int8_t> input = {-128, -127, -126, -125, -124,
                                     123,  124,  125,  126,  127};
  std::vector<float> output(10, -1);

  PerChannelDequantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelDequantize(op_params, shape, input.data(), shape,
                                      output.data());
  EXPECT_THAT(output,
              ElementsAreArray(ArrayFloatNear({-63.5, -32, -62.5, -31.5, -124,
                                               62, 30.75, 63, 31.25, 127})));
}

TEST(PerChannelDequantize, TestInt8ToFloat_4DDim0) {
  const std::vector<float> scales = {0.5, 0.25};
  const std::vector<int> zero_points = {-1, 1};
  const int quantized_dimension = 0;

  RuntimeShape shape({2, 2, 5, 1});

  const std::vector<int8_t> input = {-128, -127, -126, -125, -124, 123,  124,
                                     125,  126,  127,  -128, -127, -126, -125,
                                     -124, 123,  124,  125,  126,  127};
  std::vector<float> output(20, -1);

  PerChannelDequantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelDequantize(op_params, shape, input.data(), shape,
                                      output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {-63.5,  -63,  -62.5, -62,    -61.5, 62,     62.5,
                           63,     63.5, 64,    -32.25, -32,   -31.75, -31.5,
                           -31.25, 30.5, 30.75, 31,     31.25, 31.5})));
}

TEST(PerChannelDequantize, TestInt8ToFloat_4DDim3) {
  const std::vector<float> scales = {0.5, 0.25, 0.5, 0.25, 1.0};
  const std::vector<int> zero_points = {-1, 1, -1, 1, 0};
  const int quantized_dimension = 3;

  RuntimeShape shape({1, 2, 2, 5});

  const std::vector<int8_t> input = {-128, -127, -126, -125, -124, 123,  124,
                                     125,  126,  127,  -128, -127, -126, -125,
                                     -124, 123,  124,  125,  126,  127};
  std::vector<float> output(20, -1);

  PerChannelDequantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelDequantize(op_params, shape, input.data(), shape,
                                      output.data());
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {-63.5, -32,   -62.5, -31.5, -124,  62,    30.75,
                           63,    31.25, 127,   -63.5, -32,   -62.5, -31.5,
                           -124,  62,    30.75, 63,    31.25, 127})));
}

TEST(PerChannelDequantize, TestInt4ToFloat_2D) {
  const std::vector<float> scales = {0.5, 0.25};
  const std::vector<int> zero_points = {-1, -1};
  const int quantized_dimension = 0;

  const RuntimeShape unpacked_shape({2, 4});

  const std::vector<int8_t> packed_int4_input = {-1, 0, 65, -127};
  std::vector<float> output(8, -1);
  const size_t bytes_unpacked = packed_int4_input.size() * 2;
  auto unpacked_input_data = std::make_unique<int8_t[]>(bytes_unpacked);
  tflite::tensor_utils::UnpackPackedIntToInt8(packed_int4_input.data(),
                                              bytes_unpacked, /*bit_width=*/4,
                                              unpacked_input_data.get());
  EXPECT_THAT(std::vector<int8_t>(unpacked_input_data.get(),
                                  unpacked_input_data.get() + bytes_unpacked),
              ElementsAreArray(ArrayFloatNear({-1, -1, 0, 0, 1, 4, 1, -8})));

  PerChannelDequantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelDequantize(op_params, unpacked_shape,
                                      unpacked_input_data.get(), unpacked_shape,
                                      output.data());
  // This comes from (UNPACKED - zero_point) * scale.
  EXPECT_THAT(output, ElementsAreArray(ArrayFloatNear(
                          {0, 0, 0.5, 0.5, 0.5, 1.25, 0.5, -1.75})));
}

}  // namespace
}  // namespace tflite
