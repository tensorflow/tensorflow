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
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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

}  // namespace
}  // namespace tflite
