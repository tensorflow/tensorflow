/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

TEST(PerChannelQuantize, TestFloatToInt8_2D) {
  const std::vector<float> scales = {0.5, 0.25};
  const std::vector<int> zero_points = {-1, -1};
  const int quantized_dimension = 0;

  const RuntimeShape shape({2, 5});

  const std::vector<float> input = {-63.5, -63,   -62.5, -62,   -61.5,
                                    31,    31.25, 31.5,  31.75, 32};
  std::vector<int8_t> output(10, -1);

  PerChannelQuantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelQuantize(op_params, shape, input.data(), shape,
                                    output.data());
  EXPECT_THAT(output, ElementsAreArray({-128, -127, -126, -125, -124, 123, 124,
                                        125, 126, 127}));
}

TEST(PerChannelQuantize, TestFloatToInt8_3D) {
  const std::vector<float> scales = {0.5, 0.25, 0.5, 0.25, 1.0};
  const std::vector<int> zero_points = {-1, 1, -1, 1, 0};
  const int quantized_dimension = 2;

  const RuntimeShape shape({1, 2, 5});

  const std::vector<float> input = {-63.5, -32,   -62.5, -31.5, -124,
                                    62,    30.75, 63,    31.25, 127};
  std::vector<int8_t> output(10, -1);

  PerChannelQuantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelQuantize(op_params, shape, input.data(), shape,
                                    output.data());
  EXPECT_THAT(output, ElementsAreArray({-128, -127, -126, -125, -124, 123, 124,
                                        125, 126, 127}));
}

TEST(PerChannelQuantize, TestFloatToInt8_4DDim0) {
  const std::vector<float> scales = {0.5, 0.25};
  const std::vector<int> zero_points = {-1, 1};
  const int quantized_dimension = 0;

  RuntimeShape shape({2, 2, 5, 1});

  const std::vector<float> input = {
      -63.5,  -63, -62.5,  -62,   -61.5,  62,   62.5,  63, 63.5,  64,
      -32.25, -32, -31.75, -31.5, -31.25, 30.5, 30.75, 31, 31.25, 31.5};
  std::vector<int8_t> output(20, -1);

  PerChannelQuantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelQuantize(op_params, shape, input.data(), shape,
                                    output.data());
  EXPECT_THAT(output,
              ElementsAreArray({-128, -127, -126, -125, -124, 123,  124,
                                125,  126,  127,  -128, -127, -126, -125,
                                -124, 123,  124,  125,  126,  127}));
}

TEST(PerChannelQuantize, TestFloatToInt8_4DDim3) {
  const std::vector<float> scales = {0.5, 0.25, 0.5, 0.25, 1.0};
  const std::vector<int> zero_points = {-1, 1, -1, 1, 0};
  const int quantized_dimension = 3;

  RuntimeShape shape({1, 2, 2, 5});

  const std::vector<float> input = {
      -63.5, -32, -62.5, -31.5, -124, 62, 30.75, 63, 31.25, 127,
      -63.5, -32, -62.5, -31.5, -124, 62, 30.75, 63, 31.25, 127};
  std::vector<int8_t> output(20, -1);

  PerChannelQuantizationParams op_params;
  op_params.zero_point = zero_points.data();
  op_params.scale = scales.data();
  op_params.quantized_dimension = quantized_dimension;
  reference_ops::PerChannelQuantize(op_params, shape, input.data(), shape,
                                    output.data());
  EXPECT_THAT(output,
              ElementsAreArray({-128, -127, -126, -125, -124, 123,  124,
                                125,  126,  127,  -128, -127, -126, -125,
                                -124, 123,  124,  125,  126,  127}));
}

// Positive control: non-empty portable quantization scales values as expected.
TEST(PerChannelQuantize, PortableQuantizeFloatsValidNonEmptyControl) {
  const float values[2] = {-12.7f, 12.7f};
  int8_t sym_out[2] = {0, 0};
  float min_val = 0.0f;
  float max_val = 0.0f;
  float sym_scale = 0.0f;
  tensor_utils::PortableSymmetricQuantizeFloats(values, 2, sym_out, &min_val,
                                                &max_val, &sym_scale);
  EXPECT_FLOAT_EQ(min_val, -12.7f);
  EXPECT_FLOAT_EQ(max_val, 12.7f);
  EXPECT_EQ(sym_out[0], -127);
  EXPECT_EQ(sym_out[1], 127);

  int8_t asym_out[2] = {0, 0};
  float asym_scale = 0.0f;
  int32_t offset = 0;
  tensor_utils::PortableAsymmetricQuantizeFloats(values, 2, asym_out,
                                                 &asym_scale, &offset);
  EXPECT_GT(asym_scale, 0.0f);
  EXPECT_LT(asym_out[0], asym_out[1]);
}

TEST(PerChannelQuantize, PortableSymmetricQuantizeFloatsNonPositiveSizeSafe) {
  auto backing = std::make_unique<float[]>(1);
  backing[0] = 42.0f;
  const float* one_past_end = backing.get() + 1;
  int8_t sentinel = 77;
  float min_val = -5.0f;
  float max_val = 9.0f;
  float scale = 0.0f;

  tensor_utils::PortableSymmetricQuantizeFloats(one_past_end, 0, &sentinel,
                                                &min_val, &max_val, &scale);
  EXPECT_FLOAT_EQ(min_val, 0.0f);
  EXPECT_FLOAT_EQ(max_val, 0.0f);
  EXPECT_FLOAT_EQ(scale, 1.0f);
  EXPECT_EQ(sentinel, 77);

  scale = 0.0f;
  tensor_utils::PortableSymmetricQuantizeFloats(nullptr, 0, nullptr, 0.0f, 0.0f,
                                                &scale);
  EXPECT_FLOAT_EQ(scale, 1.0f);

  min_val = -3.0f;
  max_val = 3.0f;
  scale = 0.0f;
  tensor_utils::PortableSymmetricQuantizeFloats(one_past_end, -1, &sentinel,
                                                &min_val, &max_val, &scale);
  EXPECT_FLOAT_EQ(min_val, 0.0f);
  EXPECT_FLOAT_EQ(max_val, 0.0f);
  EXPECT_FLOAT_EQ(scale, 1.0f);
  EXPECT_EQ(sentinel, 77);
}

TEST(PerChannelQuantize, PortableAsymmetricQuantizeFloatsNonPositiveSizeSafe) {
  auto backing = std::make_unique<float[]>(1);
  backing[0] = -19.0f;
  const float* one_past_end = backing.get() + 1;
  int8_t sentinel = 55;
  float scale = 0.0f;
  int32_t offset = -9;

  tensor_utils::PortableAsymmetricQuantizeFloats(one_past_end, 0, &sentinel,
                                                 &scale, &offset);
  EXPECT_FLOAT_EQ(scale, 1.0f);
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(sentinel, 55);

  scale = 0.0f;
  offset = -9;
  tensor_utils::PortableAsymmetricQuantizeFloats(nullptr, 0, nullptr, &scale,
                                                 &offset);
  EXPECT_FLOAT_EQ(scale, 1.0f);
  EXPECT_EQ(offset, 0);

  scale = 0.0f;
  offset = -9;
  tensor_utils::PortableAsymmetricQuantizeFloats(one_past_end, -1, &sentinel,
                                                 &scale, &offset);
  EXPECT_FLOAT_EQ(scale, 1.0f);
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(sentinel, 55);
}

}  // namespace
}  // namespace tflite
