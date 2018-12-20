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
#include "tensorflow/lite/tools/optimize/quantization_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace optimize {
namespace utils {
namespace {

TEST(QuantizationUtilsTest, NumElements) {
  TensorT tensor;
  tensor.shape = {1, 2, 3, 4};
  uint64_t num_elements;
  EXPECT_EQ(kTfLiteOk, NumElements(tensor, &num_elements));
  EXPECT_EQ(num_elements, 1 * 2 * 3 * 4);

  tensor.shape = {5};
  EXPECT_EQ(kTfLiteOk, NumElements(tensor, &num_elements));
  EXPECT_EQ(num_elements, 5);

  tensor.shape = {};
  EXPECT_EQ(kTfLiteError, NumElements(tensor, &num_elements));
}

TEST(QuantizationUtilsTest, GetAsymmetricQuantizationParamsUnitRange) {
  const float float_min = -128.0;
  const float float_max = 127.0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], float_max);
  EXPECT_EQ(params.min[0], float_min);

  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_EQ(zero_point, 0);
  EXPECT_NEAR(scale, 1, eps);
}

TEST(QuantizationUtilsTest, AsymmetricQuantizationParamsWithAllPositiveRange) {
  const float float_min = 1.0;
  const float float_max = 5.0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], float_max);
  EXPECT_EQ(params.min[0], float_min);
  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_EQ(zero_point, -128);
  EXPECT_NEAR(scale, 4 / 255.0f, eps);
}

TEST(QuantizationUtilsTest, AsymmetricQuantizationParamsWithAllNegativeRange) {
  const float float_min = -5.0;
  const float float_max = -1.0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], float_max);
  EXPECT_EQ(params.min[0], float_min);
  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_EQ(zero_point, 127);
  EXPECT_NEAR(scale, 4 / 255.0f, eps);
}

TEST(QuantizationUtilsTest, AsymmetricQuantizationParamsWithZeroInRange) {
  const float float_min = -5.0;
  const float float_max = 1.0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], float_max);
  EXPECT_EQ(params.min[0], float_min);
  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_NEAR(scale, 6 / 255.0f, eps);
  EXPECT_GT(zero_point, quant_min);
  EXPECT_LT(zero_point, quant_max);
}

}  // namespace
}  // namespace utils
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
