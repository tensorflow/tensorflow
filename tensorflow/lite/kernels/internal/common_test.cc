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
#include "tensorflow/lite/kernels/internal/common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

constexpr float GetMinValue() {
  return std::numeric_limits<float>::has_infinity
             ? -std::numeric_limits<float>::infinity()
             : std::numeric_limits<float>::lowest();
}

constexpr float GetMaxValue() {
  return std::numeric_limits<float>::has_infinity
             ? std::numeric_limits<float>::infinity()
             : std::numeric_limits<float>::max();
}

TEST(ActivationFunctionWithMinMaxTest, FloatNoneActivation) {
  // For float with kTfLiteActNone, both sides are unbounded, so the output must
  // be the same with input.
  std::vector<float> input{
      GetMinValue(), std::numeric_limits<float>::lowest(), -1.0,         0.0,
      1.0,           std::numeric_limits<float>::max(),    GetMaxValue()};
  std::vector<float> output(input.size());

  float activation_min, activation_max;
  CalculateActivationRange(TfLiteFusedActivation::kTfLiteActNone,
                           &activation_min, &activation_max);

  std::transform(input.begin(), input.end(), output.begin(), [&](float x) {
    return ActivationFunctionWithMinMax(x, activation_min, activation_max);
  });
  EXPECT_THAT(output, ElementsAreArray(input));
}

TEST(ActivationFunctionWithMinMaxTest, FloatReluActivation) {
  // For float with kTfLiteActRelu, positive side is unbounded, so the output
  // must be the same with positive input.
  std::vector<float> input{0.0, 1.0, std::numeric_limits<float>::max(),
                           GetMaxValue()};
  std::vector<float> output(input.size());

  float activation_min, activation_max;
  CalculateActivationRange(TfLiteFusedActivation::kTfLiteActRelu,
                           &activation_min, &activation_max);

  std::transform(input.begin(), input.end(), output.begin(), [&](float x) {
    return ActivationFunctionWithMinMax(x, activation_min, activation_max);
  });
  EXPECT_THAT(output, ElementsAreArray(input));
}

}  // namespace
}  // namespace tflite