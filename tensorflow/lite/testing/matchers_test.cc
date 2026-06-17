/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/testing/matchers.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace {

using ::testing::tflite::Approximately;
using ::testing::tflite::EqualsTensor;
using ::testing::tflite::SimpleConstTensor;

TEST(TensorMatcherTest, ExactlyEqualsSelf) {
  float data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2}, absl::MakeSpan(data));
  EXPECT_THAT(a, EqualsTensor(a));
}

TEST(TensorMatcherTest, ExactlyEqualsSame) {
  float a_data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(a_data));
  float b_data[] = {2.71828f, 3.14159f};
  SimpleConstTensor b(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(b_data));
  EXPECT_THAT(a, EqualsTensor(b));
}

TEST(TensorMatcherTest, DoesNotExactlyEqualDifferentType) {
  float data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2}, absl::MakeSpan(data));
  SimpleConstTensor b(TfLiteType::kTfLiteInt32, {1, 2}, absl::MakeSpan(data));
  EXPECT_THAT(a, Not(EqualsTensor(b)));
}

TEST(TensorMatcherTest, DoesNotExactlyEqualDifferentDims) {
  float data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2}, absl::MakeSpan(data));
  SimpleConstTensor b(TfLiteType::kTfLiteFloat32, {2, 1}, absl::MakeSpan(data));
  EXPECT_THAT(a, Not(EqualsTensor(b)));
}

TEST(TensorMatcherTest, DoesNotExactlyEqualDifferentData) {
  float a_data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(a_data));
  float b_data[] = {3.14159f, 2.71828f};
  SimpleConstTensor b(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(b_data));
  EXPECT_THAT(a, Not(EqualsTensor(b)));
}

TEST(TensorMatcherTest, ApproximatelyEqualsDefaultMargin) {
  float a_data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(a_data));
  float b_data[] = {2.718277f, 3.141593f};
  SimpleConstTensor b(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(b_data));
  EXPECT_THAT(a, Approximately(EqualsTensor(b)));
}

TEST(TensorMatcherTest, ApproximatelyEqualsWithLooseMargin) {
  float a_data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(a_data));
  float b_data[] = {2.72f, 3.14f};
  SimpleConstTensor b(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(b_data));
  EXPECT_THAT(a, Approximately(EqualsTensor(b), /*margin=*/0.01));
}

TEST(TensorMatcherTest, DoesNotApproximatelyEqualWithTightMargin) {
  float a_data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(a_data));
  float b_data[] = {2.72f, 3.14f};
  SimpleConstTensor b(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(b_data));
  EXPECT_THAT(a, Not(Approximately(EqualsTensor(b), /*margin=*/0.001)));
}

TEST(TensorMatcherTest, ApproximatelyEqualsWithLooseFraction) {
  float a_data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(a_data));
  float b_data[] = {2.72f, 3.14f};
  SimpleConstTensor b(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(b_data));
  EXPECT_THAT(
      a, Approximately(EqualsTensor(b), /*margin=*/0.0, /*fraction=*/0.999));
}

TEST(TensorMatcherTest, DoesNotApproximatelyEqualWithTightFraction) {
  float a_data[] = {2.71828f, 3.14159f};
  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(a_data));
  float b_data[] = {2.72f, 3.14f};
  SimpleConstTensor b(TfLiteType::kTfLiteFloat32, {1, 2},
                      absl::MakeSpan(b_data));
  EXPECT_THAT(a, Not(Approximately(EqualsTensor(b), /*margin=*/0.0,
                                   /*fraction=*/0.0001)));
}

}  // namespace
}  // namespace tflite
