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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

template <typename T>
tflite::TensorType GetTTEnum();

template <>
tflite::TensorType GetTTEnum<float>() {
  return tflite::TensorType_FLOAT32;
}

template <>
tflite::TensorType GetTTEnum<double>() {
  return tflite::TensorType_FLOAT64;
}

template <>
tflite::TensorType GetTTEnum<int>() {
  return tflite::TensorType_INT32;
}

template <>
tflite::TensorType GetTTEnum<int64_t>() {
  return tflite::TensorType_INT64;
}

class MultinomialOpModel : public tflite::SingleOpModel {
 public:
  MultinomialOpModel(tflite::TensorData logits, int num_samples,
                     tflite::TensorData output) {
    logits_ = AddInput(logits);
    num_samples_ = AddConstInput(tflite::TensorType_INT32, {num_samples}, {});
    output_ = AddOutput(output);
    SetCustomOp("Multinomial", {}, ops::custom::Register_MULTINOMIAL);
    BuildInterpreter({GetShape(logits_), GetShape(num_samples_)});
  }

  int logits_;
  int num_samples_;
  int output_;

  int logits() { return logits_; }
  int num_samples() { return num_samples_; }
  int output() { return output_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

}  // namespace
}  // namespace tflite

template <typename Type1, typename Type2>
struct TypePair {
  using T1 = Type1;
  using T2 = Type2;
};

template <typename TypePair>
class MultinomialTest : public ::testing::Test {
 public:
  using FloatType = typename TypePair::T1;
  using IntegralType = typename TypePair::T2;
};

using TestTypes =
    ::testing::Types<TypePair<float, int>, TypePair<double, int>,
                     TypePair<float, int64_t>, TypePair<double, int64_t> >;

TYPED_TEST_SUITE(MultinomialTest, TestTypes);

TYPED_TEST(MultinomialTest, TestMultiBatch) {
  const int kNumSamples = 1000;
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {3, 3}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});
  // Add 3 batches of 3 logits each.
  m.PopulateTensor<Float>(m.logits(),
                          std::vector<Float>(9, static_cast<Float>(0.0f)));

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), kNumSamples * 3);

  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);
  int c2 = std::count(output.begin(), output.end(), 2);

  EXPECT_EQ(c0 + c1 + c2, 3 * kNumSamples);

  // Make sure they're all sampled with roughly equal probability.
  EXPECT_GT(c0, 750);
  EXPECT_GT(c1, 750);
  EXPECT_GT(c2, 750);
  EXPECT_LT(c0, 1250);
  EXPECT_LT(c1, 1250);
  EXPECT_LT(c2, 1250);
}

// Test that higher log odds are sampled more often.
TYPED_TEST(MultinomialTest, TestSampleHighLogOdds) {
  const int kNumSamples = 1000;
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {1, 3}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});

  // Add 1 batch of 3 logits.
  m.PopulateTensor<Float>(m.logits(),
                          {static_cast<Float>(0.0f), static_cast<Float>(1.0f),
                           static_cast<Float>(0.0f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), kNumSamples);

  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);
  int c2 = std::count(output.begin(), output.end(), 2);
  EXPECT_EQ(c0 + c1 + c2, kNumSamples);
  EXPECT_GT(c1, c0);
  EXPECT_GT(c1, c2);
}

// Test that very low log odds are never sampled.
TYPED_TEST(MultinomialTest, TestVeryLowLogOdds) {
  const int kNumSamples = 1000;
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {1, 3}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});

  // Add 1 batch of 3 logits.
  m.PopulateTensor<Float>(
      m.logits(), {static_cast<Float>(-1000.0f), static_cast<Float>(-1000.0f),
                   static_cast<Float>(0.0f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), kNumSamples);

  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);
  int c2 = std::count(output.begin(), output.end(), 2);
  EXPECT_EQ(c0, 0);
  EXPECT_EQ(c1, 0);
  EXPECT_EQ(c2, kNumSamples);
}

TYPED_TEST(MultinomialTest, TestSamplesDifferent) {
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  const int kNumSamples = 5;
  const int kNumLogits = 1000;

  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {1, kNumLogits}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});

  std::vector<Float> logits(kNumLogits, static_cast<Float>(0.0f));
  m.PopulateTensor<Float>(m.logits(), logits);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto output1 = m.GetOutput<Int>();
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto output2 = m.GetOutput<Int>();

  bool successive_samples_are_different = false;
  for (int i = 0; i < kNumSamples; ++i) {
    if (output1[i] == output2[i]) continue;
    successive_samples_are_different = true;
    break;
  }
  EXPECT_TRUE(successive_samples_are_different);
}

TYPED_TEST(MultinomialTest, TestSamplesPrecise) {
  using Float = typename TestFixture::FloatType;
  using Int = typename TestFixture::IntegralType;
  const int kNumSamples = 100000;
  const int kNumLogits = 2;

  tflite::MultinomialOpModel m({tflite::GetTTEnum<Float>(), {1, kNumLogits}},
                               kNumSamples, {tflite::GetTTEnum<Int>(), {}});

  std::vector<Float> logits(
      {static_cast<Float>(1000.0), static_cast<float>(1001.0)});
  m.PopulateTensor<Float>(m.logits(), logits);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  int c0 = std::count(output.begin(), output.end(), 0);
  int c1 = std::count(output.begin(), output.end(), 1);

  double p0 = static_cast<double>(c0) / (c0 + c1);
  EXPECT_LT(std::abs(p0 - 0.26894142137), 0.01);
}
