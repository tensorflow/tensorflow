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
#include <cstdint>

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
tflite::TensorType GetTTEnum<int8_t>() {
  return tflite::TensorType_INT8;
}

template <>
tflite::TensorType GetTTEnum<int32_t>() {
  return tflite::TensorType_INT32;
}

template <>
tflite::TensorType GetTTEnum<int64_t>() {
  return tflite::TensorType_INT64;
}

template <typename INPUT_TYPE>
class RandomUniformOpModel : public tflite::SingleOpModel {
 public:
  RandomUniformOpModel(const std::initializer_list<INPUT_TYPE>& input,
                       TensorType input_type, tflite::TensorData output,
                       bool dynamic_input) {
    if (dynamic_input) {
      input_ = AddInput({input_type, {3}});
    } else {
      input_ =
          AddConstInput(input_type, input, {static_cast<int>(input.size())});
    }
    output_ = AddOutput(output);
    SetCustomOp("RandomUniform", {}, ops::custom::Register_RANDOM_UNIFORM);
    BuildInterpreter({GetShape(input_)});
    if (dynamic_input) {
      PopulateTensor<INPUT_TYPE>(input_, std::vector<INPUT_TYPE>(input));
    }
  }

  int input_;
  int output_;

  int input() { return input_; }
  int output() { return output_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

template <typename INPUT_TYPE>
class RandomUniformIntOpModel : public tflite::SingleOpModel {
 public:
  RandomUniformIntOpModel(const std::initializer_list<INPUT_TYPE>& input,
                          TensorType input_type, tflite::TensorData output,
                          INPUT_TYPE min_val, INPUT_TYPE max_val) {
    input_ = AddConstInput(input_type, input, {static_cast<int>(input.size())});
    input_minval_ = AddConstInput(input_type, {min_val}, {1});
    input_maxval_ = AddConstInput(input_type, {max_val}, {1});
    output_ = AddOutput(output);
    SetCustomOp("RandomUniformInt", {},
                ops::custom::Register_RANDOM_UNIFORM_INT);
    BuildInterpreter({GetShape(input_)});
  }

  int input_;
  int input_minval_;
  int input_maxval_;

  int output_;

  int input() { return input_; }
  int output() { return output_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
};

}  // namespace
}  // namespace tflite

template <typename FloatType>
class RandomUniformTest : public ::testing::Test {
 public:
  using Float = FloatType;
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(RandomUniformTest, TestTypes);

TYPED_TEST(RandomUniformTest, TestOutput) {
  using Float = typename TestFixture::Float;
  for (const auto dynamic : {true, false}) {
    tflite::RandomUniformOpModel<int32_t> m(
        {1000, 50, 5}, tflite::TensorType_INT32,
        {tflite::GetTTEnum<Float>(), {}}, dynamic);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    auto output = m.GetOutput<Float>();
    EXPECT_EQ(output.size(), 1000 * 50 * 5);

    double sum = 0;
    for (const auto r : output) {
      sum += r;
    }
    double avg = sum / output.size();
    ASSERT_LT(std::abs(avg - 0.5), 0.05);  // Average should approximately 0.5

    double sum_squared = 0;
    for (const auto r : output) {
      sum_squared += std::pow(r - avg, 2);
    }
    double var = sum_squared / output.size();
    EXPECT_LT(std::abs(1. / 12 - var),
              0.05);  // Variance should be approximately 1./12
  }
}

TYPED_TEST(RandomUniformTest, TestOutputInt64) {
  using Float = typename TestFixture::Float;
  for (const auto dynamic : {true, false}) {
    tflite::RandomUniformOpModel<int64_t> m(
        {1000, 50, 5}, tflite::TensorType_INT64,
        {tflite::GetTTEnum<Float>(), {}}, dynamic);
    ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
    auto output = m.GetOutput<Float>();
    EXPECT_EQ(output.size(), 1000 * 50 * 5);

    double sum = 0;
    for (const auto r : output) {
      sum += r;
    }
    double avg = sum / output.size();
    ASSERT_LT(std::abs(avg - 0.5), 0.05);  // Average should approximately 0.5

    double sum_squared = 0;
    for (const auto r : output) {
      sum_squared += std::pow(r - avg, 2);
    }
    double var = sum_squared / output.size();
    EXPECT_LT(std::abs(1. / 12 - var),
              0.05);  // Variance should be approximately 1./12
  }
}

template <typename IntType>
class RandomUniformIntTest : public ::testing::Test {
 public:
  using Int = IntType;
};

using TestTypesInt = ::testing::Types<int8_t, int32_t, int64_t>;

TYPED_TEST_SUITE(RandomUniformIntTest, TestTypesInt);

TYPED_TEST(RandomUniformIntTest, TestOutput) {
  using Int = typename TestFixture::Int;
  tflite::RandomUniformIntOpModel<int32_t> m(
      {1000, 50, 5}, tflite::TensorType_INT32, {tflite::GetTTEnum<Int>(), {}},
      0, 5);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), 1000 * 50 * 5);

  int counters[] = {0, 0, 0, 0, 0, 0};
  for (const auto r : output) {
    ASSERT_GE(r, 0);
    ASSERT_LE(r, 5);
    ++counters[r];
  }
  // Check that all numbers are meet with near the same frequency.
  for (int i = 1; i < 6; ++i) {
    EXPECT_LT(std::abs(counters[i] - counters[0]), 1000);
  }
}

TYPED_TEST(RandomUniformIntTest, TestOutputInt64) {
  using Int = typename TestFixture::Int;
  tflite::RandomUniformIntOpModel<int64_t> m(
      {1000, 50, 5}, tflite::TensorType_INT64, {tflite::GetTTEnum<Int>(), {}},
      0, 5);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto output = m.GetOutput<Int>();
  EXPECT_EQ(output.size(), 1000 * 50 * 5);

  int counters[] = {0, 0, 0, 0, 0, 0};
  for (const auto r : output) {
    ASSERT_GE(r, 0);
    ASSERT_LE(r, 5);
    ++counters[r];
  }
  // Check that all numbers are meet with near the same frequency.
  for (int i = 1; i < 6; ++i) {
    EXPECT_LT(std::abs(counters[i] - counters[0]), 1000);
  }
}
