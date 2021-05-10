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

class RandomStandardNormalOpModel : public tflite::SingleOpModel {
 public:
  RandomStandardNormalOpModel(const std::initializer_list<int>& input,
                              tflite::TensorData output) {
    input_ = AddConstInput(tflite::TensorType_INT32, input,
                           {static_cast<int>(input.size())});
    output_ = AddOutput(output);
    SetCustomOp("RandomStandardNormal", {},
                ops::custom::Register_RANDOM_STANDARD_NORMAL);
    BuildInterpreter({GetShape(input_)});
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

}  // namespace
}  // namespace tflite

template <typename FloatType>
class RandomStandardNormalTest : public ::testing::Test {
 public:
  using Float = FloatType;
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(RandomStandardNormalTest, TestTypes);

TYPED_TEST(RandomStandardNormalTest, TestOutput) {
  using Float = typename TestFixture::Float;
  tflite::RandomStandardNormalOpModel m({1000, 50, 5},
                                        {tflite::GetTTEnum<Float>(), {}});
  m.Invoke();
  auto output = m.GetOutput<Float>();
  EXPECT_EQ(output.size(), 1000 * 50 * 5);

  double sum = 0;
  for (auto r : output) {
    sum += r;
  }
  double avg = sum / output.size();
  ASSERT_LT(std::abs(avg), 0.05);  // Average should approximately 0.

  double sum_squared = 0;
  for (auto r : output) {
    sum_squared += std::pow(r - avg, 2);
  }
  double var = sum_squared / output.size();
  EXPECT_LT(std::abs(1 - var), 0.05);  // Variance should be approximately 1.
}
