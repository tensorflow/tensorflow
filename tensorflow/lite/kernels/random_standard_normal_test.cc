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

#include <initializer_list>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

template <typename T>
TensorType GetTTEnum();

template <>
TensorType GetTTEnum<float>() {
  return TensorType_FLOAT32;
}

template <>
TensorType GetTTEnum<double>() {
  return TensorType_FLOAT64;
}

class RandomStandardNormalOpModel : public SingleOpModel {
 public:
  RandomStandardNormalOpModel(const std::initializer_list<int>& input,
                              TensorData output, bool dynamic_input) {
    if (dynamic_input) {
      input_ = AddInput({TensorType_INT32, {3}});
    } else {
      input_ = AddConstInput(TensorType_INT32, input,
                             {static_cast<int>(input.size())});
    }
    output_ = AddOutput(output);
    SetCustomOp("RandomStandardNormal", {},
                ops::custom::Register_RANDOM_STANDARD_NORMAL);
    BuildInterpreter({GetShape(input_)});
    if (dynamic_input) {
      PopulateTensor<int32_t>(input_, std::vector<int32_t>(input));
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

}  // namespace
}  // namespace tflite

template <typename InputType>
struct RandomStandardNormalTest : public ::testing::Test {
  using Type = InputType;
};
using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(RandomStandardNormalTest, TestTypes);

TYPED_TEST(RandomStandardNormalTest, TestOutput) {
  using Type = typename TestFixture::Type;
  for (const auto dynamic : {false, true}) {
    tflite::RandomStandardNormalOpModel m(
        {1000, 50, 5}, {tflite::GetTTEnum<Type>(), {}}, dynamic);
    m.Invoke();
    auto output = m.GetOutput<Type>();
    EXPECT_EQ(output.size(), 1000 * 50 * 5);

    double sum = 0;
    for (auto r : output) {
      sum += r;
    }
    double avg = sum / output.size();
    ASSERT_LT(std::abs(avg), 0.05);  // Average should be approximately 0.

    double sum_squared = 0;
    for (auto r : output) {
      sum_squared += std::pow(r - avg, 2);
    }
    double var = sum_squared / output.size();
    EXPECT_LT(std::abs(1 - var), 0.05);  // Variance should be approximately 1.
  }
}

TYPED_TEST(RandomStandardNormalTest, TestOutputDistributionRange) {
  using Type = typename TestFixture::Type;
  tflite::RandomStandardNormalOpModel m({1000, 50, 5},
                                        {tflite::GetTTEnum<Type>(), {}}, false);
  // Initialize output tensor to infinity to validate that all of its values are
  // updated and are normally distributed after Invoke().
  const std::vector<Type> output_data(1000 * 50 * 5,
                                      std::numeric_limits<Type>::infinity());
  m.PopulateTensor(m.output(), output_data);
  m.Invoke();
  auto output = m.GetOutput<Type>();
  EXPECT_EQ(output.size(), 1000 * 50 * 5);
  double sum = 0;
  for (auto r : output) {
    sum += r;
  }
  double avg = sum / output.size();
  ASSERT_LT(std::abs(avg), 0.05);  // Average should be approximately 0.

  double sum_squared = 0;
  for (auto r : output) {
    sum_squared += std::pow(r - avg, 2);
  }
  double var = sum_squared / output.size();
  EXPECT_LT(std::abs(1 - var), 0.05);  // Variance should be approximately 1.
}
