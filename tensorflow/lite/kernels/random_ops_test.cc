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

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

enum class InputType {
  kConst = 0,
  kDynamic = 1,
};

class RandomOpModel : public SingleOpModel {
 public:
  RandomOpModel(BuiltinOperator op_code, InputType input_type,
                const std::initializer_list<int32_t>& shape,
                int32_t seed = 0, int32_t seed2 = 0) {
    bool is_input_const = (input_type == InputType::kConst);
    if (is_input_const) {
      input_ = AddConstInput(TensorType_INT32, shape,
                             {static_cast<int32_t>(shape.size())});
    } else {
      input_ =
          AddInput({TensorType_INT32, {static_cast<int32_t>(shape.size())}});
    }
    output_ = AddOutput({TensorType_FLOAT32, {}});
    SetBuiltinOp(op_code, BuiltinOptions_RandomOptions,
                 CreateRandomOptions(builder_, seed, seed2).Union());
    BuildInterpreter({GetShape(input_)});
    if (!is_input_const) {
      PopulateTensor<int32_t>(input_, std::vector<int32_t>(shape));
    }
  }

  int input() { return input_; }
  int output() { return output_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

class TestSuite : public testing::TestWithParam<std::tuple<
    BuiltinOperator, InputType>> {
};

TEST_P(TestSuite, NonDeterministicOutputWithSeedsEqualToZero)
{
  BuiltinOperator op_code = std::get<0>(GetParam());
  InputType input_type = std::get<1>(GetParam());

  RandomOpModel m1(op_code, input_type,
                   /*shape=*/{100, 50, 5}, /*seed=*/0, /*seed2=*/0);
  m1.Invoke();
  std::vector<float> output1a = m1.GetOutput();
  EXPECT_EQ(output1a.size(), 100 * 50 * 5);
  m1.Invoke();
  std::vector<float> output1b = m1.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output1a, output1b);

  RandomOpModel m2(op_code, input_type,
                   /*shape=*/{100, 50, 5}, /*seed=*/0, /*seed2=*/0);
  m2.Invoke();
  std::vector<float> output2a = m2.GetOutput();
  EXPECT_EQ(output2a.size(), 100 * 50 * 5);
  m2.Invoke();
  std::vector<float> output2b = m2.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output2a, output2b);

  // Verify that outputs are non-deterministic (different random sequences)
  EXPECT_NE(output1a, output2a);
  EXPECT_NE(output1b, output2b);
}

TEST_P(TestSuite, DeterministicOutputWithNonZeroSeeds) {
  BuiltinOperator op_code = std::get<0>(GetParam());
  InputType input_type = std::get<1>(GetParam());

  RandomOpModel m1(op_code, input_type, /*shape=*/{100, 50, 5},
                   /*seed=*/1234, /*seed2=*/5678);
  m1.Invoke();
  std::vector<float> output1a = m1.GetOutput();
  EXPECT_EQ(output1a.size(), 100 * 50 * 5);
  m1.Invoke();
  std::vector<float> output1b = m1.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output1a, output1b);

  RandomOpModel m2(op_code, input_type, /*shape=*/{100, 50, 5},
                   /*seed=*/1234, /*seed2=*/5678);
  m2.Invoke();
  std::vector<float> output2a = m2.GetOutput();
  EXPECT_EQ(output2a.size(), 100 * 50 * 5);
  m2.Invoke();
  std::vector<float> output2b = m2.GetOutput();
  // Verify that consecutive outputs are different.
  EXPECT_NE(output2a, output2b);

  // Verify that outputs are determinisitc (same random sequence)
  EXPECT_EQ(output1a, output2a);
  EXPECT_EQ(output1b, output2b);
}

INSTANTIATE_TEST_SUITE_P(
    RandomOpTest, TestSuite,
    testing::Combine(
        testing::Values(BuiltinOperator_RANDOM_UNIFORM,
                        BuiltinOperator_RANDOM_STANDARD_NORMAL),
        testing::Values(InputType::kConst, InputType::kDynamic)),
     [](const testing::TestParamInfo<TestSuite::ParamType>& info) {
      std::string name = absl::StrCat(
          std::get<0>(info.param) == BuiltinOperator_RANDOM_UNIFORM ?
            "_RandomUniformOp" : "_RandomStandardNormalOp",
          std::get<1>(info.param) == InputType::kConst ?
            "_ConstInput" : "_DynamicInput");
      return name;
    }
    );

TEST(RandomUniformOpTest, TestOutputMeanAndVariance) {
  RandomOpModel m(/*op_code*/BuiltinOperator_RANDOM_UNIFORM,
                  /*input_type=*/InputType::kConst,
                  /*shape=*/{100, 50, 5}, /*seed=*/1234, /*seed2=*/5678);

  // Initialize output tensor to infinity to validate that all of its values are
  // updated and are normally distributed after Invoke().
  const std::vector<float> output_data(100 * 50 * 5,
                                       std::numeric_limits<float>::infinity());
  m.PopulateTensor(m.output(), output_data);
  m.Invoke();
  auto output = m.GetOutput();
  EXPECT_EQ(output.size(), 100 * 50 * 5);

  // For uniform distribution with min=0 and max=1:
  // * Mean = (max-min)/2 = 0.5
  // * Variance = 1/12 * (max-min)^2 = 1/12

  // Mean should be approximately 0.5
  double sum = 0;
  for (const auto r : output) {
    sum += r;
  }
  double mean = sum / output.size();
  ASSERT_LT(std::abs(mean - 0.5), 0.05);

  // Variance should be approximately 1/12
  double sum_squared = 0;
  for (const auto r : output) {
    sum_squared += std::pow(r - mean, 2);
  }
  double var = sum_squared / output.size();
  EXPECT_LT(std::abs(1. / 12 - var), 0.05);
}

TEST(RandomStandardNormalOpTest, TestOutputMeanAndVariance) {
  RandomOpModel m(/*op_code*/BuiltinOperator_RANDOM_STANDARD_NORMAL,
                  /*input_type=*/InputType::kConst,
                  /*shape=*/{100, 50, 5}, /*seed=*/1234, /*seed2=*/5678);

  // Initialize output tensor to infinity to validate that all of its values are
  // updated and are normally distributed after Invoke().
  const std::vector<float> output_data(100 * 50 * 5,
                                       std::numeric_limits<float>::infinity());
  m.PopulateTensor(m.output(), output_data);
  m.Invoke();
  auto output = m.GetOutput();
  EXPECT_EQ(output.size(), 100 * 50 * 5);

  // For uniform distribution with min=0 and max=1:
  // * Mean = (max-min)/2 = 0.5
  // * Variance = 1/12 * (max-min)^2 = 1/12

  // Mean should be approximately 0.5
  double sum = 0;
  for (const auto r : output) {
    sum += r;
  }
  double mean = sum / output.size();
  ASSERT_LT(std::abs(mean), 0.05);

  // Variance should be approximately 1/12
  double sum_squared = 0;
  for (const auto r : output) {
    sum_squared += std::pow(r - mean, 2);
  }
  double var = sum_squared / output.size();
  EXPECT_LT(std::abs(1.0 - var), 0.05);
}


}  // namespace
}  // namespace tflite
