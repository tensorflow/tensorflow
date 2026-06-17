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

#include <initializer_list>
#include <vector>

#include "Eigen/Core"
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/types/half.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class RoundOpModel : public SingleOpModel {
 public:
  RoundOpModel(std::initializer_list<int> input_shape) {
    input_ = AddInput(GetTensorType<T>());
    output_ = AddOutput(GetTensorType<T>());
    SetBuiltinOp(BuiltinOperator_ROUND, BuiltinOptions_NONE, 0);
    BuildInterpreter({
        input_shape,
    });
  }

  int input() { return input_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(RoundOpTest, SingleDim) {
  RoundOpModel<float> model({6});
  model.PopulateTensor<float>(model.input(), {8.5, 0.0, 3.5, 4.2, -3.5, -4.5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({8, 0, 4, 4, -4, -4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
}

TEST(RoundOpTest, MultiDims) {
  RoundOpModel<float> model({2, 1, 1, 6});
  model.PopulateTensor<float>(
      model.input(), {0.0001, 8.0001, 0.9999, 9.9999, 0.5, -0.0001, -8.0001,
                      -0.9999, -9.9999, -0.5, -2.5, 1.5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({0, 8, 1, 10, 0, 0, -8, -1, -10, -0, -2, 2}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 6}));
}

TEST(RoundOpTest, Float16SingleDim) {
  RoundOpModel<half> model({6});
  model.PopulateTensor<half>(model.input(),
                             {half(8.5f), half(0.0f), half(3.5f), half(4.2f),
                              half(-3.5f), half(-4.5f)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  {half(8), half(0), half(4), half(4), half(-4), half(-4)}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
}

TEST(RoundOpTest, Float16MultiDims) {
  RoundOpModel<half> model({2, 1, 1, 6});
  model.PopulateTensor<half>(
      model.input(),
      {half(0.0001f), half(8.0001f), half(0.9999f), half(9.9999f), half(0.5f),
       half(-0.0001f), half(-8.0001f), half(-0.9999f), half(-9.9999f),
       half(-0.5f), half(-2.5f), half(1.5f)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({half(0), half(8), half(1), half(10), half(0),
                                half(0), half(-8), half(-1), half(-10),
                                half(-0), half(-2), half(2)}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 6}));
}

TEST(RoundOpTest, BFloat16SingleDim) {
  RoundOpModel<Eigen::bfloat16> model({6});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input(),
      {Eigen::bfloat16(8.5), Eigen::bfloat16(0.0), Eigen::bfloat16(3.5),
       Eigen::bfloat16(4.2), Eigen::bfloat16(-3.5), Eigen::bfloat16(-4.5)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({Eigen::bfloat16(8), Eigen::bfloat16(0),
                                Eigen::bfloat16(4), Eigen::bfloat16(4),
                                Eigen::bfloat16(-4), Eigen::bfloat16(-4)}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
}

TEST(RoundOpTest, BFloat16MultiDims) {
  RoundOpModel<Eigen::bfloat16> model({2, 1, 1, 6});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input(),
      {Eigen::bfloat16(0.0001), Eigen::bfloat16(8.0001),
       Eigen::bfloat16(0.9999), Eigen::bfloat16(9.9999), Eigen::bfloat16(0.5),
       Eigen::bfloat16(-0.0001), Eigen::bfloat16(-8.0001),
       Eigen::bfloat16(-0.9999), Eigen::bfloat16(-9.9999),
       Eigen::bfloat16(-0.5), Eigen::bfloat16(-2.5), Eigen::bfloat16(1.5)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray(
          {Eigen::bfloat16(0), Eigen::bfloat16(8), Eigen::bfloat16(1),
           Eigen::bfloat16(10), Eigen::bfloat16(0), Eigen::bfloat16(0),
           Eigen::bfloat16(-8), Eigen::bfloat16(-1), Eigen::bfloat16(-10),
           Eigen::bfloat16(-0), Eigen::bfloat16(-2), Eigen::bfloat16(2)}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 6}));
}

}  // namespace
}  // namespace tflite
