
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <gtest/gtest.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  CONST = 0,
  DYNAMIC = 1,
};

template <typename InputType>
class ExpandDimsOpModel : public SingleOpModel {
 public:
  ExpandDimsOpModel(int axis, std::initializer_list<int> input_shape,
                    std::initializer_list<InputType> input_data,
                    TestType input_tensor_types) {
    if (input_tensor_types == TestType::DYNAMIC) {
      input_ = AddInput(GetTensorType<InputType>());
      axis_ = AddInput(TensorType_INT32);
    } else {
      input_ =
          AddConstInput(GetTensorType<InputType>(), input_data, input_shape);
      axis_ = AddConstInput(TensorType_INT32, {axis}, {1});
    }
    output_ = AddOutput(GetTensorType<InputType>());
    SetBuiltinOp(BuiltinOperator_EXPAND_DIMS, BuiltinOptions_ExpandDimsOptions,
                 0);

    BuildInterpreter({input_shape, {1}});

    if (input_tensor_types == TestType::DYNAMIC) {
      PopulateTensor<InputType>(input_, input_data);
      PopulateTensor<int32_t>(axis_, {axis});
    }
  }
  std::vector<InputType> GetValues() {
    return ExtractVector<InputType>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int axis_;
  int output_;
};

class ExpandDimsOpTest : public ::testing::TestWithParam<TestType> {};

TEST_P(ExpandDimsOpTest, PositiveAxis) {
  std::initializer_list<float> values = {-1.f, 1.f, -2.f, 2.f};

  ExpandDimsOpModel<float> axis_0(0, {2, 2}, values, GetParam());
  axis_0.Invoke();
  EXPECT_THAT(axis_0.GetValues(), ElementsAreArray(values));
  EXPECT_THAT(axis_0.GetOutputShape(), ElementsAreArray({1, 2, 2}));

  ExpandDimsOpModel<float> axis_1(1, {2, 2}, values, GetParam());
  axis_1.Invoke();
  EXPECT_THAT(axis_1.GetValues(), ElementsAreArray(values));
  EXPECT_THAT(axis_1.GetOutputShape(), ElementsAreArray({2, 1, 2}));

  ExpandDimsOpModel<float> axis_2(2, {2, 2}, values, GetParam());
  axis_2.Invoke();
  EXPECT_THAT(axis_2.GetValues(), ElementsAreArray(values));
  EXPECT_THAT(axis_2.GetOutputShape(), ElementsAreArray({2, 2, 1}));
}

TEST_P(ExpandDimsOpTest, NegativeAxis) {
  std::initializer_list<float> values = {-1.f, 1.f, -2.f, 2.f};

  ExpandDimsOpModel<float> m(-1, {2, 2}, values, GetParam());
  m.Invoke();
  EXPECT_THAT(m.GetValues(), ElementsAreArray(values));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1}));
}

TEST_F(ExpandDimsOpTest, StrTensor) {
  std::initializer_list<std::string> values = {"abc", "de", "fghi"};

  ExpandDimsOpModel<std::string> m(0, {3}, values, TestType::DYNAMIC);
  m.Invoke();
  EXPECT_THAT(m.GetValues(), ElementsAreArray(values));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
}

INSTANTIATE_TEST_SUITE_P(ExpandDimsOpTest, ExpandDimsOpTest,
                         ::testing::Values(TestType::DYNAMIC, TestType::CONST));
}  // namespace
}  // namespace tflite
