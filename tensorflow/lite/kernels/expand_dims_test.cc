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
#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename InputType>
class ExpandDimsOpModel : public SingleOpModel {
 public:
  ExpandDimsOpModel(int axis, std::initializer_list<int> input_shape,
                    std::initializer_list<InputType> input_data,
                    TestType input_tensor_types) {
    if (input_tensor_types == TestType::kDynamic) {
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

    if (input_tensor_types == TestType::kDynamic) {
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

template <typename T>
class ExpandDimsOpTest : public ::testing::Test {
 public:
  static std::vector<TestType> range_;
};

template <>
std::vector<TestType> ExpandDimsOpTest<TestType>::range_{TestType::kConst,
                                                         TestType::kDynamic};

using DataTypes = ::testing::Types<float, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(ExpandDimsOpTest, DataTypes);

TYPED_TEST(ExpandDimsOpTest, PositiveAxisInplace) {
  std::initializer_list<TypeParam> values = {-1, 1, -2, 2};

  ExpandDimsOpModel<TypeParam> axis_0(0, {2, 2}, values, TestType::kConst);
  const int kInplaceInputTensorIdx = 0;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor =
      axis_0.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = axis_0.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  ASSERT_EQ(axis_0.Invoke(), kTfLiteOk);
  EXPECT_THAT(axis_0.GetValues(), ElementsAreArray(values));
  EXPECT_THAT(axis_0.GetOutputShape(), ElementsAreArray({1, 2, 2}));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TYPED_TEST(ExpandDimsOpTest, PositiveAxis) {
  for (TestType test_type : ExpandDimsOpTest<TestType>::range_) {
    std::initializer_list<TypeParam> values = {-1, 1, -2, 2};

    ExpandDimsOpModel<TypeParam> axis_0(0, {2, 2}, values, test_type);
    ASSERT_EQ(axis_0.Invoke(), kTfLiteOk);
    EXPECT_THAT(axis_0.GetValues(), ElementsAreArray(values));
    EXPECT_THAT(axis_0.GetOutputShape(), ElementsAreArray({1, 2, 2}));

    ExpandDimsOpModel<TypeParam> axis_1(1, {2, 2}, values, test_type);
    ASSERT_EQ(axis_1.Invoke(), kTfLiteOk);
    EXPECT_THAT(axis_1.GetValues(), ElementsAreArray(values));
    EXPECT_THAT(axis_1.GetOutputShape(), ElementsAreArray({2, 1, 2}));

    ExpandDimsOpModel<TypeParam> axis_2(2, {2, 2}, values, test_type);
    ASSERT_EQ(axis_2.Invoke(), kTfLiteOk);
    EXPECT_THAT(axis_2.GetValues(), ElementsAreArray(values));
    EXPECT_THAT(axis_2.GetOutputShape(), ElementsAreArray({2, 2, 1}));
  }
}

TYPED_TEST(ExpandDimsOpTest, NegativeAxis) {
  for (TestType test_type : ExpandDimsOpTest<TestType>::range_) {
    std::initializer_list<TypeParam> values = {-1, 1, -2, 2};

    ExpandDimsOpModel<TypeParam> m(-1, {2, 2}, values, test_type);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetValues(), ElementsAreArray(values));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1}));
  }
}

TEST(ExpandDimsOpTest, StrTensor) {
  std::initializer_list<std::string> values = {"abc", "de", "fghi"};

  // this test will fail on TestType::CONST
  ExpandDimsOpModel<std::string> m(0, {3}, values, TestType::kDynamic);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetValues(), ElementsAreArray(values));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
}

}  // namespace
}  // namespace tflite
