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
#include <stdint.h>

#include <initializer_list>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <class T>
class MaxMinOpModel : public SingleOpModel {
 public:
  MaxMinOpModel(tflite::BuiltinOperator op, const TensorData& input1,
                const TensorData& input2, const TensorType& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(op, BuiltinOptions_MaximumMinimumOptions,
                 CreateMaximumMinimumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  MaxMinOpModel(tflite::BuiltinOperator op, const TensorData& input1,
                const TensorData& input2,
                std::initializer_list<T> input2_values,
                const TensorType& output) {
    input1_ = AddInput(input1);
    input2_ = AddConstInput<T>(input2, input2_values);
    output_ = AddOutput(output);
    SetBuiltinOp(op, BuiltinOptions_MaximumMinimumOptions,
                 CreateMaximumMinimumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  void SetInput1(std::initializer_list<T> data) {
    PopulateTensor(input1_, data);
  }

  void SetInput2(std::initializer_list<T> data) {
    PopulateTensor(input2_, data);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

template <typename data_type>
void TestModel(tflite::BuiltinOperator op, const TensorData& input1,
               const TensorData& input2, const TensorData& output,
               std::initializer_list<data_type> input1_values,
               std::initializer_list<data_type> input2_values,
               std::initializer_list<data_type> output_values,
               int is_constant = false) {
  std::unique_ptr<MaxMinOpModel<data_type>> m;
  if (is_constant) {
    m = std::make_unique<MaxMinOpModel<data_type>>(op, input1, input2,
                                                   input2_values, output.type);
  } else {
    m = std::make_unique<MaxMinOpModel<data_type>>(op, input1, input2,
                                                   output.type);
    m->SetInput2(input2_values);
  }
  m->SetInput1(input1_values);

  ASSERT_EQ(m->InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m->GetOutputShape(), ElementsAreArray(output.shape));
  EXPECT_THAT(m->GetOutput(), ElementsAreArray(output_values));
}

TEST(MaximumOpTest, FloatTest) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  std::initializer_list<float> data2 = {-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  TestModel<float>(BuiltinOperator_MAXIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}}, data1, data2,
                   {1.0, 0.0, 1.0, 12.0, -2.0, -1.43});
  TestModel<float>(BuiltinOperator_MINIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}}, data1, data2,
                   {-1.0, 0.0, -1.0, 11.0, -3.0, -1.44});
}

TEST(MaxMinOpTest, Uint8Test) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  TestModel<uint8_t>(BuiltinOperator_MAXIMUM, {TensorType_UINT8, {3, 1, 2}},
                     {TensorType_UINT8, {3, 1, 2}},
                     {TensorType_UINT8, {3, 1, 2}}, data1, data2,
                     {1, 0, 2, 12, 255, 23});
  TestModel<uint8_t>(BuiltinOperator_MINIMUM, {TensorType_UINT8, {3, 1, 2}},
                     {TensorType_UINT8, {3, 1, 2}},
                     {TensorType_UINT8, {3, 1, 2}}, data1, data2,
                     {0, 0, 1, 11, 2, 1});
}

TEST(MaxMinOpTest, Int8Test) {
  std::initializer_list<int8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<int8_t> data2 = {0, 0, 1, 12, 123, 1};
  TestModel<int8_t>(BuiltinOperator_MAXIMUM, {TensorType_INT8, {3, 1, 2}},
                    {TensorType_INT8, {3, 1, 2}}, {TensorType_INT8, {3, 1, 2}},
                    data1, data2, {1, 0, 2, 12, 123, 23});
  TestModel<int8_t>(BuiltinOperator_MINIMUM, {TensorType_INT8, {3, 1, 2}},
                    {TensorType_INT8, {3, 1, 2}}, {TensorType_INT8, {3, 1, 2}},
                    data1, data2, {0, 0, 1, 11, 2, 1});
}

TEST(MaxMinOpTest, Int16Test) {
  std::initializer_list<int16_t> data1 = {-32768, 0, 2, 11, 2, 23};
  std::initializer_list<int16_t> data2 = {0, 0, 1, 32767, 123, 1};
  TestModel<int16_t>(BuiltinOperator_MAXIMUM, {TensorType_INT16, {3, 1, 2}},
                     {TensorType_INT16, {3, 1, 2}},
                     {TensorType_INT16, {3, 1, 2}}, data1, data2,
                     {0, 0, 2, 32767, 123, 23});
  TestModel<int16_t>(BuiltinOperator_MINIMUM, {TensorType_INT16, {3, 1, 2}},
                     {TensorType_INT16, {3, 1, 2}},
                     {TensorType_INT16, {3, 1, 2}}, data1, data2,
                     {-32768, 0, 1, 11, 2, 1});
}

TEST(MaximumOpTest, FloatWithBroadcastTest) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  std::initializer_list<float> data2 = {0.5, 2.0};
  TestModel<float>(BuiltinOperator_MAXIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {3, 1, 2}},
                   data1, data2, {1.0, 2.0, 0.5, 2.0, 0.5, 11.0});
  TestModel<float>(BuiltinOperator_MINIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {3, 1, 2}},
                   data1, data2, {0.5, 0.0, -1.0, -2.0, -1.44, 2.0});
}

TEST(MaximumOpTest, FloatWithBroadcastTest_ScalarY) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  std::initializer_list<float> data2 = {0.5};
  TestModel<float>(BuiltinOperator_MAXIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {3, 1, 2}},
                   data1, data2, {1.0, 0.5, 0.5, 0.5, 0.5, 11.0},
                   /*is_constant=*/true);
  TestModel<float>(BuiltinOperator_MINIMUM, {TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {3, 1, 2}},
                   data1, data2, {0.5, 0.0, -1.0, -2.0, -1.44, 0.5},
                   /*is_constant=*/true);
}

TEST(MaximumOpTest, Int32WithBroadcastTest) {
  std::initializer_list<int32_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int32_t> data2 = {2};
  TestModel<int32_t>(BuiltinOperator_MAXIMUM, {TensorType_INT32, {3, 1, 2}},
                     {TensorType_INT32, {1}}, {TensorType_INT32, {3, 1, 2}},
                     data1, data2, {2, 2, 2, 2, 3, 11});
  TestModel<int32_t>(BuiltinOperator_MINIMUM, {TensorType_INT32, {3, 1, 2}},
                     {TensorType_INT32, {1}}, {TensorType_INT32, {3, 1, 2}},
                     data1, data2, {1, 0, -1, -2, 2, 2});
}

TEST(MaximumOpTest, Int32WithBroadcastTest_ScalarY) {
  std::initializer_list<int32_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int32_t> data2 = {2};
  TestModel<int32_t>(BuiltinOperator_MAXIMUM, {TensorType_INT32, {3, 1, 2}},
                     {TensorType_INT32, {}}, {TensorType_INT32, {3, 1, 2}},
                     data1, data2, {2, 2, 2, 2, 3, 11}, /*is_constant=*/true);
  TestModel<int32_t>(BuiltinOperator_MINIMUM, {TensorType_INT32, {3, 1, 2}},
                     {TensorType_INT32, {}}, {TensorType_INT32, {3, 1, 2}},
                     data1, data2, {1, 0, -1, -2, 2, 2}, /*is_constant=*/true);
}

TEST(MaximumOpTest, Int8WithBroadcastTest_ScalarY) {
  std::initializer_list<int8_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int8_t> data2 = {2};
  TestModel<int8_t>(BuiltinOperator_MAXIMUM, {TensorType_INT8, {3, 1, 2}},
                    {TensorType_INT8, {}}, {TensorType_INT8, {3, 1, 2}}, data1,
                    data2, {2, 2, 2, 2, 3, 11}, /*is_constant=*/true);
  TestModel<int8_t>(BuiltinOperator_MINIMUM, {TensorType_INT8, {3, 1, 2}},
                    {TensorType_INT8, {}}, {TensorType_INT8, {3, 1, 2}}, data1,
                    data2, {1, 0, -1, -2, 2, 2}, /*is_constant=*/true);
}

TEST(MaxMinOpTest, Int8Test8D) {
  std::initializer_list<int8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<int8_t> data2 = {0, 0, 1, 12, 123, 1};
  TestModel<int8_t>(BuiltinOperator_MAXIMUM,
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}},
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}},
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}}, data1, data2,
                    {1, 0, 2, 12, 123, 23});
  TestModel<int8_t>(BuiltinOperator_MINIMUM,
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}},
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}},
                    {TensorType_INT8, {3, 1, 2, 1, 1, 1, 1, 1}}, data1, data2,
                    {0, 0, 1, 11, 2, 1});
}

TEST(MaximumOpTest, FloatWithBroadcastTest5D) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  std::initializer_list<float> data2 = {0.5, 2.0};
  TestModel<float>(
      BuiltinOperator_MAXIMUM, {TensorType_FLOAT32, {3, 1, 1, 1, 2}},
      {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {3, 1, 1, 1, 2}}, data1,
      data2, {1.0, 2.0, 0.5, 2.0, 0.5, 11.0});
  TestModel<float>(
      BuiltinOperator_MINIMUM, {TensorType_FLOAT32, {3, 1, 1, 1, 2}},
      {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {3, 1, 1, 1, 2}}, data1,
      data2, {0.5, 0.0, -1.0, -2.0, -1.44, 2.0});
}

TEST(MaximumOpTest, Int32WithBroadcastTest5D) {
  std::initializer_list<int32_t> data1 = {1, 0, -1, -2, 3, 11};
  std::initializer_list<int32_t> data2 = {2};
  TestModel<int32_t>(
      BuiltinOperator_MAXIMUM, {TensorType_INT32, {3, 1, 2, 1, 1}},
      {TensorType_INT32, {1}}, {TensorType_INT32, {3, 1, 2, 1, 1}}, data1,
      data2, {2, 2, 2, 2, 3, 11});
  TestModel<int32_t>(
      BuiltinOperator_MINIMUM, {TensorType_INT32, {3, 1, 2, 1, 1}},
      {TensorType_INT32, {1}}, {TensorType_INT32, {3, 1, 2, 1, 1}}, data1,
      data2, {1, 0, -1, -2, 2, 2});
}
}  // namespace
}  // namespace tflite
