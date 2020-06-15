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

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

template <typename data_type>
class MinMaxOpModel : public SingleOpModelWithHexagon {
 public:
  MinMaxOpModel(tflite::BuiltinOperator op, const TensorData& input1,
                const TensorData& input2, const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(op, BuiltinOptions_MaximumMinimumOptions,
                 CreateMaximumMinimumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  MinMaxOpModel(tflite::BuiltinOperator op, const TensorData& input1,
                std::initializer_list<data_type> input1_values,
                const TensorData& input2,
                std::initializer_list<data_type> input2_values,
                const TensorData& output, bool input1_const) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(op, BuiltinOptions_MaximumMinimumOptions,
                 CreateMaximumMinimumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});

    // A workaround to mark the tensors as constant.
    if (input1_const) {
      auto* input1_tensor = interpreter_->tensor(input1_);
      input1_tensor->allocation_type = kTfLiteMmapRo;
    } else {
      auto* input2_tensor = interpreter_->tensor(input2_);
      input2_tensor->allocation_type = kTfLiteMmapRo;
    }
  }

  void SetInput1(std::vector<data_type> data) { PopulateTensor(input1_, data); }

  void SetInput2(std::vector<data_type> data) { PopulateTensor(input2_, data); }

  std::vector<data_type> GetOutput() {
    return ExtractVector<data_type>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

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
               std::initializer_list<data_type> input2_values) {
  std::unique_ptr<MinMaxOpModel<data_type>> m;
  m = std::make_unique<MinMaxOpModel<data_type>>(op, input1, input2, output);
  m->SetInput1(input1_values);
  m->SetInput2(input2_values);

  m->Invoke();
  const auto reference_output = m->GetOutput();
  const auto reference_output_shape = m->GetOutputShape();
  m->ApplyDelegateAndInvoke();
  EXPECT_THAT(m->GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m->GetOutput(), ElementsAreArray(reference_output));
}

template <typename data_type>
void TestModelConstInput(tflite::BuiltinOperator op, const TensorData& input1,
                         const TensorData& input2, const TensorData& output,
                         std::initializer_list<data_type> input1_values,
                         std::initializer_list<data_type> input2_values,
                         bool input1_const) {
  std::unique_ptr<MinMaxOpModel<data_type>> m;
  m = std::make_unique<MinMaxOpModel<data_type>>(
      op, input1, input1_values, input2, input2_values, output, input1_const);
  m->SetInput1(input1_values);
  m->SetInput2(input2_values);

  m->Invoke();
  const auto reference_output = m->GetOutput();
  const auto reference_output_shape = m->GetOutputShape();
  m->ApplyDelegateAndInvoke();
  EXPECT_THAT(m->GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m->GetOutput(), ElementsAreArray(reference_output));
}

TEST(MinMaxOpTest, Maximum_Uint8Test) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  TestModel<uint8_t>(BuiltinOperator_MAXIMUM,
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255}, data1, data2);
}

TEST(MinMaxOpTest, Maximum_Uint8Test_Const) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  TestModelConstInput<uint8_t>(
      BuiltinOperator_MAXIMUM, {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
      {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
      {TensorType_UINT8, {1, 3, 1, 2}, -1, 255}, data1, data2, false);
}

TEST(MinMaxOpTest, Minimum_Uint8Test) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 255, 1};
  TestModel<uint8_t>(BuiltinOperator_MINIMUM,
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255},
                     {TensorType_UINT8, {1, 3, 1, 2}, -1, 255}, data1, data2);
}

TEST(MinMaxOpTest, Minimum_Uint8Test_Const) {
  std::initializer_list<uint8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<uint8_t> data2 = {0, 0, 1, 12, 20, 1};
  TestModelConstInput<uint8_t>(
      BuiltinOperator_MINIMUM, {TensorType_UINT8, {1, 3, 1, 2}, -1, 25},
      {TensorType_UINT8, {1, 3, 1, 2}, -1, 25},
      {TensorType_UINT8, {1, 3, 1, 2}, -1, 25}, data1, data2, false);
}

TEST(MinMaxOpTest, Maximum_Int8Test) {
  std::initializer_list<int8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<int8_t> data2 = {0, 0, 1, 12, 123, 1};
  TestModel<int8_t>(BuiltinOperator_MAXIMUM,
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 125},
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 125},
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 125}, data1, data2);
}

TEST(MinMaxOpTest, Minimum_Int8Test) {
  std::initializer_list<int8_t> data1 = {1, 0, 2, 11, 2, 23};
  std::initializer_list<int8_t> data2 = {0, 0, 1, 12, 12, 1};
  TestModel<int8_t>(BuiltinOperator_MINIMUM,
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 25},
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 25},
                    {TensorType_INT8, {1, 3, 1, 2}, -1, 25}, data1, data2);
}

}  // namespace tflite
