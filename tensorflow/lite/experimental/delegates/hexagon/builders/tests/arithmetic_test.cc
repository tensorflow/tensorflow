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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

class ArithmeticOpBaseModel : public SingleOpModelWithHexagon {
 public:
  ArithmeticOpBaseModel(const TensorData& input1, const TensorData& input2,
                        const TensorData& output)
      : SingleOpModelWithHexagon() {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
  }
  ArithmeticOpBaseModel(const TensorData& input1, const TensorData& input2,
                        const TensorData& output,
                        const std::initializer_list<uint8_t>& input1_data,
                        const std::initializer_list<uint8_t>& input2_data) {
    if (input1_data.size() > 0)
      input1_ = AddConstInput(input1, input1_data);
    else
      input1_ = AddInput(input1);
    if (input2_data.size() > 0)
      input2_ = AddConstInput(input2, input2_data);
    else
      input2_ = AddInput(input2);
    output_ = AddOutput(output);
  }

  void InitInterpreter() {
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  template <typename T>
  void SetInput1(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input1_, data);
  }

  template <typename T>
  void SetInput2(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input2_, data);
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

class AddOpModel : public ArithmeticOpBaseModel {
 public:
  AddOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output)
      : ArithmeticOpBaseModel(input1, input2, output) {}
  AddOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output,
             const std::initializer_list<uint8_t>& input1_data,
             const std::initializer_list<uint8_t>& input2_data)
      : ArithmeticOpBaseModel(input1, input2, output, input1_data,
                              input2_data) {}

  void InitInterpreter() {
    SetBuiltinOp(
        BuiltinOperator_ADD, BuiltinOptions_AddOptions,
        CreateAddOptions(builder_, ActivationFunctionType_NONE).Union());
    ArithmeticOpBaseModel::InitInterpreter();
  }
};

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation() {
  const float kQuantizedTolerance = 2.0 / 255.0;
  std::vector<std::vector<float>> inputs1 = {
      {0.1, 0.2, 0.3, 0.4}, {-0.8, 0.2, 0.4, 0.7}, {-0.8, 0.2, 0.7, 0.3}};
  std::vector<std::vector<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.8}, {0.6, 0.4, -0.8, 0.5}};
  for (size_t i = 0; i < 1; ++i) {
    AddOpModel m({tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                 {tensor_type, {1, 2, 2, 1}, -1.0, 1.0},
                 {tensor_type, {1, 2, 2, 1}, -1.0, 1.0});
    m.InitInterpreter();
    m.SetInput1<integer_dtype>(inputs1[i]);
    m.SetInput2<integer_dtype>(inputs2[i]);
    m.Invoke();
    auto reference_output = m.GetDequantizedOutput<integer_dtype>();
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(
        m.GetDequantizedOutput<integer_dtype>(),
        ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationUInt8) {
  QuantizedTestsNoActivation<TensorType_UINT8, uint8_t>();
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>();
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationUInt8_ConstInput_1) {
  const float kQuantizedTolerance = 2.0 / 255.0;
  AddOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {110, 142, 156, 171}, {});
  m.InitInterpreter();
  m.SetInput1<uint8_t>({0.1, 0.2, 0.3, 0.4});
  m.Invoke();
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)));
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationUInt8_ConstInput_2) {
  const float kQuantizedTolerance = 2.0 / 255.0;
  AddOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0}, {},
               {110, 142, 156, 171});
  m.InitInterpreter();
  m.SetInput2<uint8_t>({0.1, 0.2, 0.3, 0.4});
  m.Invoke();
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)));
}

TEST(QuantizedAddOpModel, QuantizedTestsNoActivationInt8_ConstInput) {
  const float kQuantizedTolerance = 2.0 / 255.0;
  AddOpModel m({TensorType_INT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_INT8, {1, 2, 2, 1}, -1.0, 1.0},
               {TensorType_INT8, {1, 2, 2, 1}, -1.0, 1.0}, {},
               {110, 101, 105, 120});
  m.InitInterpreter();
  m.SetInput2<int8_t>({0.1, 0.2, 0.3, 0.4});
  m.Invoke();
  auto reference_output = m.GetDequantizedOutput<int8_t>();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)));
}

}  // namespace tflite
