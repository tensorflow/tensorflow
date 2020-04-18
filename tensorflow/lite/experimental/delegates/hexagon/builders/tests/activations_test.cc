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

namespace tflite {
using testing::ElementsAreArray;

class ActivationOpModel : public SingleOpModelWithHexagon {
 public:
  explicit ActivationOpModel(BuiltinOperator type, const TensorData& input,
                             const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  BuiltinOperator op_code_;

  int input_;
  int output_;
};

template <typename integer_type, TensorType tensor_dtype>
void ReluTestImpl() {
  const float kMin = -6;
  const float kMax = 6;
  ActivationOpModel model(BuiltinOperator_RELU,
                          /*input=*/{tensor_dtype, {1, 3}, kMin, kMax},
                          /*output=*/{tensor_dtype, {1, 3}, kMin, kMax});
  model.SetInput<integer_type>({1, 5, 7});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(
                  ArrayFloatNear({1.0, 5.0, 6.0}, /*max_abs_error=*/0.03)));
}

template <typename integer_type, TensorType tensor_dtype>
void Relu6TestImpl() {
  const float kMin = -8;
  const float kMax = 8;
  ActivationOpModel model(BuiltinOperator_RELU6,
                          /*input=*/{tensor_dtype, {1, 3}, kMin, kMax},
                          /*output=*/{tensor_dtype, {1, 3}, kMin, kMax});
  model.SetInput<integer_type>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(
                  ArrayFloatNear({4.0, 0.0, 6.0}, /*max_abs_error=*/0.03)));
}

template <typename integer_type, TensorType tensor_dtype>
void TanhTestImpl() {
  // Tanh values are always in this range.
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  ActivationOpModel model(BuiltinOperator_TANH,
                          /*input=*/{tensor_dtype, {1, 3}, 8 * kMin, 8 * kMax},
                          /*output=*/{tensor_dtype, {1, 3}, kMin, kMax});
  model.SetInput<integer_type>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear({1.00392, -0.752941, 1.00392},
                                              /*max_abs_error=*/0.03)));
}

template <typename integer_type, TensorType tensor_dtype>
void SigmoidTestImpl() {
  const float kMin = -8;
  const float kMax = 8;
  TensorData output;
  if (tensor_dtype == TensorType_UINT8) {
    output = {tensor_dtype, {}, 0, 0, 1. / 256};
  } else if (tensor_dtype == TensorType_INT8) {
    output = {tensor_dtype, {}, 0, 0, 1. / 256, -128};
  }
  // Sigmoid requires output min/max to be set to these numbers.
  ActivationOpModel model(BuiltinOperator_LOGISTIC,
                          /*input=*/{tensor_dtype, {1, 3}, kMin, kMax},
                          /*output=*/output);
  model.SetInput<integer_type>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear({0.977, 0.266, 0.996},
                                              /*max_abs_error=*/0.03)));
}

TEST(ActivationOpModel, ReluOutput_UInt8) {
  ReluTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ActivationOpModel, ReluOutput_Int8) {
  ReluTestImpl<int8_t, TensorType_INT8>();
}

TEST(ActivationOpModel, Relu6Output_UInt8) {
  Relu6TestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ActivationOpModel, Relu6Output_Int8) {
  Relu6TestImpl<int8_t, TensorType_INT8>();
}

TEST(ActivationOpModel, SigmoidOutput_UInt8) {
  SigmoidTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ActivationOpModel, SigmoidOutput_Int8) {
  SigmoidTestImpl<int8_t, TensorType_INT8>();
}

TEST(ActivationOpModel, TanhOutput_UInt8) {
  TanhTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ActivationOpModel, TanhOutput_Int8) {
  TanhTestImpl<int8_t, TensorType_INT8>();
}

}  // namespace tflite
