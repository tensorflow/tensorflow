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

TEST(ActivationOpModel, ReluOutput) {
  const float kMin = -6;
  const float kMax = 6;
  ActivationOpModel model(BuiltinOperator_RELU,
                          /*input=*/{TensorType_UINT8, {1, 3}, kMin, kMax},
                          /*output=*/{TensorType_UINT8, {1, 3}, kMin, kMax});
  model.SetInput<uint8_t>({1, 5, 7});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(
      model.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({1.0, 5.0, 6.0}, /*tolerance=*/0.03)));
}

TEST(ActivationOpModel, Relu6Output) {
  const float kMin = -8;
  const float kMax = 8;
  ActivationOpModel model(BuiltinOperator_RELU6,
                          /*input=*/{TensorType_UINT8, {1, 3}, kMin, kMax},
                          /*output=*/{TensorType_UINT8, {1, 3}, kMin, kMax});
  model.SetInput<uint8_t>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(
      model.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({4.0, 0.0, 6.0}, /*tolerance=*/0.03)));
}

TEST(ActivationOpModel, TanhOutput) {
  const float kMin = -8;
  const float kMax = 8;
  ActivationOpModel model(BuiltinOperator_TANH,
                          /*input=*/{TensorType_UINT8, {1, 3}, kMin, kMax},
                          /*output=*/{TensorType_UINT8, {1, 3}, kMin, kMax});
  model.SetInput<uint8_t>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({7.96, -6.09, 7.97}, /*tolerance=*/0.03)));
}

TEST(ActivationOpModel, SigmoidOutput) {
  const float kMin = -8;
  const float kMax = 8;
  // Sigmoid requires output min/max to be set to these numbers.
  ActivationOpModel model(
      BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_UINT8, {1, 3}, kMin, kMax},
      /*output=*/{TensorType_UINT8, {1, 3}, 0, 0, 1. / 256});
  model.SetInput<uint8_t>({4, -1.0, 8});
  model.ApplyDelegateAndInvoke();

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({0.977, 0.266, 0.996}, /*tolerance=*/0.03)));
}

}  // namespace tflite
