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
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

class MulOpModel : public SingleOpModelWithHexagon {
 public:
  explicit MulOpModel(const TensorData& input1, const TensorData& input2,
                      const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MUL, BuiltinOptions_MulOptions,
                 CreateMulOptions(builder_, activation_type).Union());
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
  BuiltinOperator op_code_;

  int input1_;
  int input2_;
  int output_;
};

TEST(MulOpModel, MulOutput) {
  MulOpModel model(
      /*input1=*/{TensorType_UINT8, {2, 3}, -0.44f, 8.0f},
      /*input2=*/{TensorType_UINT8, {1, 3}, 0, 0.999f},
      /*output=*/{TensorType_UINT8, {2, 3}, -0.44f, 4.996f});
  model.SetInput1<uint8_t>({1, 2, 3, 4, 5, 6});
  model.SetInput2<uint8_t>({0.1f, 0.2f, 0.3f});

  // Reference output.
  model.Invoke();
  auto reference_out = model.GetDequantizedOutput<uint8_t>();

  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_out, 0.03)));
}

TEST(MulOpModel, MulOutput_LargeInputRange) {
  MulOpModel model(
      /*input1=*/{TensorType_UINT8, {1, 2, 2, 3}, -0.44f, 55.7f},
      /*input2=*/{TensorType_UINT8, {1, 1, 2, 3}, 0, 0.999f},
      /*output=*/{TensorType_UINT8, {1, 2, 2, 3}, -0.44f, 4.996f});
  model.SetInput1<uint8_t>({1, 2, 3, 4, 5, 6, 20, 30, 40, 50, 52, 55});
  model.SetInput2<uint8_t>({0.8f, 0.9f, 0.99f, 0.8f, 0.9f, 0.99f});

  // Reference output.
  model.Invoke();
  auto reference_out = model.GetDequantizedOutput<uint8_t>();

  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 3}));
  EXPECT_THAT(model.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_out, 0.03)));
}

}  // namespace tflite
