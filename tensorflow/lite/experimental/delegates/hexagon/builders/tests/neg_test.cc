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

class NegOpModel : public SingleOpModelWithHexagon {
 public:
  NegOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_NEG, BuiltinOptions_NegOptions,
                 CreateNegOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;
};

TEST(NegOpModel, NegTest) {
  NegOpModel m({TensorType_UINT8, {2, 3}, -10, 10},
               {TensorType_UINT8, {2, 3}, -10, 10});
  m.SetQuantizedInput({-2.0f, -1.0f, 0.f, 1.0f, 2.0f, 3.0f});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput(),
      ElementsAreArray(ArrayFloatNear({2.0f, 1.0f, 0.f, -1.0f, -2.0f, -3.0f},
                                      /*max_abs_error=*/0.1)));
}

}  // namespace tflite
