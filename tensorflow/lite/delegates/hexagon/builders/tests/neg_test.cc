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
#include <initializer_list>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

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

  template <typename integer_type>
  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<integer_type>(input_, data);
  }

  template <typename integer_type>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_type>(ExtractVector<integer_type>(output_),
                                    GetScale(output_), GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;
};

TEST(NegOpModel, NegTest_UInt8) {
  NegOpModel m({TensorType_UINT8, {2, 3}, -4, 4},
               {TensorType_UINT8, {2, 3}, -4, 4});
  m.SetQuantizedInput<uint8_t>({-2.0f, -1.0f, 0.f, 1.0f, 2.0f, 3.0f});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<uint8_t>(),
      ElementsAreArray(ArrayFloatNear({2.0f, 1.0f, 0.f, -1.0f, -2.0f, -3.0f},
                                      /*max_abs_error=*/0.05)));
}

TEST(NegOpModel, NegTest_Int8) {
  NegOpModel m({TensorType_INT8, {2, 3}, -4, 4},
               {TensorType_INT8, {2, 3}, -4, 4});
  m.SetQuantizedInput<int8_t>({-2.0f, -1.0f, 0.f, 1.0f, 2.0f, 3.0f});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(
      m.GetDequantizedOutput<int8_t>(),
      ElementsAreArray(ArrayFloatNear({2.0f, 1.0f, 0.f, -1.0f, -2.0f, -3.0f},
                                      /*max_abs_error=*/0.05)));
}

}  // namespace tflite
