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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

class PadOpConstModel : public SingleOpModelWithHexagon {
 public:
  PadOpConstModel(const TensorData& input,
                  std::initializer_list<int> paddings_shape,
                  std::initializer_list<int> paddings,
                  const TensorData& output) {
    this->input_ = AddInput(input);
    paddings_ = AddConstInput(TensorType_INT32, paddings, paddings_shape);
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_PAD, BuiltinOptions_PadOptions,
                 CreatePadOptions(builder_).Union());
    BuildInterpreter({input.shape});
  }

  template <typename integer_type>
  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<integer_type>(input_, data);
  }

  void SetPaddings(std::initializer_list<int> paddings) {
    PopulateTensor<int>(paddings_, paddings);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename integer_type>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_type>(ExtractVector<integer_type>(output_),
                                    GetScale(output_), GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;
  int paddings_;
};

template <typename integer_type, TensorType tensor_dtype>
void SimpleConstTestImpl() {
  const float quantization_tolerance = 2 / 255.0;
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadOpConstModel m({tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2},
                    {0, 0, 1, 1, 1, 1, 0, 0}, {tensor_dtype, {}, -1.0, 1.0});
  m.SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  quantization_tolerance)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedConstTestImpl() {
  const float quantization_tolerance = 2 / 255.0;
  PadOpConstModel m({tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2},
                    {0, 0, 0, 2, 1, 3, 0, 0}, {tensor_dtype, {}, -1.0, 1.0});
  m.SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<integer_type>(),
              ElementsAreArray(ArrayFloatNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  quantization_tolerance)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST(PadOpConstModel, SimpleConstTest_UInt8) {
  SimpleConstTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(PadOpConstModel, SimpleConstTest_Int8) {
  SimpleConstTestImpl<int8_t, TensorType_INT8>();
}

TEST(PadOpConstModel, AdvancedConstTest_UInt8) {
  AdvancedConstTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(PadOpConstModel, AdvancedConstTest_Int8) {
  AdvancedConstTestImpl<int8_t, TensorType_INT8>();
}

}  // namespace tflite
