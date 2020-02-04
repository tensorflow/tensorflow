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

class AveragePoolingOpModel : public SingleOpModelWithHexagon {
 public:
  explicit AveragePoolingOpModel(const TensorData& input, int filter_width,
                                 int filter_height, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_AVERAGE_POOL_2D, BuiltinOptions_Pool2DOptions,
                 CreatePool2DOptions(builder_, Padding_VALID, /*stride_w=*/2,
                                     /*stride_h=*/2, filter_width,
                                     filter_height, ActivationFunctionType_NONE)
                     .Union());

    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

 private:
  int input_;
  int output_;
};

TEST(QuantizedPoolingOpTest, AveragePool) {
  AveragePoolingOpModel m(
      /*input=*/{TensorType_UINT8, {1, 16, 8, 1}, 0, 10},
      /*filter_width=*/8, /*filter_height=*/8,
      /*output=*/{TensorType_UINT8, {}, 0, 10});
  m.SetInput({
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      0, 6, 2,  4, 0, 6, 2,  4,  //
      3, 2, 10, 7, 3, 2, 10, 7,  //
  });
  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {4.58824, 4.58824, 4.90196, 4.58824, 4.27451})));
}

}  // namespace tflite
