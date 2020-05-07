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

const float kTolerance = 2 * (1. / 256);

class SoftmaxOpModel : public SingleOpModelWithHexagon {
 public:
  SoftmaxOpModel(float softmax_beta, const TensorData& input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else if (input.type == TensorType_INT8) {
      output_ = AddOutput({TensorType_INT8, {}, 0, 0, 1. / 256, -128});
    }
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, softmax_beta).Union());
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

 protected:
  int input_;
  int output_;
};

TEST(SoftmaxOpModel, Softmax4DUint8) {
  SoftmaxOpModel m(0.1,
                   /*input=*/{TensorType_UINT8, {1, 2, 1, 4}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kTolerance)));
}

TEST(SoftmaxOpModel, Softmax4DUint8_MultipleBatch) {
  SoftmaxOpModel m(0.1,
                   /*input=*/{TensorType_UINT8, {4, 1, 1, 2}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.645656, 0.354344,  //
                      0.450166, 0.549834,  //
                      0.622459, 0.377541,  //
                      0.710949, 0.28905,   //
                  },
                  kTolerance)));
}

TEST(SoftmaxOpModel, Softmax4DInt8) {
  SoftmaxOpModel m(0.1,
                   /*input=*/{TensorType_INT8, {1, 2, 1, 4}, -10, 10});
  m.SetInput<int8_t>({
      0, -6, 2, 4,   // depth = 0
      3, -2, 10, 1,  // depth = 1
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      .23463, .12877, .28658, .35003,  //
                      .22528, .13664, .45365, .18443,  //
                  },
                  kTolerance)));
}

TEST(SoftmaxOpModel, Softmax4DInt8_MultipleBatch) {
  SoftmaxOpModel m(0.1,
                   /*input=*/{TensorType_INT8, {4, 1, 1, 2}, -10, 10});
  m.SetInput<int8_t>({
      0, -6,  //
      2, 4,   //
      3, -2,  //
      10, 1,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        0.645656, 0.354344,  //
                                                        0.450166, 0.549834,  //
                                                        0.622459, 0.377541,  //
                                                        0.710949, 0.28905,   //
                                                    },
                                                    kTolerance)));
}

}  // namespace tflite
