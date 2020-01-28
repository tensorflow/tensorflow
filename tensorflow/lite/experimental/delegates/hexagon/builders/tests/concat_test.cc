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

class QuantizedConcatenationOpModel : public SingleOpModelWithHexagon {
 public:
  QuantizedConcatenationOpModel(const std::vector<TensorData>& input_template,
                                int axis, const TensorData& output_template) {
    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < input_template.size(); ++i) {
      all_input_shapes.push_back(input_template[i].shape);
      AddInput(input_template[i]);
    }
    output_ = AddOutput({output_template.type, /*shape=*/{},
                         output_template.min, output_template.max});
    SetBuiltinOp(
        BuiltinOperator_CONCATENATION, BuiltinOptions_ConcatenationOptions,
        CreateConcatenationOptions(builder_, axis, ActivationFunctionType_NONE)
            .Union());
    BuildInterpreter(all_input_shapes);
  }

  template <typename T>
  void SetInput(int index, std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(index, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 private:
  int output_;
};

TEST(QuantizedConcatenationOpModel, FourInputsQuantizedSameRange) {
  QuantizedConcatenationOpModel m0(
      {{TensorType_UINT8, {2, 1, 1, 2}, -12.7, 12.8},
       {TensorType_UINT8, {2, 1, 1, 2}, -12.7, 12.8},
       {TensorType_UINT8, {2, 1, 1, 2}, -12.7, 12.8},
       {TensorType_UINT8, {2, 1, 1, 2}, -12.7, 12.8}},
      /*axis=*/3, {TensorType_UINT8, {}, -12.7, 12.8});

  m0.SetInput<uint8_t>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<uint8_t>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<uint8_t>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<uint8_t>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.ApplyDelegateAndInvoke();
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                      4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
                  },
                  /*max_abs_error=*/0.2)));
}

TEST(QuantizedConcatenationOpModel, FourInputsQuantizedMixedRange) {
  QuantizedConcatenationOpModel m0(
      {{TensorType_UINT8, {2, 1, 1, 2}, -10.7, 10.8},
       {TensorType_UINT8, {2, 1, 1, 2}, 0, 12.8},
       {TensorType_UINT8, {2, 1, 1, 2}, -11, 11.8},
       {TensorType_UINT8, {2, 1, 1, 2}, 0, 7.4}},
      /*axis=*/3, {TensorType_UINT8, {}, -12.7, 12.8});

  m0.SetInput<uint8_t>(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput<uint8_t>(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput<uint8_t>(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput<uint8_t>(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.ApplyDelegateAndInvoke();
  EXPECT_THAT(m0.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                      4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
                  },
                  /*max_abs_error=*/0.2)));
}

}  // namespace tflite
