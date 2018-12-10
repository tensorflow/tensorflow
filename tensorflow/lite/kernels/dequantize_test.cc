/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class DequantizeOpModel : public SingleOpModel {
 public:
  DequantizeOpModel(TensorType type, std::initializer_list<int> shape,
                    float scale, int32_t zero_point) {
    TensorData input_tensor_data;
    input_tensor_data.type = type;
    input_tensor_data.shape = shape;
    input_tensor_data.min = 0;
    input_tensor_data.max = 0;
    input_tensor_data.scale = scale;
    input_tensor_data.zero_point = zero_point;
    input_ = AddInput(input_tensor_data);
    output_ = AddOutput({TensorType_FLOAT32, shape});
    SetBuiltinOp(BuiltinOperator_DEQUANTIZE, BuiltinOptions_DequantizeOptions,
                 CreateDequantizeOptions(builder_).Union());

    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(DequantizeOpTest, UINT8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  DequantizeOpModel m(TensorType_UINT8, {2, 5}, 0.5, 127);

  m.SetInput<uint8>({0, 1, 2, 3, 4, 251, 252, 253, 254, 255});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64})));
}

TEST(DequantizeOpTest, INT8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  DequantizeOpModel m(TensorType_INT8, {2, 5}, 0.5, -1);

  m.SetInput<int8>({-128, -127, -126, -125, -124, 123, 124, 125, 126, 127});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64})));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
