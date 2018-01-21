/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseMulOpModel : public SingleOpModel {
 public:
  BaseMulOpModel(TensorData input, TensorData output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input);
    input2_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MUL, BuiltinOptions_MulOptions,
                 CreateMulOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatMulOpModel : public BaseMulOpModel {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// For quantized Mul, the error shouldn't exceed (2*step + step^2).
// The param min=-1.0f & max=1.0f is used in the following tests.
// The tolerance value is ~0.0157f.
const float kQuantizedStep = 2.0f / 255.0f;
const float kQuantizedTolerance =
    2.0f * kQuantizedStep + kQuantizedStep * kQuantizedStep;

class QuantizedMulOpModel : public BaseMulOpModel {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST(FloatMulOpTest, NoActivation) {
  FloatMulOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0f, 0.2f, 0.7f, 0.8f});
  m.PopulateTensor<float>(m.input2(), {0.1f, 0.2f, 0.3f, 0.5f});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2f, 0.04f, 0.21f, 0.4f})));
}

TEST(FloatMulOpTest, ActivationRELU1) {
  FloatMulOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU1);
  m.PopulateTensor<float>(m.input1(), {-2.0f, 0.2f, 0.7f, 0.8f});
  m.PopulateTensor<float>(m.input2(), {0.1f, 0.2f, 0.3f, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2f, 0.04f, 0.21f, 1.0f})));
}

TEST(FloatMulOpTest, VariousInputShapes) {
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0f, 0.2f, 0.7f, 0.8f, 1.1f, 2.0f});
    m.PopulateTensor<float>(m.input2(), {0.1f, 0.2f, 0.3f, 0.5f, 1.1f, 0.1f});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-0.2f, 0.04f, 0.21f, 0.4f, 1.21f, 0.2f})))
        << "With shape number " << i;
  }
}

TEST(QuantizedMulOpTest, NoActivation) {
  QuantizedMulOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0f, 1.0f},
                        {TensorType_UINT8, {}, -1.0f, 1.0f},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<uint8_t>(m.input1(), {-0.8f, 0.2f, 0.9f, 0.7f});
  m.QuantizeAndPopulate<uint8_t>(m.input2(), {0.6f, 0.4f, 0.9f, 0.8f});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-0.48f, 0.08f, 0.81f, 0.56f},
                                              kQuantizedTolerance)));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
