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
  BaseMulOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
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
// The param min=-1.0 & max=1.0 is used in the following tests.
// The tolerance value is ~0.0157.
const float kQuantizedStep = 2.0 / 255.0;
const float kQuantizedTolerance =
    2.0 * kQuantizedStep + kQuantizedStep * kQuantizedStep;
const float kQuantizedStepInt16 = 2.0 / 32767.0;
const float kQuantizedToleranceInt16 =
    2.0 * kQuantizedStepInt16 + kQuantizedStepInt16 * kQuantizedStepInt16;

class QuantizedMulOpModel : public BaseMulOpModel {
 public:
  using BaseMulOpModel::BaseMulOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

  std::vector<float> GetDequantizedOutputInt16() {
    return Dequantize<int16_t>(ExtractVector<int16_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST(FloatMulOpTest, NoActivation) {
  FloatMulOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4})));
}

TEST(FloatMulOpTest, ActivationRELU_N1_TO_1) {
  FloatMulOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 1.0})));
}

TEST(FloatMulOpTest, VariousInputShapes) {
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5, 1.1, 0.1});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4, 1.21, 0.2})))
        << "With shape number " << i;
  }
}

TEST(FloatMulOpTest, WithBroadcast) {
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatMulOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-0.2, 0.02, 0.07, 0.08, 0.11, 0.2})))
        << "With shape number " << i;
  }
}

TEST(QuantizedMulOpTest, NoActivation) {
  QuantizedMulOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
                        {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
                        {TensorType_UINT8, {}, -1.0, 1.0},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<uint8_t>(m.input1(), {-0.8, 0.2, 0.9, 0.7});
  m.QuantizeAndPopulate<uint8_t>(m.input2(), {0.6, 0.4, 0.9, 0.8});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedTolerance)));
}

TEST(QuantizedMulOpTest, NoActivationInt16) {
  const float kMin = -1.f;
  const float kMax = 32767.f / 32768.f;
  QuantizedMulOpModel m({TensorType_INT16, {1, 2, 2, 1}, kMin, kMax},
                        {TensorType_INT16, {1, 2, 2, 1}, kMin, kMax},
                        {TensorType_INT16, {}, kMin, kMax},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<int16_t>(m.input1(), {-0.8, 0.2, 0.9, 0.7});
  m.QuantizeAndPopulate<int16_t>(m.input2(), {0.6, 0.4, 0.9, 0.8});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutputInt16(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedToleranceInt16)));
}

TEST(QuantizedMulOpTest, NoActivationInt16WithUint8Output) {
  const float kMinInt16 = -1.f;
  const float kMaxInt16 = 32767.f / 32768.f;
  const float kMinUint8 = -1.f;
  const float kMaxUint8 = 127.f / 128.f;
  QuantizedMulOpModel m({TensorType_INT16, {1, 2, 2, 1}, kMinInt16, kMaxInt16},
                        {TensorType_INT16, {1, 2, 2, 1}, kMinInt16, kMaxInt16},
                        {TensorType_UINT8, {}, kMinUint8, kMaxUint8},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<int16_t>(m.input1(), {-0.8, 0.2, 0.9, 0.7});
  m.QuantizeAndPopulate<int16_t>(m.input2(), {0.6, 0.4, 0.9, 0.8});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-0.48, 0.08, 0.81, 0.56},
                                              kQuantizedTolerance)));
}

// for quantized Mul, the error shouldn't exceed 2*step
float GetTolerance(int min, int max) {
  float kQuantizedStep = (max - min) / 255.0;
  float kQuantizedTolerance = 2.0 * kQuantizedStep;
  return kQuantizedTolerance;
}

TEST(QuantizedMulOpTest, WithBroadcast) {
  float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedMulOpModel m({TensorType_UINT8, test_shapes[i], -3.0, 3.0},
                          {TensorType_UINT8, {}, -3.0, 3.0},  // always a scalar
                          {TensorType_UINT8, {}, -3.0, 3.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<uint8_t>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<uint8_t>(m.input2(), {0.1});
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-0.2, 0.02, 0.07, 0.08, 0.11, 0.2}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
