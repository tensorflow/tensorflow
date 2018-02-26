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

class BaseDivOpModel : public SingleOpModel {
 public:
  BaseDivOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_DIV, BuiltinOptions_DivOptions,
                 CreateDivOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

// For quantized Div, the error shouldn't exceed (2*step + step^2).
// The param min=-1.0 & max=1.0 is used in the following tests.
// The tolerance value is ~0.0157.
const float kQuantizedStep = 2.0 / 255.0;
const float kQuantizedTolerance =
    2.0 * kQuantizedStep + kQuantizedStep * kQuantizedStep;

class QuantizedDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST(FloatDivOpTest, NoActivation) {
  FloatDivOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.5, 0.2, -1.5, 0.5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.4, 1.0, 0.8, 1.6})));
}

TEST(FloatDivOpTest, ActivationRELU_N1_TO_1) {
  FloatDivOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, -1.2, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, -1.5, 0.5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 1.0, 0.8, 1.0})));
}

TEST(FloatDivOpTest, VariousInputShapes) {
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatDivOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.6, 0.5, -1.1, -0.1});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-20.0, 1.0, 0.5, 1.6, -1.0, 20.0})))
        << "With shape number " << i;
  }
}

TEST(FloatDivOpTest, WithBroadcast) {
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatDivOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-0.2, 0.2, 0.07, 0.08, 0.11, -0.123});
    m.PopulateTensor<float>(m.input2(), {0.1});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.0, 2.0, 0.7, 0.8, 1.1, -1.23})))
        << "With shape number " << i;
  }
}

TEST(QuantizedDivOpTest, NoActivation) {
  QuantizedDivOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
                        {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
                        {TensorType_UINT8, {}, -1.0, 1.0},
                        ActivationFunctionType_NONE);
  m.QuantizeAndPopulate<uint8_t>(m.input1(), {-0.6, 0.2, 0.9, -0.7});
  m.QuantizeAndPopulate<uint8_t>(m.input2(), {0.8, 0.4, 0.9, -0.8});
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({-0.75, 0.5, 1.0, 0.875},
                                              kQuantizedTolerance)));
}

// for quantized Div, the error shouldn't exceed 2*step
float GetTolerance(int min, int max) {
  float kQuantizedStep = (max - min) / 255.0;
  float kQuantizedTolerance = 2.0 * kQuantizedStep;
  return kQuantizedTolerance;
}

TEST(QuantizedDivOpTest, WithBroadcast) {
  float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedDivOpModel m({TensorType_UINT8, test_shapes[i], -3.0, 3.0},
                          {TensorType_UINT8, {}, -3.0, 3.0},  // always a scalar
                          {TensorType_UINT8, {}, -3.0, 3.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<uint8_t>(m.input1(), {-0.2,  0.2,   0.07,
                                                0.08, 0.11, -0.123});
    m.QuantizeAndPopulate<uint8_t>(m.input2(), {0.1});
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23}, kQuantizedTolerance)))
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
