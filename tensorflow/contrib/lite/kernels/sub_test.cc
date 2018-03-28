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

class BaseSubOpModel : public SingleOpModel {
 public:
  BaseSubOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output,
                 ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SUB, BuiltinOptions_SubOptions,
                 CreateSubOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

class FloatSubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class QuantizedSubOpModel : public BaseSubOpModel {
 public:
  using BaseSubOpModel::BaseSubOpModel;

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

// for quantized Sub, the error shouldn't exceed 2*step
float GetTolerance(int min, int max) {
  float kQuantizedStep = (max - min) / 255.0;
  float kQuantizedTolerance = 2.0 * kQuantizedStep;
  return kQuantizedTolerance;
}

TEST(FloatSubOpModel, NoActivation) {
  FloatSubOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3})));
}

TEST(FloatSubOpModel, ActivationRELU_N1_TO_1) {
  FloatSubOpModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-1.0, 0.0, 1.0, -0.3})));
}

TEST(FloatSubOpModel, VariousInputShapes) {
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.8, -1.1, 0.1});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 1.4, -0.3, 0.0, 1.9})))
        << "With shape number " << i;
  }
}

TEST(FloatSubOpModel, WithBroadcast) {
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    FloatSubOpModel m({TensorType_FLOAT32, test_shapes[i]},
                      {TensorType_FLOAT32, {}},  // always a scalar
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0});
    m.PopulateTensor<float>(m.input2(), {0.5});
    m.Invoke();
    EXPECT_THAT(
        m.GetOutput(),
        ElementsAreArray(ArrayFloatNear({-2.5, -0.3, 1.2, 0.0, -1.6, 1.5})))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsNoActivation) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<std::initializer_list<float>> inputs1 = {
      {0.1, 0.2, 0.3, 0.4}, {-0.2, 0.2, 0.4, 0.7}, {-0.01, 0.2, 0.7, 0.3}};
  std::vector<std::initializer_list<float>> inputs2 = {
      {0.6, 0.4, 0.3, 0.1}, {0.6, 0.4, 0.5, -0.2}, {0.6, 0.4, -0.18, 0.5}};
  std::vector<std::initializer_list<float>> results = {
      {-0.5, -0.2, 0.0, 0.3},
      {-0.8, -0.2, -0.1, 0.9},
      {-0.61, -0.2, 0.88, -0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
                          {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
                          {TensorType_UINT8, {}, -1.0, 1.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<uint8_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<uint8_t>(m.input2(), inputs2[i]);
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                              results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedTestsActivationRELU_N1_TO_1) {
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::vector<std::initializer_list<float>> inputs1 = {{-0.8, 0.2, 0.9, 0.7},
                                                       {-0.8, 0.2, 0.7, 0.5}};
  std::vector<std::initializer_list<float>> inputs2 = {{0.6, 0.4, 0.9, -0.8},
                                                       {0.6, 0.4, -0.8, 0.3}};
  std::vector<std::initializer_list<float>> results = {{-1.0, -0.2, 0.0, 1.0},
                                                       {-1.0, -0.2, 1.0, 0.2}};
  for (int i = 0; i < inputs1.size(); ++i) {
    QuantizedSubOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
                          {TensorType_UINT8, {1, 2, 2, 1}, -1.0, 1.0},
                          {TensorType_UINT8, {}, -1.0, 1.0},
                          ActivationFunctionType_RELU_N1_TO_1);
    m.QuantizeAndPopulate<uint8_t>(m.input1(), inputs1[i]);
    m.QuantizeAndPopulate<uint8_t>(m.input2(), inputs2[i]);
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput(), ElementsAreArray(ArrayFloatNear(
                                              results[i], kQuantizedTolerance)))
        << "With test number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedVariousInputShapes) {
  float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({TensorType_UINT8, test_shapes[i], -3.0, 3.0},
                          {TensorType_UINT8, test_shapes[i], -3.0, 3.0},
                          {TensorType_UINT8, {}, -3.0, 3.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<uint8_t>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<uint8_t>(m.input2(), {0.1, 0.3, 0.3, 0.5, 1.1, 0.1});
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.1, -0.1, 0.4, 0.3, 0.0, 1.9}, kQuantizedTolerance)))
        << "With shape number " << i;
  }
}

TEST(QuantizedSubOpModel, QuantizedWithBroadcast) {
  float kQuantizedTolerance = GetTolerance(-3.0, 3.0);
  std::vector<std::initializer_list<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    QuantizedSubOpModel m({TensorType_UINT8, test_shapes[i], -3.0, 3.0},
                          {TensorType_UINT8, {}, -3.0, 3.0},
                          {TensorType_UINT8, {}, -3.0, 3.0},
                          ActivationFunctionType_NONE);
    m.QuantizeAndPopulate<uint8_t>(m.input1(), {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0});
    m.QuantizeAndPopulate<uint8_t>(m.input2(), {0.7});
    m.Invoke();
    EXPECT_THAT(m.GetDequantizedOutput(),
                ElementsAreArray(ArrayFloatNear(
                    {-2.7, -0.5, 0.0, 0.1, 0.4, 1.3}, kQuantizedTolerance)))
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
