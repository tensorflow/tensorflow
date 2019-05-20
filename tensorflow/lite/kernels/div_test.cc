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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

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

class IntegerDivOpModel : public BaseDivOpModel {
 public:
  using BaseDivOpModel::BaseDivOpModel;

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
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
  std::vector<std::vector<int>> test_shapes = {
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
  std::vector<std::vector<int>> test_shapes = {
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

TEST(IntegerDivOpTest, NoActivation) {
  IntegerDivOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_NONE);
  m.PopulateTensor<int32_t>(m.input1(), {-2, 2, -15, 8});
  m.PopulateTensor<int32_t>(m.input2(), {5, -2, -3, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, -1, 5, 1}));
}

TEST(IntegerDivOpTest, ActivationRELU_N1_TO_1) {
  IntegerDivOpModel m({TensorType_INT32, {1, 2, 2, 1}},
                      {TensorType_INT32, {1, 2, 2, 1}}, {TensorType_INT32, {}},
                      ActivationFunctionType_RELU_N1_TO_1);
  m.PopulateTensor<int32_t>(m.input1(), {-2, 2, -12, 8});
  m.PopulateTensor<int32_t>(m.input2(), {1, 2, -15, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1, 1, 0, 1}));
}

TEST(IntegerDivOpTest, VariousInputShapes) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerDivOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 2, 3, 8, 11, -20});
    m.PopulateTensor<int32_t>(m.input2(), {1, 2, 6, 5, -11, -1});
    m.Invoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-20, 1, 0, 1, -1, 20}))
        << "With shape number " << i;
  }
}

TEST(IntegerDivOpTest, WithBroadcast) {
  std::vector<std::vector<int>> test_shapes = {
      {6}, {2, 3}, {2, 1, 3}, {1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    IntegerDivOpModel m({TensorType_INT32, test_shapes[i]},
                        {TensorType_INT32, {}},  // always a scalar
                        {TensorType_INT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<int32_t>(m.input1(), {-20, 21, 7, 8, 11, -123});
    m.PopulateTensor<int32_t>(m.input2(), {3});
    m.Invoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-6, 7, 2, 2, 3, -41}))
        << "With shape number " << i;
  }
}

}  // namespace
}  // namespace tflite
