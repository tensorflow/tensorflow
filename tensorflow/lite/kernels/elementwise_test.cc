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

class ElementWiseOpBaseModel : public SingleOpModel {
 public:
  int input() const { return input_; }
  int output() const { return output_; }

 protected:
  int input_;
  int output_;
};

class ElementWiseOpFloatModel : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpFloatModel(BuiltinOperator op,
                          std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_shape});
  }
};

class ElementWiseOpBoolModel : public ElementWiseOpBaseModel {
 public:
  ElementWiseOpBoolModel(BuiltinOperator op,
                         std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_BOOL);
    output_ = AddOutput(TensorType_BOOL);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_shape});
  }
};

TEST(ElementWise, Sin) {
  ElementWiseOpFloatModel m(BuiltinOperator_SIN, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 3.1415926, -3.1415926, 1});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 0, 0, 0.84147})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Log) {
  ElementWiseOpFloatModel m(BuiltinOperator_LOG, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 3.1415926, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 1.14473, 0, 0})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(FloatActivationsOpTest, Abs) {
  ElementWiseOpFloatModel m(BuiltinOperator_ABS, {1, 2, 4, 1});
  m.PopulateTensor<float>(m.input(), {
                                         0.f, -6.2f, 2.f, 4.f,  //
                                         3.f, -2.f, 10.f, 1.f,  //
                                     });
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()), ElementsAreArray({
                                                      0.f, 6.2f, 2.f, 4.f,  //
                                                      3.f, 2.f, 10.f, 1.f,  //
                                                  }));
}

TEST(ElementWise, Sqrt) {
  ElementWiseOpFloatModel m(BuiltinOperator_SQRT, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 1, 2, 4});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 1, 1.41421, 2})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Rsqrt) {
  ElementWiseOpFloatModel m(BuiltinOperator_RSQRT, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 2, 4, 9});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({1, 0.7071, 0.5, 0.33333})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Square) {
  ElementWiseOpFloatModel m(BuiltinOperator_SQUARE, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 2, 0.5, -3.0});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({1, 4.0, 0.25, 9.0})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, LogicalNot) {
  ElementWiseOpBoolModel m(BuiltinOperator_LOGICAL_NOT, {1, 1, 4, 1});
  m.PopulateTensor<bool>(m.input(), {true, false, true, false});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<bool>(m.output()),
              ElementsAreArray({false, true, false, true}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
