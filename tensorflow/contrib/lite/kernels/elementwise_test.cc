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
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class ElementWiseOpModel : public SingleOpModel {
 public:
  ElementWiseOpModel(BuiltinOperator op,
                     std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(op, BuiltinOptions_NONE, 0);
    BuildInterpreter({input_shape});
  }

  int input() const { return input_; }
  int output() const { return output_; }

 private:
  int input_;
  int output_;
};

TEST(ElementWise, Sin) {
  ElementWiseOpModel m(BuiltinOperator_SIN, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {0, 3.1415926, -3.1415926, 1});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 0, 0, 0.84147})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

TEST(ElementWise, Log) {
  ElementWiseOpModel m(BuiltinOperator_LOG, {1, 1, 4, 1});
  m.PopulateTensor<float>(m.input(), {1, 3.1415926, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear({0, 1.14473, 0, 0})));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 1, 4, 1}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
