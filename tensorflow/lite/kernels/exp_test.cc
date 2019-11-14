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

class ExpOpModel : public SingleOpModel {
 public:
  ExpOpModel(const TensorData& input, const TensorType& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_EXP, BuiltinOptions_ExpOptions,
                 CreateExpOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <class T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int output_;
};

TEST(ExpOpTest, FloatTest) {
  std::initializer_list<float> data = {1.0, 0.0, -1.0, 1.0, 1.0, -1.0};
  ExpOpModel m({TensorType_FLOAT32, {3, 1, 2}}, TensorType_FLOAT32);
  m.SetInput<float>(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 2}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {2.71828, 1, 0.367879, 2.71828, 2.71828, 0.367879})));
}

}  // namespace
}  // namespace tflite
