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

class MaximumOpModel : public SingleOpModel {
 public:
  MaximumOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorType& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MAXIMUM, BuiltinOptions_MaximumOptions,
                 CreateMaximumOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  template <class T>
  void SetInput1(std::initializer_list<T> data) {
    PopulateTensor(input1_, data);
  }

  template <class T>
  void SetInput2(std::initializer_list<T> data) {
    PopulateTensor(input2_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(MaximumOpTest, FloatTest) {
  std::initializer_list<float> data1 = {1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  std::initializer_list<float> data2 = {-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  MaximumOpModel m({TensorType_FLOAT32, {3, 1, 2}},
                   {TensorType_FLOAT32, {3, 1, 2}}, TensorType_FLOAT32);
  m.SetInput1<float>(data1);
  m.SetInput2<float>(data2);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 2}));
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({1.0, 0.0, 1.0, 12.0, -2.0, -1.43})));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
