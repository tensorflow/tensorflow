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

class RoundOpModel : public SingleOpModel {
 public:
  RoundOpModel(std::initializer_list<int> input_shape, TensorType input_type) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_ROUND, BuiltinOptions_NONE, 0);
    BuildInterpreter({
        input_shape,
    });
  }

  int input() { return input_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(RoundOpTest, SingleDim) {
  RoundOpModel model({6}, TensorType_FLOAT32);
  model.PopulateTensor<float>(model.input(), {8.5, 0.0, 3.5, 4.2, -3.5, -4.5});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({8, 0, 4, 4, -4, -4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({6}));
}

TEST(RoundOpTest, MultiDims) {
  RoundOpModel model({2, 1, 1, 6}, TensorType_FLOAT32);
  model.PopulateTensor<float>(
      model.input(), {0.0001, 8.0001, 0.9999, 9.9999, 0.5, -0.0001, -8.0001,
                      -0.9999, -9.9999, -0.5, -2.5, 1.5});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({0, 8, 1, 10, 0, 0, -8, -1, -10, -0, -2, 2}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 6}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
