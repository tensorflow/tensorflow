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

class BaseMeanOpModel : public SingleOpModel {
 public:
  BaseMeanOpModel(const TensorData& input, const TensorData& output,
                  std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_MEAN, BuiltinOptions_MeanOptions,
        CreateMeanOptions(builder_, builder_.CreateVector<int>(axis), keep_dims)
            .Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

 protected:
  int input_;
  int output_;
};

class FloatMeanOpModel : public BaseMeanOpModel {
 public:
  using BaseMeanOpModel::BaseMeanOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

TEST(FloatMeanOpTest, NotKeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  FloatMeanOpModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {2}},
                     {1, 0, -3, -3}, false);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({12, 13})));
}

TEST(FloatMeanOpTest, KeepDims) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  FloatMeanOpModel m({TensorType_FLOAT32, {4, 3, 2}}, {TensorType_FLOAT32, {3}},
                     {0, 2}, true);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({10.5, 12.5, 14.5})));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
