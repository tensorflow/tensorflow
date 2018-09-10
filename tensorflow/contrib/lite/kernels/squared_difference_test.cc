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

using ::testing::ElementsAre;

template <typename T>
class SquaredDifferenceModel : public SingleOpModel {
 public:
  SquaredDifferenceModel(const TensorData& input1, const TensorData& input2,
                         const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SQUARED_DIFFERENCE,
                 BuiltinOptions_SquaredDifferenceOptions,
                 CreateSquaredDifferenceOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input1_;
  int input2_;
  int output_;
};

TEST(SquaredDifferenceOpModel, Simple) {
  SquaredDifferenceModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                                        {TensorType_INT32, {1, 2, 2, 1}},
                                        {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, 9, 11, 5});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, 3, 4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(64, 49, 64, 1));
}

TEST(SquaredDifferenceOpModel, Float) {
  SquaredDifferenceModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                                      {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {0.7, 0.4, 0.7, 5.8});
  model.PopulateTensor<float>(model.input2(), {0.5, 1.4, 1.9, 3.2});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear({0.04, 1.0, 1.44, 6.76}, 1e-3)));
}

TEST(SquaredDifferenceOpModel, BroadcastTest) {
  SquaredDifferenceModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                                        {TensorType_INT32, {1}},
                                        {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(64, 4, 9, 16));
}

TEST(SquaredDifferenceOpModel, BroadcastTestFloat) {
  SquaredDifferenceModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                                      {TensorType_FLOAT32, {1}},
                                      {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {1.2, 0.2, 0.7, 0.8});
  model.PopulateTensor<float>(model.input2(), {0.2});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear({1.0, 0.0, 0.25, 0.36}, 1e-3)));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
