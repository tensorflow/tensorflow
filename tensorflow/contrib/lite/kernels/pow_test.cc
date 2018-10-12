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
using ::testing::ElementsAreArray;

template <typename T>
class PowOpModel : public SingleOpModel {
 public:
  PowOpModel(const TensorData& input1, const TensorData& input2,
             const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_POW, BuiltinOptions_PowOptions,
                 CreatePowOptions(builder_).Union());
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

TEST(PowOpModel, Simple) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3, 1});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(12, 4, 343, 8));
}

TEST(PowOpModel, NegativeAndZeroValue) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {0, 2, -7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3, 0});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 4, -343, 1));
}

TEST(PowOpModel, Float) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {0.3, 0.4, 0.7, 5.8});
  model.PopulateTensor<float>(model.input2(), {0.5, 2.7, 3.1, 3.2});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5477226, 0.08424846, 0.33098164, 277.313}, 1e-3)));
}

TEST(PowOpModel, NegativeFloatTest) {
  PowOpModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {1, 2, 2, 1}},
                          {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {0.3, 0.4, 0.7, 5.8});
  model.PopulateTensor<float>(model.input2(), {0.5, -2.7, 3.1, -3.2});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.5477226, 11.869653, 0.33098164, 0.003606}, 1e-3)));
}

TEST(PowOpModel, BroadcastTest) {
  PowOpModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                            {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {12, 2, 7, 8});
  model.PopulateTensor<int32_t>(model.input2(), {4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(20736, 16, 2401, 4096));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
