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

using ::testing::ElementsAre;

template <typename T>
class FloorModModel : public SingleOpModel {
 public:
  FloorModModel(const TensorData& input1, const TensorData& input2,
                const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_FLOOR_MOD, BuiltinOptions_FloorModOptions,
                 CreateFloorModOptions(builder_).Union());
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

TEST(FloorModModel, Simple) {
  FloorModModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, 9, 11, 3});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, 3, 4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(FloorModModel, NegativeValue) {
  FloorModModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {2, 2, -3, -4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, -2, -1));
}

TEST(FloorModModel, BroadcastFloorMod) {
  FloorModModel<int32_t> model({TensorType_INT32, {1, 2, 2, 1}},
                               {TensorType_INT32, {1}}, {TensorType_INT32, {}});
  model.PopulateTensor<int32_t>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<int32_t>(model.input2(), {-3});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(-2, 0, -2, -2));
}

TEST(FloorModModel, Int64WithBroadcast) {
  FloorModModel<int64_t> model({TensorType_INT64, {1, 2, 2, 1}},
                               {TensorType_INT64, {1}}, {TensorType_INT64, {}});
  model.PopulateTensor<int64_t>(model.input1(), {10, -9, -11, (1LL << 34) + 9});
  model.PopulateTensor<int64_t>(model.input2(), {-(1LL << 33)});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(-8589934582, -9, -11, -8589934583));
}

TEST(FloorModModel, FloatSimple) {
  FloorModModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10, 9, 11, 3});
  model.PopulateTensor<float>(model.input2(), {2, 2, 3, 4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(FloorModModel, FloatNegativeValue) {
  FloorModModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<float>(model.input2(), {2, 2, -3, -4});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, -2, -1));
}

TEST(FloorModModel, FloatBroadcastFloorMod) {
  FloorModModel<float> model({TensorType_FLOAT32, {1, 2, 2, 1}},
                             {TensorType_FLOAT32, {1}},
                             {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input1(), {10, -9, -11, 7});
  model.PopulateTensor<float>(model.input2(), {-3});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 2, 2, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAre(-2, 0, -2, -2));
}

}  // namespace
}  // namespace tflite
