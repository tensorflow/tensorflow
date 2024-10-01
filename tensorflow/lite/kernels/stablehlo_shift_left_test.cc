/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::ElementsAreArray;

class ShiftLeftOpModel : public SingleOpModel {
 public:
  ShiftLeftOpModel(const TensorData& input1, const TensorData& input2) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(TensorData(input1.type, GetShape(input1_)));
    SetBuiltinOp(BuiltinOperator_STABLEHLO_SHIFT_LEFT, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(ShiftLeftOpTest, ShiftLeftInt32) {
  ShiftLeftOpModel model({TensorType_INT32, {3}}, {TensorType_INT32, {3}});
  model.PopulateTensor<int32_t>(model.input1(), {-1, 0, 1});
  model.PopulateTensor<int32_t>(model.input2(), {1, 2, 3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int32_t>(), ElementsAreArray({-2, 0, 8}));
}

TEST(ShiftLeftOpTest, ShiftLeftInt16) {
  ShiftLeftOpModel model({TensorType_INT16, {2, 2}},
                         {TensorType_INT16, {2, 2}});
  model.PopulateTensor<int16_t>(model.input1(), {-5, -5, 0, 6});
  model.PopulateTensor<int16_t>(model.input2(), {0, 2, 0, 2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int16_t>(), ElementsAreArray({-5, -20, 0, 24}));
}

TEST(ShiftLeftOpTest, ShiftLeftInt8) {
  ShiftLeftOpModel model({TensorType_INT8, {2, 2}}, {TensorType_INT8, {2, 2}});
  model.PopulateTensor<int8_t>(model.input1(), {2, -2, -2, -4});
  model.PopulateTensor<int8_t>(model.input2(), {0, 1, 0, 5});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAreArray({2, -4, -2, -128}));
}

}  // namespace
}  // namespace tflite
