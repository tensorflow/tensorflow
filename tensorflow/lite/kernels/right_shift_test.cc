/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class RightShiftOpModel : public SingleOpModel {
 public:
  RightShiftOpModel(std::initializer_list<int> input1_shape,
                    std::initializer_list<int> input2_shape,
                    TensorType tensor_type) {
    input1_ = AddInput(tensor_type);
    input2_ = AddInput(tensor_type);
    output_ = AddOutput(tensor_type);
    SetBuiltinOp(BuiltinOperator_RIGHT_SHIFT, BuiltinOptions_RightShiftOptions,
                 CreateRightShiftOptions(builder_).Union());
    BuildInterpreter({input1_shape, input2_shape});
  }

  int input1() const { return input1_; }
  int input2() const { return input2_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(RightShiftOpTest, SimpleTestInt8) {
  RightShiftOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT8);
  model.PopulateTensor<int8_t>(model.input1(), {-1, -5, -3, -14});
  model.PopulateTensor<int8_t>(model.input2(), {5, 0, 7, 11});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<int8_t>(), ElementsAreArray({-1, -5, -1, -1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(RightShiftOpTest, SimpleTestInt16) {
  RightShiftOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT16);
  model.PopulateTensor<int16_t>(model.input1(), {-1, -5, -3, -14});
  model.PopulateTensor<int16_t>(model.input2(), {5, 0, 7, 11});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<int16_t>(), ElementsAreArray({-1, -5, -1, -1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(RightShiftOpTest, SimpleTestInt32) {
  RightShiftOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_INT32);
  model.PopulateTensor<int32_t>(model.input1(), {-1, -5, -3, -14});
  model.PopulateTensor<int32_t>(model.input2(), {5, 0, 7, 11});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<int32_t>(), ElementsAreArray({-1, -5, -1, -1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(RightShiftOpTest, SimpleTestUInt8) {
  RightShiftOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_UINT8);
  model.PopulateTensor<uint8_t>(model.input1(), {1, 5, 3, 14});
  model.PopulateTensor<uint8_t>(model.input2(), {5, 0, 7, 11});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<uint8_t>(), ElementsAreArray({0, 5, 0, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(RightShiftOpTest, SimpleTestUInt16) {
  RightShiftOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_UINT16);
  model.PopulateTensor<uint16_t>(model.input1(), {1, 5, 3, 14});
  model.PopulateTensor<uint16_t>(model.input2(), {5, 0, 7, 11});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<uint16_t>(), ElementsAreArray({0, 5, 0, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(RightShiftOpTest, SimpleTestUInt32) {
  RightShiftOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, TensorType_UINT32);
  model.PopulateTensor<uint32_t>(model.input1(), {1, 5, 3, 14});
  model.PopulateTensor<uint32_t>(model.input2(), {5, 0, 7, 11});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<uint32_t>(), ElementsAreArray({0, 5, 0, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(BitwiseXorOpTest, BroadcastRhsInt) {
  RightShiftOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_INT32);
  model.PopulateTensor<int32_t>(model.input1(), {-1, -5, -3, -14});
  model.PopulateTensor<int32_t>(model.input2(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<int32_t>(), ElementsAreArray({-1, -2, -1, -4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(BitwiseXorOpTest, BroadcastLhsInt) {
  RightShiftOpModel model({1, 1, 1, 1}, {1, 1, 1, 4}, TensorType_INT32);
  model.PopulateTensor<int32_t>(model.input1(), {4});
  model.PopulateTensor<int32_t>(model.input2(), {1, -2, 3, -4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<int32_t>(), ElementsAreArray({2, 4, 0, 4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(BitwiseXorOpTest, BroadcastRhsUInt) {
  RightShiftOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, TensorType_UINT32);
  model.PopulateTensor<uint32_t>(model.input1(), {5, 0, 7, 11});
  model.PopulateTensor<uint32_t>(model.input2(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<uint32_t>(), ElementsAreArray({1, 0, 1, 2}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(BitwiseXorOpTest, BroadcastLhsUInt) {
  RightShiftOpModel model({1, 1, 1, 1}, {1, 1, 1, 4}, TensorType_UINT32);
  model.PopulateTensor<uint32_t>(model.input1(), {4});
  model.PopulateTensor<uint32_t>(model.input2(), {1, 2, 3, 4});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput<uint32_t>(), ElementsAreArray({2, 1, 0, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

}  // namespace
}  // namespace tflite
