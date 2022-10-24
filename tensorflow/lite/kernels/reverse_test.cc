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
#include <stdint.h>

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class ReverseOpModel : public SingleOpModel {
 public:
  ReverseOpModel(const TensorData& input, const TensorData& axis) {
    input_ = AddInput(input);
    axis_ = AddInput(axis);

    output_ = AddOutput({input.type, {}});

    SetBuiltinOp(BuiltinOperator_REVERSE_V2, BuiltinOptions_ReverseV2Options,
                 CreateReverseV2Options(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  int axis() { return axis_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int axis_;
  int output_;
};

// float32 tests.
TEST(ReverseOpTest, FloatOneDimension) {
  ReverseOpModel<float> model({TensorType_FLOAT32, {4}},
                              {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.input(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 3, 2, 1}));
}

TEST(ReverseOpTest, FloatMultiDimensions) {
  ReverseOpModel<float> model({TensorType_FLOAT32, {4, 3, 2}},
                              {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.input(),
                              {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

// int32 tests
TEST(ReverseOpTest, Int32OneDimension) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {4}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.input(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 3, 2, 1}));
}

TEST(ReverseOpTest, Int32MultiDimensions) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {4, 3, 2}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

// int64 tests
TEST(ReverseOpTest, Int64OneDimension) {
  ReverseOpModel<int64_t> model({TensorType_INT64, {4}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int64_t>(model.input(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 3, 2, 1}));
}

TEST(ReverseOpTest, Int64MultiDimensions) {
  ReverseOpModel<int64_t> model({TensorType_INT64, {4, 3, 2}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int64_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

// uint8 tests
TEST(ReverseOpTest, Uint8OneDimension) {
  ReverseOpModel<uint8_t> model({TensorType_UINT8, {4}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<uint8_t>(model.input(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 3, 2, 1}));
}

TEST(ReverseOpTest, Uint8MultiDimensions) {
  ReverseOpModel<uint8_t> model({TensorType_UINT8, {4, 3, 2}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<uint8_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

// int8 tests
TEST(ReverseOpTest, Int8OneDimension) {
  ReverseOpModel<int8_t> model({TensorType_INT8, {4}}, {TensorType_INT32, {1}});
  model.PopulateTensor<int8_t>(model.input(), {1, 2, -1, -2});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({-2, -1, 2, 1}));
}

TEST(ReverseOpTest, Int8MultiDimensions) {
  ReverseOpModel<int8_t> model({TensorType_INT8, {4, 3, 2}},
                               {TensorType_INT32, {1}});
  model.PopulateTensor<int8_t>(
      model.input(), {-1, -2, -3, -4, 5,  6,  7,  8,  9,   10,  11,  12,
                      13, 14, 15, 16, 17, 18, 19, 20, -21, -22, -23, -24});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  -3, -4, -1, -2, 11,  12,  9,   10,  7,  8,
                        17, 18, 15, 16, 13, 14, -23, -24, -21, -22, 19, 20}));
}

// int16 tests
TEST(ReverseOpTest, Int16OneDimension) {
  ReverseOpModel<int16_t> model({TensorType_INT16, {4}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int16_t>(model.input(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 3, 2, 1}));
}

TEST(ReverseOpTest, Int16MultiDimensions) {
  ReverseOpModel<int16_t> model({TensorType_INT16, {4, 3, 2}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int16_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                        17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20}));
}

}  // namespace
}  // namespace tflite
