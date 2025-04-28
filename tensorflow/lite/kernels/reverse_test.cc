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

#include <gmock/gmock.h>
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
    BuildInterpreter({GetShape(input_), GetShape(axis_)});
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

TEST(ReverseOpTest, Int32MultiDimensionsFirst) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {3, 3, 3}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3, 3));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({19, 20, 21, 22, 23, 24, 25, 26, 27, 10, 11, 12, 13, 14,
                        15, 16, 17, 18, 1,  2,  3,  4,  5,  6,  7,  8,  9}));
}

TEST(ReverseOpTest, Int32MultiDimensionsSecond) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {3, 3, 3}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3, 3));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({7,  8,  9,  4,  5,  6,  1,  2,  3,  16, 17, 18, 13, 14,
                        15, 10, 11, 12, 25, 26, 27, 22, 23, 24, 19, 20, 21}));
}

TEST(ReverseOpTest, Int32MultiDimensionsThird) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {3, 3, 3}},
                                {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
  model.PopulateTensor<int32_t>(model.axis(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3, 3));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({3,  2,  1,  6,  5,  4,  9,  8,  7,  12, 11, 10, 15, 14,
                        13, 18, 17, 16, 21, 20, 19, 24, 23, 22, 27, 26, 25}));
}

TEST(ReverseOpTest, Int32MultiDimensionsFirstSecond) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {3, 3, 3}},
                                {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
  model.PopulateTensor<int32_t>(model.axis(), {0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3, 3));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({25, 26, 27, 22, 23, 24, 19, 20, 21, 16, 17, 18, 13, 14,
                        15, 10, 11, 12, 7,  8,  9,  4,  5,  6,  1,  2,  3}));
}

TEST(ReverseOpTest, Int32MultiDimensionsSecondThird) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {3, 3, 3}},
                                {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
  model.PopulateTensor<int32_t>(model.axis(), {1, 2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3, 3));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({9,  8,  7,  6,  5,  4,  3,  2,  1,  18, 17, 16, 15, 14,
                        13, 12, 11, 10, 27, 26, 25, 24, 23, 22, 21, 20, 19}));
}

TEST(ReverseOpTest, Int32MultiDimensionsSecondFirst) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {3, 3, 3}},
                                {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
  model.PopulateTensor<int32_t>(model.axis(), {1, 0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3, 3));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({25, 26, 27, 22, 23, 24, 19, 20, 21, 16, 17, 18, 13, 14,
                        15, 10, 11, 12, 7,  8,  9,  4,  5,  6,  1,  2,  3}));
}

TEST(ReverseOpTest, Int32MultiDimensionsAll) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {3, 3, 3}},
                                {TensorType_INT32, {3}});
  model.PopulateTensor<int32_t>(
      model.input(), {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
  model.PopulateTensor<int32_t>(model.axis(), {0, 1, 2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3, 3));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
                        13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1}));
}

TEST(ReverseOpTest, Int32MultiDimensions8D) {
  ReverseOpModel<int32_t> model({TensorType_INT32, {1, 1, 1, 1, 1, 1, 1, 3}},
                                {TensorType_INT32, {8}});
  model.PopulateTensor<int32_t>(model.input(), {1, 2, 3});
  model.PopulateTensor<int32_t>(model.axis(), {7, 6, 5, 4, 3, 2, 1, 0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 1, 1, 1, 1, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({3, 2, 1}));
}

#if GTEST_HAS_DEATH_TEST
TEST(ReverseOpTest, Int32MultiDimensions9D) {
  EXPECT_DEATH(
      ReverseOpModel<int32_t>({TensorType_INT32, {1, 1, 1, 1, 1, 1, 1, 1, 3}},
                              {TensorType_INT32, {9}}),
      "Cannot allocate tensors");
}
#endif  // GTEST_HAS_DEATH_TEST

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

// float16 tests.
TEST(ReverseOpTest, Float16OneDimension) {
  ReverseOpModel<Eigen::half> model({TensorType_FLOAT16, {4}},
                                    {TensorType_INT32, {1}});
  model.PopulateTensor<Eigen::half>(
      model.input(),
      {Eigen::half(1), Eigen::half(2), Eigen::half(3), Eigen::half(4)});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({Eigen::half(4), Eigen::half(3), Eigen::half(2),
                                Eigen::half(1)}));
}

TEST(ReverseOpTest, Float16MultiDimensions) {
  ReverseOpModel<Eigen::half> model({TensorType_FLOAT16, {4, 3, 2}},
                                    {TensorType_INT32, {1}});
  model.PopulateTensor<Eigen::half>(
      model.input(),
      {Eigen::half(1),  Eigen::half(2),  Eigen::half(3),  Eigen::half(4),
       Eigen::half(5),  Eigen::half(6),  Eigen::half(7),  Eigen::half(8),
       Eigen::half(9),  Eigen::half(10), Eigen::half(11), Eigen::half(12),
       Eigen::half(13), Eigen::half(14), Eigen::half(15), Eigen::half(16),
       Eigen::half(17), Eigen::half(18), Eigen::half(19), Eigen::half(20),
       Eigen::half(21), Eigen::half(22), Eigen::half(23), Eigen::half(24)});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({Eigen::half(5),  Eigen::half(6),  Eigen::half(3),
                        Eigen::half(4),  Eigen::half(1),  Eigen::half(2),
                        Eigen::half(11), Eigen::half(12), Eigen::half(9),
                        Eigen::half(10), Eigen::half(7),  Eigen::half(8),
                        Eigen::half(17), Eigen::half(18), Eigen::half(15),
                        Eigen::half(16), Eigen::half(13), Eigen::half(14),
                        Eigen::half(23), Eigen::half(24), Eigen::half(21),
                        Eigen::half(22), Eigen::half(19), Eigen::half(20)}));
}

// bfloat16 tests.
TEST(ReverseOpTest, BFloat16OneDimension) {
  ReverseOpModel<Eigen::bfloat16> model({TensorType_BFLOAT16, {4}},
                                        {TensorType_INT32, {1}});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input(), {Eigen::bfloat16(1), Eigen::bfloat16(2),
                      Eigen::bfloat16(3), Eigen::bfloat16(4)});
  model.PopulateTensor<int32_t>(model.axis(), {0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({Eigen::bfloat16(4), Eigen::bfloat16(3),
                                Eigen::bfloat16(2), Eigen::bfloat16(1)}));
}

TEST(ReverseOpTest, BFloat16MultiDimensions) {
  ReverseOpModel<Eigen::bfloat16> model({TensorType_BFLOAT16, {4, 3, 2}},
                                        {TensorType_INT32, {1}});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input(),
      {Eigen::bfloat16(1),  Eigen::bfloat16(2),  Eigen::bfloat16(3),
       Eigen::bfloat16(4),  Eigen::bfloat16(5),  Eigen::bfloat16(6),
       Eigen::bfloat16(7),  Eigen::bfloat16(8),  Eigen::bfloat16(9),
       Eigen::bfloat16(10), Eigen::bfloat16(11), Eigen::bfloat16(12),
       Eigen::bfloat16(13), Eigen::bfloat16(14), Eigen::bfloat16(15),
       Eigen::bfloat16(16), Eigen::bfloat16(17), Eigen::bfloat16(18),
       Eigen::bfloat16(19), Eigen::bfloat16(20), Eigen::bfloat16(21),
       Eigen::bfloat16(22), Eigen::bfloat16(23), Eigen::bfloat16(24)});
  model.PopulateTensor<int32_t>(model.axis(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4, 3, 2));
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray(
          {Eigen::bfloat16(5),  Eigen::bfloat16(6),  Eigen::bfloat16(3),
           Eigen::bfloat16(4),  Eigen::bfloat16(1),  Eigen::bfloat16(2),
           Eigen::bfloat16(11), Eigen::bfloat16(12), Eigen::bfloat16(9),
           Eigen::bfloat16(10), Eigen::bfloat16(7),  Eigen::bfloat16(8),
           Eigen::bfloat16(17), Eigen::bfloat16(18), Eigen::bfloat16(15),
           Eigen::bfloat16(16), Eigen::bfloat16(13), Eigen::bfloat16(14),
           Eigen::bfloat16(23), Eigen::bfloat16(24), Eigen::bfloat16(21),
           Eigen::bfloat16(22), Eigen::bfloat16(19), Eigen::bfloat16(20)}));
}

}  // namespace
}  // namespace tflite
