/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace {
using ::testing::ElementsAreArray;

template <class InputType, class ShapeType = int32_t>
class BroadcastToOpModel : public SingleOpModel {
 public:
  // BroadcastTo with dynamic shape.
  BroadcastToOpModel(std::initializer_list<int> input_shape,
                     std::initializer_list<int> shape_shape) {
    input_ = AddInput({GetTensorType<InputType>(), input_shape});
    shape_ = AddInput({GetTensorType<ShapeType>(), shape_shape});
    output_ = AddOutput(GetTensorType<InputType>());
    SetBuiltinOp(BuiltinOperator_BROADCAST_TO,
                 BuiltinOptions_BroadcastToOptions,
                 CreateBroadcastToOptions(builder_).Union());
    BuildInterpreter({input_shape, shape_shape});
  }

  // BroadcastTo with const shape.
  BroadcastToOpModel(std::initializer_list<int> input_shape,
                     std::initializer_list<int> shape_shape,
                     std::initializer_list<ShapeType> shape_values) {
    input_ = AddInput({GetTensorType<InputType>(), input_shape});
    shape_ =
        AddConstInput(GetTensorType<ShapeType>(), shape_values, shape_shape);
    output_ = AddOutput(GetTensorType<InputType>());
    SetBuiltinOp(BuiltinOperator_BROADCAST_TO,
                 BuiltinOptions_BroadcastToOptions,
                 CreateBroadcastToOptions(builder_).Union());
    BuildInterpreter({input_shape, shape_shape});
  }

  void SetInput(std::initializer_list<InputType> data) {
    PopulateTensor(input_, data);
  }

  void SetShape(std::initializer_list<ShapeType> data) {
    PopulateTensor(shape_, data);
  }

  std::vector<InputType> GetOutput() {
    return ExtractVector<InputType>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int shape_;
  int output_;
};

template <typename T>
class BroadcastToOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, uint8_t, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(BroadcastToOpTest, DataTypes);

#ifdef GTEST_HAS_DEATH_TEST
TYPED_TEST(BroadcastToOpTest, ShapeMustBe1D) {
  EXPECT_DEATH(
      BroadcastToOpModel<TypeParam>({2, 3, 4, 4}, {2, 2}, {2, 3, 4, 4}), "");
  // Non-constant Shape tensor.
  BroadcastToOpModel<TypeParam> m({2, 3, 4, 4}, {2, 2});
  m.SetShape({2, 3, 4, 4});
  EXPECT_THAT(m.Invoke(), kTfLiteError);
}

TYPED_TEST(BroadcastToOpTest, TooManyDimensions) {
  EXPECT_DEATH(BroadcastToOpModel<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {9},
                                             {2, 2, 3, 4, 5, 6, 7, 8, 9}),
               "BroadcastTo only supports 1-8D tensor.");
  EXPECT_DEATH(BroadcastToOpModel<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 9}, {9}),
               "BroadcastTo only supports 1-8D tensor.");
}

TYPED_TEST(BroadcastToOpTest, MismatchDimension) {
  EXPECT_DEATH(BroadcastToOpModel<TypeParam>({2, 4, 1, 2}, {4}, {2, 4, 1, 3}),
               "Output shape must be broadcastable from input shape.");
  EXPECT_DEATH(
      BroadcastToOpModel<TypeParam>({2, 4, 1, 2, 3}, {4}, {2, 4, 1, 2}),
      "Output shape must be broadcastable from input shape.");

  // Non-constant Shape tensor.
  BroadcastToOpModel<TypeParam> m1({2, 4, 1, 2}, {4});
  m1.SetShape({2, 3, 4, 4});
  EXPECT_THAT(m1.Invoke(), kTfLiteError);
  BroadcastToOpModel<TypeParam> m2({2, 4, 1, 2}, {5});
  m2.SetShape({1, 2, 3, 4, 4});
  EXPECT_THAT(m2.Invoke(), kTfLiteError);
}
#endif

TYPED_TEST(BroadcastToOpTest, BroadcastTo1DConstTest) {
  BroadcastToOpModel<TypeParam> m({1}, {1}, {4});
  m.SetInput({3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 3}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo4DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 1, 1, 2}, {4}, {1, 1, 2, 2});
  m.SetInput({3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 4, 3, 4}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo8DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 1, 1, 1, 1, 1, 2, 1}, {8},
                                  {1, 1, 1, 1, 1, 1, 2, 2});
  m.SetInput({3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 4, 4}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo1DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1}, {1});
  m.SetInput({3});
  m.SetShape({4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 3}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo4DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1, 1, 1, 2}, {4});
  m.SetInput({3, 4});
  m.SetShape({1, 1, 2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 4, 3, 4}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastTo8DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1, 1, 1, 1, 1, 1, 2, 1}, {8});
  m.SetInput({3, 4});
  m.SetShape({1, 1, 1, 1, 1, 1, 2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 4, 4}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast4DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 3, 1, 2}, {4}, {3, 3, 2, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 2, 2}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4,
                        3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast4DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1, 3, 1, 2}, {4});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetShape({3, 3, 2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 2, 2}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4,
                        3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast6DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 2, 1, 3, 1, 2}, {6}, {2, 2, 1, 3, 2, 2});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 3, 2, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 1, 2, 3, 4,  3, 4,  5,  6,  5,  6,
                                7, 8, 7, 8, 9, 10, 9, 10, 11, 12, 11, 12,
                                1, 2, 1, 2, 3, 4,  3, 4,  5,  6,  5,  6,
                                7, 8, 7, 8, 9, 10, 9, 10, 11, 12, 11, 12}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast6DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({1, 2, 1, 3, 1, 2}, {6});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetShape({2, 2, 1, 3, 2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 3, 2, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 1, 2, 3, 4,  3, 4,  5,  6,  5,  6,
                                7, 8, 7, 8, 9, 10, 9, 10, 11, 12, 11, 12,
                                1, 2, 1, 2, 3, 4,  3, 4,  5,  6,  5,  6,
                                7, 8, 7, 8, 9, 10, 9, 10, 11, 12, 11, 12}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast8DConstTest) {
  BroadcastToOpModel<TypeParam> m({1, 3, 1, 2, 1, 4, 1, 1}, {8},
                                  {2, 3, 1, 2, 2, 4, 1, 1});
  m.SetInput({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 1, 2, 2, 4, 1, 1}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  5,  6,
                        7,  8,  9,  10, 11, 12, 9,  10, 11, 12, 13, 14, 15, 16,
                        13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22,
                        23, 24, 21, 22, 23, 24, 1,  2,  3,  4,  1,  2,  3,  4,
                        5,  6,  7,  8,  5,  6,  7,  8,  9,  10, 11, 12, 9,  10,
                        11, 12, 13, 14, 15, 16, 13, 14, 15, 16, 17, 18, 19, 20,
                        17, 18, 19, 20, 21, 22, 23, 24, 21, 22, 23, 24}));
}

TYPED_TEST(BroadcastToOpTest, ComplexBroadcast8DDynamicTest) {
  BroadcastToOpModel<TypeParam> m({2, 1, 1, 2, 1, 4, 1, 1}, {8});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetShape({2, 3, 2, 2, 2, 4, 1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2, 2, 2, 4, 1, 1}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(
          {1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           1, 2,  3,  4,  1, 2,  3,  4,  5,  6,  7,  8,  5,  6,  7,  8,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16,
           9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 13, 14, 15, 16}));
}

TYPED_TEST(BroadcastToOpTest, ExtendingShape4DConstTest) {
  BroadcastToOpModel<TypeParam> m({3, 1, 2}, {4}, {3, 3, 2, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 3, 2, 2}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4,
                        3, 4, 5, 6, 5, 6, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, NoBroadcastingConstTest) {
  BroadcastToOpModel<TypeParam> m({3, 1, 2}, {3}, {3, 1, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, NoBroadcasting8DConstTest) {
  BroadcastToOpModel<TypeParam> m({3, 1, 1, 1, 1, 1, 1, 2}, {8},
                                  {3, 1, 1, 1, 1, 1, 1, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1, 1, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(BroadcastToOpTest, Int64ShapeConstTest) {
  BroadcastToOpModel<TypeParam, int64_t> m({1, 1, 1, 1, 1, 1, 2, 1}, {8},
                                           {1, 1, 1, 1, 1, 1, 2, 2});
  m.SetInput({3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 4, 4}));
}

TYPED_TEST(BroadcastToOpTest, Int64ShapeDDynamicTest) {
  BroadcastToOpModel<TypeParam, int64_t> m({1, 1, 1, 1, 1, 1, 2, 1}, {8});
  m.SetInput({3, 4});
  m.SetShape({1, 1, 1, 1, 1, 1, 2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 1, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 4, 4}));
}

TYPED_TEST(BroadcastToOpTest, BroadcastToEmtpyShapeTest) {
  BroadcastToOpModel<TypeParam> m({3, 1, 2}, {3}, {3, 0, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 0, 2}));
}

}  // namespace
}  // namespace tflite
