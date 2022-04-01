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

#include <initializer_list>
#include <type_traits>
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
class PackOpModel : public SingleOpModel {
 public:
  PackOpModel(const TensorData& input_template, int axis, int values_count) {
    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < values_count; ++i) {
      all_input_shapes.push_back(input_template.shape);
      AddInput(input_template);
    }
    output_ = AddOutput({input_template.type, /*shape=*/{}, input_template.min,
                         input_template.max});
    SetBuiltinOp(BuiltinOperator_PACK, BuiltinOptions_PackOptions,
                 CreatePackOptions(builder_, values_count, axis).Union());
    BuildInterpreter(all_input_shapes);
  }

  void SetInput(int index, std::initializer_list<T> data) {
    PopulateTensor(index, data);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int output_;
};

// float32 tests.
TEST(PackOpTest, FloatThreeInputs) {
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, 0, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TEST(PackOpTest, FloatThreeInputsDifferentAxis) {
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, 1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, FloatThreeInputsNegativeAxis) {
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, -1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, FloatMultilDimensions) {
  PackOpModel<float> model({TensorType_FLOAT32, {2, 3}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, 9, 10, 11, 12});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

TEST(PackOpTest, FloatFiveDimensions) {
  PackOpModel<float> model({TensorType_FLOAT32, {2, 2, 2, 2}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  model.SetInput(
      1, {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 2, 2, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  17, 18, 19,
                                20, 21, 22, 23, 24, 9,  10, 11, 12, 13, 14,
                                15, 16, 25, 26, 27, 28, 29, 30, 31, 32}));
}

// int32 tests.
TEST(PackOpTest, Int32ThreeInputs) {
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, 0, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TEST(PackOpTest, Int32ThreeInputsDifferentAxis) {
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, 1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, Int32ThreeInputsNegativeAxis) {
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, -1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, Int32MultilDimensions) {
  PackOpModel<int32_t> model({TensorType_INT32, {2, 3}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, 9, 10, 11, 12});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

// int64 tests.
TEST(PackOpTest, Int64ThreeInputs) {
  PackOpModel<int64_t> model({TensorType_INT64, {2}}, 0, 3);
  model.SetInput(0, {1LL << 33, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, -(1LL << 34)});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 4LL, 2LL, 5LL, 3LL, -(1LL << 34)}));
}

TEST(PackOpTest, Int64ThreeInputsDifferentAxis) {
  PackOpModel<int64_t> model({TensorType_INT64, {2}}, 1, 3);
  model.SetInput(0, {1LL << 33, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, -(1LL << 34)});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 2LL, 3LL, 4LL, 5LL, -(1LL << 34)}));
}

TEST(PackOpTest, Int64ThreeInputsNegativeAxis) {
  PackOpModel<int64_t> model({TensorType_INT64, {2}}, -1, 3);
  model.SetInput(0, {1LL << 33, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, -(1LL << 34)});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 2LL, 3LL, 4LL, 5LL, -(1LL << 34)}));
}

TEST(PackOpTest, Int64MultilDimensions) {
  PackOpModel<int64_t> model({TensorType_INT64, {2, 3}}, 1, 2);
  model.SetInput(0, {1LL << 33, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, -(1LL << 34), 10, 11, 12});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 2LL, 3LL, 7LL, 8LL, -(1LL << 34),
                                4LL, 5LL, 6LL, 10LL, 11LL, 12LL}));
}

template <typename InputType>
struct PackOpTestInt : public ::testing::Test {
  using TypeToTest = InputType;
  TensorType TENSOR_TYPE =
      (std::is_same<InputType, int16_t>::value
           ? TensorType_INT16
           : (std::is_same<InputType, uint8_t>::value ? TensorType_UINT8
                                                      : TensorType_INT8));
};

using TestTypes = testing::Types<int8_t, uint8_t, int16_t>;
TYPED_TEST_CASE(PackOpTestInt, TestTypes);

TYPED_TEST(PackOpTestInt, ThreeInputs) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2}}, 0, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TYPED_TEST(PackOpTestInt, ThreeInputsDifferentAxis) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2}}, 1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(PackOpTestInt, ThreeInputsNegativeAxis) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2}}, -1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(PackOpTestInt, MultilDimensions) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2, 3}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, 9, 10, 11, 12});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

}  // namespace
}  // namespace tflite
