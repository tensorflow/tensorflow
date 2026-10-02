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

#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
#include "tensorflow/lite/kernels/internal/float8.h"
#endif
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
template <typename Float8T>
std::vector<uint8_t> Float8Bytes(std::initializer_list<float> values) {
  std::vector<uint8_t> result;
  result.reserve(values.size());
  for (float value : values) {
    result.push_back(Float8T::ConvertFrom(value).rep());
  }
  return result;
}
#endif

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

  void SetRawInput(int index, const std::vector<uint8_t>& data) {
    TfLiteTensor* tensor = GetInputTensor(index);
    ASSERT_EQ(tensor->bytes, data.size());
    std::memcpy(tensor->data.uint8, data.data(), data.size());
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
  std::vector<uint8_t> GetRawOutput() {
    const TfLiteTensor* tensor = GetOutputTensor(0);
    return std::vector<uint8_t>(tensor->data.uint8,
                                tensor->data.uint8 + tensor->bytes);
  }

 private:
  int output_;
};

// float32 tests.
TEST(PackOpTest, FloatThreeInputs) {
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, 0, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TEST(PackOpTest, FloatThreeInputsDifferentAxis) {
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, 1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, FloatThreeInputsNegativeAxis) {
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, -1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, FloatMultilDimensions) {
  PackOpModel<float> model({TensorType_FLOAT32, {2, 3}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, 9, 10, 11, 12});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

TEST(PackOpTest, FloatFiveDimensions) {
  PackOpModel<float> model({TensorType_FLOAT32, {2, 2, 2, 2}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  model.SetInput(
      1, {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 2, 2, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  17, 18, 19,
                                20, 21, 22, 23, 24, 9,  10, 11, 12, 13, 14,
                                15, 16, 25, 26, 27, 28, 29, 30, 31, 32}));
}

// uint32 tests.
TEST(PackOpTest, UInt32ThreeInputs) {
  PackOpModel<uint32_t> model({TensorType_UINT32, {2}}, 0, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TEST(PackOpTest, UInt32ThreeInputsDifferentAxis) {
  PackOpModel<uint32_t> model({TensorType_UINT32, {2}}, 1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, UInt32ThreeInputsNegativeAxis) {
  PackOpModel<uint32_t> model({TensorType_UINT32, {2}}, -1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, UInt32MultilDimensions) {
  PackOpModel<uint32_t> model({TensorType_UINT32, {2, 3}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, 9, 10, 11, 12});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
TEST(PackOpTest, Float8E4M3FNThreeInputs) {
  PackOpModel<uint8_t> model({TensorType_FLOAT8_E4M3FN, {2}}, 0, 3);
  const std::vector<uint8_t> input0 =
      Float8Bytes<float8_internal::Float8E4M3FN>({1.f, 4.f});
  const std::vector<uint8_t> input1 =
      Float8Bytes<float8_internal::Float8E4M3FN>({2.f, 5.f});
  const std::vector<uint8_t> input2 =
      Float8Bytes<float8_internal::Float8E4M3FN>({3.f, 6.f});
  model.SetRawInput(0, input0);
  model.SetRawInput(1, input1);
  model.SetRawInput(2, input2);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetRawOutput(),
              ElementsAreArray({input0[0], input0[1], input1[0], input1[1],
                                input2[0], input2[1]}));
}

TEST(PackOpTest, Float8E5M2ThreeInputs) {
  PackOpModel<uint8_t> model({TensorType_FLOAT8_E5M2, {2}}, 0, 3);
  const std::vector<uint8_t> input0 =
      Float8Bytes<float8_internal::Float8E5M2>({1.f, 4.f});
  const std::vector<uint8_t> input1 =
      Float8Bytes<float8_internal::Float8E5M2>({2.f, 5.f});
  const std::vector<uint8_t> input2 =
      Float8Bytes<float8_internal::Float8E5M2>({3.f, 6.f});
  model.SetRawInput(0, input0);
  model.SetRawInput(1, input1);
  model.SetRawInput(2, input2);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetRawOutput(),
              ElementsAreArray({input0[0], input0[1], input1[0], input1[1],
                                input2[0], input2[1]}));
}
#endif

// int32 tests.
TEST(PackOpTest, Int32ThreeInputs) {
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, 0, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TEST(PackOpTest, Int32ThreeInputsDifferentAxis) {
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, 1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, Int32ThreeInputsNegativeAxis) {
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, -1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(PackOpTest, Int32MultilDimensions) {
  PackOpModel<int32_t> model({TensorType_INT32, {2, 3}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, 9, 10, 11, 12});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 4LL, 2LL, 5LL, 3LL, -(1LL << 34)}));
}

TEST(PackOpTest, Int64ThreeInputsDifferentAxis) {
  PackOpModel<int64_t> model({TensorType_INT64, {2}}, 1, 3);
  model.SetInput(0, {1LL << 33, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, -(1LL << 34)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 2LL, 3LL, 4LL, 5LL, -(1LL << 34)}));
}

TEST(PackOpTest, Int64ThreeInputsNegativeAxis) {
  PackOpModel<int64_t> model({TensorType_INT64, {2}}, -1, 3);
  model.SetInput(0, {1LL << 33, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, -(1LL << 34)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 2LL, 3LL, 4LL, 5LL, -(1LL << 34)}));
}

TEST(PackOpTest, Int64MultilDimensions) {
  PackOpModel<int64_t> model({TensorType_INT64, {2, 3}}, 1, 2);
  model.SetInput(0, {1LL << 33, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, -(1LL << 34), 10, 11, 12});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
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
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TYPED_TEST(PackOpTestInt, ThreeInputsDifferentAxis) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2}}, 1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(PackOpTestInt, ThreeInputsNegativeAxis) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2}}, -1, 3);
  model.SetInput(0, {1, 4});
  model.SetInput(1, {2, 5});
  model.SetInput(2, {3, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(PackOpTestInt, MultilDimensions) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2, 3}}, 1, 2);
  model.SetInput(0, {1, 2, 3, 4, 5, 6});
  model.SetInput(1, {7, 8, 9, 10, 11, 12});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

}  // namespace
}  // namespace tflite
