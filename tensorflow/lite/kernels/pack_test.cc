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
  PackOpModel(const TensorData& input_template,
              std::vector<std::vector<T>> input_data, int axis,
              int values_count, bool constant_tensors) {
    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < values_count; ++i) {
      all_input_shapes.push_back(input_template.shape);
      if (constant_tensors) {
        AddConstInput(input_template, input_data[i]);
      } else {
        AddInput(input_template);
      }
    }
    output_ = AddOutput({input_template.type, /*shape=*/{}, input_template.min,
                         input_template.max});
    SetBuiltinOp(BuiltinOperator_PACK, BuiltinOptions_PackOptions,
                 CreatePackOptions(builder_, values_count, axis).Union());
    BuildInterpreter(all_input_shapes);
    if (!constant_tensors) {
      for (int i = 0; i < values_count; ++i) {
        SetInput(i, input_data[i]);
      }
    }
  }

  const TfLiteTensor* GetOutputTensor(int index) {
    return interpreter_->output_tensor(index);
  }

  void SetInput(int index, std::vector<T>& data) {
    PopulateTensor(index, data);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int output_;
};

struct PackOpTest : public testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(ConstantTensor, PackOpTest, testing::Bool());

// float32 tests.
TEST_P(PackOpTest, FloatThreeInputs) {
  const bool constant_tensors = GetParam();
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, {{1, 4}, {2, 5}, {3, 6}},
                           0, 3, constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, FloatThreeInputsDifferentAxis) {
  const bool constant_tensors = GetParam();
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, {{1, 4}, {2, 5}, {3, 6}},
                           1, 3, constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, FloatThreeInputsNegativeAxis) {
  const bool constant_tensors = GetParam();
  PackOpModel<float> model({TensorType_FLOAT32, {2}}, {{1, 4}, {2, 5}, {3, 6}},
                           -1, 3, constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, FloatMultilDimensions) {
  const bool constant_tensors = GetParam();
  PackOpModel<float> model({TensorType_FLOAT32, {2, 3}},
                           {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}}, 1, 2,
                           constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
  if (constant_tensors) {
    // The output is too big to be calculated during Prepare.
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
  }
}

TEST_P(PackOpTest, FloatFiveDimensions) {
  const bool constant_tensors = GetParam();
  PackOpModel<float> model(
      {TensorType_FLOAT32, {2, 2, 2, 2}},
      {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
       {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}},
      1, 2, constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 2, 2, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  17, 18, 19,
                                20, 21, 22, 23, 24, 9,  10, 11, 12, 13, 14,
                                15, 16, 25, 26, 27, 28, 29, 30, 31, 32}));
  if (constant_tensors) {
    // The output is too big to be calculated during Prepare.
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
  }
}

// int32 tests.
TEST_P(PackOpTest, Int32ThreeInputs) {
  const bool constant_tensors = GetParam();
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, {{1, 4}, {2, 5}, {3, 6}},
                             0, 3, constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, Int32ThreeInputsDifferentAxis) {
  const bool constant_tensors = GetParam();
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, {{1, 4}, {2, 5}, {3, 6}},
                             1, 3, constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, Int32ThreeInputsNegativeAxis) {
  const bool constant_tensors = GetParam();
  PackOpModel<int32_t> model({TensorType_INT32, {2}}, {{1, 4}, {2, 5}, {3, 6}},
                             -1, 3, constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, Int32MultilDimensions) {
  const bool constant_tensors = GetParam();
  PackOpModel<int32_t> model({TensorType_INT32, {2, 3}},
                             {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}}, 1, 2,
                             constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
  if (constant_tensors) {
    // The output is too big to be calculated during Prepare.
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
  }
}

// int64 tests.
TEST_P(PackOpTest, Int64ThreeInputs) {
  const bool constant_tensors = GetParam();
  PackOpModel<int64_t> model({TensorType_INT64, {2}},
                             {{1LL << 33, 4}, {2, 5}, {3, -(1LL << 34)}}, 0, 3,
                             constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 4LL, 2LL, 5LL, 3LL, -(1LL << 34)}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, Int64ThreeInputsDifferentAxis) {
  const bool constant_tensors = GetParam();
  PackOpModel<int64_t> model({TensorType_INT64, {2}},
                             {{1LL << 33, 4}, {2, 5}, {3, -(1LL << 34)}}, 1, 3,
                             constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 2LL, 3LL, 4LL, 5LL, -(1LL << 34)}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, Int64ThreeInputsNegativeAxis) {
  const bool constant_tensors = GetParam();
  PackOpModel<int64_t> model({TensorType_INT64, {2}},
                             {{1LL << 33, 4}, {2, 5}, {3, -(1LL << 34)}}, -1, 3,
                             constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 2LL, 3LL, 4LL, 5LL, -(1LL << 34)}));
  if (constant_tensors) {
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLitePersistentRo);
  }
}

TEST_P(PackOpTest, Int64MultilDimensions) {
  const bool constant_tensors = GetParam();
  PackOpModel<int64_t> model(
      {TensorType_INT64, {2, 3}},
      {{1LL << 33, 2, 3, 4, 5, 6}, {7, 8, -(1LL << 34), 10, 11, 12}}, 1, 2,
      constant_tensors);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1LL << 33, 2LL, 3LL, 7LL, 8LL, -(1LL << 34),
                                4LL, 5LL, 6LL, 10LL, 11LL, 12LL}));
  if (constant_tensors) {
    // The output is too big to be calculated during Prepare.
    EXPECT_EQ(model.GetOutputTensor(0)->allocation_type, kTfLiteArenaRw);
  }
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
      {TestFixture::TENSOR_TYPE, {2}}, {{1, 4}, {2, 5}, {3, 6}}, 0, 3, false);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TYPED_TEST(PackOpTestInt, ThreeInputsDifferentAxis) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2}}, {{1, 4}, {2, 5}, {3, 6}}, 1, 3, false);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(PackOpTestInt, ThreeInputsNegativeAxis) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2}}, {{1, 4}, {2, 5}, {3, 6}}, -1, 3, false);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(PackOpTestInt, MultilDimensions) {
  PackOpModel<typename TestFixture::TypeToTest> model(
      {TestFixture::TENSOR_TYPE, {2, 3}},
      {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}}, 1, 2, false);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
}

}  // namespace
}  // namespace tflite
