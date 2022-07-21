/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename InputType>
class TopKV2OpModel : public SingleOpModel {
 public:
  TopKV2OpModel(int top_k, std::initializer_list<int> input_shape,
                std::initializer_list<InputType> input_data,
                TestType input_tensor_types) {
    input_ = AddInput(GetTensorType<InputType>());
    if (input_tensor_types == TestType::kDynamic) {
      top_k_ = AddInput(TensorType_INT32);
    } else {
      top_k_ = AddConstInput(TensorType_INT32, {top_k}, {1});
    }
    output_values_ = AddOutput(GetTensorType<InputType>());
    output_indexes_ = AddOutput(TensorType_INT32);
    SetBuiltinOp(BuiltinOperator_TOPK_V2, BuiltinOptions_TopKV2Options, 0);
    BuildInterpreter({input_shape, {1}});

    PopulateTensor<InputType>(input_, input_data);
    if (input_tensor_types == TestType::kDynamic) {
      PopulateTensor<int32_t>(top_k_, {top_k});
    }
  }

  std::vector<int32_t> GetIndexes() {
    return ExtractVector<int32_t>(output_indexes_);
  }

  std::vector<InputType> GetValues() {
    return ExtractVector<InputType>(output_values_);
  }

 protected:
  int input_;
  int top_k_;
  int output_indexes_;
  int output_values_;
};

class TopKV2OpTest : public ::testing::TestWithParam<TestType> {};

// The test where the tensor dimension is equal to top.
TEST_P(TopKV2OpTest, EqualFloat) {
  TopKV2OpModel<float> m(2, {2, 2}, {-2.0, 0.2, 0.8, 0.1}, GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({1, 0, 0, 1}));
  EXPECT_THAT(m.GetValues(),
              ElementsAreArray(ArrayFloatNear({0.2, -2.0, 0.8, 0.1})));
}

// Test when internal dimension is k+1.
TEST_P(TopKV2OpTest, BorderFloat) {
  TopKV2OpModel<float> m(2, {2, 3}, {-2.0, -3.0, 0.2, 0.8, 0.1, -0.1},
                         GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 0, 0, 1}));
  EXPECT_THAT(m.GetValues(),
              ElementsAreArray(ArrayFloatNear({0.2, -2.0, 0.8, 0.1})));
}
// Test when internal dimension is higher than k.
TEST_P(TopKV2OpTest, LargeFloat) {
  TopKV2OpModel<float> m(
      2, {2, 4}, {-2.0, -3.0, -4.0, 0.2, 0.8, 0.1, -0.1, -0.8}, GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({3, 0, 0, 1}));
  EXPECT_THAT(m.GetValues(),
              ElementsAreArray(ArrayFloatNear({0.2, -2.0, 0.8, 0.1})));
}

// Test 1D case.
TEST_P(TopKV2OpTest, VectorFloat) {
  TopKV2OpModel<float> m(2, {8}, {-2.0, -3.0, -4.0, 0.2, 0.8, 0.1, -0.1, -0.8},
                         GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({4, 3}));
  EXPECT_THAT(m.GetValues(), ElementsAreArray(ArrayFloatNear({0.8, 0.2})));
}

// Check that int32_t works.
TEST_P(TopKV2OpTest, TypeInt32) {
  TopKV2OpModel<int32_t> m(2, {2, 3}, {1, 2, 3, 10251, 10250, 10249},
                           GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 1, 0, 1}));
  EXPECT_THAT(m.GetValues(), ElementsAreArray({3, 2, 10251, 10250}));
}

INSTANTIATE_TEST_SUITE_P(TopKV2OpTest, TopKV2OpTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));

// Check that uint8_t works.
TEST_P(TopKV2OpTest, TypeUint8) {
  TopKV2OpModel<uint8_t> m(2, {2, 3}, {1, 2, 3, 251, 250, 249}, GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 1, 0, 1}));
  EXPECT_THAT(m.GetValues(), ElementsAreArray({3, 2, 251, 250}));
}

TEST_P(TopKV2OpTest, TypeInt8) {
  TopKV2OpModel<int8_t> m(2, {2, 3}, {1, 2, 3, -126, 125, -24}, GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 1, 1, 2}));
  EXPECT_THAT(m.GetValues(), ElementsAreArray({3, 2, 125, -24}));
}

// Check that int64 works.
TEST_P(TopKV2OpTest, TypeInt64) {
  TopKV2OpModel<int64_t> m(2, {2, 3}, {1, 2, 3, -1, -2, -3}, GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 1, 0, 1}));
  EXPECT_THAT(m.GetValues(), ElementsAreArray({3, 2, -1, -2}));
}

}  // namespace
}  // namespace tflite
