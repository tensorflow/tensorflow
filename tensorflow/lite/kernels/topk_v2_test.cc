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
#include <limits>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace topk_v2 {
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node);
}  // namespace topk_v2
}  // namespace builtin
}  // namespace ops
}  // namespace tflite

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

using TfLiteIntArrayPtr =
    std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)>;

void NoOpErrorReporter(TfLiteContext*, const char*, ...) {}

TfLiteIntArrayPtr MakeTfLiteIntArray(std::initializer_list<int> dims) {
  TfLiteIntArrayPtr array(TfLiteIntArrayCreate(dims.size()),
                          TfLiteIntArrayFree);
  int i = 0;
  for (const int dim : dims) {
    array->data[i++] = dim;
  }
  return array;
}

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename InputType>
class TopKV2OpModel : public SingleOpModel {
 public:
  TopKV2OpModel(int top_k, const std::vector<int>& input_shape,
                const std::vector<InputType>& input_data,
                TestType input_tensor_types,
                TensorType top_k_type = TensorType_INT32,
                TensorType output_index_type = TensorType_INT32,
                bool allocate_and_delegate = true) {
    input_ = AddInput(GetTensorType<InputType>());
    if (input_tensor_types == TestType::kDynamic) {
      top_k_ = AddInput(top_k_type);
    } else if (top_k_type == TensorType_INT16) {
      top_k_ =
          AddConstInput(TensorType_INT16, {static_cast<int16_t>(top_k)}, {1});
    } else if (top_k_type == TensorType_INT64) {
      top_k_ =
          AddConstInput(TensorType_INT64, {static_cast<int64_t>(top_k)}, {1});
    } else {
      top_k_ = AddConstInput(top_k_type, {top_k}, {1});
    }
    output_values_ = AddOutput(GetTensorType<InputType>());
    output_indexes_ = AddOutput(output_index_type);
    SetBuiltinOp(BuiltinOperator_TOPK_V2, BuiltinOptions_TopKV2Options, 0);
    BuildInterpreter({input_shape, {1}}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true, allocate_and_delegate);

    if (allocate_and_delegate && !input_data.empty()) {
      PopulateTensor<InputType>(input_, input_data);
    }
    if (allocate_and_delegate && input_tensor_types == TestType::kDynamic) {
      if (top_k_type == TensorType_INT16) {
        PopulateTensor<int16_t>(top_k_, {static_cast<int16_t>(top_k)});
      } else {
        PopulateTensor<int32_t>(top_k_, {top_k});
      }
    }
  }

  std::vector<int32_t> GetIndexes() {
    return ExtractVector<int32_t>(output_indexes_);
  }
  std::vector<int16_t> GetInt16Indexes() {
    return ExtractVector<int16_t>(output_indexes_);
  }

  std::vector<InputType> GetValues() {
    return ExtractVector<InputType>(output_values_);
  }
  int input() const { return input_; }
  int top_k() const { return top_k_; }

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

// Test case for k = 0
TEST_P(TopKV2OpTest, KIsZeroFloat) {
  TopKV2OpModel<float> m(                 // top_k = 0
      0, {2, 3},                          // input_shape
      {-2.0, -3.0, 0.2, 0.8, 0.1, -0.1},  // input_data
      GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), IsEmpty());
  EXPECT_THAT(m.GetValues(), IsEmpty());
}

// Test case for k < 0
TEST_P(TopKV2OpTest, KIsNegativeFloat) {
  TopKV2OpModel<float> m(                 // top_k = -1
      -1, {2, 3},                         // input_shape
      {-2.0, -3.0, 0.2, 0.8, 0.1, -0.1},  // input_data
      GetParam());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), IsEmpty());
  EXPECT_THAT(m.GetValues(), IsEmpty());
}

TEST(TopKV2OpEdgeTest, Int16TopKTypeWorks) {
  TopKV2OpModel<float> m(2, {1, 3}, {1.0f, 3.0f, 2.0f}, TestType::kConst,
                         TensorType_INT16);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetValues(), ElementsAreArray(ArrayFloatNear({3.0f, 2.0f})));
}

TEST(TopKV2OpEdgeTest, UnsupportedTopKTypeRejectedByPrepare) {
  TopKV2OpModel<float> m(1, {1, 3}, {1.0f, 3.0f, 2.0f}, TestType::kConst,
                         TensorType_INT64, TensorType_INT32,
                         /*allocate_and_delegate=*/false);
  EXPECT_EQ(m.AllocateTensors(), kTfLiteError);
}

TEST(TopKV2OpEdgeTest, Int16OutputIndexRejectsLargeNonEmptyRow) {
  std::vector<float> input(32768, 0.0f);
  input[32767] = 1.0f;
  TopKV2OpModel<float> m(1, {1, 32768}, input, TestType::kDynamic,
                         TensorType_INT32, TensorType_INT16);
  EXPECT_EQ(m.Invoke(), kTfLiteError);
}

TEST(TopKV2OpEdgeTest, Rank64SmallTensorWorks) {
  std::vector<int> input_shape(64, 1);
  input_shape.back() = 3;
  TopKV2OpModel<float> m(2, input_shape, {1.0f, 3.0f, 2.0f},
                         TestType::kDynamic);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetValues(), ElementsAreArray(ArrayFloatNear({3.0f, 2.0f})));
}

TEST(TopKV2OpEdgeTest, IntermediateRowProductOverflowRejected) {
  auto input_dims = MakeTfLiteIntArray(
      {std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 0, 1});
  auto top_k_dims = MakeTfLiteIntArray({1});
  auto output_dims = MakeTfLiteIntArray({0});
  auto inputs = MakeTfLiteIntArray({0, 1});
  auto outputs = MakeTfLiteIntArray({2, 3});

  float input_data = 0.0f;
  int32_t top_k = 1;
  float output_value = 0.0f;
  int32_t output_index = 0;
  TfLiteTensor tensors[4] = {};
  tensors[0].type = kTfLiteFloat32;
  tensors[0].data.f = &input_data;
  tensors[0].dims = input_dims.get();
  tensors[0].allocation_type = kTfLiteArenaRw;
  tensors[1].type = kTfLiteInt32;
  tensors[1].data.i32 = &top_k;
  tensors[1].dims = top_k_dims.get();
  tensors[1].allocation_type = kTfLiteArenaRw;
  tensors[2].type = kTfLiteFloat32;
  tensors[2].data.f = &output_value;
  tensors[2].dims = output_dims.get();
  tensors[2].allocation_type = kTfLiteArenaRw;
  tensors[3].type = kTfLiteInt32;
  tensors[3].data.i32 = &output_index;
  tensors[3].dims = output_dims.get();
  tensors[3].allocation_type = kTfLiteArenaRw;

  TfLiteContext context = {};
  context.tensors = tensors;
  context.tensors_size = 4;
  context.ReportError = NoOpErrorReporter;

  TfLiteNode node = {};
  node.inputs = inputs.get();
  node.outputs = outputs.get();

  EXPECT_EQ(ops::builtin::topk_v2::Eval(&context, &node), kTfLiteError);
}

}  // namespace
}  // namespace tflite
