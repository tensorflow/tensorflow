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
#include <algorithm>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter_test_util.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_subgraph_test_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::testing::ElementsAreArray;
using ::tflite::variants::TensorArray;

namespace tflite {
namespace {

using UtilsTest = ListOpsSubgraphTest;

// This test just validates the test fixture. It doesn't test any business
// logic.
TEST_F(UtilsTest, SimpleAddConst) {
  builder_.BuildAddConstSubgraph(&interpreter_.primary_subgraph());

  TfLiteTensor* cst1 = interpreter_.tensor(0);
  ASSERT_THAT(cst1, DimsAre({2}));
  EXPECT_EQ(cst1->data.i32[0], 2);
  EXPECT_EQ(cst1->data.i32[1], 2);

  TfLiteTensor* cst2 = interpreter_.tensor(1);
  ASSERT_THAT(cst2, DimsAre({2}));
  EXPECT_EQ(cst2->data.i32[0], 3);
  EXPECT_EQ(cst2->data.i32[1], 3);

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  TfLiteTensor* out = interpreter_.tensor(2);
  ASSERT_THAT(out, DimsAre({2}));
  EXPECT_EQ(out->data.i32[0], 5);
  EXPECT_EQ(out->data.i32[1], 5);
}

struct ListReserveSubgraphTestParams {
  const TensorType tensor_type;
  const TfLiteType expected_type;
  const std::vector<int> element_shape_shape;
  const std::vector<int> element_shape_data;
  const std::vector<int> expected_element_shape;
  const int num_elements;
};

class ListReserveSubgraphTest
    : public ListOpsSubgraphTest,
      public ::testing::WithParamInterface<ListReserveSubgraphTestParams> {};

TEST_P(ListReserveSubgraphTest, InterpreterOutputsTensorArray) {
  const ListReserveSubgraphTestParams& params = GetParam();

  builder_.BuildReserveSubgraph(&interpreter_.primary_subgraph(),
                                params.tensor_type);

  ASSERT_EQ(interpreter_.ResizeInputTensor(0, params.element_shape_shape),
            kTfLiteOk);
  ASSERT_EQ(interpreter_.ResizeInputTensor(1, {}), kTfLiteOk);
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* element_shape = interpreter_.input_tensor(0);
  std::copy(params.element_shape_data.begin(), params.element_shape_data.end(),
            element_shape->data.i32);

  TfLiteTensor* num_elements = interpreter_.input_tensor(1);
  num_elements->data.i32[0] = params.num_elements;

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_.output_tensor(0);
  ASSERT_EQ(output->type, kTfLiteVariant);
  ASSERT_EQ(output->allocation_type, kTfLiteVariantObject);
  ASSERT_TRUE(output->data.data != nullptr);

  TensorArray* result =
      static_cast<TensorArray*>(static_cast<VariantData*>(output->data.data));
  EXPECT_EQ(result->NumElements(), params.num_elements);
  EXPECT_THAT(result->ElementShape(), DimsAre(params.expected_element_shape));
  EXPECT_EQ(result->ElementType(), params.expected_type);
  for (int i = 0; i < params.num_elements; ++i) {
    EXPECT_EQ(result->At(i), nullptr);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ListOpsSubgraphParamTests, ListReserveSubgraphTest,
    testing::ValuesIn({
        ListReserveSubgraphTestParams{
            TensorType_INT32, kTfLiteInt32, {}, {-1}, {}, 2},
        ListReserveSubgraphTestParams{
            TensorType_FLOAT32, kTfLiteFloat32, {}, {-1}, {}, 2},
        ListReserveSubgraphTestParams{
            TensorType_FLOAT32, kTfLiteFloat32, {1}, {-1}, {-1}, 2},
        ListReserveSubgraphTestParams{
            TensorType_FLOAT32, kTfLiteFloat32, {2}, {2, 2}, {2, 2}, 0},
        ListReserveSubgraphTestParams{
            TensorType_FLOAT32, kTfLiteFloat32, {2}, {2, -1}, {2, -1}, 10},
    }));

struct ListStackSubgraphDynamicTestParams {
  // Reserve params.
  const std::vector<int> element_shape_shape;
  const std::vector<int> element_shape_data;
  const int num_elements;
  // Stack params.
  const std::vector<int> stack_shape_shape;
  const std::vector<int> stack_shape_data;
  // Expected.
  const std::vector<int> expected_shape;
};

class ListStackDynamicSubgraphTest
    : public ListOpsSubgraphTest,
      public ::testing::WithParamInterface<ListStackSubgraphDynamicTestParams> {
};

TEST_P(ListStackDynamicSubgraphTest,
       InterpreterOutputsStackTensor_DynamicOutput) {
  const ListStackSubgraphDynamicTestParams& params = GetParam();

  builder_.BuildReserveStackSubgraph(&interpreter_.primary_subgraph());

  ASSERT_EQ(interpreter_.ResizeInputTensor(0, params.element_shape_shape),
            kTfLiteOk);
  ASSERT_EQ(interpreter_.ResizeInputTensor(1, {}), kTfLiteOk);
  ASSERT_EQ(interpreter_.ResizeInputTensor(2, params.stack_shape_shape),
            kTfLiteOk);
  interpreter_.output_tensor(0)->allocation_type = kTfLiteDynamic;
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* element_shape = interpreter_.input_tensor(0);
  std::copy(params.element_shape_data.begin(), params.element_shape_data.end(),
            element_shape->data.i32);

  TfLiteTensor* num_elements = interpreter_.input_tensor(1);
  num_elements->data.i32[0] = params.num_elements;

  TfLiteTensor* stack_shape = interpreter_.input_tensor(2);
  std::copy(params.stack_shape_data.begin(), params.stack_shape_data.end(),
            stack_shape->data.i32);

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_.output_tensor(0);
  ASSERT_EQ(output->type, kTfLiteInt32);
  ASSERT_EQ(output->allocation_type, kTfLiteDynamic);

  const int output_num_elements = NumElements(output);
  ASSERT_TRUE((output_num_elements > 0 && output->data.data != nullptr) ||
              (output_num_elements == 0 && output->data.data == nullptr));

  ASSERT_THAT(output, DimsAre(params.expected_shape));
  for (int i = 0; i < NumElements(output); ++i) {
    EXPECT_EQ(output->data.i32[i], 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ListOpsSubgraphParamTests, ListStackDynamicSubgraphTest,
    testing::ValuesIn({
        ListStackSubgraphDynamicTestParams{{1}, {2}, 4, {}, {-1}, {4, 2}},
        ListStackSubgraphDynamicTestParams{
            {}, {-1}, 4, {3}, {2, 3, 4}, {4, 2, 3, 4}},
        ListStackSubgraphDynamicTestParams{{1}, {2}, 4, {}, {-1}, {4, 2}},
        ListStackSubgraphDynamicTestParams{{1}, {2}, 0, {}, {-1}, {0, 2}},
        ListStackSubgraphDynamicTestParams{{1}, {1}, 2, {}, {-1}, {2}},
    }));

// Fixture that constructs a model that uses a "While" op to
// populate each element in a `TensorArray` with a constant tensor.
// See documentation of `BuildLessThanSubgraph`,
// `BuildSetItemAndIncrementSubgraph` and `BuildWhileSubgraph` for more
// detail.
class WhileIncrementListOpsTest : public InterpreterTest {
 public:
  WhileIncrementListOpsTest() {
    AddSubgraphs(2);
    builder_.BuildLessThanSubgraph(interpreter_->subgraph(1));
    builder_.BuildSetItemAndIncrementSubgraph(interpreter_->subgraph(2));
    builder_.BuildWhileSubgraph(&interpreter_->primary_subgraph());
    TFLITE_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);
  }

 protected:
  // Allocates a `TensorArray` behind the `kTfLiteVariant` tensor at given
  // index.
  void PopulateListTensor(int index, absl::Span<const int> element_shape_data,
                          int num_elements, TfLiteType element_type) {
    TfLiteTensor* tensor = interpreter_->tensor(index);

    TF_LITE_ASSERT_EQ(tensor->type, kTfLiteVariant);
    tensor->allocation_type = kTfLiteVariantObject;
    tensor->buffer_handle = kTfLiteNullBufferHandle;
    tensor->quantization = {kTfLiteNoQuantization};

    IntArrayUniquePtr element_shape =
        BuildTfLiteArray(element_shape_data.size(), element_shape_data.data());

    TfLiteStatus stat = TfLiteTensorVariantRealloc<TensorArray>(
        tensor, element_type, std::move(element_shape));
    TF_LITE_ASSERT_EQ(stat, kTfLiteOk);

    TensorArray* arr =
        static_cast<TensorArray*>(static_cast<VariantData*>(tensor->data.data));
    arr->Resize(num_elements);
  }

  // Retreives a pointer to the `TensorArray` sitting behind the
  //  `kTfLiteVariant` tensor at given index.
  const TensorArray* GetOutputTensorArray(int tensor_id) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_id);
    TFLITE_CHECK(tensor != nullptr && tensor->type == kTfLiteVariant &&
                 tensor->allocation_type == kTfLiteVariantObject);
    return static_cast<const TensorArray*>(
        static_cast<const VariantData*>(tensor->data.data));
  }

  ListOpsSubgraphBuilder builder_;
};

TEST_F(WhileIncrementListOpsTest, PopulateListWithWhile) {
  interpreter_->tensor(interpreter_->inputs()[0])->data.i32[0] = 0;
  PopulateListTensor(interpreter_->inputs()[1], {2, 2}, 3, kTfLiteInt32);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  const TensorArray* output = GetOutputTensorArray(interpreter_->outputs()[1]);
  ASSERT_TRUE(output != nullptr);

  ASSERT_EQ(output->NumElements(), 3);

  for (int i = 0; i < 3; ++i) {
    const TfLiteTensor* item = output->At(i);
    ASSERT_TRUE(item != nullptr);

    ASSERT_THAT(item, DimsAre({2}));
    EXPECT_THAT(std::vector<int>(item->data.i32, item->data.i32 + 2),
                ElementsAreArray({2, 2}));
  }
}

TEST_F(WhileIncrementListOpsTest,
       PartiallyPopulateListWithWhile_UnsetItemsZeroed) {
  interpreter_->tensor(interpreter_->inputs()[0])->data.i32[0] = 1;
  PopulateListTensor(interpreter_->inputs()[1], {2, 2}, 3, kTfLiteInt32);

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  const TensorArray* output = GetOutputTensorArray(interpreter_->outputs()[1]);
  ASSERT_TRUE(output != nullptr);

  ASSERT_EQ(output->NumElements(), 3);
  ASSERT_EQ(output->At(0), nullptr);

  for (int i = 1; i < 3; ++i) {
    const TfLiteTensor* item = output->At(i);
    ASSERT_TRUE(item != nullptr);

    ASSERT_THAT(item, DimsAre({2}));
    EXPECT_THAT(std::vector<int>(item->data.i32, item->data.i32 + 2),
                ElementsAreArray({2, 2}));
  }
}

class ListReserveLengthSubgraphTest
    : public ListOpsSubgraphTest,
      public ::testing::WithParamInterface<int> {};

TEST_P(ListReserveLengthSubgraphTest, InterpreterOutputsListLength) {
  const int length = GetParam();

  builder_.BuildReserveLengthSubgraph(&interpreter_.primary_subgraph());

  ASSERT_EQ(interpreter_.ResizeInputTensor(0, {1}), kTfLiteOk);
  ASSERT_EQ(interpreter_.ResizeInputTensor(1, {}), kTfLiteOk);
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* element_shape = interpreter_.input_tensor(0);
  element_shape->data.i32[0] = 2;

  TfLiteTensor* num_elements = interpreter_.input_tensor(1);
  num_elements->data.i32[0] = length;

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_.output_tensor(0);
  ASSERT_EQ(output->type, kTfLiteInt32);
  ASSERT_EQ(output->allocation_type, kTfLiteArenaRw);
  ASSERT_THAT(output, DimsAre({}));
  ASSERT_TRUE(output->data.data != nullptr);

  ASSERT_EQ(output->data.i32[0], length);
}

INSTANTIATE_TEST_SUITE_P(ListOpsSubgraphParamTests,
                         ListReserveLengthSubgraphTest,
                         testing::Values(0, 1, 2, 5, 10));

}  // namespace
}  // namespace tflite
