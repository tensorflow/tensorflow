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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_subgraph_test_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::tflite::variants::TensorArray;

namespace tflite {
namespace {

using UtilsTest = ListOpsSubgraphTest;
// This test just validates the test fixture. It doesn't test any business
// logic.
TEST_F(UtilsTest, SimpleAddConst) {
  builder_.AddConstSubgraph(&interpreter_.primary_subgraph());

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

  builder_.AddReserveSubgraph(&interpreter_.primary_subgraph(),
                              params.tensor_type);

  ASSERT_THAT(interpreter_.ResizeInputTensor(0, params.element_shape_shape),
              kTfLiteOk);
  ASSERT_THAT(interpreter_.ResizeInputTensor(1, {}), kTfLiteOk);
  ASSERT_THAT(interpreter_.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* element_shape = interpreter_.input_tensor(0);
  std::copy(params.element_shape_data.begin(), params.element_shape_data.end(),
            element_shape->data.i32);

  TfLiteTensor* num_elements = interpreter_.input_tensor(1);
  num_elements->data.i32[0] = params.num_elements;

  ASSERT_THAT(interpreter_.Invoke(), kTfLiteOk);

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

}  // namespace
}  // namespace tflite
