// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
// NOLINTNEXTLINE

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_options.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {
TEST(GetOpOptionTest, TestGetAddOptions) {
  auto model = lrt::testing::LoadTestFileModel("simple_add_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(LrtAddGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetBatchMatmulOptions) {
  auto model = lrt::testing::LoadTestFileModel("simple_batch_matmul_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  bool adj_x;
  ASSERT_STATUS_OK(LrtBatchMatmulGetAdjXOption(op, &adj_x));
  ASSERT_EQ(adj_x, false);

  bool adj_y;
  ASSERT_STATUS_OK(LrtBatchMatmulGetAdjYOption(op, &adj_y));
  ASSERT_EQ(adj_y, false);

  bool asymmetric_quantize_input;
  ASSERT_STATUS_OK(LrtBatchMatmulGetAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));
  ASSERT_EQ(asymmetric_quantize_input, false);
}

TEST(GetOpOptionTest, TestGetConcatenationOptions) {
  auto model =
      lrt::testing::LoadTestFileModel("simple_concatenation_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(
      LrtConcatenationGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);

  int32_t axis;
  ASSERT_STATUS_OK(LrtConcatenationGetAxisOption(op, &axis));
  ASSERT_EQ(axis, 2);
}

TEST(GetOpOptionTest, TestGetDivOptions) {
  auto model = lrt::testing::LoadTestFileModel("simple_div_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(LrtDivGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetFullyConnectedOptions) {
  auto model =
      lrt::testing::LoadTestFileModel("simple_fully_connected_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(
      LrtFullyConnectedGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);

  uint32_t weights_format;
  ASSERT_STATUS_OK(
      LrtFullyConnectedGetWeightsFormatOption(op, &weights_format));
  ASSERT_EQ(weights_format, 0);

  bool keep_num_dims;
  ASSERT_STATUS_OK(LrtFullyConnectedGetKeepNumDimsOption(op, &keep_num_dims));
  ASSERT_EQ(keep_num_dims, true);

  uint32_t quantized_bias_type;
  ASSERT_STATUS_OK(
      LrtFullyConnectedGetQuantizedBiasTypeOption(op, &quantized_bias_type));
  ASSERT_EQ(quantized_bias_type, 0);

  bool asymmetric_quantize_input;
  ASSERT_STATUS_OK(LrtFullyConnectedGetAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));
  ASSERT_EQ(asymmetric_quantize_input, false);
}

TEST(GetOpOptionTest, TestGetMulOptions) {
  auto model = lrt::testing::LoadTestFileModel("simple_mul_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(LrtMulGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetSoftmaxOptions) {
  auto model = lrt::testing::LoadTestFileModel("simple_softmax_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  float beta;
  ASSERT_STATUS_OK(LrtSoftmaxGetBetaOption(op, &beta));
  EXPECT_FLOAT_EQ(beta, 1.0);
}

TEST(GetOpOptionTest, TestGetStridedSliceOptions) {
  auto model =
      lrt::testing::LoadTestFileModel("simple_strided_slice_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  int32_t begin_mask;
  ASSERT_STATUS_OK(LrtStridedSliceGetBeginMaskOption(op, &begin_mask));
  ASSERT_EQ(begin_mask, 0);

  int32_t end_mask;
  ASSERT_STATUS_OK(LrtStridedSliceGetEndMaskOption(op, &end_mask));
  ASSERT_EQ(end_mask, 0);

  int32_t ellipsis_mask;
  ASSERT_STATUS_OK(LrtStridedSliceGetEllipsisMaskOption(op, &ellipsis_mask));
  ASSERT_EQ(ellipsis_mask, 0);

  int32_t new_axis_mask;
  ASSERT_STATUS_OK(LrtStridedSliceGetNewAxisMaskOption(op, &new_axis_mask));
  ASSERT_EQ(new_axis_mask, 0);

  int32_t shrink_axis_mask;
  ASSERT_STATUS_OK(
      LrtStridedSliceGetShrinkAxisMaskOption(op, &shrink_axis_mask));
  ASSERT_EQ(shrink_axis_mask, 0);

  bool offset;
  ASSERT_STATUS_OK(LrtStridedSliceGetOffsetOption(op, &offset));
  ASSERT_EQ(offset, false);
}

TEST(GetOpOptionTest, TestGetSubOptions) {
  auto model = lrt::testing::LoadTestFileModel("simple_sub_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(LrtSubGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetReshapeOptions) {
  auto model = lrt::testing::LoadTestFileModel("simple_reshape_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  int32_t* new_shape = nullptr;
  int32_t new_shape_size;
  ASSERT_STATUS_OK(
      LrtReshapeGetNewShapeOption(op, &new_shape, &new_shape_size));
  ASSERT_EQ(new_shape_size, -1);
}

}  // namespace
