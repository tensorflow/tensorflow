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
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_options.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {
TEST(GetOpOptionTest, TestGetAddOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_add_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(LiteRtAddGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetBatchMatmulOptions) {
  auto model =
      litert::testing::LoadTestFileModel("simple_batch_matmul_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  bool adj_x;
  ASSERT_STATUS_OK(LiteRtBatchMatmulGetAdjXOption(op, &adj_x));
  ASSERT_EQ(adj_x, false);

  bool adj_y;
  ASSERT_STATUS_OK(LiteRtBatchMatmulGetAdjYOption(op, &adj_y));
  ASSERT_EQ(adj_y, false);

  bool asymmetric_quantize_input;
  ASSERT_STATUS_OK(LiteRtBatchMatmulGetAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));
  ASSERT_EQ(asymmetric_quantize_input, false);
}

TEST(GetOpOptionTest, TestGetConcatenationOptions) {
  auto model =
      litert::testing::LoadTestFileModel("simple_concatenation_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(
      LiteRtConcatenationGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);

  int32_t axis;
  ASSERT_STATUS_OK(LiteRtConcatenationGetAxisOption(op, &axis));
  ASSERT_EQ(axis, 2);
}

TEST(GetOpOptionTest, TestGetDivOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_div_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(LiteRtDivGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetFullyConnectedOptions) {
  auto model =
      litert::testing::LoadTestFileModel("simple_fully_connected_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(
      LiteRtFullyConnectedGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);

  uint32_t weights_format;
  ASSERT_STATUS_OK(
      LiteRtFullyConnectedGetWeightsFormatOption(op, &weights_format));
  ASSERT_EQ(weights_format, 0);

  bool keep_num_dims;
  ASSERT_STATUS_OK(
      LiteRtFullyConnectedGetKeepNumDimsOption(op, &keep_num_dims));
  ASSERT_EQ(keep_num_dims, true);

  uint32_t quantized_bias_type;
  ASSERT_STATUS_OK(
      LiteRtFullyConnectedGetQuantizedBiasTypeOption(op, &quantized_bias_type));
  ASSERT_EQ(quantized_bias_type, 0);

  bool asymmetric_quantize_input;
  ASSERT_STATUS_OK(LiteRtFullyConnectedGetAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));
  ASSERT_EQ(asymmetric_quantize_input, false);
}

TEST(GetOpOptionTest, TestGetMulOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_mul_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(LiteRtMulGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetSoftmaxOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_softmax_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  float beta;
  ASSERT_STATUS_OK(LiteRtSoftmaxGetBetaOption(op, &beta));
  EXPECT_FLOAT_EQ(beta, 1.0);
}

TEST(GetOpOptionTest, TestGetStridedSliceOptions) {
  auto model =
      litert::testing::LoadTestFileModel("simple_strided_slice_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  int32_t begin_mask;
  ASSERT_STATUS_OK(LiteRtStridedSliceGetBeginMaskOption(op, &begin_mask));
  ASSERT_EQ(begin_mask, 0);

  int32_t end_mask;
  ASSERT_STATUS_OK(LiteRtStridedSliceGetEndMaskOption(op, &end_mask));
  ASSERT_EQ(end_mask, 0);

  int32_t ellipsis_mask;
  ASSERT_STATUS_OK(LiteRtStridedSliceGetEllipsisMaskOption(op, &ellipsis_mask));
  ASSERT_EQ(ellipsis_mask, 0);

  int32_t new_axis_mask;
  ASSERT_STATUS_OK(LiteRtStridedSliceGetNewAxisMaskOption(op, &new_axis_mask));
  ASSERT_EQ(new_axis_mask, 0);

  int32_t shrink_axis_mask;
  ASSERT_STATUS_OK(
      LiteRtStridedSliceGetShrinkAxisMaskOption(op, &shrink_axis_mask));
  ASSERT_EQ(shrink_axis_mask, 0);

  bool offset;
  ASSERT_STATUS_OK(LiteRtStridedSliceGetOffsetOption(op, &offset));
  ASSERT_EQ(offset, false);
}

TEST(GetOpOptionTest, TestGetSubOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_sub_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  uint32_t fused_activation;
  ASSERT_STATUS_OK(LiteRtSubGetFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetReshapeOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_reshape_op.tflite");
  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  auto op = ops[0];

  int32_t* new_shape = nullptr;
  int32_t new_shape_size;
  ASSERT_STATUS_OK(
      LiteRtReshapeGetNewShapeOption(op, &new_shape, &new_shape_size));
  ASSERT_EQ(new_shape_size, -1);
}

}  // namespace
