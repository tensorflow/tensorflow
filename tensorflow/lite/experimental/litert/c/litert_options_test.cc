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

#include "tensorflow/lite/experimental/litert/c/litert_options.h"

#include <cstdint>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace {
using testing::litert::IsError;

TEST(GetOpOptionTest, TestGetAddOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_add_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  uint32_t fused_activation;
  LITERT_ASSERT_OK(LiteRtGetAddFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetBatchMatmulOptions) {
  auto model =
      litert::testing::LoadTestFileModel("simple_batch_matmul_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  bool adj_x;
  LITERT_ASSERT_OK(LiteRtGetBatchMatmulAdjXOption(op, &adj_x));
  ASSERT_EQ(adj_x, false);

  bool adj_y;
  LITERT_ASSERT_OK(LiteRtGetBatchMatmulAdjYOption(op, &adj_y));
  ASSERT_EQ(adj_y, false);

  bool asymmetric_quantize_input;
  LITERT_ASSERT_OK(LiteRtGetBatchMatmulAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));
  ASSERT_EQ(asymmetric_quantize_input, false);
}

TEST(GetOpOptionTest, TestGetConcatenationOptions) {
  auto model =
      litert::testing::LoadTestFileModel("simple_concatenation_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  uint32_t fused_activation;
  LITERT_ASSERT_OK(
      LiteRtGetConcatenationFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);

  int32_t axis;
  LITERT_ASSERT_OK(LiteRtGetConcatenationAxisOption(op, &axis));
  ASSERT_EQ(axis, 2);
}

TEST(GetOpOptionTest, TestGetDivOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_div_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  uint32_t fused_activation;
  LITERT_ASSERT_OK(LiteRtGetDivFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetFullyConnectedOptions) {
  auto model =
      litert::testing::LoadTestFileModel("simple_fully_connected_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  uint32_t fused_activation;
  LITERT_ASSERT_OK(
      LiteRtGetFullyConnectedFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);

  uint32_t weights_format;
  LITERT_ASSERT_OK(
      LiteRtGetFullyConnectedWeightsFormatOption(op, &weights_format));
  ASSERT_EQ(weights_format, 0);

  bool keep_num_dims;
  LITERT_ASSERT_OK(
      LiteRtGetFullyConnectedKeepNumDimsOption(op, &keep_num_dims));
  ASSERT_EQ(keep_num_dims, true);

  uint32_t quantized_bias_type;
  LITERT_ASSERT_OK(
      LiteRtFullyConnectedGetQuantizedBiasTypeOption(op, &quantized_bias_type));
  ASSERT_EQ(quantized_bias_type, 0);

  bool asymmetric_quantize_input;
  LITERT_ASSERT_OK(LiteRtGetFullyConnectedAsymmetricQuantizeInputOption(
      op, &asymmetric_quantize_input));
  ASSERT_EQ(asymmetric_quantize_input, false);
}

TEST(GetOpOptionTest, TestGetMulOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_mul_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  uint32_t fused_activation;
  LITERT_ASSERT_OK(LiteRtGetMulFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetSoftmaxOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_softmax_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  float beta;
  LITERT_ASSERT_OK(LiteRtGetSoftmaxBetaOption(op, &beta));
  EXPECT_FLOAT_EQ(beta, 1.0);
}

TEST(GetOpOptionTest, TestGetStridedSliceOptions) {
  auto model =
      litert::testing::LoadTestFileModel("simple_strided_slice_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  int32_t begin_mask;
  LITERT_ASSERT_OK(LiteRtGetStridedSliceBeginMaskOption(op, &begin_mask));
  ASSERT_EQ(begin_mask, 0);

  int32_t end_mask;
  LITERT_ASSERT_OK(LiteRtGetStridedSliceEndMaskOption(op, &end_mask));
  ASSERT_EQ(end_mask, 0);

  int32_t ellipsis_mask;
  LITERT_ASSERT_OK(LiteRtGetStridedSliceEllipsisMaskOption(op, &ellipsis_mask));
  ASSERT_EQ(ellipsis_mask, 0);

  int32_t new_axis_mask;
  LITERT_ASSERT_OK(LiteRtGetStridedSliceNewAxisMaskOption(op, &new_axis_mask));
  ASSERT_EQ(new_axis_mask, 0);

  int32_t shrink_axis_mask;
  LITERT_ASSERT_OK(
      LiteRtGetStridedSliceShrinkAxisMaskOption(op, &shrink_axis_mask));
  ASSERT_EQ(shrink_axis_mask, 0);

  bool offset;
  LITERT_ASSERT_OK(LiteRtGetStridedSliceOffsetOption(op, &offset));
  ASSERT_EQ(offset, false);
}

TEST(GetOpOptionTest, TestGetSubOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_sub_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  uint32_t fused_activation;
  LITERT_ASSERT_OK(LiteRtGetSubFusedActivationOption(op, &fused_activation));
  ASSERT_EQ(fused_activation, 0);
}

TEST(GetOpOptionTest, TestGetNullReshapeOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_reshape_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  const int32_t* new_shape = nullptr;
  int32_t new_shape_size;

  EXPECT_THAT(LiteRtGetReshapeNewShapeOption(op, &new_shape, &new_shape_size),
              IsError(kLiteRtStatusErrorInvalidArgument));
  ASSERT_EQ(new_shape_size, -1);
}

TEST(GetOpOptionTest, TestGetSumOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_sum_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  bool keepdims;
  LITERT_ASSERT_OK(LiteRtGetSumKeepDimsOption(op, &keepdims));
  ASSERT_EQ(keepdims, true);
}

TEST(GetOpOptionTest, TestGetPackOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_pack_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  int32_t axis;
  LITERT_ASSERT_OK(LiteRtGetPackAxisOption(op, &axis));
  ASSERT_EQ(axis, 0);
}

TEST(GetOpOptionTest, TestGetGatherOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_gather_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  int32_t axis;
  LITERT_ASSERT_OK(LiteRtGetGatherAxisOption(op, &axis));
  ASSERT_EQ(axis, 0);

  int32_t batch_dims;
  LITERT_ASSERT_OK(LiteRtGetGatherBatchDimsOption(op, &batch_dims));
  ASSERT_EQ(batch_dims, 0);
}

TEST(GetOpOptionTest, TestGetMeanOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_mean_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  bool keepdims;
  LITERT_ASSERT_OK(LiteRtGetMeanKeepDimsOption(op, &keepdims));
  ASSERT_EQ(keepdims, false);
}

TEST(GetOpOptionTest, TestGetSplitOptions) {
  auto model = litert::testing::LoadTestFileModel("simple_split_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();
  auto op = ops.front().Get();

  int32_t num_splits;
  LITERT_ASSERT_OK(LiteRtGetSplitNumSplitsOption(op, &num_splits));
  ASSERT_EQ(num_splits, 3);
}

}  // namespace
