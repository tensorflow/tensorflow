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

#include "tensorflow/lite/experimental/litert/c/litert_model.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"

namespace {

using ::litert::BufferRef;
using ::litert::internal::MakeTflBuffer;
using ::testing::ElementsAreArray;

TEST(LiteRtWeightsTest, GetNullWeights) {
  LiteRtWeightsT weights = {};

  const void* addr;
  size_t size;
  LITERT_ASSERT_STATUS_OK(LiteRtGetWeightsBytes(&weights, &addr, &size));

  EXPECT_EQ(addr, nullptr);
  EXPECT_EQ(size, 0);
}

TEST(LiteRtWeightsTest, GetWeights) {
  LiteRtWeightsT weights;
  detail::SetTflBuffer(weights, MakeTflBuffer({1, 2, 3}));

  const void* addr;
  size_t size;
  LITERT_ASSERT_STATUS_OK(LiteRtGetWeightsBytes(&weights, &addr, &size));

  EXPECT_NE(addr, nullptr);
  EXPECT_EQ(size, 3 * sizeof(int32_t));

  EXPECT_THAT(absl::MakeConstSpan(reinterpret_cast<const int32_t*>(addr), 3),
              ElementsAreArray({1, 2, 3}));
}

TEST(LiteRtTensorTest, GetUnrankedType) {
  static constexpr auto kElementType = kLiteRtElementTypeFloat32;
  static constexpr auto kId = kLiteRtUnrankedTensorType;

  TensorType type;
  type.first = kId;
  type.second.unranked_tensor_type.element_type = kElementType;

  LiteRtTensorT tensor;
  tensor.SetType(std::move(type));

  LiteRtTensorTypeId id;
  LITERT_ASSERT_STATUS_OK(LiteRtGetTensorTypeId(&tensor, &id));
  ASSERT_EQ(id, kId);

  LiteRtUnrankedTensorType unranked;
  LITERT_ASSERT_STATUS_OK(LiteRtGetUnrankedTensorType(&tensor, &unranked));
  EXPECT_EQ(unranked.element_type, kElementType);
}

TEST(LiteRtTensorTest, GetRankedTensorType) {
  static constexpr auto kElementType = kLiteRtElementTypeFloat32;
  static constexpr auto kId = kLiteRtRankedTensorType;

  LiteRtTensorT tensor;
  tensor.SetType(MakeRankedTensorType(kElementType, {3, 3}));

  LiteRtTensorTypeId id;
  LITERT_ASSERT_STATUS_OK(LiteRtGetTensorTypeId(&tensor, &id));
  ASSERT_EQ(id, kId);

  LiteRtRankedTensorType ranked;
  LITERT_ASSERT_STATUS_OK(LiteRtGetRankedTensorType(&tensor, &ranked));
  EXPECT_EQ(ranked.element_type, kElementType);
  ASSERT_EQ(ranked.layout.rank, 2);
  EXPECT_THAT(absl::MakeConstSpan(ranked.layout.dimensions, 2),
              ElementsAreArray({3, 3}));
}

TEST(LiteRtTensorTest, GetUses) {
  LiteRtTensorT tensor;

  LiteRtOpT user;
  tensor.Users().push_back(&user);
  tensor.UserArgInds().push_back(0);

  LiteRtOpT other_user;
  tensor.Users().push_back(&other_user);
  tensor.UserArgInds().push_back(1);

  LiteRtParamIndex num_uses;
  LITERT_ASSERT_STATUS_OK(LiteRtGetNumTensorUses(&tensor, &num_uses));
  ASSERT_EQ(num_uses, 2);

  LiteRtOp actual_user;
  LiteRtParamIndex actual_user_arg_index;
  LITERT_ASSERT_STATUS_OK(LiteRtGetTensorUse(
      &tensor, /*use_index=*/0, &actual_user, &actual_user_arg_index));
  ASSERT_EQ(actual_user, &user);
  ASSERT_EQ(actual_user_arg_index, 0);

  LITERT_ASSERT_STATUS_OK(LiteRtGetTensorUse(
      &tensor, /*use_index=*/1, &actual_user, &actual_user_arg_index));
  ASSERT_EQ(actual_user, &other_user);
  ASSERT_EQ(actual_user_arg_index, 1);
}

TEST(LiteRtTensorTest, GetDefiningOp) {
  LiteRtTensorT tensor;

  LiteRtOpT def_op;
  tensor.SetDefiningOp(def_op, 0);

  LiteRtTensorDefiningOp actual_def_op;
  bool has_defining_op;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetTensorDefiningOp(&tensor, &has_defining_op, &actual_def_op));
  ASSERT_TRUE(has_defining_op);
  EXPECT_EQ(actual_def_op.op, &def_op);
  EXPECT_EQ(actual_def_op.op_output_index, 0);
}

TEST(LiteRtTensorTest, NoDefiningOp) {
  LiteRtTensorT tensor;

  LiteRtTensorDefiningOp actual_def_op;
  bool has_defining_op;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetTensorDefiningOp(&tensor, &has_defining_op, &actual_def_op));
  ASSERT_FALSE(has_defining_op);
}

TEST(LiteRtTensorTest, Name) {
  static constexpr const char kName[] = "foo";

  LiteRtTensorT tensor;
  tensor.SetName(std::string(kName));

  const char* name;
  LITERT_ASSERT_STATUS_OK(LiteRtGetTensorName(&tensor, &name));
  EXPECT_STREQ(name, kName);
}

TEST(LiteRtTensorTest, QuantizationNone) {
  LiteRtTensorT tensor;

  LiteRtQuantizationTypeId q_type_id;
  LITERT_ASSERT_STATUS_OK(LiteRtGetQuantizationTypeId(&tensor, &q_type_id));
  EXPECT_EQ(q_type_id, kLiteRtQuantizationNone);

  LiteRtQuantizationPerTensor per_tensor_quantization;
  EXPECT_NE(LiteRtGetPerTensorQuantization(&tensor, &per_tensor_quantization),
            kLiteRtStatusOk);
}

TEST(LiteRtTensorTest, QuantizationPerTensor) {
  static constexpr auto kScale = 1.0;
  static constexpr auto kZeroPoint = 1;

  LiteRtTensorT tensor;
  tensor.SetQarams(MakePerTensorQuantization(kScale, kZeroPoint));

  LiteRtQuantizationTypeId q_type_id;
  LITERT_ASSERT_STATUS_OK(LiteRtGetQuantizationTypeId(&tensor, &q_type_id));
  ASSERT_EQ(q_type_id, kLiteRtQuantizationPerTensor);

  LiteRtQuantizationPerTensor per_tensor_quantization;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetPerTensorQuantization(&tensor, &per_tensor_quantization));

  EXPECT_EQ(per_tensor_quantization.scale, kScale);
  EXPECT_EQ(per_tensor_quantization.zero_point, kZeroPoint);
}

TEST(LiteRtTensorTest, QuantizationPerChannel) {
  static constexpr size_t kNumChannels = 2;
  static constexpr size_t kQuantizedDimension = 0;
  static constexpr float kScales[kNumChannels] = {1.0, 2.0};
  static constexpr int64_t kZps[kNumChannels] = {2, 3};

  LiteRtTensorT tensor;

  {
    auto per_channel =
        MakePerChannelQuantization(kScales, kZps, kQuantizedDimension, tensor);
    tensor.SetQarams(per_channel);
  }

  LiteRtQuantizationTypeId q_type_id;
  LITERT_ASSERT_STATUS_OK(LiteRtGetQuantizationTypeId(&tensor, &q_type_id));
  ASSERT_EQ(q_type_id, kLiteRtQuantizationPerChannel);

  LiteRtQuantizationPerChannel per_channel_quantization;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetPerChannelQuantization(&tensor, &per_channel_quantization));

  EXPECT_THAT(
      absl::MakeConstSpan(per_channel_quantization.scales, kNumChannels),
      testing::ElementsAreArray(kScales));
  EXPECT_THAT(
      absl::MakeConstSpan(per_channel_quantization.zero_points, kNumChannels),
      testing::ElementsAreArray(kZps));
  ASSERT_EQ(per_channel_quantization.num_channels, kNumChannels);
  ASSERT_EQ(per_channel_quantization.quantized_dimension, kQuantizedDimension);
}

TEST(LiteRtOpTest, GetOpCode) {
  static constexpr auto kCode = kLiteRtOpCodeTflCustom;

  LiteRtOpT op;
  op.SetOpCode(kCode);

  LiteRtOpCode code;
  LITERT_ASSERT_STATUS_OK(LiteRtGetOpCode(&op, &code));
  EXPECT_EQ(code, kCode);
}

TEST(LiteRtOpTest, GetInputs) {
  LiteRtTensorT input1;
  LiteRtTensorT input2;

  LiteRtOpT op;
  op.Inputs().push_back(&input1);
  op.Inputs().push_back(&input2);

  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_STATUS_OK(LiteRtGetNumOpInputs(&op, &num_inputs));
  ASSERT_EQ(num_inputs, 2);

  LiteRtTensor actual_input;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetOpInput(&op, /*input_index=*/0, &actual_input));
  EXPECT_EQ(actual_input, &input1);

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetOpInput(&op, /*input_index=*/1, &actual_input));
  EXPECT_EQ(actual_input, &input2);
}

TEST(LiteRtOpTest, GetOutputs) {
  LiteRtTensorT output1;
  LiteRtTensorT output2;

  LiteRtOpT op;
  op.Outputs().push_back(&output1);
  op.Outputs().push_back(&output2);

  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_STATUS_OK(LiteRtGetNumOpOutputs(&op, &num_outputs));
  ASSERT_EQ(num_outputs, 2);

  LiteRtTensor actual_output;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetOpOutput(&op, /*output_index=*/0, &actual_output));
  EXPECT_EQ(actual_output, &output1);

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetOpOutput(&op, /*output_index=*/1, &actual_output));
  EXPECT_EQ(actual_output, &output2);
}

TEST(LiteRtSubgraphTest, GetInputs) {
  LiteRtTensorT input1;
  LiteRtTensorT input2;

  LiteRtSubgraphT subgraph;
  subgraph.Inputs().push_back(&input1);
  subgraph.Inputs().push_back(&input2);

  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_STATUS_OK(LiteRtGetNumSubgraphInputs(&subgraph, &num_inputs));

  LiteRtTensor actual_input;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetSubgraphInput(&subgraph, /*input_index=*/0, &actual_input));
  EXPECT_EQ(actual_input, &input1);

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetSubgraphInput(&subgraph, /*input_index=*/1, &actual_input));
  EXPECT_EQ(actual_input, &input2);
}

TEST(LiteRtSubgraphTest, GetOutputs) {
  LiteRtTensorT output1;
  LiteRtTensorT output2;

  LiteRtSubgraphT subgraph;
  subgraph.Outputs().push_back(&output1);
  subgraph.Outputs().push_back(&output2);

  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_STATUS_OK(LiteRtGetNumSubgraphOutputs(&subgraph, &num_outputs));

  LiteRtTensor actual_output;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetSubgraphOutput(&subgraph, /*output_index=*/0, &actual_output));
  EXPECT_EQ(actual_output, &output1);

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetSubgraphOutput(&subgraph, /*output_index=*/1, &actual_output));
  EXPECT_EQ(actual_output, &output2);
}

TEST(LiteRtSubgraphTest, GetOps) {
  LiteRtSubgraphT subgraph;
  auto& op1 = subgraph.EmplaceOp();
  auto& op2 = subgraph.EmplaceOp();

  LiteRtParamIndex num_ops;
  LITERT_ASSERT_STATUS_OK(LiteRtGetNumSubgraphOps(&subgraph, &num_ops));
  ASSERT_EQ(num_ops, 2);

  LiteRtOp actual_op;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetSubgraphOp(&subgraph, /*op_index=*/0, &actual_op));
  ASSERT_EQ(actual_op, &op1);

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetSubgraphOp(&subgraph, /*op_index=*/1, &actual_op));
  ASSERT_EQ(actual_op, &op2);
}

TEST(LiteRtModelTest, GetMetadata) {
  static constexpr absl::string_view kKey = "KEY";
  static constexpr absl::string_view kData = "DATA";

  LiteRtModelT model;
  model.PushMetadata(kKey, kData);

  const void* metadata;
  size_t metadata_size;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetModelMetadata(&model, kKey.data(), &metadata, &metadata_size));
  EXPECT_EQ(BufferRef(metadata, metadata_size).StrView(), kData);
}

TEST(LiteRtModelTest, GetSubgraph) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  LiteRtSubgraph actual_subgraph;
  LITERT_ASSERT_STATUS_OK(LiteRtGetModelSubgraph(&model, 0, &actual_subgraph));
  EXPECT_EQ(actual_subgraph, &subgraph);
}

TEST(LiteRtModelTest, GetSubgraphOOB) {
  LiteRtModelT model;

  LiteRtSubgraph actual_subgraph;
  LITERT_ASSERT_STATUS_HAS_CODE(
      LiteRtGetModelSubgraph(&model, 0, &actual_subgraph),
      kLiteRtStatusErrorIndexOOB);
}

TEST(LiteRtOpListTest, PushOps) {
  LiteRtOpListT op_list;
  LiteRtOpT op;

  LITERT_ASSERT_STATUS_OK(LiteRtPushOp(&op_list, &op));
  auto vec = op_list.Vec();
  ASSERT_EQ(vec.size(), 1);
  EXPECT_EQ(vec.front(), &op);
}

}  // namespace
