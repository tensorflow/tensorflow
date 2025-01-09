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

#include "tensorflow/lite/experimental/litert/core/model/model.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

//
// Model
//

TEST(ModelTest, GetMetadata) {
  static constexpr absl::string_view kMetadata = "VALUE";
  static constexpr absl::string_view kKey = "KEY";

  LiteRtModelT model;
  LITERT_ASSERT_STATUS_OK(model.PushMetadata(kKey, kMetadata));
  auto found_metadata = model.FindMetadata(kKey);
  ASSERT_TRUE(found_metadata);
  EXPECT_EQ(found_metadata->StrView(), kMetadata);
}

TEST(ModelTest, MetadataDNE) {
  LiteRtModelT model;
  auto res = model.FindMetadata("FOO");
  ASSERT_FALSE(res.HasValue());
}

TEST(ModelTest, PopMetadata) {
  static constexpr absl::string_view kMetadata = "VALUE";
  static constexpr absl::string_view kKey = "KEY";

  LiteRtModelT model;
  LITERT_ASSERT_STATUS_OK(model.PushMetadata(kKey, kMetadata));

  auto popped_metadata = model.PopMetadata(kKey);
  ASSERT_TRUE(popped_metadata);
  EXPECT_EQ(popped_metadata->StrView(), kMetadata);

  EXPECT_FALSE(model.FindMetadata(kKey));
}

TEST(ModelTest, EmplaceSubgraph) {
  LiteRtModelT model;
  model.EmplaceSubgraph();
  EXPECT_EQ(model.Subgraphs().size(), 1);
}

TEST(ModelTest, Signature) {
  static constexpr absl::string_view kSignatureName = "MY_SIGNATURE";

  const std::vector<std::string> inputs = {"input_1", "input_2"};
  const std::vector<std::string> outputs = {"output_1"};

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  auto& signature = model.EmplaceSignature(&subgraph, inputs, outputs,
                                           std::string(kSignatureName));

  auto found_signature = model.FindSignature(kSignatureName);
  ASSERT_TRUE(found_signature);
  EXPECT_EQ(found_signature->get(), signature);
}

TEST(ModelTest, SignatureDNE) {
  static constexpr absl::string_view kSignatureName = "MY_SIGNATURE";
  LiteRtModelT model;
  auto found_signature = model.FindSignature(kSignatureName);
  EXPECT_FALSE(found_signature);
}

//
// Subgraph
//

TEST(ModelSubgraphTest, Input) {
  LiteRtTensorT tensor;
  LiteRtSubgraphT subgraph;
  subgraph.Inputs().push_back(&tensor);
  EXPECT_EQ(&subgraph.Input(0), subgraph.Inputs().front());
}

TEST(ModelSubgraphTest, Output) {
  LiteRtTensorT tensor;
  LiteRtSubgraphT subgraph;
  subgraph.Outputs().push_back(&tensor);
  EXPECT_EQ(&subgraph.Output(0), subgraph.Outputs().front());
}

TEST(ModelSubgraphTest, EmplaceTensor) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  ASSERT_EQ(subgraph.Tensors().size(), 1);
  EXPECT_THAT(subgraph.Tensors(), ElementsAreArray({&tensor}));
}

TEST(ModelSubgraphTest, EmplaceOp) {
  LiteRtSubgraphT subgraph;
  auto& op = subgraph.EmplaceOp();
  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_THAT(subgraph.Ops(), ElementsAreArray({&op}));
}

//
// Op
//

TEST(ModelOpTest, Input) {
  LiteRtOpT op;
  LiteRtTensorT tensor;
  op.Inputs().push_back(&tensor);
  EXPECT_EQ(&op.Input(0), op.Inputs().front());
}

TEST(ModelOpTest, Output) {
  LiteRtOpT op;
  LiteRtTensorT tensor;
  op.Outputs().push_back(&tensor);
  EXPECT_EQ(&op.Output(0), op.Outputs().front());
}

TEST(ModelOpTest, CustomOptions) {
  static constexpr absl::string_view kOpts = "OPTIONS";

  LiteRtOpT op;
  op.SetCustomOptions(kOpts);
  EXPECT_EQ(op.CustomOptions().StrView(), kOpts);
}

TEST(ModelOpTest, Options) {
  static constexpr auto kOptsType = ::tflite::BuiltinOptions_AddOptions;

  TflOptions options;
  options.type = kOptsType;
  options.Set(::tflite::AddOptionsT());

  LiteRtOpT op;
  detail::SetTflOptions(op, std::move(options));

  ASSERT_EQ(detail::GetTflOptions(op).type, kOptsType);
}

TEST(ModelOpTest, OpCode) {
  constexpr static auto kOpCode = kLiteRtOpCodeTflMul;

  LiteRtOpT op;
  op.SetOpCode(kOpCode);
  EXPECT_EQ(op.OpCode(), kOpCode);
}

//
// Tensor
//

TEST(ModelTensorTypeTest, MakeRankedTensorType) {
  static constexpr const int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);
  static constexpr auto kElementType = kLiteRtElementTypeFloat32;
  const auto tensor_type = MakeRankedTensorType(kElementType, kDimsSpan);
  ASSERT_EQ(tensor_type.first, kLiteRtRankedTensorType);
  EXPECT_EQ(tensor_type.second.ranked_tensor_type.element_type, kElementType);
  const auto& layout = tensor_type.second.ranked_tensor_type.layout;
  ASSERT_EQ(layout.rank, kDimsSpan.size());
  EXPECT_THAT(absl::MakeConstSpan(layout.dimensions, kDimsSpan.size()),
              ElementsAreArray(kDimsSpan));
}

TEST(ModelQuantizationTypeTest, MakePerTensor) {
  static constexpr auto kScale = 1.0f;
  static constexpr auto kZero = 1L;
  const auto quant = MakePerTensorQuantization(kScale, kZero);
  ASSERT_EQ(quant.first, kLiteRtQuantizationPerTensor);
  const auto& per_tensor = quant.second.per_tensor;
  EXPECT_EQ(per_tensor.scale, kScale);
  EXPECT_EQ(per_tensor.zero_point, kZero);
}

TEST(ModelQuantizationTypeTest, MakePerChannel) {
  static constexpr std::array kScale = {1.0f, 2.0f};
  static constexpr std::array kZero = {1L, 2L};
  static constexpr int32_t kQdim = 0;

  LiteRtTensorT tensor;
  const auto quant = MakePerChannelQuantization(
      kScale, kZero, kQdim,
      [&tensor](auto s) { return tensor.RequestBuffer(s); });

  ASSERT_EQ(quant.first, kLiteRtQuantizationPerChannel);
  const auto& per_channel = quant.second.per_channel;

  const auto size = per_channel.num_channels;
  ASSERT_EQ(size, 2);
  EXPECT_EQ(per_channel.quantized_dimension, 0);

  auto scales = absl::MakeConstSpan(per_channel.scales, size);
  auto zeros = absl::MakeConstSpan(per_channel.zero_points, size);

  EXPECT_THAT(scales, ElementsAreArray(kScale));
  EXPECT_THAT(zeros, ElementsAreArray(kZero));
}

TEST(ModelWeightsTest, WeightsFromBuf) {
  static constexpr absl::string_view kData = "some_data";

  LiteRtWeightsT weights;
  weights.SetFromBuf(BufferRef<uint8_t>(kData.data(), kData.size()));
  EXPECT_EQ(weights.Buf().StrView(), kData);
}

TEST(ModelTensorTest, Name) {
  static constexpr absl::string_view kName = "TENSOR_NAME";

  LiteRtTensorT tensor;
  tensor.SetName(std::string(kName.begin(), kName.end()));
  EXPECT_EQ(tensor.Name(), kName);
}

TEST(ModelTensorTest, Use) {
  LiteRtTensorT tensor;
  tensor.Users().emplace_back();
  tensor.UserArgInds().push_back(0);
  auto [user, ind] = tensor.GetUse(0);
  EXPECT_EQ(user, tensor.Users().front());
  EXPECT_EQ(ind, 0);
}

TEST(ModelTensorTest, DefiningOp) {
  LiteRtTensorT tensor;
  LiteRtOpT op;
  tensor.SetDefiningOp(op, 0);
  EXPECT_EQ(tensor.DefiningOp(), &op);
  EXPECT_EQ(tensor.DefiningOpOutInd(), 0);
}

//
// Util
//

TEST(ModelOpListTest, Push) {
  LiteRtOpListT op_list;
  LiteRtOpT op;
  op_list.Push(&op);
  auto vec = op_list.Vec();
  EXPECT_EQ(vec.front(), &op);
}

}  // namespace
}  // namespace litert::internal
