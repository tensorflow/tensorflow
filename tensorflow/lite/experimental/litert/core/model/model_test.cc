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
#include "tensorflow/lite/experimental/litert/core/model/buffer_manager.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
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
  LITERT_ASSERT_OK(model.PushMetadata(kKey, kMetadata));
  auto found_metadata = model.FindMetadata(kKey);
  ASSERT_TRUE(found_metadata);
  EXPECT_EQ(found_metadata->StrView(), kMetadata);
}

TEST(ModelTest, MetadataDNE) {
  LiteRtModelT model;
  auto res = model.FindMetadata("FOO");
  ASSERT_FALSE(res.HasValue());
}

TEST(ModelTest, EmplaceSubgraph) {
  LiteRtModelT model;
  auto& sg = model.EmplaceSubgraph();
  EXPECT_EQ(model.Subgraphs().size(), 1);
  auto& tensor = sg.EmplaceTensor();
  EXPECT_EQ(tensor.Weights().GetBufferManager(), model.Buffers());
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

TEST(ModelTest, AttachExternalBufferToOp) {
  static constexpr absl::string_view kBufferData = "BUFFER_DATA";
  static constexpr absl::string_view kOpName = "OP1";
  static constexpr absl::string_view kOp2Name = "OP2";

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op = subgraph.EmplaceOp();
  auto& op2 = subgraph.EmplaceOp();

  OwningBufferRef<uint8_t> external_buf(kBufferData);

  auto buf1_id = model.Buffers()->RegisterOwnedBuffer(std::move(external_buf));

  model.AttachAssetToOp(&op, buf1_id, std::string(kOpName));
  model.AttachAssetToOp(&op2, buf1_id, std::string(kOp2Name));

  auto op_1_res = model.FindOpAsset(&op);
  ASSERT_TRUE(op_1_res);
  EXPECT_EQ(op_1_res->second, kOpName);
  EXPECT_EQ(op_1_res->first, buf1_id);

  auto op_2_res = model.FindOpAsset(&op2);
  ASSERT_TRUE(op_2_res);
  EXPECT_EQ(op_2_res->second, kOp2Name);
  EXPECT_EQ(op_2_res->first, buf1_id);
}

TEST(ModelTest, ExternalBufferNotFound) {
  LiteRtModelT model;
  LiteRtOpT op;
  ASSERT_FALSE(model.FindOpAsset(&op));
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
      [&tensor](auto s) { return tensor.RequestScratchBuffer(s); });

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

TEST(ModelWeightsTest, EmptyWeights) {
  LiteRtWeightsT weights;
  EXPECT_EQ(weights.Buffer().Size(), 0);
}

TEST(ModelWeightsTest, WeightsWithExternalBufferManager) {
  static constexpr absl::string_view kData = "some_data";
  BufferManager manager;

  LiteRtWeightsT weights;
  weights.SetBufferManager(&manager);

  BufferRef<uint8_t> buf(kData.data(), kData.size());
  SetWeightsFromUnownedBuffer(weights, buf);

  EXPECT_EQ(manager.GetBuffer(weights.GetBufferId())->StrView(), kData);
  EXPECT_EQ(weights.Buffer().StrView(), kData);
}

TEST(ModelWeightsTest, WeightsFromUnownedBuffer) {
  static constexpr absl::string_view kData = "some_data";

  LiteRtWeightsT weights;
  BufferRef<uint8_t> buf(kData.data(), kData.size());
  SetWeightsFromUnownedBuffer(weights, buf);

  EXPECT_EQ(weights.Buffer().StrView(), kData);
}

TEST(ModelWeightsTest, WeightsFromOwnedBuffer) {
  static constexpr absl::string_view kData = "some_data";

  LiteRtWeightsT weights;

  OwningBufferRef<uint8_t> buf(kData);
  SetWeightsFromUnownedBuffer(weights, std::move(buf));

  EXPECT_EQ(weights.Buffer().StrView(), kData);
}

TEST(ModelWeightsTest, OverwriteBuffer) {
  static constexpr absl::string_view kData = "some_data";
  static constexpr absl::string_view kData2 = "some_data2";

  LiteRtWeightsT weights;

  {
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }

  {
    OwningBufferRef<uint8_t> buf(kData2);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }

  EXPECT_EQ(weights.Buffer().StrView(), kData2);
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
// Misc Ir Containers
//

TEST(ModelOpListTest, Push) {
  LiteRtOpListT op_list;
  LiteRtOpT op;
  op_list.Push(&op);
  auto vec = op_list.Values();
  EXPECT_EQ(vec.front().first, &op);
}

TEST(ModelOpListTest, PushWithIndex) {
  LiteRtOpListT op_list;
  LiteRtOpT op;
  op_list.Push(&op, 1);
  auto vec = op_list.Values();
  EXPECT_EQ(vec.front().first, &op);
  EXPECT_EQ(vec.front().second, 1);
}

//
// Traversal Utils
//

TEST(CcForEachIrTest, OpF3) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model, [&](LiteRtSubgraph subgraph, int32_t subgraph_index,
                        LiteRtOp op) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(CcForEachIrTest, OpF1) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model, [&](LiteRtOp op) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(CcForEachIrTest, OpF2) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model, [&](LiteRtSubgraph subgraph, LiteRtOp op) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(CcForEachIrTest, SgF1) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model, [&](LiteRtSubgraph subgraph) { count++; });
  EXPECT_EQ(count, 1);
}

TEST(CcForEachIrTest, SgF2) {
  LiteRtModelT model;
  model.EmplaceSubgraph().EmplaceOp();

  int count = 0;
  ForEachIr(&model,
            [&](LiteRtSubgraph subgraph, int32_t subgraph_index) { count++; });
  EXPECT_EQ(count, 1);
}

}  // namespace
}  // namespace litert::internal
