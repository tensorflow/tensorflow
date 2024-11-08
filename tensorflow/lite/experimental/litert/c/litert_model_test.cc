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
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

using ::litert::BufferRef;
using ::testing::ElementsAreArray;

template <typename T>
LiteRtWeightsT MakeWeights(std::initializer_list<T> data, size_t offset = 0) {
  LiteRtWeightsT weights;
  weights.fb_buffer = std::make_unique<tflite::BufferT>();
  weights.fb_buffer->data.resize(data.size() * sizeof(T));
  auto data_it = data.begin();
  for (int i = 0; i < data.size(); ++i) {
    *(reinterpret_cast<T*>(weights.fb_buffer->data.data()) + i) = *data_it;
    ++data_it;
  }
  weights.fb_buffer->size = weights.fb_buffer->data.size();
  weights.fb_buffer->offset = offset;
  return weights;
}

TEST(LiteRtWeightsTest, GetNullWeights) {
  LiteRtWeightsT weights = {};

  const void* addr;
  size_t size;
  LITERT_ASSERT_STATUS_OK(LiteRtGetWeightsBytes(&weights, &addr, &size));

  EXPECT_EQ(addr, nullptr);
  EXPECT_EQ(size, 0);
}

TEST(LiteRtWeightsTest, GetWeights) {
  auto weights = MakeWeights<int32_t>({1, 2, 3});

  const void* addr;
  size_t size;
  LITERT_ASSERT_STATUS_OK(LiteRtGetWeightsBytes(&weights, &addr, &size));

  EXPECT_NE(addr, nullptr);
  EXPECT_EQ(size, 3 * sizeof(int32_t));

  EXPECT_THAT(absl::MakeConstSpan(reinterpret_cast<const int32_t*>(addr), 3),
              ElementsAreArray({1, 2, 3}));
}

TEST(LiteRtTensorTest, GetUnrankedType) {
  LiteRtTensorT tensor;
  tensor.type_id = kLiteRtUnrankedTensorType;
  tensor.type_detail.unranked_tensor_type.element_type =
      kLiteRtElementTypeFloat32;

  LiteRtTensorTypeId id;
  LITERT_ASSERT_STATUS_OK(LiteRtGetTensorTypeId(&tensor, &id));
  ASSERT_EQ(id, kLiteRtUnrankedTensorType);

  LiteRtUnrankedTensorType unranked;
  LITERT_ASSERT_STATUS_OK(LiteRtGetUnrankedTensorType(&tensor, &unranked));
  EXPECT_EQ(unranked.element_type, kLiteRtElementTypeFloat32);
}

TEST(LiteRtTensorTest, GetRankedTensorType) {
  LiteRtTensorT tensor;
  tensor.type_id = kLiteRtRankedTensorType;
  tensor.type_detail.ranked_tensor_type.element_type =
      kLiteRtElementTypeFloat32;
  LITERT_STACK_ARRAY(int32_t, dims, 2, 3);
  tensor.type_detail.ranked_tensor_type.layout.dimensions = dims;
  tensor.type_detail.ranked_tensor_type.layout.rank = 2;

  LiteRtTensorTypeId id;
  LITERT_ASSERT_STATUS_OK(LiteRtGetTensorTypeId(&tensor, &id));
  ASSERT_EQ(id, kLiteRtRankedTensorType);

  LiteRtRankedTensorType ranked;
  LITERT_ASSERT_STATUS_OK(LiteRtGetRankedTensorType(&tensor, &ranked));
  EXPECT_EQ(ranked.element_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(ranked.layout.rank, 2);
  EXPECT_THAT(absl::MakeConstSpan(ranked.layout.dimensions, 2),
              ElementsAreArray({3, 3}));
}

TEST(LiteRtTensorTest, GetUses) {
  LiteRtTensorT tensor;

  LiteRtOpT user;
  tensor.users.push_back(&user);
  tensor.user_arg_inds.push_back(0);

  LiteRtOpT other_user;
  tensor.users.push_back(&other_user);
  tensor.user_arg_inds.push_back(1);

  LiteRtParamIndex num_uses;
  LiteRtOpArray actual_users;
  LiteRtParamIndex* user_arg_inds;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetTensorUses(&tensor, &num_uses, &actual_users, &user_arg_inds));

  ASSERT_EQ(num_uses, 2);
  EXPECT_THAT(absl::MakeConstSpan(actual_users, 2),
              ElementsAreArray({&user, &other_user}));
  EXPECT_THAT(absl::MakeConstSpan(user_arg_inds, 2), ElementsAreArray({0, 1}));
}

TEST(LiteRtTensorTest, GetDefiningOp) {
  LiteRtTensorT tensor;

  LiteRtOpT def_op;
  tensor.defining_op = &def_op;
  tensor.defining_op_out_ind = 0;

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

TEST(LiteRtOpTest, GetOpCode) {
  LiteRtOpT op;
  op.op_code = kLiteRtOpCodeTflCustom;

  LiteRtOpCode code;
  LITERT_ASSERT_STATUS_OK(LiteRtGetOpCode(&op, &code));
  EXPECT_EQ(code, kLiteRtOpCodeTflCustom);
}

TEST(LiteRtOpTest, GetInputs) {
  LiteRtTensorT input1;
  LiteRtTensorT input2;

  LiteRtOpT op;
  op.inputs.push_back(&input1);
  op.inputs.push_back(&input2);

  LiteRtTensorArray inputs;
  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_STATUS_OK(LiteRtGetOpInputs(&op, &num_inputs, &inputs));
  ASSERT_EQ(num_inputs, 2);
  EXPECT_THAT(absl::MakeConstSpan(inputs, num_inputs),
              ElementsAreArray({&input1, &input2}));
}

TEST(LiteRtOpTest, GetOutputs) {
  LiteRtTensorT output1;
  LiteRtTensorT output2;

  LiteRtOpT op;
  op.outputs.push_back(&output1);
  op.outputs.push_back(&output2);

  LiteRtTensorArray outputs;
  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_STATUS_OK(LiteRtGetOpOutputs(&op, &num_outputs, &outputs));
  ASSERT_EQ(num_outputs, 2);
  EXPECT_THAT(absl::MakeConstSpan(outputs, num_outputs),
              ElementsAreArray({&output1, &output2}));
}

TEST(LiteRtSubgraphTest, GetInputs) {
  LiteRtTensorT input1;
  LiteRtTensorT input2;

  LiteRtSubgraphT subgraph;
  subgraph.inputs.push_back(&input1);
  subgraph.inputs.push_back(&input2);

  LiteRtTensorArray inputs;
  LiteRtParamIndex num_inputs;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetSubgraphInputs(&subgraph, &num_inputs, &inputs));
  ASSERT_EQ(num_inputs, 2);
  EXPECT_THAT(absl::MakeConstSpan(inputs, num_inputs),
              ElementsAreArray({&input1, &input2}));
}

TEST(LiteRtSubgraphTest, GetOutputs) {
  LiteRtTensorT output1;
  LiteRtTensorT output2;

  LiteRtSubgraphT subgraph;
  subgraph.outputs.push_back(&output1);
  subgraph.outputs.push_back(&output2);

  LiteRtTensorArray outputs;
  LiteRtParamIndex num_outputs;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetSubgraphOutputs(&subgraph, &num_outputs, &outputs));
  ASSERT_EQ(num_outputs, 2);
  EXPECT_THAT(absl::MakeConstSpan(outputs, num_outputs),
              ElementsAreArray({&output1, &output2}));
}

TEST(LiteRtSubgraphTest, GetOps) {
  LiteRtOpT op1;
  LiteRtOpT op2;

  LiteRtSubgraphT subgraph;
  subgraph.ops.push_back(&op1);
  subgraph.ops.push_back(&op2);

  LiteRtOpArray ops;
  LiteRtParamIndex num_ops;
  LITERT_ASSERT_STATUS_OK(LiteRtGetSubgraphOps(&subgraph, &num_ops, &ops));
  ASSERT_EQ(num_ops, 2);
  EXPECT_THAT(absl::MakeConstSpan(ops, num_ops),
              ElementsAreArray({&op1, &op2}));
}

TEST(LiteRtModelTest, GetMetadata) {
  LiteRtModelT model;
  model.flatbuffer_model = std::make_unique<tflite::ModelT>();
  litert::OwningBufferRef<uint8_t> buf("Bar");
  model.PushMetadata("Foo", buf);

  const void* metadata;
  size_t metadata_size;
  LITERT_ASSERT_STATUS_OK(
      LiteRtGetModelMetadata(&model, "Foo", &metadata, &metadata_size));
  ASSERT_EQ(metadata_size, 3);
  EXPECT_EQ(BufferRef(metadata, metadata_size).StrView(), "Bar");
}

TEST(LiteRtModelTest, GetSubgraph) {
  LiteRtModelT model;
  auto& subgraph = model.subgraphs.emplace_back();

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
