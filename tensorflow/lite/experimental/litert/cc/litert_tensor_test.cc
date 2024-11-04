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

#include "tensorflow/lite/experimental/litert/cc/litert_tensor.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/core/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

using ::litert::LiteRtTensorManager;

TEST(TestLiteRtTensorManager, SimpleRankedTensorSubgraphInput) {
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto inputs,
                          ::graph_tools::GetSubgraphInputs(subgraph));

  LiteRtTensorManager::Unique tensor;
  ASSERT_STATUS_OK(LiteRtTensorManager::MakeFromTensor(inputs[0], tensor));

  ASSERT_EQ(tensor->Rank(), 2);
  EXPECT_EQ(tensor->Dims(), absl::MakeConstSpan({2, 2}));
  EXPECT_EQ(tensor->ElementType(), kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor->Tensor(), inputs[0]);
  EXPECT_TRUE(tensor->IsSubgraphInput());
  EXPECT_FALSE(tensor->IsSubgraphOutput());
}

TEST(TestLiteRtTensorManager, SimpleRankedTensorSubgraphOutput) {
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto outputs,
                          ::graph_tools::GetSubgraphOutputs(subgraph));

  LiteRtTensorManager::Unique tensor;
  ASSERT_STATUS_OK(LiteRtTensorManager::MakeFromTensor(outputs[0], tensor));

  ASSERT_EQ(tensor->Rank(), 2);
  EXPECT_EQ(tensor->Dims(), absl::MakeConstSpan({2, 2}));
  EXPECT_EQ(tensor->ElementType(), kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor->Tensor(), outputs[0]);
  EXPECT_TRUE(tensor->IsSubgraphOutput());
  EXPECT_FALSE(tensor->IsSubgraphInput());
}

TEST(TestLiteRtTensorManager, SimpleRankedTensor) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, ::graph_tools::GetSubgraphOps(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto op_outs, ::graph_tools::GetOpOuts(ops[1]));

  LiteRtTensorManager::Unique tensor;
  ASSERT_STATUS_OK(LiteRtTensorManager::MakeFromTensor(op_outs[0], tensor));

  ASSERT_EQ(tensor->Rank(), 2);
  EXPECT_EQ(tensor->Dims(), absl::MakeConstSpan({2, 2}));
  EXPECT_EQ(tensor->ElementType(), kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor->Tensor(), op_outs[0]);
  EXPECT_FALSE(tensor->IsSubgraphOutput());
  EXPECT_FALSE(tensor->IsSubgraphInput());
}

TEST(TestLiteRtTensorManager, NoStrides) {
  int32_t dimensions[] = {1, 2, 3};

  LiteRtTensorT tensor;
  tensor.type_id = kLiteRtRankedTensorType;
  tensor.type_detail.ranked_tensor_type.element_type =
      kLiteRtElementTypeFloat32;
  tensor.type_detail.ranked_tensor_type.layout.rank =
      sizeof(dimensions) / sizeof(dimensions[0]);
  tensor.type_detail.ranked_tensor_type.layout.dimensions = dimensions;
  tensor.type_detail.ranked_tensor_type.layout.strides = nullptr;

  LiteRtTensorManager::Unique tensor_manager;
  ASSERT_STATUS_OK(
      LiteRtTensorManager::MakeFromTensor(&tensor, tensor_manager));
  EXPECT_FALSE(tensor_manager->HasStrides());
}

TEST(TestLiteRtTensorManager, Strides) {
  int32_t dimensions[] = {1, 2, 3};
  uint32_t strides[] = {6, 3, 1};

  LiteRtTensorT tensor;
  tensor.type_id = kLiteRtRankedTensorType;
  tensor.type_detail.ranked_tensor_type.element_type =
      kLiteRtElementTypeFloat32;
  tensor.type_detail.ranked_tensor_type.layout.rank =
      sizeof(dimensions) / sizeof(dimensions[0]);
  tensor.type_detail.ranked_tensor_type.layout.dimensions = dimensions;
  tensor.type_detail.ranked_tensor_type.layout.strides = strides;

  LiteRtTensorManager::Unique tensor_manager;
  ASSERT_STATUS_OK(
      LiteRtTensorManager::MakeFromTensor(&tensor, tensor_manager));
  EXPECT_TRUE(tensor_manager->HasStrides());
}

}  // namespace
