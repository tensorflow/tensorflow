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

#include <optional>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

TEST(Tensor, SimpleModel) {
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");
  EXPECT_EQ(model.NumSubgraphs(), 1);

  auto subgraph = model.Subgraph(0);

  auto inputs = subgraph->Inputs();
  ASSERT_EQ(inputs.size(), 2);

  litert::Tensor input_tensor(inputs[0]);
  ASSERT_EQ(input_tensor.TypeId(), kLiteRtRankedTensorType);

  auto input_ranked_tensor_type = input_tensor.RankedTensorType();
  ASSERT_EQ(input_ranked_tensor_type.ElementType(),
            litert::ElementType::Float32);

  EXPECT_FALSE(input_tensor.HasWeights());

  auto input_weights = input_tensor.Weights();
  ASSERT_EQ(input_weights.Bytes().size(), 0);

  ASSERT_EQ(input_tensor.DefiningOp(), std::nullopt);

  absl::Span<LiteRtOp> input_uses;
  absl::Span<LiteRtParamIndex> input_user_arg_indices;
  input_tensor.Uses(input_uses, input_user_arg_indices);
  ASSERT_EQ(input_uses.size(), 1);
  ASSERT_EQ(input_user_arg_indices.size(), 1);

  auto outputs = subgraph->Outputs();
  ASSERT_EQ(outputs.size(), 1);

  litert::Tensor output_tensor(outputs[0]);
  ASSERT_EQ(output_tensor.TypeId(), kLiteRtRankedTensorType);

  auto output_defining_op = output_tensor.DefiningOp();
  EXPECT_TRUE(output_defining_op.has_value());

  absl::Span<LiteRtOp> output_uses;
  absl::Span<LiteRtParamIndex> output_user_arg_indices;
  output_tensor.Uses(output_uses, output_user_arg_indices);
  ASSERT_EQ(output_uses.size(), 0);
  ASSERT_EQ(output_user_arg_indices.size(), 0);
}

}  // namespace
