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

#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"

#include <optional>

#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace litert {

namespace {

using ::litert::testing::LoadTestFileModel;

TEST(MatchRankedTensorTypeTest, HasAll) {
  auto litert_model = LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  const auto& input = inputs.front();
  auto input_tensor_type = input.RankedTensorType();
  EXPECT_TRUE(input_tensor_type);
  EXPECT_TRUE(MatchRankedTensorType(
      *input_tensor_type, TensorTypeInfo(ElementType::Float32, {2, 2})));
}

TEST(MatchRankedTensorTypeTest, NoMatch) {
  auto litert_model = LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  const auto& input = inputs.front();
  auto input_tensor_type = input.RankedTensorType();
  EXPECT_TRUE(input_tensor_type);
  EXPECT_FALSE(MatchRankedTensorType(
      *input_tensor_type, TensorTypeInfo(ElementType::Float32, {3, 2})));
}

TEST(MatchRankedTensorTypeTest, AnyDims) {
  auto litert_model = LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  const auto& input = inputs.front();
  auto input_tensor_type = input.RankedTensorType();
  EXPECT_TRUE(input_tensor_type);
  EXPECT_TRUE(MatchRankedTensorType(*input_tensor_type,
                                    TensorTypeInfo(ElementType::Float32)));
}

TEST(MatchRankedTensorTypeTest, AnyElementType) {
  auto litert_model = LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  const auto& input = inputs.front();
  auto input_tensor_type = input.RankedTensorType();
  EXPECT_TRUE(input_tensor_type);
  EXPECT_TRUE(
      MatchRankedTensorType(*input_tensor_type, TensorTypeInfo({2, 2})));
}

TEST(MatchOpTypeTest, HasAll) {
  auto litert_model = LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  TensorTypeInfo expected_type(ElementType::Float32, {2, 2});
  EXPECT_TRUE(MatchOpType(ops.front(), {expected_type, expected_type},
                          {expected_type}));
}

TEST(MatchOpTypeTest, NoMatch) {
  auto litert_model = LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  TensorTypeInfo expected_type(ElementType::Float32, {2, 2});
  TensorTypeInfo not_expected_type(ElementType::Int32, {2, 2});
  EXPECT_FALSE(MatchOpType(ops.front(), {not_expected_type, expected_type},
                           {expected_type}));
}

TEST(MatchOpTypeTest, AnyInput) {
  auto litert_model = LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  TensorTypeInfo expected_type(ElementType::Float32, {2, 2});
  EXPECT_TRUE(
      MatchOpType(ops.front(), {std::nullopt, expected_type}, {expected_type}));
}

TEST(MatchOpTypeTest, AnyOutput) {
  auto litert_model = LoadTestFileModel("one_mul.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  TensorTypeInfo expected_type(ElementType::Float32, {2, 2});
  EXPECT_TRUE(
      MatchOpType(ops.front(), {std::nullopt, expected_type}, {std::nullopt}));
}

TEST(MatchWeightsTest, Matches) {
  auto litert_model = LoadTestFileModel("add_cst.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  const auto& cst = inputs.back();
  EXPECT_TRUE(MatchWeights(cst, absl::Span<const float>({1.0, 2.0, 3.0, 4.0})));
}

TEST(MatchWeightsTest, NoMatchBadType) {
  auto litert_model = LoadTestFileModel("add_cst.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  const auto& cst = inputs.back();
  EXPECT_FALSE(
      MatchWeights(cst, absl::Span<const double>({1.0, 2.0, 3.0, 4.0})));
}
TEST(MatchWeightsTest, NoMatchBadVals) {
  auto litert_model = LoadTestFileModel("add_cst.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  const auto& cst = inputs.back();
  EXPECT_FALSE(
      MatchWeights(cst, absl::Span<const float>({3.0, 2.0, 3.0, 5.0})));
}

TEST(MatchUseTest, Match) {
  auto litert_model = LoadTestFileModel("add_cst.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  EXPECT_TRUE(MatchUse(inputs.back(), UseInfo{kLiteRtOpCodeTflAdd, 1}));
}

TEST(MatchUseTest, MatchAnyCode) {
  auto litert_model = LoadTestFileModel("add_cst.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  EXPECT_TRUE(MatchUse(inputs.back(), UseInfo{std::nullopt, 1}));
}

TEST(MatchUseTest, NoMatch) {
  auto litert_model = LoadTestFileModel("add_cst.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto ops = subgraph->Ops();
  const auto inputs = ops.front().Inputs();
  EXPECT_FALSE(MatchUse(inputs.back(), UseInfo{std::nullopt, 2}));
}

TEST(MatchUsesTest, StrictMatch) {
  auto litert_model = LoadTestFileModel("add_simple.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto subgraph_inputs = subgraph->Inputs();
  const auto& tensor = subgraph_inputs.front();
  EXPECT_TRUE(
      MatchUses(tensor, {{kLiteRtOpCodeTflAdd, 0}, {kLiteRtOpCodeTflAdd, 1}}));
}

TEST(MatchUsesTest, StrictNoMatch) {
  auto litert_model = LoadTestFileModel("add_simple.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto subgraph_inputs = subgraph->Inputs();
  const auto& tensor = subgraph_inputs.front();
  EXPECT_FALSE(MatchUses(tensor, {{kLiteRtOpCodeTflAdd, 0}}));
}

TEST(MatchUsesTest, NonStrict) {
  auto litert_model = LoadTestFileModel("add_simple.tflite");
  auto subgraph = litert_model.MainSubgraph();
  ABSL_CHECK(subgraph);
  auto subgraph_inputs = subgraph->Inputs();
  const auto& tensor = subgraph_inputs.front();
  EXPECT_TRUE(MatchUses(tensor, {{kLiteRtOpCodeTflAdd, 0}}, /*strict=*/false));
}

}  // namespace

}  // namespace litert
