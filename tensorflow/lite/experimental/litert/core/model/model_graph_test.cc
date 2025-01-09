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

#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"

#include <cstddef>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/core/model/graph_validation.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {
namespace {

using ::testing::UnorderedElementsAreArray;

// Custom matcher; example:
// ```
// LiteRtTensor tensor ...
// EXPECT_THAT(tensor, HasRankedType(kLiteRtInt, absl::MakeSpan({2, 2})));
// ```
// TODO: Update to use dumping API directly and move to shared header.
MATCHER_P2(HasRankedType, element_type, shape, "") {
  if (arg.Type().first != kLiteRtRankedTensorType) {
    *result_listener << "Not ranked tensor type";
    return false;
  }
  const auto& ranked_tensor_type = arg.Type().second.ranked_tensor_type;
  const auto& layout = ranked_tensor_type.layout;

  const auto element_type_eq = ranked_tensor_type.element_type == element_type;
  const auto rank_eq = layout.rank == std::size(shape);

  auto actual_shape = absl::MakeConstSpan(layout.dimensions, layout.rank);
  auto expected_shape =
      absl::MakeConstSpan(std::cbegin(shape), std::cend(shape));
  const auto shape_eq = actual_shape == expected_shape;

  if (shape_eq && element_type_eq && rank_eq) {
    return true;
  }

  *result_listener << "\n";
  if (!shape_eq) {
    *result_listener << "Not correct shape\n";
  }
  if (!element_type_eq) {
    *result_listener << "Not correct element type\n";
  }
  if (!rank_eq) {
    *result_listener << "Not correct rank\n";
  }

  *result_listener << absl::StreamFormat("Actual ElementType is: %d\n",
                                         ranked_tensor_type.element_type);
  *result_listener << absl::StreamFormat("Actual Rank is: %lu\n", layout.rank);
  *result_listener << "Actual shape is: { ";
  for (const auto d : actual_shape) {
    *result_listener << absl::StreamFormat("%d, ", d);
  }
  *result_listener << "}\n";

  return false;
}

using ::testing::ElementsAreArray;

static constexpr size_t kRank = 1;
static constexpr int32_t kDims[] = {2};
static constexpr absl::Span<const int32_t> kDimsSpan(kDims);
static constexpr auto kType = kLiteRtElementTypeInt32;
static constexpr absl::string_view kCustomOptions = "OPTIONS";
static constexpr auto kOpCode = kLiteRtOpCodeTflMul;

LiteRtTensorT TestTensor() {
  LiteRtTensorT tensor;
  tensor.Type().first = kLiteRtRankedTensorType;
  tensor.Type().second.ranked_tensor_type.element_type = kType;
  tensor.Type().second.ranked_tensor_type.layout.dimensions[0] = kDims[0];
  tensor.Type().second.ranked_tensor_type.layout.rank = kRank;
  return tensor;
}

LiteRtOpT TestOp() {
  LiteRtOpT op;
  op.SetOpCode(kOpCode);
  op.SetCustomOptions(kCustomOptions);
  return op;
}

TEST(ModelGraphTest, CloneTensor) {
  LiteRtTensorT dest;
  CloneTo(TestTensor(), dest);
  EXPECT_THAT(dest, HasRankedType(kType, kDimsSpan));
}

TEST(ModelGraphTest, MakeCloneTensor) {
  LiteRtSubgraphT subgraph;
  auto& dest = MakeClone(subgraph, TestTensor());
  EXPECT_THAT(dest, HasRankedType(kType, kDimsSpan));
}

TEST(ModelGraphTest, CloneOp) {
  LiteRtOpT dest;
  CloneTo(TestOp(), dest);
  EXPECT_EQ(dest.OpCode(), kOpCode);
  EXPECT_EQ(dest.CustomOptions().StrView(), kCustomOptions);
}

TEST(ModelGraphTest, MakeCloneOp) {
  LiteRtSubgraphT subgraph;
  auto& dest = MakeClone(subgraph, TestOp());
  EXPECT_EQ(dest.OpCode(), kOpCode);
  EXPECT_EQ(dest.CustomOptions().StrView(), kCustomOptions);
}

TEST(ModelGraphTest, OpFindInput) {
  auto op = TestOp();
  auto tensor = TestTensor();
  AttachInput(&tensor, op);
  auto input = FindInput(op, tensor);
  ASSERT_TRUE(input);
  EXPECT_EQ(*input, 0);
}

TEST(ModelGraphTest, OpFindOutput) {
  auto op = TestOp();
  auto tensor = TestTensor();
  AttachOutput(&tensor, op);
  auto output = FindOutput(op, tensor);
  ASSERT_TRUE(output);
  EXPECT_EQ(*output, 0);
}

TEST(ModelGraphTest, SubgraphFindInput) {
  LiteRtSubgraphT subgraph;
  auto tensor = TestTensor();
  subgraph.Inputs().push_back(&tensor);
  auto input = FindInput(subgraph, tensor);
  ASSERT_TRUE(input);
  EXPECT_EQ(*input, 0);
}

TEST(ModelGraphTest, SubgraphFindOutput) {
  LiteRtSubgraphT subgraph;
  auto tensor = TestTensor();
  subgraph.Outputs().push_back(&tensor);
  auto output = FindOutput(subgraph, tensor);
  ASSERT_TRUE(output);
  EXPECT_EQ(*output, 0);
}

TEST(ModelGraphTest, TensorFindUseInds) {
  auto op1 = TestOp();
  auto op2 = TestOp();
  auto tensor = TestTensor();

  AttachInput(&tensor, op1);
  AttachInput(&tensor, op2);
  AttachInput(&tensor, op1);

  auto use_inds = FindUseInds(tensor, op1);
  auto uses = GetTensorUses(tensor, use_inds);
  ASSERT_EQ(uses.size(), 2);

  LiteRtTensorT::UseVec expected = {{&op1, 0}, {&op1, 1}};
  EXPECT_THAT(uses, UnorderedElementsAreArray(expected));
}

TEST(ModelGraphTest, OpAttachInput) {
  auto op = TestOp();
  auto tensor = TestTensor();
  AttachInput(&tensor, op);
  EXPECT_THAT(op.Inputs(), ElementsAreArray({&tensor}));
  EXPECT_THAT(tensor.Users(), ElementsAreArray({&op}));
  EXPECT_THAT(tensor.UserArgInds(), ElementsAreArray({0}));
}

TEST(ModelGraphTest, OpAttachOutput) {
  auto op = TestOp();
  auto tensor = TestTensor();
  AttachOutput(&tensor, op);
  EXPECT_THAT(op.Outputs(), ElementsAreArray({&tensor}));
  EXPECT_EQ(tensor.DefiningOp(), &op);
  EXPECT_EQ(tensor.DefiningOpOutInd(), 0);
}

TEST(ModelGraphTest, DisconnectInputOp) {
  auto op = TestOp();
  auto tensor = TestTensor();
  AttachInput(&tensor, op);
  auto disconnected = DisconnectInput(op, 0);
  EXPECT_EQ(disconnected, &tensor);
  EXPECT_TRUE(op.Inputs().empty());
  EXPECT_TRUE(tensor.Users().empty());
  EXPECT_TRUE(tensor.UserArgInds().empty());
}

TEST(ModelGraphTest, DisconnectMiddleInputOp) {
  auto op = TestOp();

  auto tensor1 = TestTensor();
  auto tensor2 = TestTensor();
  auto tensor3 = TestTensor();

  AttachInput(&tensor1, op);
  AttachInput(&tensor2, op);
  AttachInput(&tensor3, op);

  auto disconnected = DisconnectInput(op, 1);

  EXPECT_EQ(disconnected, &tensor2);
  ASSERT_EQ(op.Inputs().size(), 2);
  EXPECT_EQ(op.Inputs().front(), &tensor1);
  EXPECT_EQ(op.Inputs().back(), &tensor3);
  ASSERT_TRUE(tensor2.Users().empty());
  ASSERT_TRUE(tensor2.UserArgInds().empty());

  ASSERT_TRUE(ValidateLocalTopology(op));
}

TEST(ModelGraphTest, DisconnectOutputOp) {
  auto op = TestOp();
  auto tensor = TestTensor();
  AttachOutput(&tensor, op);
  auto disconnected = DisconnectOutput(op, 0);
  EXPECT_EQ(disconnected, &tensor);
  EXPECT_EQ(tensor.DefiningOp(), nullptr);
  EXPECT_TRUE(op.Outputs().empty());
}

TEST(ModelGraphTest, DropOp) {
  LiteRtOpT op;

  LiteRtTensorT input1;
  LiteRtTensorT input2;
  LiteRtTensorT output;

  AttachInput(&input1, op);
  AttachInput(&input2, op);
  AttachOutput(&output, op);

  Drop(op);

  EXPECT_TRUE(op.Inputs().empty());
  EXPECT_TRUE(op.Outputs().empty());
  EXPECT_TRUE(input1.Users().empty());
  EXPECT_TRUE(input2.Users().empty());
  EXPECT_EQ(output.DefiningOp(), nullptr);
}

TEST(ModelGraphTestDCE, NoDeadCode) {
  LiteRtSubgraphT subgraph;

  auto& input = subgraph.EmplaceTensor();
  auto& output = subgraph.EmplaceTensor();

  auto& op = subgraph.EmplaceOp();

  AttachInput(&input, op);
  AttachOutput(&output, op);

  subgraph.Inputs().push_back(&input);
  subgraph.Outputs().push_back(&output);

  ASSERT_FALSE(DCE(subgraph));
  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Tensors().size(), 2);

  ASSERT_TRUE(
      ValidateLocalTopology(subgraph.Ops().cbegin(), subgraph.Ops().cend()));
  ASSERT_TRUE(ValidateSubgraphIO(subgraph));
}

TEST(ModelGraphTestDCE, DeadTensor) {
  LiteRtSubgraphT subgraph;
  subgraph.EmplaceTensor();

  ASSERT_TRUE(DCE(subgraph));
  EXPECT_TRUE(subgraph.Tensors().empty());

  ASSERT_TRUE(
      ValidateLocalTopology(subgraph.Ops().cbegin(), subgraph.Ops().cend()));
  ASSERT_TRUE(ValidateSubgraphIO(subgraph));
}

TEST(ModelGraphTestDCE, DeadOp) {
  LiteRtSubgraphT subgraph;
  subgraph.EmplaceOp();

  ASSERT_TRUE(DCE(subgraph));
  EXPECT_TRUE(subgraph.Ops().empty());

  ASSERT_TRUE(
      ValidateLocalTopology(subgraph.Ops().cbegin(), subgraph.Ops().cend()));
  ASSERT_TRUE(ValidateSubgraphIO(subgraph));
}

TEST(ModelGraphTestDCE, SomeDead) {
  LiteRtSubgraphT subgraph;

  auto& input = subgraph.EmplaceTensor();
  auto& output = subgraph.EmplaceTensor();

  auto& op = subgraph.EmplaceOp();

  AttachInput(&input, op);
  AttachOutput(&output, op);

  // Dead
  subgraph.EmplaceTensor();
  subgraph.EmplaceOp();

  subgraph.Inputs().push_back(&input);
  subgraph.Outputs().push_back(&output);

  ASSERT_TRUE(DCE(subgraph));
  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Tensors().size(), 2);

  ASSERT_TRUE(
      ValidateLocalTopology(subgraph.Ops().cbegin(), subgraph.Ops().cend()));
  ASSERT_TRUE(ValidateSubgraphIO(subgraph));
}

}  // namespace
}  // namespace litert::internal
