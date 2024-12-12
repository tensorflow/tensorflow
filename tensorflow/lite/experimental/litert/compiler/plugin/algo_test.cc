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

#include "tensorflow/lite/experimental/litert/compiler/plugin/algo.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/core/model/graph_validation.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace litert::internal {
namespace {

TEST(TestPartitionsFromFlatList, SimpleMultiOp) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  {
    std::vector<LiteRtOp> partition;
    partition.push_back(ops.at(1).Get());
    partition.push_back(ops.at(2).Get());

    auto partitions = GroupPartitions(partition);
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions.front().size(), 2);

    EXPECT_EQ(partitions.front().at(0), partition.at(0));
    EXPECT_EQ(partitions.front().at(1), partition.at(1));
  }

  {
    std::vector<LiteRtOp> partition;
    partition.push_back(ops.at(1).Get());
    partition.push_back(ops.at(3).Get());

    auto partitions = GroupPartitions(partition);
    ASSERT_EQ(partitions.size(), 2);
    ASSERT_EQ(partitions.front().size(), 1);
    ASSERT_EQ(partitions.back().size(), 1);

    auto p1_op_code = partitions.front().front()->OpCode();
    auto p2_op_code = partitions.back().front()->OpCode();

    ASSERT_TRUE((p1_op_code == kLiteRtOpCodeTflMul &&
                 p2_op_code == kLiteRtOpCodeTflAdd) ||
                (p1_op_code == kLiteRtOpCodeTflAdd &&
                 p2_op_code == kLiteRtOpCodeTflMul));
  }

  {
    std::vector<LiteRtOp> partition;

    auto partitions = GroupPartitions(partition);
    ASSERT_EQ(partitions.size(), 0);
  }

  {
    std::vector<LiteRtOp> partition;
    partition.push_back(ops.at(0).Get());
    partition.push_back(ops.at(1).Get());
    partition.push_back(ops.at(2).Get());
    partition.push_back(ops.at(3).Get());

    auto partitions = GroupPartitions(partition);
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions.front().size(), 4);

    EXPECT_EQ(partitions.front().at(0), partition.at(0));
    EXPECT_EQ(partitions.front().at(1), partition.at(1));
    EXPECT_EQ(partitions.front().at(2), partition.at(2));
    EXPECT_EQ(partitions.front().at(3), partition.at(3));
  }
}

TEST(TestSliceSubgraphSimpleMultiOp, OnePartition) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  std::vector<LiteRtOp> partition;
  partition.push_back(ops.at(1).Get());
  partition.push_back(ops.at(2).Get());

  auto sliced_graph = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  auto* dispatch_op =
      OutlinePartition(*subgraph->Get(), sliced_graph.Get(), partition);

  const auto& internal_sliced = *sliced_graph.Get();
  ASSERT_TRUE(ValidateSubgraphIO(internal_sliced));
  ASSERT_TRUE(ValidateLocalTopology(internal_sliced.Ops().cbegin(),
                                    internal_sliced.Ops().cend()));

  auto edited_subgraph_ops = subgraph->Ops();

  ASSERT_EQ(edited_subgraph_ops.size(), 3);
  ASSERT_EQ(edited_subgraph_ops.at(0).Code(), kLiteRtOpCodeTflAdd);
  ASSERT_EQ(edited_subgraph_ops.at(1).Code(), kLiteRtOpCodeTflCustom);
  ASSERT_EQ(edited_subgraph_ops.at(2).Code(), kLiteRtOpCodeTflAdd);

  auto sliced_subgraph_ops = sliced_graph.Ops();

  ASSERT_EQ(sliced_subgraph_ops.size(), 2);
  ASSERT_EQ(sliced_subgraph_ops[0].Code(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(sliced_subgraph_ops[1].Code(), kLiteRtOpCodeTflMul);

  ASSERT_EQ(dispatch_op, edited_subgraph_ops.at(1).Get());
  const Op hal_call(dispatch_op);

  {
    const auto dispatch_op_ins = hal_call.Inputs();

    ASSERT_EQ(dispatch_op_ins.size(), 1);

    auto hal_input_defining_op = dispatch_op_ins.front().DefiningOp();
    ASSERT_EQ(hal_input_defining_op->op, edited_subgraph_ops.at(0).Get());
    ASSERT_EQ(hal_input_defining_op->op_output_index, 0);

    const auto sliced_subgraph_inputs = sliced_graph.Inputs();

    ASSERT_EQ(sliced_subgraph_inputs.size(), 1);

    ASSERT_TRUE(MatchUses(sliced_subgraph_inputs.front(),
                          {UseInfo{sliced_subgraph_ops.front().Code(), 0},
                           UseInfo{sliced_subgraph_ops.front().Code(), 0}}));
    ASSERT_TRUE(sliced_subgraph_inputs.front().IsSubgraphInput());
  }

  {
    const auto hal_call_outs = hal_call.Outputs();
    ASSERT_EQ(hal_call_outs.size(), 1);
    const auto& hal_call_out = hal_call_outs.front();

    ASSERT_TRUE(MatchUses(hal_call_out,
                          {UseInfo{edited_subgraph_ops.back().Code(), 0},
                           UseInfo{edited_subgraph_ops.back().Code(), 1}}));

    auto sliced_subgraph_outputs = sliced_graph.Outputs();

    ASSERT_EQ(sliced_subgraph_outputs.size(), 1);

    const auto defining_op = sliced_subgraph_outputs.front().DefiningOp();
    ASSERT_EQ(defining_op->op, sliced_subgraph_ops.back().Get());
    ASSERT_EQ(defining_op->op_output_index, 0);

    ASSERT_TRUE(sliced_subgraph_outputs.front().Uses().empty());
  }
}

TEST(TestSliceSubgraphSimpleMultiOp, TwoPartitions) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  std::vector<LiteRtOp> partition_1;
  partition_1.push_back(ops.at(0).Get());

  auto sliced_graph_1 = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  OutlinePartition(*(subgraph->Get()), sliced_graph_1.Get(), partition_1);

  const auto& internal_slice_1 = *sliced_graph_1.Get();
  ASSERT_TRUE(ValidateSubgraphIO(internal_slice_1));
  ASSERT_TRUE(ValidateLocalTopology(internal_slice_1.Ops().cbegin(),
                                    internal_slice_1.Ops().cend()));

  std::vector<LiteRtOp> partition_2;
  partition_2.push_back(ops.at(2).Get());
  partition_2.push_back(ops.at(3).Get());

  auto sliced_graph_2 = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  OutlinePartition(*(subgraph->Get()), sliced_graph_2.Get(), partition_2);

  const auto& internal_slice_2 = *sliced_graph_2.Get();
  ASSERT_TRUE(ValidateSubgraphIO(internal_slice_2));
  ASSERT_TRUE(ValidateLocalTopology(internal_slice_2.Ops().cbegin(),
                                    internal_slice_2.Ops().cend()));

  auto edited_subgraph_ops = subgraph->Ops();

  ASSERT_EQ(edited_subgraph_ops.size(), 3);
  ASSERT_EQ(edited_subgraph_ops.at(0).Code(), kLiteRtOpCodeTflCustom);
  ASSERT_EQ(edited_subgraph_ops.at(1).Code(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(edited_subgraph_ops.at(2).Code(), kLiteRtOpCodeTflCustom);

  {
    auto sliced_ops = sliced_graph_1.Ops();

    ASSERT_EQ(sliced_ops.size(), 1);
    ASSERT_EQ(sliced_ops.at(0).Code(), kLiteRtOpCodeTflAdd);
  }

  {
    auto sliced_ops = sliced_graph_2.Ops();

    ASSERT_EQ(sliced_ops.size(), 2);
    ASSERT_EQ(sliced_ops.at(0).Code(), kLiteRtOpCodeTflMul);
    ASSERT_EQ(sliced_ops.at(1).Code(), kLiteRtOpCodeTflAdd);
  }
}

}  // namespace
}  // namespace litert::internal
