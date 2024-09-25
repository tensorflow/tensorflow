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

#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/algo.h"

#include <memory>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/test_data/test_data_util.h"

namespace {

using ::algo::DisjointSets;
using ::algo::GraphSlicer;

// NOLINTBEGIN
bool HasValidGeneralTopology(LrtSubgraph subgraph) {
  if (!::graph_tools::ValidateTopology(subgraph->ops)) {
    _LRT_D_MSG("Failed validate op tolopology");
    return false;
  }

  std::unordered_set<LrtTensor> implied_subgraph_outs;
  for (auto tensor : subgraph->tensors) {
    if (tensor->users.empty()) {
      implied_subgraph_outs.insert(tensor);
    }
  }

  if (implied_subgraph_outs.size() != subgraph->outputs.size()) {
    _LRT_D_MSG("Outs not same size");
    return false;
  }

  for (auto tensor : subgraph->outputs) {
    if (implied_subgraph_outs.find(tensor) == implied_subgraph_outs.end()) {
      _LRT_D_MSG("Mismatched subgraph outs");
      return false;
    }
  }

  std::unordered_set<LrtTensor> implied_subgraph_ins;
  for (auto tensor : subgraph->tensors) {
    if (tensor->defining_op == nullptr &&
        tensor->buffer.fb_buffer->data.empty()) {
      implied_subgraph_ins.insert(tensor);
    }
  }

  if (implied_subgraph_ins.size() != subgraph->inputs.size()) {
    _LRT_D_MSG("Ins not same size");
    return false;
  }

  for (auto tensor : subgraph->inputs) {
    if (implied_subgraph_ins.find(tensor) == implied_subgraph_ins.end()) {
      _LRT_D_MSG("Mismatched subgraph ins");
      return false;
    }
  }

  return true;
}
// NOLINTEND

TEST(TestPartitionsFromFlatList, SimpleMultiOp) {
  auto model = LoadTestFileModel("simple_multi_op.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph, graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  {
    std::vector<LrtOp> partition;
    partition.push_back(ops[1]);
    partition.push_back(ops[2]);

    auto partitions = DisjointSets::GetPartitionsFromFlatList(partition);
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions.front().size(), 2);

    EXPECT_EQ(partitions.front().at(0), partition.at(0));
    EXPECT_EQ(partitions.front().at(1), partition.at(1));
  }

  {
    std::vector<LrtOp> partition;
    partition.push_back(ops[1]);
    partition.push_back(ops[3]);

    auto partitions = DisjointSets::GetPartitionsFromFlatList(partition);
    ASSERT_EQ(partitions.size(), 2);
    ASSERT_EQ(partitions.front().size(), 1);
    ASSERT_EQ(partitions.back().size(), 1);

    auto p1_op_code = partitions.front().front()->op_code;
    auto p2_op_code = partitions.back().front()->op_code;

    ASSERT_TRUE(
        (p1_op_code == kLrtOpCodeTflMul && p2_op_code == kLrtOpCodeTflAdd) ||
        (p1_op_code == kLrtOpCodeTflAdd && p2_op_code == kLrtOpCodeTflMul));
  }

  {
    std::vector<LrtOp> partition;

    auto partitions = DisjointSets::GetPartitionsFromFlatList(partition);
    ASSERT_EQ(partitions.size(), 0);
  }

  {
    std::vector<LrtOp> partition;
    partition.push_back(ops[0]);
    partition.push_back(ops[1]);
    partition.push_back(ops[2]);
    partition.push_back(ops[3]);

    auto partitions = DisjointSets::GetPartitionsFromFlatList(partition);
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions.front().size(), 4);

    EXPECT_EQ(partitions.front().at(0), partition.at(0));
    EXPECT_EQ(partitions.front().at(1), partition.at(1));
    EXPECT_EQ(partitions.front().at(2), partition.at(2));
    EXPECT_EQ(partitions.front().at(3), partition.at(3));
  }
}

TEST(TestSliceSubgraphSimpleMultiOp, OnePartition) {
  auto model = LoadTestFileModel("simple_multi_op.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph, graph_tools::GetSubgraph(model.get()));

  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  std::vector<LrtOp> partition;
  partition.push_back(ops[1]);
  partition.push_back(ops[2]);

  LrtSubgraph sliced_graph = &model->subgraphs.emplace_back();
  auto* hal_cal_op =
      GraphSlicer::SlicePartitionFromGraph(*subgraph, sliced_graph, partition);

  ASSERT_TRUE(HasValidGeneralTopology(sliced_graph));
  ASSERT_TRUE(HasValidGeneralTopology(subgraph));

  ASSERT_RESULT_OK_ASSIGN(auto edited_subgraph_ops,
                          graph_tools::GetSubgraphOps(subgraph));

  ASSERT_EQ(edited_subgraph_ops.size(), 3);
  ASSERT_EQ(edited_subgraph_ops[0]->op_code, kLrtOpCodeTflAdd);
  ASSERT_EQ(edited_subgraph_ops[1]->op_code, kLrtOpCodeTflCustom);
  ASSERT_EQ(edited_subgraph_ops[2]->op_code, kLrtOpCodeTflAdd);

  ASSERT_RESULT_OK_ASSIGN(auto sliced_subgraph_ops,
                          graph_tools::GetSubgraphOps(sliced_graph));

  ASSERT_EQ(sliced_subgraph_ops.size(), 2);
  ASSERT_EQ(sliced_subgraph_ops[0]->op_code, kLrtOpCodeTflMul);
  ASSERT_EQ(sliced_subgraph_ops[1]->op_code, kLrtOpCodeTflMul);

  ASSERT_EQ(hal_cal_op, edited_subgraph_ops[1]);

  {
    ASSERT_RESULT_OK_ASSIGN(auto hal_cal_op_ins,
                            graph_tools::GetOpIns(hal_cal_op));

    ASSERT_EQ(hal_cal_op_ins.size(), 1);

    ASSERT_TRUE(graph_tools::MatchTensorDefiningOp(hal_cal_op_ins[0], 0,
                                                   edited_subgraph_ops[0]));

    ASSERT_RESULT_OK_ASSIGN(auto sliced_subgraph_inputs,
                            graph_tools::GetSubgraphInputs(sliced_graph));

    ASSERT_EQ(sliced_subgraph_inputs.size(), 1);

    ASSERT_TRUE(graph_tools::MatchTensorHasUses(
        sliced_subgraph_inputs[0],
        {{sliced_subgraph_ops[0], 0}, {sliced_subgraph_ops[0], 1}}));

    ASSERT_TRUE(
        graph_tools::MatchTensorNoDefiningOp(sliced_subgraph_inputs[0]));
  }

  {
    ASSERT_RESULT_OK_ASSIGN(auto hal_cal_op_out,
                            graph_tools::GetOnlyOpOut(hal_cal_op));

    ASSERT_TRUE(graph_tools::MatchTensorHasUses(
        hal_cal_op_out,
        {{edited_subgraph_ops.back(), 0}, {edited_subgraph_ops.back(), 1}}));

    ASSERT_RESULT_OK_ASSIGN(auto sliced_subgraph_outputs,
                            graph_tools::GetSubgraphOutputs(sliced_graph));

    ASSERT_EQ(sliced_subgraph_outputs.size(), 1);
    ASSERT_TRUE(graph_tools::MatchTensorDefiningOp(
        sliced_subgraph_outputs[0], 0, sliced_subgraph_ops.back()));
    ASSERT_TRUE(graph_tools::MatchkTensorNoUses(sliced_subgraph_outputs[0]));
  }
}

TEST(TestSliceSubgraphSimpleMultiOp, TwoPartitions) {
  auto model = LoadTestFileModel("simple_multi_op.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph, graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  std::vector<LrtOp> partition_1;
  partition_1.push_back(ops[0]);

  LrtSubgraph sliced_graph_1 = &model->subgraphs.emplace_back();
  GraphSlicer::SlicePartitionFromGraph(*subgraph, sliced_graph_1, partition_1);

  ASSERT_TRUE(HasValidGeneralTopology(sliced_graph_1));
  ASSERT_TRUE(HasValidGeneralTopology(subgraph));

  std::vector<LrtOp> partition_2;
  partition_2.push_back(ops[2]);
  partition_2.push_back(ops[3]);

  LrtSubgraph sliced_graph_2 = &model->subgraphs.emplace_back();
  GraphSlicer::SlicePartitionFromGraph(*subgraph, sliced_graph_2, partition_2);

  ASSERT_TRUE(HasValidGeneralTopology(sliced_graph_2));
  ASSERT_TRUE(HasValidGeneralTopology(subgraph));

  ASSERT_RESULT_OK_ASSIGN(auto edited_subgraph_ops,
                          graph_tools::GetSubgraphOps(subgraph));

  ASSERT_EQ(edited_subgraph_ops.size(), 3);
  ASSERT_EQ(edited_subgraph_ops[0]->op_code, kLrtOpCodeTflCustom);
  ASSERT_EQ(edited_subgraph_ops[1]->op_code, kLrtOpCodeTflMul);
  ASSERT_EQ(edited_subgraph_ops[2]->op_code, kLrtOpCodeTflCustom);

  {
    ASSERT_RESULT_OK_ASSIGN(auto sliced_ops,
                            graph_tools::GetSubgraphOps(sliced_graph_1));

    ASSERT_EQ(sliced_ops.size(), 1);
    ASSERT_EQ(sliced_ops[0]->op_code, kLrtOpCodeTflAdd);
  }

  {
    ASSERT_RESULT_OK_ASSIGN(auto sliced_ops,
                            graph_tools::GetSubgraphOps(sliced_graph_2));

    ASSERT_EQ(sliced_ops.size(), 2);
    ASSERT_EQ(sliced_ops[0]->op_code, kLrtOpCodeTflMul);
    ASSERT_EQ(sliced_ops[1]->op_code, kLrtOpCodeTflAdd);
  }
}

}  // namespace
