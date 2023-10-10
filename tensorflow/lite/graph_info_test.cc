/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/graph_info.h"

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

using ::testing::Eq;
using ::testing::ExplainMatchResult;
using ::testing::Pointwise;

using NodeSubsets = std::vector<NodeSubset>;

// Makes a TfLiteIntArray* from std::vector, must free with TfLiteIntFree().
TfLiteIntArray* ConvertVector(const std::vector<int>& x) {
  TfLiteIntArray* lite = TfLiteIntArrayCreate(x.size());
  for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];
  return lite;
}

// A very simple test graph that supports setting in/out tensors on nodes.
class SimpleTestGraph : public GraphInfo {
 public:
  SimpleTestGraph(
      const std::vector<int>& inputs, const std::vector<int>& outputs,
      const std::vector<std::tuple<std::vector<int>, std::vector<int>, bool>>&
          nodes,
      int node_index_offset = 0)
      : inputs_(inputs),
        outputs_(outputs),
        node_index_offset_(node_index_offset) {
    NeedsTensors(inputs_);
    NeedsTensors(outputs_);
    for (int i = 0; i < node_index_offset; ++i) AddNode({}, {}, false);
    for (const auto& [inputs, outputs, might_have_side_effect] : nodes) {
      AddNode(inputs, outputs, might_have_side_effect);
    }
    registrations_.resize(nodes.size());
  }

  ~SimpleTestGraph() override {
    for (auto& node : nodes_) {
      TfLiteIntArrayFree(node.inputs);
      TfLiteIntArrayFree(node.outputs);
    }
  }

  size_t num_total_nodes() const override { return nodes_.size(); }
  size_t num_execution_nodes() const override {
    return nodes_.size() - node_index_offset_;
  }
  const TfLiteNode& node(size_t index) const override {
    return nodes_[index + node_index_offset_];
  }
  size_t node_index(size_t index) const override {
    return index + node_index_offset_;
  }
  size_t num_tensors() const override { return tensors_.size(); }
  const TfLiteRegistration& registration(size_t index) const override {
    return registrations_[index + node_index_offset_];
  }
  TfLiteTensor* tensor(size_t index) override { return &tensors_[index]; }
  TfLiteTensor* tensors() override { return tensors_.data(); }
  const std::vector<int>& inputs() const override { return inputs_; }
  const std::vector<int>& outputs() const override { return outputs_; }
  const std::vector<int>& variables() const override { return variables_; }

 private:
  void AddNode(const std::vector<int>& inputs, const std::vector<int>& outputs,
               bool might_have_side_effect) {
    NeedsTensors(inputs);
    NeedsTensors(outputs);
    nodes_.push_back(TfLiteNode());
    TfLiteNode& node = nodes_.back();
    node.inputs = ConvertVector(inputs);
    node.outputs = ConvertVector(outputs);
    node.might_have_side_effect = might_have_side_effect;
  }

  void NeedsTensors(const std::vector<int>& tensors) {
    for (const int tensor : tensors)
      tensors_.resize(std::max<int>(tensor + 1, tensors_.size()));
  }

  std::vector<TfLiteNode> nodes_;
  std::vector<TfLiteTensor> tensors_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> variables_;
  std::vector<TfLiteRegistration> registrations_;
  size_t node_index_offset_;
};

// Partition a graph to generate a list of subgraphs. This wraps the API call
// we are testing and handles memory management and conversion to
// TfLiteIntArray. Populates `subgraphs` with the resulting generated
// subgraphs.
void PartitionGraphOrDie(const SimpleTestGraph& graph,
                         const std::vector<int>& nodes_to_partition,
                         NodeSubsets* subgraphs, const bool greedily,
                         const ControlEdges* control_edges) {
  TfLiteIntArray* nodes_to_partition_int_array =
      ConvertVector(nodes_to_partition);
  ASSERT_EQ(PartitionGraphIntoIndependentNodeSubsets(
                &graph, nodes_to_partition_int_array, subgraphs, greedily,
                control_edges),
            kTfLiteOk);
  TfLiteIntArrayFree(nodes_to_partition_int_array);
}

NodeSubsets PartitionGraph(const SimpleTestGraph& graph,
                           const std::vector<int>& nodes_to_partition,
                           const bool greedily = true,
                           const ControlEdges* control_edges = nullptr) {
  NodeSubsets subgraphs;
  PartitionGraphOrDie(graph, nodes_to_partition, &subgraphs, greedily,
                      control_edges);
  return subgraphs;
}

MATCHER(EqNodeSubset, "") {
  const NodeSubset& a = std::get<0>(arg);
  const NodeSubset& b = std::get<1>(arg);
  if (a.type != b.type) {
    *result_listener << "mismatched .type ";
    return ExplainMatchResult(Eq(b.type), a.type, result_listener);
  }
  if (a.nodes != b.nodes) {
    *result_listener << "mismatched .nodes ";
    return ExplainMatchResult(Pointwise(Eq(), b.nodes), a.nodes,
                              result_listener);
  }
  if (a.input_tensors != b.input_tensors) {
    *result_listener << "mismatched .input_tensors ";
    return ExplainMatchResult(Pointwise(Eq(), b.input_tensors), a.input_tensors,
                              result_listener);
  }
  if (a.output_tensors != b.output_tensors) {
    *result_listener << "mismatched .output_tensors ";
    return ExplainMatchResult(Pointwise(Eq(), b.output_tensors),
                              a.output_tensors, result_listener);
  }
  return true;
}

// Test an empty trivial graph with no partitions.
TEST(PartitionTest, Nodes0PartitionNodes0) {
  EXPECT_THAT(PartitionGraph({
                                 /*inputs=*/{},
                                 /*outputs=*/{},
                                 /*nodes=*/{},
                             },
                             /*nodes_to_partition=*/{}),
              Pointwise(EqNodeSubset(), NodeSubsets({})));
}

// Test a trivial graph with no node and only 1 tensor.
// The tensor is input & output of the graph at the same time.
// Note: This is a regression test to ensure the partitioning logic
// handles this case without crashing.
TEST(PartitionTest, Nodes0PartitionNodes0Tensors1) {
  EXPECT_THAT(PartitionGraph({
                                 /*inputs=*/{0},
                                 /*outputs=*/{0},
                                 /*nodes=*/{},
                             },
                             /*nodes_to_partition=*/{}),
              Pointwise(EqNodeSubset(), NodeSubsets({})));
}

// Test a 1-node graph with no partitions.
// Input: tensor(0) -> node(0) -> tensor(1), nodes_to_partition=[]
// Output: [kTfNoPartition, tensor(0) -> node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes0) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{1},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                         },
                     },
                     /*nodes_to_partition=*/{}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{0},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1},
                                    },
                                })));
}

TEST(PartitionTest, Nodes1PartitionNodes0_WithOffset) {
  constexpr int node_index_offset = 17;
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{1},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                         },
                         node_index_offset,
                     },
                     /*nodes_to_partition_=*/{}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{node_index_offset},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1},
                                    },
                                })));
}

// Test a 1-node graph with no inputs that is fully partitioned.
// Input: node(0) -> tensor(1), nodes_to_partition=[node0]
// Output: [kTfPartition, node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes0Inputs0) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{},
                         /*outputs=*/{0},
                         /*nodes=*/
                         {
                             {{}, {0}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0},
                                        /*input_tensors=*/{},
                                        /*output_tensors=*/{0},
                                    },
                                })));
}

// Test a 1-node graph that is partitioned completely.
// Input: tensor(0) -> node(0) -> tensor(1), nodes_to_partition=[node0]
// Output: [kTfPartition, tensor(0) -> node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes1) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{1},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1},
                                    },
                                })));
}

// Test a 2-node graph where node 1 is partitioned and node 0 is not.
// Input: tensor(0) -> node(0) -> tensor(1) -> node(1) -> tensor(2),
//    nodes_to_partition = [1]
// Output: [kTfNonPartition, tensor(0) -> node(0) -> tensor(1),
//          kTfPartition, tensor(1) -> node(1)-> tensor(2)]
TEST(PartitionTest, Nodes2PartitionNodes1) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{2},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                             {{1}, {2}, false},
                         },
                     },
                     /*nodes_to_partition=*/{1}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{0},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{1},
                                        /*input_tensors=*/{1},
                                        /*output_tensors=*/{2},
                                    },
                                })));
}

// Same as above, but with node offset to ensure correct handling of original vs
// execution plan indices.
TEST(PartitionTest, Nodes2PartitionNodes1_WithOffset) {
  constexpr int node_index_offset = 17;
  EXPECT_THAT(
      PartitionGraph({/*inputs=*/{0},
                      /*outputs=*/{2},
                      /*nodes=*/
                      {
                          {{0}, {1}, false},
                          {{1}, {2}, false},
                      },
                      node_index_offset},
                     /*nodes_to_partition=*/{node_index_offset + 1}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{node_index_offset + 0},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{node_index_offset + 1},
                                        /*input_tensors=*/{1},
                                        /*output_tensors=*/{2},
                                    },
                                })));
}

// Test a 2-node graph where both nodes are fully partitioned.
// Input: tensor(0) -> node(0) -> tensor(1) -> node(1) -> tensor(2),
//    nodes_to_partition = [0, 1]
// Output: [kTfPartition, tensor(0) -> node(0) -> node(1) -> tensor(1)]
TEST(PartitionTest, Nodes2PartitionNodes2) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{2},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                             {{1}, {2}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0, 1}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0, 1},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{2},
                                    },
                                })));
}

// Test a 3-node model where we want to partition node 0 and node
// 2, but node 0 and node 2 cannot be in the same subgraph since node 2
// depends on node 1 which depends on node 0. Thus, we need to produce three
// subgraphs.
//
// Input: tensor(0) -> node(0) -> tensor(1)
//        tensor(1) -> node(1) -> tensor(2)
//        [tensor(1), tensor(2)] -> node(2) -> tensor(3)
//    nodes_to_partition = [0, 2]
// Output: [[kTfPartition, tensor(0) -> node(0) -> tensor(1),
//          [kTfNonPartition, tensor(1) -> node(1) -> tensor(2)],
//          [kTfPartition, [tensor(2), tensor(1)] -> node(2) -> tensor(3)]
TEST(PartitionTest, Nodes3PartitionNodes2) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{3},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                             {{1}, {2}, false},
                             {{1, 2}, {3}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0, 2}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{1},
                                        /*input_tensors=*/{1},
                                        /*output_tensors=*/{2},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{2},
                                        /*input_tensors=*/{1, 2},
                                        /*output_tensors=*/{3},
                                    },
                                })));
}

// Test a 3-node model where we want to partition node 0 and node
// 2, and node 0 and node 2 can be in the same subgraph since node 2
// depends only on node 0.
//
// Input: tensor(0) -> node(0) -> tensor(1)
//        tensor(1) -> node(1) -> tensor(2)
//        [tensor(1), tensor(0)] -> node(2) -> tensor(3)
//    nodes_to_partition = [0, 2]
// Output: [[kTfPartition, tensor(0) -> node(0) -> tensor(1),
//          [kTfNonPartition, tensor(1) -> node(1) -> tensor(2)],
//          [kTfPartition, [tensor(0), tensor(1)] -> node(2) -> tensor(3)]
TEST(PartitionTest, Nodes3PartitionNodes2Greedily) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{2, 3},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                             {{1}, {2}, false},
                             {{1}, {3}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0, 2}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0, 2},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1, 3},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{1},
                                        /*input_tensors=*/{1},
                                        /*output_tensors=*/{2},
                                    },
                                })));
}

// Same case as above, but partitioning non-greedily. This time,
// nodes 0 and 2 can't be in the same subgraph because they aren't
// immediate successors in the original schedule. Thus, we
// need to produce three subgraphs.
//
// Input: tensor(0) -> node(0) -> tensor(1)
//        tensor(1) -> node(1) -> tensor(2)
//        [tensor(0), tensor(1)] -> node(2) -> tensor(3)
//    nodes_to_partition = [0, 2]
// Output: [[kTfPartition, tensor(0) -> node(0) -> tensor(1),
//          [kTfNonPartition, tensor(1) -> node(1) -> tensor(2)],
//          [kTfPartition, [tensor(0), tensor(1)] -> node(2) -> tensor(3)]
TEST(PartitionTest, Nodes3PartitionNodes2ClusteredNonGreedily) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{2, 3},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                             {{1}, {2}, false},
                             {{1}, {3}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0, 2},
                     /*greedily=*/false),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{1},
                                        /*input_tensors=*/{1},
                                        /*output_tensors=*/{2},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{2},
                                        /*input_tensors=*/{1},
                                        /*output_tensors=*/{3},
                                    },
                                })));
}

// Test correct partition for graph with control dependency.
// Graph for test is like
// varhandleOp -> ReadVariableOp -> Add -> AssignVariableOp
//             |_________________________^    ^^
//             |------------------------->ReadVariableOp -> (Output)
// ^^ is control dependency, in this case we don't want to invoke the
// last ReadVariableOp before AssignVariableOp finishes executing.
// '>' and '^' represents data dependency.
TEST(PartitionTest, Nodes4PartitionNodes3_WithControlDependency) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{4},
                         /*nodes=*/
                         {
                             {{0}, {1}, /*might_have_side_effect=*/true},
                             {{1}, {2}, /*might_have_side_effect=*/true},
                             {{2}, {3}, false},
                             {{1, 3}, {}, /*might_have_side_effect=*/true},
                             {{1}, {4}, /*might_have_side_effect=*/true},
                         },
                     },
                     /*nodes_to_partition=*/{0, 1, 3, 4}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0, 1},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1, 2},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{2},
                                        /*input_tensors=*/{2},
                                        /*output_tensors=*/{3},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{3, 4},
                                        /*input_tensors=*/{1, 3},
                                        /*output_tensors=*/{4},
                                    },
                                })));
}

// The same as above, but the control dependency is given by an external edge
// set.
TEST(PartitionTest, Nodes4PartitionNodes3_WithExternalControlDependency) {
  // Nodes {0,1,3,4} are stateful.
  const ControlEdges control_edges = {
      {0, 1},
      {1, 3},
      {3, 4},
  };
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{4},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                             {{1}, {2}, false},
                             {{2}, {3}, false},
                             {{1, 3}, {}, false},
                             {{1}, {4}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0, 1, 3, 4},
                     /*greedily=*/true, &control_edges),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0, 1},
                                        /*input_tensors=*/{0},
                                        /*output_tensors=*/{1, 2},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{2},
                                        /*input_tensors=*/{2},
                                        /*output_tensors=*/{3},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{3, 4},
                                        /*input_tensors=*/{1, 3},
                                        /*output_tensors=*/{4},
                                    },
                                })));
}

//                                         ________________
// A more complex case: [tensor], (node), (partitioned node)
//        _           _           _           _
// [0]-->(0)-->[1]-->(1)-->[5]-->(4)-->[6]-->(5)-->[7]
//                     \
//                      \
//                       \>[2]-->(2)-->[3]-->(3)-->[4]
//
// Greedy partitioning;
//         ____
//  [0]-->(0145)-->[7]
//            \
//             \-->[2]-->(23)-->[4]
//
TEST(PartitionTest, ComplexGreedily) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{4, 7},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                             {{1}, {2, 5}, false},
                             {{2}, {3}, false},
                             {{3}, {4}, false},
                             {{5}, {6}, false},
                             {{6}, {7}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0, 1, 4, 5}),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0, 1, 4, 5},
                                        /*inputs=*/{0},
                                        /*outputs=*/{2, 7},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{2, 3},
                                        /*inputs=*/{2},
                                        /*outputs=*/{4},
                                    },
                                })));
}

// Same, non-greedy partitioning:
//         __           __
//  [0]-->(01)-->[5]-->(45)-->[7]
//          \
//           \-->[2]-->(23)-->[4]
//
TEST(PartitionTest, ComplexNonGreedily) {
  EXPECT_THAT(
      PartitionGraph({
                         /*inputs=*/{0},
                         /*outputs=*/{4, 7},
                         /*nodes=*/
                         {
                             {{0}, {1}, false},
                             {{1}, {2, 5}, false},
                             {{2}, {3}, false},
                             {{3}, {4}, false},
                             {{5}, {6}, false},
                             {{6}, {7}, false},
                         },
                     },
                     /*nodes_to_partition=*/{0, 1, 4, 5},
                     /*greedily=*/false),
      Pointwise(EqNodeSubset(), NodeSubsets({
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{0, 1},
                                        /*inputs=*/{0},
                                        /*outputs=*/{2, 5},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfNonPartition,
                                        /*nodes=*/{2, 3},
                                        /*inputs=*/{2},
                                        /*outputs=*/{4},
                                    },
                                    {
                                        /*type=*/NodeSubset::kTfPartition,
                                        /*nodes=*/{4, 5},
                                        /*inputs=*/{5},
                                        /*outputs=*/{7},
                                    },
                                })));
}

}  // namespace
}  // namespace tflite
