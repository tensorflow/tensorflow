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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

// Makes a TfLiteIntArray* from std::vector, must free with TfLiteIntFree().
TfLiteIntArray* ConvertVector(const std::vector<int>& x) {
  TfLiteIntArray* lite = TfLiteIntArrayCreate(x.size());
  for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];
  return lite;
}

// A very simple test graph that supports setting in/out tensors on nodes.
class SimpleTestGraph : public GraphInfo {
 public:
  ~SimpleTestGraph() override {
    for (auto& node : nodes_) {
      TfLiteIntArrayFree(node.inputs);
      TfLiteIntArrayFree(node.outputs);
    }
  }

  size_t num_tensors() const override { return tensors_.size(); }
  size_t num_nodes() const override { return nodes_.size(); }
  const TfLiteNode& node(size_t index) const override { return nodes_[index]; }
  TfLiteTensor* tensor(size_t index) override { return &tensors_[index]; }
  const std::vector<int>& inputs() const override { return inputs_; }
  const std::vector<int>& outputs() const override { return outputs_; }
  const std::vector<int>& variables() const override { return variables_; }

  void AddNode(const std::vector<int>& inputs,
               const std::vector<int>& outputs) {
    nodes_.push_back(TfLiteNode());
    TfLiteNode& node = nodes_.back();
    node.inputs = ConvertVector(inputs);
    node.outputs = ConvertVector(outputs);
  }

  void AddTensors(int count) { tensors_.resize(count + tensors_.size()); }

  void SetInputsAndOutputs(const std::vector<int>& inputs,
                           const std::vector<int>& outputs) {
    inputs_ = inputs;
    outputs_ = outputs;
  }

 private:
  std::vector<TfLiteNode> nodes_;
  std::vector<TfLiteTensor> tensors_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> variables_;
};

// Partition a graph to generate a list of subgraphs. This wraps the API call
// we are testing and handles memory management and conversion to
// TfLiteIntArray. Populates `subgraphs` with resulting generated subgraphs.
void PartitionGraph(const SimpleTestGraph& graph,
                    const std::vector<int>& nodes_to_partition,
                    std::vector<NodeSubset>* subgraphs) {
  TfLiteIntArray* nodes_to_partition_int_array =
      ConvertVector(nodes_to_partition);
  PartitionGraphIntoIndependentNodeSubsets(&graph, nodes_to_partition_int_array,
                                           subgraphs);
  TfLiteIntArrayFree(nodes_to_partition_int_array);
}

// Check a generated list of subgraphs against the expected list of subgraphs.
void CheckPartitionSubgraphs(
    const std::vector<NodeSubset>& generated_subgraphs,
    const std::vector<NodeSubset>& expected_subgraphs) {
  ASSERT_EQ(generated_subgraphs.size(), expected_subgraphs.size());
  for (int subgraph_index = 0; subgraph_index < generated_subgraphs.size();
       subgraph_index++) {
    EXPECT_EQ(generated_subgraphs[subgraph_index].nodes,
              expected_subgraphs[subgraph_index].nodes);
    EXPECT_EQ(generated_subgraphs[subgraph_index].input_tensors,
              expected_subgraphs[subgraph_index].input_tensors);
    EXPECT_EQ(generated_subgraphs[subgraph_index].output_tensors,
              expected_subgraphs[subgraph_index].output_tensors);
  }
}

// Test an empty trivial graph with no partitions.
TEST(PartitionTest, Nodes0PartitionNodes0) {
  SimpleTestGraph graph;
  std::vector<int> nodes_to_partition = {};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);
  CheckPartitionSubgraphs(generated_subgraphs, {});
}

// Test a trivial graph with no node and only 1 tensor.
// The tensor is input & output of the graph at the same time.
// Note: This is a regression test to ensure the partitioning logic
// handles this case without crashing.
TEST(PartitionTest, Nodes0PartitionNodes0Tensors1) {
  SimpleTestGraph graph;
  graph.AddTensors(1);
  graph.SetInputsAndOutputs({0}, {0});
  std::vector<int> nodes_to_partition = {};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);
  CheckPartitionSubgraphs(generated_subgraphs, {});
}

// Test a 1 node graph with no partitions.
// Input: tensor(0) -> node(0) -> tensor(1), nodes_to_partition=[]
// Output: [kTfNoPartition, tensor(0) -> node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes0) {
  SimpleTestGraph graph;
  graph.AddTensors(2);
  graph.AddNode({0}, {1});
  graph.SetInputsAndOutputs({0}, {1});
  std::vector<int> nodes_to_partition = {};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph;
  expected_subgraph.type = NodeSubset::kTfNonPartition;
  expected_subgraph.nodes = {0};
  expected_subgraph.input_tensors = {0};
  expected_subgraph.output_tensors = {1};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph});
}

// Test a 1 node graph with no inputs that is fully partitioned.
// Input: node(0) -> tensor(1), nodes_to_partition=[node0]
// Output: [kTfPartition, node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes0Inputs0) {
  SimpleTestGraph graph;
  graph.AddTensors(1);
  graph.AddNode({}, {0});
  graph.SetInputsAndOutputs({}, {0});
  std::vector<NodeSubset> generated_subgraphs;
  std::vector<int> nodes_to_partition = {0};
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph;
  expected_subgraph.type = NodeSubset::kTfPartition;
  expected_subgraph.nodes = {0};
  expected_subgraph.input_tensors = {};
  expected_subgraph.output_tensors = {0};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph});
}

// Test a 1 node graph that is partitioned completely.
// Input: tensor(0) -> node(0) -> tensor(1), nodes_to_partition=[node0]
// Output: [kTfPartition, tensor(0) -> node(0) -> tensor(1)]
TEST(PartitionTest, Nodes1PartitionNodes1) {
  SimpleTestGraph graph;
  graph.AddTensors(2);
  graph.AddNode({0}, {1});
  graph.SetInputsAndOutputs({0}, {1});
  std::vector<int> nodes_to_partition = {0};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph;
  expected_subgraph.type = NodeSubset::kTfPartition;
  expected_subgraph.nodes = {0};
  expected_subgraph.input_tensors = {0};
  expected_subgraph.output_tensors = {1};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph});
}

// Test a 2 node graph where 1 node is partitioned and the other is not.
// Input: tensor(0) -> node(0) -> tensor(1) -> node(1) -> tensor(2),
//    nodes_to_partition = [1]
// Output: [kTfNonPartition, tensor(0) -> node(0) -> tensor(1),
//          kTfPartition, tensor(1) -> node(1), tensor(2)]
TEST(PartitionTest, Nodes2PartitionNodes1) {
  SimpleTestGraph graph;
  graph.AddTensors(3);
  graph.AddNode({0}, {1});
  graph.AddNode({1}, {2});
  graph.SetInputsAndOutputs({0}, {2});
  std::vector<int> nodes_to_partition = {1};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph0;
  expected_subgraph0.type = NodeSubset::kTfPartition;
  expected_subgraph0.nodes = {0};
  expected_subgraph0.input_tensors = {0};
  expected_subgraph0.output_tensors = {1};
  NodeSubset expected_subgraph1;
  expected_subgraph1.type = NodeSubset::kTfPartition;
  expected_subgraph1.nodes = {1};
  expected_subgraph1.input_tensors = {1};
  expected_subgraph1.output_tensors = {2};
  CheckPartitionSubgraphs(generated_subgraphs,
                          {expected_subgraph0, expected_subgraph1});
}

// Test a 2 node graph where both nodes are fully partitioned.
// Input: tensor(0) -> node(0) -> tensor(1) -> node(1) -> tensor(2),
//    nodes_to_partition = [0, 1]
// Output: [kTfPartition, tensor(0) -> node(0) -> node(1) -> tensor(1)]
TEST(PartitionTest, Nodes2PartitionNodes2) {
  SimpleTestGraph graph;
  graph.AddTensors(3);
  graph.AddNode({0}, {1});
  graph.AddNode({1}, {2});
  graph.SetInputsAndOutputs({0}, {2});
  std::vector<int> nodes_to_partition = {0, 1};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph0;
  expected_subgraph0.type = NodeSubset::kTfPartition;
  expected_subgraph0.nodes = {0, 1};
  expected_subgraph0.input_tensors = {0};
  expected_subgraph0.output_tensors = {2};
  CheckPartitionSubgraphs(generated_subgraphs, {expected_subgraph0});
}

// Test a three node model where we want to partition node 0 and node
// 2, but node 0 and node 2 cannot be in the same subgraph since node 2
// depends on node 1 which depends on node 0. Thus, we need to produce three
// subgraphs.
//
// Input: tensor(0) -> node(0) -> tensor(1)
//        tensor(1) -> node(1) -> tensor(2)
//        [tensor(2), tensor(1)] -> node(2) -> tensor(3)
//    nodes_to_partition = [0, 2]
// Output: [[kTfPartition, tensor(0) -> node(0) -> tensor(1),
//          [kTfNonPartition, tensor(1) -> node(1) -> tensor(2)],
//          [kTfPartition, [tensor(2), tensor(1)] -> node(2) -> node(3)]
TEST(PartitionTest, Nodes3PartitionNodes2) {
  SimpleTestGraph graph;
  graph.AddTensors(4);
  graph.AddNode({0}, {1});
  graph.AddNode({1}, {2});
  graph.AddNode({1, 2}, {3});
  graph.SetInputsAndOutputs({0}, {3});
  std::vector<int> nodes_to_partition = {0, 2};
  std::vector<NodeSubset> generated_subgraphs;
  PartitionGraph(graph, nodes_to_partition, &generated_subgraphs);

  NodeSubset expected_subgraph0;
  expected_subgraph0.type = NodeSubset::kTfPartition;
  expected_subgraph0.nodes = {0};
  expected_subgraph0.input_tensors = {0};
  expected_subgraph0.output_tensors = {1};
  NodeSubset expected_subgraph1;
  expected_subgraph1.type = NodeSubset::kTfNonPartition;
  expected_subgraph1.nodes = {1};
  expected_subgraph1.input_tensors = {1};
  expected_subgraph1.output_tensors = {2};
  NodeSubset expected_subgraph2;
  expected_subgraph2.type = NodeSubset::kTfPartition;
  expected_subgraph2.nodes = {2};
  expected_subgraph2.input_tensors = {1, 2};
  expected_subgraph2.output_tensors = {3};
  CheckPartitionSubgraphs(
      generated_subgraphs,
      {expected_subgraph0, expected_subgraph1, expected_subgraph2});
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
