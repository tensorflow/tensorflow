/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/model.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"

namespace tflite {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(Model, SingleNode) {
  // graph_input -> node -> graph_output
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node->id, graph_output->id).ok());

  EXPECT_THAT(graph.nodes(), ElementsAre(node));
  EXPECT_THAT(graph.values(), UnorderedElementsAre(graph_input, graph_output));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindInputs(node->id), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.FindOutputs(node->id), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindConsumers(graph_input->id), UnorderedElementsAre(node));
  EXPECT_THAT(graph.FindProducer(graph_output->id), ::testing::Eq(node));
  EXPECT_THAT(graph.FindConsumers(graph_output->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindProducer(graph_input->id), ::testing::Eq(nullptr));
}

TEST(Model, SingleNodeMultipleOutputs) {
  // graph_input -> node -> (graph_output1, graph_output2)
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output1 = graph.NewValue();
  Value* graph_output2 = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node->id, graph_output1->id).ok());
  ASSERT_TRUE(graph.SetProducer(node->id, graph_output2->id).ok());
  EXPECT_THAT(graph.FindOutputs(node->id),
              UnorderedElementsAre(graph_output1, graph_output2));
  EXPECT_THAT(graph.FindProducer(graph_output1->id), ::testing::Eq(node));
  EXPECT_THAT(graph.FindProducer(graph_output2->id), ::testing::Eq(node));
}

TEST(Model, SetSameConsumer) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node->id, graph_input->id).ok());
  EXPECT_FALSE(graph.AddConsumer(node->id, graph_input->id).ok());
}

TEST(Model, RemoveConsumer) {
  // (graph_input1, graph_input2) -> node
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_input1 = graph.NewValue();
  Value* graph_input2 = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node->id, graph_input1->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node->id, graph_input2->id).ok());
  EXPECT_THAT(graph.FindConsumers(graph_input1->id),
              UnorderedElementsAre(node));
  EXPECT_THAT(graph.FindConsumers(graph_input2->id),
              UnorderedElementsAre(node));
  EXPECT_THAT(graph.FindInputs(node->id),
              UnorderedElementsAre(graph_input1, graph_input2));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre());

  // Now remove graph_input1
  ASSERT_TRUE(graph.RemoveConsumer(node->id, graph_input1->id).ok());
  EXPECT_THAT(graph.FindConsumers(graph_input1->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindInputs(node->id), UnorderedElementsAre(graph_input2));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_input1));

  // Can not remove it twice
  ASSERT_FALSE(graph.RemoveConsumer(node->id, graph_input1->id).ok());
}

TEST(Model, SetSameProducer) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_output = graph.NewValue();
  ASSERT_TRUE(graph.SetProducer(node->id, graph_output->id).ok());
  EXPECT_FALSE(graph.SetProducer(node->id, graph_output->id).ok());
}

TEST(Model, ReplaceInput) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* v0 = graph.NewValue();
  Value* v1 = graph.NewValue();
  Value* v2 = graph.NewValue();
  Value* v3 = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node->id, v0->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node->id, v1->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node->id, v2->id).ok());
  EXPECT_THAT(graph.FindInputs(node->id), ElementsAre(v0, v1, v2));
  ASSERT_TRUE(graph.ReplaceInput(node->id, v1->id, v3->id).ok());
  EXPECT_THAT(graph.FindInputs(node->id), ElementsAre(v0, v3, v2));
}

TEST(Model, RemoveProducer) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value* graph_output = graph.NewValue();

  ASSERT_TRUE(graph.SetProducer(node->id, graph_output->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre());
  EXPECT_THAT(graph.FindProducer(graph_output->id), ::testing::Eq(node));

  ASSERT_TRUE(graph.RemoveProducer(graph_output->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindProducer(graph_output->id), ::testing::Eq(nullptr));

  // Can not remove producer twice
  ASSERT_FALSE(graph.RemoveProducer(graph_output->id).ok());
}

class OneNodeModel : public testing::Test {
 protected:
  void SetUp() override {
    node_ = graph_.NewNode();
    Value* graph_input = graph_.NewValue();
    Value* graph_output = graph_.NewValue();
    ASSERT_TRUE(graph_.AddConsumer(node_->id, graph_input->id).ok());
    ASSERT_TRUE(graph_.SetProducer(node_->id, graph_output->id).ok());
    EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input));
    EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output));
    EXPECT_THAT(graph_.nodes(), ElementsAre(node_));
  }
  GraphFloat32 graph_;
  Node* node_;
};

TEST_F(OneNodeModel, DeleteNodeKeepInput) {
  ASSERT_TRUE(RemoveSimpleNodeKeepInput(&graph_, node_).ok());
  EXPECT_TRUE(graph_.inputs().empty());
  EXPECT_TRUE(graph_.outputs().empty());
  EXPECT_TRUE(graph_.nodes().empty());
}

TEST_F(OneNodeModel, DeleteNodeKeepOutput) {
  ASSERT_TRUE(RemoveSimpleNodeKeepOutput(&graph_, node_).ok());
  EXPECT_TRUE(graph_.inputs().empty());
  EXPECT_TRUE(graph_.outputs().empty());
  EXPECT_TRUE(graph_.nodes().empty());
}

class TwoNodesModel : public testing::Test {
 protected:
  void SetUp() override {
    graph_input_ = graph_.NewValue();
    first_node_ = graph_.NewNode();
    value_ = graph_.NewValue();
    second_node_ = graph_.NewNode();
    graph_output_ = graph_.NewValue();

    ASSERT_TRUE(graph_.AddConsumer(first_node_->id, graph_input_->id).ok());
    ASSERT_TRUE(graph_.SetProducer(first_node_->id, value_->id).ok());
    ASSERT_TRUE(graph_.AddConsumer(second_node_->id, value_->id).ok());
    ASSERT_TRUE(graph_.SetProducer(second_node_->id, graph_output_->id).ok());
    EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
    EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
    EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_, second_node_));
  }
  GraphFloat32 graph_;
  Node* first_node_;
  Node* second_node_;
  Value* graph_input_;
  Value* value_;
  Value* graph_output_;
};

TEST_F(TwoNodesModel, DeleteFirstNodeKeepInput) {
  ASSERT_TRUE(RemoveSimpleNodeKeepInput(&graph_, first_node_).ok());
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(second_node_));
}

TEST_F(TwoNodesModel, DeleteFirstNodeKeepOutput) {
  ASSERT_TRUE(RemoveSimpleNodeKeepOutput(&graph_, first_node_).ok());
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(value_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(second_node_));
}

TEST_F(TwoNodesModel, DeleteSecondNodeKeepInput) {
  ASSERT_TRUE(RemoveSimpleNodeKeepInput(&graph_, second_node_).ok());
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(value_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_));
}

TEST_F(TwoNodesModel, DeleteSecondNodeKeepOutput) {
  ASSERT_TRUE(RemoveSimpleNodeKeepOutput(&graph_, second_node_).ok());
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_));
}

class ThreeNodesModel : public testing::Test {
 protected:
  void SetUp() override {
    first_node_ = graph_.NewNode();
    second_node_ = graph_.NewNode();
    third_node_ = graph_.NewNode();
    graph_input_ = graph_.NewValue();
    value0_ = graph_.NewValue();
    value1_ = graph_.NewValue();
    graph_output_ = graph_.NewValue();

    ASSERT_TRUE(graph_.AddConsumer(first_node_->id, graph_input_->id).ok());
    ASSERT_TRUE(graph_.SetProducer(first_node_->id, value0_->id).ok());
    ASSERT_TRUE(graph_.AddConsumer(second_node_->id, value0_->id).ok());
    ASSERT_TRUE(graph_.SetProducer(second_node_->id, value1_->id).ok());
    ASSERT_TRUE(graph_.AddConsumer(third_node_->id, value1_->id).ok());
    ASSERT_TRUE(graph_.SetProducer(third_node_->id, graph_output_->id).ok());
    EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
    EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
    EXPECT_THAT(graph_.nodes(),
                ElementsAre(first_node_, second_node_, third_node_));
  }
  GraphFloat32 graph_;
  Node* first_node_;
  Node* second_node_;
  Node* third_node_;
  Value* graph_input_;
  Value* value0_;
  Value* value1_;
  Value* graph_output_;
};

TEST_F(ThreeNodesModel, DeleteMiddleNodeKeepInput) {
  ASSERT_TRUE(RemoveSimpleNodeKeepInput(&graph_, second_node_).ok());
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_, third_node_));
  EXPECT_THAT(graph_.values(),
              UnorderedElementsAre(graph_input_, value0_, graph_output_));
}

TEST_F(ThreeNodesModel, DeleteMiddleNodeKeepOutput) {
  ASSERT_TRUE(RemoveSimpleNodeKeepOutput(&graph_, second_node_).ok());
  EXPECT_THAT(graph_.inputs(), UnorderedElementsAre(graph_input_));
  EXPECT_THAT(graph_.outputs(), UnorderedElementsAre(graph_output_));
  EXPECT_THAT(graph_.nodes(), ElementsAre(first_node_, third_node_));
  EXPECT_THAT(graph_.values(),
              UnorderedElementsAre(graph_input_, value1_, graph_output_));
}

TEST(Model, RemoveSimpleNodeKeepInputComplexCase) {
  // We have this graph and we are going to delete n1 and preserve order of
  // v0, v1 for n0 node and v2, v3 for n2 node
  //  v0   v1
  //   \  /  \
  //    n0    n1
  //    |      \
  //    o1      v2   v3
  //             \  /
  //              n2
  //              |
  //              o2
  //
  // And we are going to receive this:
  //  v0   v1
  //   \  /  \
  //    n0    \
  //    |      \
  //    o1      \   v3
  //             \  /
  //              n2
  //              |
  //              o2
  GraphFloat32 graph;
  Node* n0 = graph.NewNode();
  Node* n1 = graph.NewNode();  // node to remove
  Node* n2 = graph.NewNode();
  Value* v0 = graph.NewValue();
  Value* v1 = graph.NewValue();
  Value* v2 = graph.NewValue();  // value to be removed
  Value* v3 = graph.NewValue();
  Value* o1 = graph.NewValue();
  Value* o2 = graph.NewValue();

  ASSERT_TRUE(graph.AddConsumer(n0->id, v0->id).ok());
  ASSERT_TRUE(graph.AddConsumer(n0->id, v1->id).ok());
  ASSERT_TRUE(graph.SetProducer(n0->id, o1->id).ok());
  ASSERT_TRUE(graph.AddConsumer(n1->id, v1->id).ok());
  ASSERT_TRUE(graph.SetProducer(n1->id, v2->id).ok());
  ASSERT_TRUE(graph.AddConsumer(n2->id, v2->id).ok());
  ASSERT_TRUE(graph.AddConsumer(n2->id, v3->id).ok());
  ASSERT_TRUE(graph.SetProducer(n2->id, o2->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(v0, v1, v3));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(o1, o2));
  EXPECT_THAT(graph.nodes(), ElementsAre(n0, n1, n2));

  // Node should be the only consumer of the input value to be able to be
  // deleted with this function.
  ASSERT_FALSE(RemoveSimpleNodeKeepOutput(&graph, n1).ok());

  ASSERT_TRUE(RemoveSimpleNodeKeepInput(&graph, n1).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(v0, v1, v3));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(o1, o2));
  EXPECT_THAT(graph.nodes(), ElementsAre(n0, n2));
  EXPECT_THAT(graph.values(), UnorderedElementsAre(v0, v1, v3, o1, o2));
  EXPECT_THAT(graph.FindInputs(n0->id), ElementsAre(v0, v1));
  EXPECT_THAT(graph.FindInputs(n2->id), ElementsAre(v1, v3));
}

TEST(Model, CircularDependency) {
  {
    GraphFloat32 graph;
    Node* node = graph.NewNode();
    Value* value = graph.NewValue();
    ASSERT_TRUE(graph.AddConsumer(node->id, value->id).ok());
    EXPECT_FALSE(graph.SetProducer(node->id, value->id).ok());
  }
  {
    GraphFloat32 graph;
    Node* node = graph.NewNode();
    Value* value = graph.NewValue();
    ASSERT_TRUE(graph.SetProducer(node->id, value->id).ok());
    EXPECT_FALSE(graph.AddConsumer(node->id, value->id).ok());
  }
}

TEST(Model, ReassignValue) {
  // Before:
  //   graph_input  -> node1 -> graph_output
  //              \ -> node2
  GraphFloat32 graph;
  Node* node1 = graph.NewNode();
  Node* node2 = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node1->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node1->id, graph_output->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node2->id, graph_input->id).ok());

  // After:
  //   graph_input  -> node1
  //              \ -> node2 -> graph_output
  ASSERT_TRUE(graph.SetProducer(node2->id, graph_output->id).ok());

  EXPECT_THAT(graph.nodes(), ElementsAre(node1, node2));
  EXPECT_THAT(graph.FindInputs(node1->id), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.FindOutputs(node1->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindOutputs(node2->id), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindConsumers(graph_input->id),
              UnorderedElementsAre(node1, node2));
  EXPECT_THAT(graph.FindProducer(graph_output->id), ::testing::Eq(node2));
  EXPECT_THAT(graph.FindConsumers(graph_output->id), UnorderedElementsAre());
}

TEST(Model, DeleteValue) {
  // graph_input  -> node1 -> value -> node2 -> graph_output
  GraphFloat32 graph;
  Node* node1 = graph.NewNode();
  Node* node2 = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node1->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node1->id, value->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node2->id, value->id).ok());
  ASSERT_TRUE(graph.SetProducer(node2->id, graph_output->id).ok());

  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));
  EXPECT_THAT(graph.FindProducer(value->id), ::testing::Eq(node1));
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre(value));
  EXPECT_THAT(graph.FindOutputs(node1->id), UnorderedElementsAre(value));

  ASSERT_TRUE(graph.DeleteValue(value->id).ok());
  value = nullptr;
  EXPECT_THAT(graph.values(), UnorderedElementsAre(graph_input, graph_output));
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindOutputs(node1->id), UnorderedElementsAre());

  ASSERT_TRUE(graph.DeleteValue(graph_input->id).ok());
  graph_input = nullptr;
  EXPECT_THAT(graph.values(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre());
  EXPECT_THAT(graph.FindInputs(node1->id), UnorderedElementsAre());

  ASSERT_TRUE(graph.DeleteValue(graph_output->id).ok());
  graph_output = nullptr;
  EXPECT_THAT(graph.values(), UnorderedElementsAre());
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre());
  EXPECT_THAT(graph.FindOutputs(node2->id), UnorderedElementsAre());
}

TEST(Model, DeleteNode) {
  // graph_input -> node1 -> value  -> node2 -> graph_output
  //                               \-> node3 -> graph_output2
  GraphFloat32 graph;
  Node* node1 = graph.NewNode();
  Node* node2 = graph.NewNode();
  Node* node3 = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* graph_output2 = graph.NewValue();
  Value* value = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node1->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node1->id, value->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node2->id, value->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node3->id, value->id).ok());
  ASSERT_TRUE(graph.SetProducer(node2->id, graph_output->id).ok());
  ASSERT_TRUE(graph.SetProducer(node3->id, graph_output2->id).ok());

  EXPECT_THAT(graph.nodes(), ElementsAre(node1, node2, node3));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(),
              UnorderedElementsAre(graph_output, graph_output2));
  EXPECT_THAT(graph.FindConsumers(value->id),
              UnorderedElementsAre(node2, node3));
  EXPECT_THAT(graph.FindProducer(value->id), ::testing::Eq(node1));
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre(value));
  EXPECT_THAT(graph.FindInputs(node3->id), UnorderedElementsAre(value));

  // graph_input  -> node1 -> value -> node2 -> graph_output
  // graph_output2
  ASSERT_TRUE(graph.DeleteNode(node3->id).ok());
  node3 = nullptr;
  EXPECT_THAT(graph.nodes(), ElementsAre(node1, node2));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input, graph_output2));
  EXPECT_THAT(graph.outputs(),
              UnorderedElementsAre(graph_output, graph_output2));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));

  // value -> node2 -> graph_output
  // graph_input
  // graph_output2
  ASSERT_TRUE(graph.DeleteNode(node1->id).ok());
  node1 = nullptr;
  EXPECT_THAT(graph.nodes(), ElementsAre(node2));
  EXPECT_THAT(graph.inputs(),
              UnorderedElementsAre(value, graph_output2, graph_input));
  EXPECT_THAT(graph.outputs(),
              UnorderedElementsAre(graph_input, graph_output, graph_output2));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));
  EXPECT_THAT(graph.FindProducer(value->id), ::testing::Eq(nullptr));

  ASSERT_TRUE(graph.DeleteNode(node2->id).ok());
  node2 = nullptr;
  EXPECT_THAT(graph.nodes(), ElementsAre());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_output, graph_output2,
                                                   graph_input, value));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output, graph_output2,
                                                    graph_input, value));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindProducer(value->id), ::testing::Eq(nullptr));
}

TEST(Model, InsertNodeAfter) {
  // graph_input -> node1 -> value -> node2 -> graph_output
  GraphFloat32 graph;
  Node* node1 = graph.NewNode();
  Node* node2 = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node1->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node1->id, value->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node2->id, value->id).ok());
  ASSERT_TRUE(graph.SetProducer(node2->id, graph_output->id).ok());

  EXPECT_THAT(graph.nodes(), ElementsAre(node1, node2));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));
  EXPECT_THAT(graph.FindProducer(value->id), ::testing::Eq(node1));
  EXPECT_THAT(graph.FindInputs(node2->id), UnorderedElementsAre(value));

  Node* new_node1;
  absl::Status status = graph.InsertNodeAfter(node1->id, &new_node1);
  ASSERT_TRUE(status.ok());
  EXPECT_THAT(graph.nodes(), ElementsAre(node1, new_node1, node2));

  Node* new_node2;
  status = graph.InsertNodeAfter(/*id=*/100, &new_node2);
  EXPECT_EQ(status.code(), absl::StatusCode::kOutOfRange);

  status = graph.InsertNodeAfter(node2->id, &new_node2);
  ASSERT_TRUE(status.ok());
  EXPECT_THAT(graph.nodes(), ElementsAre(node1, new_node1, node2, new_node2));
}

TEST(BatchMatchingTest, EmptyGraph) {
  GraphFloat32 graph;
  ASSERT_TRUE(IsBatchMatchesForAllValues(graph));
}

TEST(BatchMatchingTest, AllMatch) {
  GraphFloat32 graph;
  Value* a = graph.NewValue();
  Value* b = graph.NewValue();
  a->tensor.shape = BHWC(1, 1, 1, 1);
  b->tensor.shape = BHWC(1, 1, 1, 1);
  ASSERT_TRUE(IsBatchMatchesForAllValues(graph));
}

TEST(BatchMatchingTest, NotAllMatch) {
  GraphFloat32 graph;
  Value* a = graph.NewValue();
  Value* b = graph.NewValue();
  a->tensor.shape = BHWC(1, 1, 1, 1);
  b->tensor.shape = BHWC(2, 1, 1, 1);
  ASSERT_FALSE(IsBatchMatchesForAllValues(graph));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
