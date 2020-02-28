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

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

TEST(Model, SingleNode) {
  // graph_input -> node -> graph_output
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node->id, graph_output->id).ok());

  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(node));
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
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output1 = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output2 = graph.NewValue();
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
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node->id, graph_input->id).ok());
  EXPECT_FALSE(graph.AddConsumer(node->id, graph_input->id).ok());
}

TEST(Model, RemoveConsumer) {
  // (graph_input1, graph_input2) -> node
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value<TensorRef<BHWC>>* graph_input1 = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_input2 = graph.NewValue();
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
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();
  ASSERT_TRUE(graph.SetProducer(node->id, graph_output->id).ok());
  EXPECT_FALSE(graph.SetProducer(node->id, graph_output->id).ok());
}

TEST(Model, ReplaceInput) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value<TensorRef<BHWC>>* v0 = graph.NewValue();
  Value<TensorRef<BHWC>>* v1 = graph.NewValue();
  Value<TensorRef<BHWC>>* v2 = graph.NewValue();
  Value<TensorRef<BHWC>>* v3 = graph.NewValue();
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
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();

  ASSERT_TRUE(graph.SetProducer(node->id, graph_output->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre());
  EXPECT_THAT(graph.FindProducer(graph_output->id), ::testing::Eq(node));

  ASSERT_TRUE(graph.RemoveProducer(graph_output->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.FindProducer(graph_output->id), ::testing::Eq(nullptr));

  // Can not remove producer twice
  ASSERT_FALSE(graph.RemoveProducer(graph_output->id).ok());
}

TEST(Model, RemoveSimpleNodeDegenerateCase) {
  GraphFloat32 graph;
  Node* node = graph.NewNode();
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();

  ASSERT_TRUE(graph.AddConsumer(node->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node->id, graph_output->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(node));

  ASSERT_TRUE(RemoveOneInputOneOutputNode(&graph, node).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre());
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre());
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre());
}

TEST(Model, RemoveSimpleNodeNoPreviousNode) {
  GraphFloat32 graph;
  Node* simple_node = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();
  Value<TensorRef<BHWC>>* value = graph.NewValue();

  ASSERT_TRUE(graph.AddConsumer(simple_node->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(simple_node->id, value->id).ok());
  ASSERT_TRUE(graph.AddConsumer(consumer_node->id, value->id).ok());
  ASSERT_TRUE(graph.SetProducer(consumer_node->id, graph_output->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(simple_node, consumer_node));

  ASSERT_TRUE(RemoveOneInputOneOutputNode(&graph, simple_node).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(consumer_node));
}

TEST(Model, RemoveSimpleNodeNoAfterNodes) {
  GraphFloat32 graph;
  Node* simple_node = graph.NewNode();
  Node* producer_node = graph.NewNode();
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();
  Value<TensorRef<BHWC>>* value = graph.NewValue();

  ASSERT_TRUE(graph.AddConsumer(simple_node->id, value->id).ok());
  ASSERT_TRUE(graph.SetProducer(simple_node->id, graph_output->id).ok());
  ASSERT_TRUE(graph.AddConsumer(producer_node->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(producer_node->id, value->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(simple_node, producer_node));

  ASSERT_TRUE(RemoveOneInputOneOutputNode(&graph, simple_node).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(value));
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(producer_node));
}

TEST(Model, RemoveSimpleNodeGeneralCase) {
  GraphFloat32 graph;
  Node* simple_node = graph.NewNode();
  Node* producer_node = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();
  Value<TensorRef<BHWC>>* value0 = graph.NewValue();
  Value<TensorRef<BHWC>>* value1 = graph.NewValue();

  ASSERT_TRUE(graph.AddConsumer(producer_node->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(producer_node->id, value0->id).ok());
  ASSERT_TRUE(graph.AddConsumer(simple_node->id, value0->id).ok());
  ASSERT_TRUE(graph.SetProducer(simple_node->id, value1->id).ok());
  ASSERT_TRUE(graph.AddConsumer(consumer_node->id, value1->id).ok());
  ASSERT_TRUE(graph.SetProducer(consumer_node->id, graph_output->id).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(simple_node, producer_node, consumer_node));

  ASSERT_TRUE(RemoveOneInputOneOutputNode(&graph, simple_node).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(producer_node, consumer_node));
  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value0));
}

TEST(Model, RemoveSimpleNodeComplexCase) {
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
  Value<TensorRef<BHWC>>* v0 = graph.NewValue();
  Value<TensorRef<BHWC>>* v1 = graph.NewValue();
  Value<TensorRef<BHWC>>* v2 = graph.NewValue();  // value to be removed
  Value<TensorRef<BHWC>>* v3 = graph.NewValue();
  Value<TensorRef<BHWC>>* o1 = graph.NewValue();
  Value<TensorRef<BHWC>>* o2 = graph.NewValue();

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
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(n0, n1, n2));

  ASSERT_TRUE(RemoveOneInputOneOutputNode(&graph, n1).ok());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(v0, v1, v3));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(o1, o2));
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(n0, n2));
  EXPECT_THAT(graph.values(), UnorderedElementsAre(v0, v1, v3, o1, o2));
  EXPECT_THAT(graph.FindInputs(n0->id), ElementsAre(v0, v1));
  EXPECT_THAT(graph.FindInputs(n2->id), ElementsAre(v1, v3));
}

TEST(Model, CircularDependency) {
  {
    GraphFloat32 graph;
    Node* node = graph.NewNode();
    Value<TensorRef<BHWC>>* value = graph.NewValue();
    ASSERT_TRUE(graph.AddConsumer(node->id, value->id).ok());
    EXPECT_FALSE(graph.SetProducer(node->id, value->id).ok());
  }
  {
    GraphFloat32 graph;
    Node* node = graph.NewNode();
    Value<TensorRef<BHWC>>* value = graph.NewValue();
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
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node1->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node1->id, graph_output->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node2->id, graph_input->id).ok());

  // After:
  //   graph_input  -> node1
  //              \ -> node2 -> graph_output
  ASSERT_TRUE(graph.SetProducer(node2->id, graph_output->id).ok());

  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(node1, node2));
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
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();
  Value<TensorRef<BHWC>>* value = graph.NewValue();
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
  Value<TensorRef<BHWC>>* graph_input = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output = graph.NewValue();
  Value<TensorRef<BHWC>>* graph_output2 = graph.NewValue();
  Value<TensorRef<BHWC>>* value = graph.NewValue();
  ASSERT_TRUE(graph.AddConsumer(node1->id, graph_input->id).ok());
  ASSERT_TRUE(graph.SetProducer(node1->id, value->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node2->id, value->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node3->id, value->id).ok());
  ASSERT_TRUE(graph.SetProducer(node2->id, graph_output->id).ok());
  ASSERT_TRUE(graph.SetProducer(node3->id, graph_output2->id).ok());

  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(node1, node2, node3));
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
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(node1, node2));
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input, graph_output2));
  EXPECT_THAT(graph.outputs(),
              UnorderedElementsAre(graph_output, graph_output2));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));

  // value -> node2 -> graph_output
  // graph_input
  // graph_output2
  ASSERT_TRUE(graph.DeleteNode(node1->id).ok());
  node1 = nullptr;
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre(node2));
  EXPECT_THAT(graph.inputs(),
              UnorderedElementsAre(value, graph_output2, graph_input));
  EXPECT_THAT(graph.outputs(),
              UnorderedElementsAre(graph_input, graph_output, graph_output2));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre(node2));
  EXPECT_THAT(graph.FindProducer(value->id), ::testing::Eq(nullptr));

  ASSERT_TRUE(graph.DeleteNode(node2->id).ok());
  node2 = nullptr;
  EXPECT_THAT(graph.nodes(), UnorderedElementsAre());
  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_output, graph_output2,
                                                   graph_input, value));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output, graph_output2,
                                                    graph_input, value));
  EXPECT_THAT(graph.FindConsumers(value->id), UnorderedElementsAre());
  EXPECT_THAT(graph.FindProducer(value->id), ::testing::Eq(nullptr));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
