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

#include "tensorflow/lite/delegates/gpu/common/transformations/remove_noop.h"

#include <any>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

using ::testing::UnorderedElementsAre;

TEST(RemoveSingleInputAdd, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  ASSERT_TRUE(graph.AddConsumer(first_node->id, input->id).ok());

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::ADD);
  add_node->operation.attributes = ElementwiseAttributes();

  Value* temp = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, add_node, &temp).ok());
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("noop", transformation.get());

  EXPECT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());
  ASSERT_EQ(first_node, graph.nodes()[0]);
  ASSERT_EQ(input, graph.values()[0]);
  ASSERT_EQ(output, graph.values()[1]);
}

TEST(RemoveSingleInputAdd, DoNotTrigger_TensorHWC) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  ASSERT_TRUE(graph.AddConsumer(first_node->id, input->id).ok());

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::ADD);
  ElementwiseAttributes attr;
  attr.param = Tensor<HWC, DataType::FLOAT32>();
  add_node->operation.attributes = attr;

  Value* temp = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, add_node, &temp).ok());
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("noop", transformation.get());

  EXPECT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());
}

TEST(RemoveSingleInputAdd, DoNotTrigger_LinearTensor) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  ASSERT_TRUE(graph.AddConsumer(first_node->id, input->id).ok());

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::ADD);
  ElementwiseAttributes attr;
  attr.param = Tensor<Linear, DataType::FLOAT32>();
  add_node->operation.attributes = attr;

  Value* temp = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, add_node, &temp).ok());
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("noop", transformation.get());

  EXPECT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());
}

TEST(RemoveSingleInputAdd, DoNotTrigger_Scalar) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  ASSERT_TRUE(graph.AddConsumer(first_node->id, input->id).ok());

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::ADD);
  ElementwiseAttributes attr;
  attr.param = 0.5f;
  add_node->operation.attributes = attr;

  Value* temp = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, add_node, &temp).ok());
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("noop", transformation.get());

  EXPECT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());
}

TEST(RemoveSingleInputAdd, DoNotTrigger_Multiple) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto node_a = graph.NewNode();
  auto node_b = graph.NewNode();
  ASSERT_TRUE(graph.AddConsumer(node_a->id, input->id).ok());
  ASSERT_TRUE(graph.AddConsumer(node_b->id, input->id).ok());

  auto add_node = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  add_node->operation.type = ToString(OperationType::ADD);

  Value* temp_a = nullptr;
  Value* temp_b = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, node_a, add_node, &temp_a).ok());
  ASSERT_TRUE(ConnectTwoNodes(&graph, node_b, add_node, &temp_b).ok());
  ASSERT_EQ(3, graph.nodes().size());
  ASSERT_EQ(4, graph.values().size());

  auto transformation = NewRemoveSingleInputAdd();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("noop", transformation.get());

  ASSERT_EQ(3, graph.nodes().size());
  ASSERT_EQ(4, graph.values().size());
}

TEST(RemoveDegenerateUpsampling, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  auto first_node = graph.NewNode();
  ASSERT_TRUE(graph.AddConsumer(first_node->id, input->id).ok());

  auto node_to_remove = graph.NewNode();
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, node_to_remove, &output).ok());
  output->tensor.shape = BHWC(1, 5, 5, 1);
  node_to_remove->operation.type = ToString(OperationType::RESIZE);
  Resize2DAttributes attr;
  attr.new_shape = HW(5, 5);
  attr.type = SamplingType::BILINEAR;
  node_to_remove->operation.attributes = attr;

  Value* link = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, first_node, node_to_remove, &link).ok());
  link->tensor.shape = output->tensor.shape;
  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewRemoveDegenerateUpsampling();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("noop", transformation.get());

  ASSERT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());
  EXPECT_EQ(first_node, graph.nodes()[0]);
  EXPECT_EQ(input, graph.values()[0]);
  EXPECT_EQ(output, graph.values()[1]);
}

TEST(RemoveIdentityReshape, Smoke) {
  GraphFloat32 graph;
  Node* simple_node = graph.NewNode();
  Node* producer_node = graph.NewNode();
  Node* consumer_node = graph.NewNode();
  Value* graph_input = graph.NewValue();
  Value* graph_output = graph.NewValue();
  Value* value0 = graph.NewValue();
  Value* value1 = graph.NewValue();

  value0->tensor.shape = BHWC(1, 1, 1, 11);
  simple_node->operation.type = ToString(OperationType::RESHAPE);
  ReshapeAttributes attr;
  attr.new_shape = BHWC(1, 1, 1, 11);
  simple_node->operation.attributes = attr;

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

  auto transformation = NewRemoveIdentityReshape();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("noop", transformation.get());

  EXPECT_THAT(graph.inputs(), UnorderedElementsAre(graph_input));
  EXPECT_THAT(graph.outputs(), UnorderedElementsAre(graph_output));
  EXPECT_THAT(graph.nodes(),
              UnorderedElementsAre(producer_node, consumer_node));
  EXPECT_THAT(graph.values(),
              UnorderedElementsAre(graph_input, graph_output, value0));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
