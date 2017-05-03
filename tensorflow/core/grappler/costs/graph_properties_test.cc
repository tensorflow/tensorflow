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

#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class GraphPropertiesTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Provision a single machine with 3 cpu cores
    cluster_.reset(new SingleMachine(5 * 60, 3, 0));
    TF_CHECK_OK(cluster_->Provision());
  }

  void TearDown() override { cluster_.reset(); }

 protected:
  std::unique_ptr<SingleMachine> cluster_;
};

TEST_F(GraphPropertiesTest, StaticProperties) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphProperties properties(item);
  Status s = properties.InferStatically();
  TF_CHECK_OK(s);

  for (const auto& node : item.graph.node()) {
    if (node.op() == "RandomStandardNormal") {
      // The node has one input (the shape of the tensor to generate).
      EXPECT_EQ(1, properties.GetInputProperties(node.name()).size());
      // The const node has one output.
      const auto props = properties.GetOutputProperties(node.name());
      EXPECT_EQ(1, props.size());
      const OpInfo::TensorProperties& prop = props[0];
      EXPECT_EQ(DT_FLOAT, prop.dtype());
      EXPECT_FALSE(prop.shape().unknown_rank());
      EXPECT_EQ(2, prop.shape().dim_size());
      EXPECT_EQ(10, prop.shape().dim(0).size());
      EXPECT_EQ(1, prop.shape().dim(1).size());
    } else if (node.op() == "AddN") {
      const auto in_props = properties.GetInputProperties(node.name());
      EXPECT_EQ(1, in_props.size());
      const OpInfo::TensorProperties& in_prop = in_props[0];
      EXPECT_EQ(DT_FLOAT, in_prop.dtype());
      EXPECT_FALSE(in_prop.shape().unknown_rank());
      EXPECT_EQ(2, in_prop.shape().dim_size());
      EXPECT_EQ(10, in_prop.shape().dim(0).size());
      EXPECT_EQ(1, in_prop.shape().dim(1).size());
      const auto out_props = properties.GetOutputProperties(node.name());
      EXPECT_EQ(1, out_props.size());
      string in_prop_str;
      ::tensorflow::protobuf::TextFormat::PrintToString(in_prop, &in_prop_str);
      string out_prop_str;
      ::tensorflow::protobuf::TextFormat::PrintToString(out_props[0],
                                                        &out_prop_str);
      EXPECT_EQ(in_prop_str, out_prop_str);
    }
  }
}

TEST_F(GraphPropertiesTest, DynamicProperties) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphProperties properties(item);
  TF_CHECK_OK(cluster_->Initialize(item));
  Status s = properties.InferDynamically(cluster_.get());
  TF_CHECK_OK(s);

  for (const auto& node : item.graph.node()) {
    if (node.op() == "RandomStandardNormal") {
      // The random node is missing from the cost graph (why ?)
      EXPECT_EQ(0, properties.GetInputProperties(node.name()).size());
    } else if (node.op() == "AddN") {
      // Since the random node is missing, we can't infer the input properties
      // of the first AddN node. The other AddN nodes have the expected
      // properties.
      if (node.name() == "AddN") {
        const auto props = properties.GetInputProperties(node.name());
        EXPECT_EQ(1, props.size());
        const OpInfo::TensorProperties& prop = props[0];
        EXPECT_EQ(DT_INVALID, prop.dtype());
        EXPECT_TRUE(prop.shape().unknown_rank());
      } else {
        const auto props = properties.GetInputProperties(node.name());
        EXPECT_EQ(1, props.size());
        const OpInfo::TensorProperties& prop = props[0];
        EXPECT_EQ(DT_FLOAT, prop.dtype());
        EXPECT_FALSE(prop.shape().unknown_rank());
        EXPECT_EQ(2, prop.shape().dim_size());
        EXPECT_EQ(10, prop.shape().dim(0).size());
        EXPECT_EQ(1, prop.shape().dim(1).size());
        const auto out_props = properties.GetOutputProperties(node.name());
        EXPECT_EQ(1, out_props.size());
        string prop_str;
        ::tensorflow::protobuf::TextFormat::PrintToString(prop, &prop_str);
        string out_prop_str;
        ::tensorflow::protobuf::TextFormat::PrintToString(out_props[0],
                                                          &out_prop_str);
        EXPECT_EQ(prop_str, out_prop_str);
      }
    }
  }
}

TEST_F(GraphPropertiesTest, VarHandles) {
  GrapplerItem item;
  TF_CHECK_OK(NodeDefBuilder("Var", "VarHandleOp")
                  .Attr("dtype", DT_FLOAT)
                  .Attr("shape", TensorShape({3, 7}))
                  .Finalize(item.graph.add_node()));

  TF_CHECK_OK(NodeDefBuilder("VarRead", "ReadVariableOp")
                  .Attr("dtype", DT_FLOAT)
                  .Input("Var", 0, DT_RESOURCE)
                  .Finalize(item.graph.add_node()));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically());

  const auto props = properties.GetOutputProperties("VarRead");
  EXPECT_EQ(1, props.size());
  const OpInfo::TensorProperties& prop = props[0];
  EXPECT_EQ(DT_FLOAT, prop.dtype());
  EXPECT_FALSE(prop.shape().unknown_rank());
  EXPECT_EQ(2, prop.shape().dim_size());
  EXPECT_EQ(3, prop.shape().dim(0).size());
  EXPECT_EQ(7, prop.shape().dim(1).size());
}

TEST_F(GraphPropertiesTest, Queues) {
  // Create a graph with known input shapes, and propagate the shapes through a
  // couple of queues.
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  auto q1 = ops::FIFOQueue(root.WithOpName("Queue1"), {DataType::DT_FLOAT});
  Output rnd =
      ops::RandomNormal(root.WithOpName("rnd"), {3, 7}, DataType::DT_FLOAT);
  Output square1 = ops::Square(root.WithOpName("Square1"), rnd);
  auto enqueue1 = ops::QueueEnqueue(root.WithOpName("Enqueue1"), q1, {square1});
  auto dequeue1 =
      ops::QueueDequeue(root.WithOpName("Dequeue1"), q1, {DataType::DT_FLOAT});

  auto q2 =
      ops::RandomShuffleQueue(root.WithOpName("Queue2"), {DataType::DT_FLOAT});
  Output square2 = ops::Square(root.WithOpName("Square2"), dequeue1[0]);
  auto enqueue2 = ops::QueueEnqueue(root.WithOpName("Enqueue2"), q2, {square2});
  auto dequeue2 =
      ops::QueueDequeue(root.WithOpName("Dequeue2"), q2, {DataType::DT_FLOAT});

  auto q3 =
      ops::RandomShuffleQueue(root.WithOpName("Queue3"), {DataType::DT_FLOAT});
  auto dequeue3 =
      ops::QueueDequeue(root.WithOpName("Dequeue3"), q3, {DataType::DT_FLOAT});

  auto q4 =
      ops::RandomShuffleQueue(root.WithOpName("Queue4"), {DataType::DT_FLOAT});
  auto enqueue4 = ops::QueueEnqueue(root.WithOpName("Enqueue4"), q4, {square2});
  auto enqueue4_2 =
      ops::QueueEnqueue(root.WithOpName("Enqueue4_2"), q4, {dequeue3[0]});
  auto dequeue4 =
      ops::QueueDequeue(root.WithOpName("Dequeue4"), q4, {DataType::DT_FLOAT});

  GrapplerItem item;
  TF_CHECK_OK(root.ToGraphDef(&item.graph));

  GraphProperties properties(item);
  TF_CHECK_OK(properties.InferStatically());

  const auto props1 = properties.GetOutputProperties("Dequeue1");
  EXPECT_EQ(1, props1.size());
  const OpInfo::TensorProperties& prop1 = props1[0];
  EXPECT_EQ(DT_FLOAT, prop1.dtype());
  EXPECT_FALSE(prop1.shape().unknown_rank());
  EXPECT_EQ(2, prop1.shape().dim_size());
  EXPECT_EQ(3, prop1.shape().dim(0).size());
  EXPECT_EQ(7, prop1.shape().dim(1).size());

  const auto props2 = properties.GetOutputProperties("Dequeue2");
  EXPECT_EQ(1, props2.size());
  const OpInfo::TensorProperties& prop2 = props2[0];
  EXPECT_EQ(DT_FLOAT, prop2.dtype());
  EXPECT_FALSE(prop2.shape().unknown_rank());
  EXPECT_EQ(2, prop2.shape().dim_size());
  EXPECT_EQ(3, prop2.shape().dim(0).size());
  EXPECT_EQ(7, prop2.shape().dim(1).size());

  // The dequeue3 op shape is unknown. The square2 op shape is known. Verify
  // that we merge the 2 properly to determine the shape of the data coming out
  // of the queue.
  const auto props4 = properties.GetOutputProperties("Dequeue4");
  EXPECT_EQ(1, props4.size());
  const OpInfo::TensorProperties& prop4 = props4[0];
  EXPECT_EQ(DT_FLOAT, prop4.dtype());
  EXPECT_FALSE(prop4.shape().unknown_rank());
  EXPECT_EQ(2, prop4.shape().dim_size());
  EXPECT_EQ(3, prop4.shape().dim(0).size());
  EXPECT_EQ(7, prop4.shape().dim(1).size());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
