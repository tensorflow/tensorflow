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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/optimizers/auto_mixed_precision.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace grappler {
namespace {

const std::pair<int, int> kMinGPUArch = {7, 0};

class AutoMixedPrecisionTest : public GrapplerTest {
 protected:
  void SetUp() override {
    int num_gpus = GetNumAvailableGPUs();
    // If GPUs are available, require that they all satisfy the min arch.
    gpu_available_ =
        num_gpus > 0 && num_gpus == GetNumAvailableGPUs(kMinGPUArch);

    if (gpu_available_) {
      virtual_cluster_.reset(new SingleMachine(/* timeout_s = */ 10, 1, 1));
    } else {
      DeviceProperties device_properties;
      device_properties.set_type("GPU");
      device_properties.mutable_environment()->insert({"architecture", "7"});
      virtual_cluster_.reset(
          new VirtualCluster({{"/GPU:1", device_properties}}));
    }
    TF_CHECK_OK(virtual_cluster_->Provision());
  }

  void TearDown() override { TF_CHECK_OK(virtual_cluster_->Shutdown()); }

  NodeDef* AddSimpleNode(const string& name, const string& op,
                         const std::vector<string>& inputs,
                         GraphDef* graph) const {
    std::vector<std::pair<string, AttrValue>> attributes;
    if (op == "AddN" || op == "ShapeN") {
      AttrValue num_inputs;
      num_inputs.set_i(inputs.size());
      attributes.emplace_back("N", num_inputs);
    }
    if (op == "ShapeN") {
      AttrValue out_type;
      out_type.set_type(DT_INT32);
      attributes.emplace_back("out_type", out_type);
    }
    AttrValue type;
    type.set_type(DT_FLOAT);
    if (op == "Const" || op == "Placeholder" || op == "VariableV2" ||
        op == "VarHandleOp" || op == "ReadVariableOp") {
      attributes.emplace_back("dtype", type);
    } else if (op == "SparseMatMul") {
      attributes.emplace_back("Ta", type);
      attributes.emplace_back("Tb", type);
    } else if (op == "IdentityN") {
      AttrValue type_list;
      for (int i = 0; i < (int)inputs.size(); ++i) {
        type_list.mutable_list()->add_type(DT_FLOAT);
      }
      attributes.emplace_back("T", type_list);
    } else if (op == "StackV2" || op == "StackPopV2") {
      attributes.emplace_back("elem_type", type);
    } else if (op == "Cast") {
      attributes.emplace_back("SrcT", type);
      attributes.emplace_back("DstT", type);
    } else {
      attributes.emplace_back("T", type);
    }
    return AddNode(name, op, inputs, attributes, graph);
  }

  std::unique_ptr<Cluster> virtual_cluster_;
  bool gpu_available_;
};

void VerifyGraphsEquivalent(const GraphDef& original_graph,
                            const GraphDef& optimized_graph,
                            const string& func) {
  EXPECT_EQ(original_graph.node_size(), optimized_graph.node_size()) << func;
  GraphView optimized_view(&optimized_graph);
  for (int i = 0; i < original_graph.node_size(); ++i) {
    const NodeDef& original = original_graph.node(i);
    const NodeDef& optimized = *optimized_view.GetNode(original.name());
    EXPECT_EQ(original.name(), optimized.name()) << func;
    EXPECT_EQ(original.op(), optimized.op()) << func;
    EXPECT_EQ(original.input_size(), optimized.input_size()) << func;
    if (original.input_size() == optimized.input_size()) {
      for (int j = 0; j < original.input_size(); ++j) {
        EXPECT_EQ(original.input(j), optimized.input(j)) << func;
      }
    }
  }
}

TEST_F(AutoMixedPrecisionTest, NoOp) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("B1", "Exp", {"In"}, &graph);
  AddSimpleNode("C1", "Relu", {"B1"}, &graph);
  AddSimpleNode("G1", "Sqrt", {"C1"}, &graph);
  AddSimpleNode("C2", "Relu", {"G1"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  VerifyGraphsEquivalent(item.graph, output, __FUNCTION__);

  GraphView output_view(&output);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_FLOAT);
}

TEST_F(AutoMixedPrecisionTest, AlreadyFp16) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  NodeDef* cast1 = AddSimpleNode("Cast1", "Cast", {"In"}, &graph);
  cast1->mutable_attr()->at("DstT").set_type(DT_HALF);
  NodeDef* w1 = AddSimpleNode("W1", "MatMul", {"Cast1", "Cast1"}, &graph);
  w1->mutable_attr()->at("T").set_type(DT_HALF);
  NodeDef* c1 = AddSimpleNode("C1", "Relu", {"W1"}, &graph);
  c1->mutable_attr()->at("T").set_type(DT_HALF);
  NodeDef* cast2 = AddSimpleNode("Cast2", "Cast", {"C1"}, &graph);
  cast2->mutable_attr()->at("SrcT").set_type(DT_HALF);
  AddSimpleNode("C2", "Relu", {"Cast2"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  VerifyGraphsEquivalent(item.graph, output, __FUNCTION__);

  GraphView output_view(&output);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("Cast1")->attr().at("DstT").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("Cast2")->attr().at("SrcT").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("Cast2")->attr().at("DstT").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_FLOAT);
}

TEST_F(AutoMixedPrecisionTest, Simple) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("B1", "Exp", {"In"}, &graph);
  AddSimpleNode("C1", "Relu", {"B1"}, &graph);
  AddSimpleNode("G1", "Sqrt", {"C1"}, &graph);
  AddSimpleNode("C2", "Relu", {"G1"}, &graph);
  AddSimpleNode("W1", "MatMul", {"C2", "C2"}, &graph);
  AddSimpleNode("C3", "Relu", {"W1"}, &graph);
  AddSimpleNode("B2", "Exp", {"C3"}, &graph);
  AddSimpleNode("C4", "Relu", {"B2"}, &graph);
  AddSimpleNode("B4", "SparseMatMul", {"C4", "C4"}, &graph);
  AddSimpleNode("C5", "Relu", {"B4"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("C3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("B2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C4")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B4")->attr().at("Ta").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B4")->attr().at("Tb").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C5")->attr().at("T").type(), DT_FLOAT);
}

TEST_F(AutoMixedPrecisionTest, BidirectionalClearChain) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("C1", "Relu", {"In"}, &graph);
  AddSimpleNode("C2", "Relu", {"In"}, &graph);
  AddSimpleNode("W1", "MatMul", {"C1", "C1"}, &graph);
  AddSimpleNode("C3", "ShapeN", {"C1", "C2"}, &graph);
  AddSimpleNode("C4", "Relu", {"C2"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 1);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("C3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("C4")->attr().at("T").type(), DT_HALF);
};

TEST_F(AutoMixedPrecisionTest, PreserveFetches) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("Const1", "Const", {}, &graph);
  AddSimpleNode("W1", "MatMul", {"In", "Const1"}, &graph);
  AddSimpleNode("C1", "Relu", {"W1"}, &graph);
  AddSimpleNode("G1", "Sqrt", {"C1"}, &graph);
  AddSimpleNode("B1", "Exp", {"G1"}, &graph);
  AddSimpleNode("C2", "Relu", {"B1"}, &graph);
  AddSimpleNode("W2", "MatMul", {"C2", "C2"}, &graph);
  AddSimpleNode("C3", "Relu", {"W2"}, &graph);
  AddSimpleNode("B2", "Exp", {"C3"}, &graph);
  AddSimpleNode("C4", "Relu", {"B2"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  item.fetch.push_back("W1");
  item.fetch.push_back("C2");
  item.fetch.push_back("C3");
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("Const1")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("C3")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C4")->attr().at("T").type(), DT_FLOAT);
}

TEST_F(AutoMixedPrecisionTest, PreserveCPUNodes) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("C1", "Relu", {"In"}, &graph);
  AddSimpleNode("W1", "MatMul", {"C1", "C1"}, &graph);
  AddSimpleNode("G1", "Tanh", {"W1"}, &graph);
  NodeDef* w2 = AddSimpleNode("W2", "MatMul", {"G1", "G1"}, &graph);
  w2->set_device("/job:localhost/replica:0/task:0/device:CPU:0");
  AddSimpleNode("C2", "Relu", {"W2"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_FLOAT);
}

TEST_F(AutoMixedPrecisionTest, PreserveIdentityAfterVariable) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("V1", "VariableV2", {}, &graph);
  AddSimpleNode("C1", "Identity", {"V1"}, &graph);
  AddSimpleNode("W1", "MatMul", {"In", "C1"}, &graph);
  AddSimpleNode("VarHandle1", "VarHandleOp", {}, &graph);
  AddSimpleNode("V2", "ReadVariableOp", {"VarHandle1"}, &graph);
  AddSimpleNode("W2", "MatMul", {"In", "V2"}, &graph);
  AddSimpleNode("Const1", "Const", {}, &graph);
  AddSimpleNode("C2", "Identity", {"Const1"}, &graph);
  AddSimpleNode("W3", "MatMul", {"In", "C2"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 4);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("V1")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("VarHandle1")->attr().at("dtype").type(),
            DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("V2")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("Const1")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W3")->attr().at("T").type(), DT_HALF);
}

TEST_F(AutoMixedPrecisionTest, FusedBatchNorm) {
  GraphDef graph;
  AddSimpleNode("X", "Placeholder", {}, &graph);
  AddSimpleNode("Const1", "Const", {}, &graph);
  AddSimpleNode("Scale", "Placeholder", {}, &graph);
  AddSimpleNode("Offset", "Placeholder", {}, &graph);
  AddSimpleNode("Mean", "Placeholder", {}, &graph);
  AddSimpleNode("Variance", "Placeholder", {}, &graph);
  AddSimpleNode("W1", "Conv2D", {"X", "Const1"}, &graph);
  AddSimpleNode("BN1", "FusedBatchNorm",
                {"W1", "Scale", "Offset", "Mean", "Variance"}, &graph);
  AddSimpleNode("BNG1", "FusedBatchNormGrad",
                {"BN1", "W1", "Scale", "Mean", "Variance"}, &graph);
  AddSimpleNode("G1", "Add", {"BN1", "BNG1"}, &graph);
  AddSimpleNode("W2", "Conv2D", {"G1", "Const1"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("BN1")->op(), "FusedBatchNormV2");
  EXPECT_EQ(output_view.GetNode("BN1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("BN1")->attr().at("U").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("BNG1")->op(), "FusedBatchNormGradV2");
  EXPECT_EQ(output_view.GetNode("BNG1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("BNG1")->attr().at("U").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_HALF);
}

TEST_F(AutoMixedPrecisionTest, RepeatedAndListTypeAttrs) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("W1", "MatMul", {"In", "In"}, &graph);
  AddSimpleNode("ID", "IdentityN", {"W1", "W1", "W1"}, &graph);
  AddSimpleNode("G1", "AddN", {"ID:0", "ID:1", "ID:2"}, &graph);
  AddSimpleNode("W2", "MatMul", {"G1", "G1"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 1);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  for (auto type : output_view.GetNode("ID")->attr().at("T").list().type()) {
    EXPECT_EQ(type, DT_HALF);
  }
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_HALF);
}

TEST_F(AutoMixedPrecisionTest, ExistingCast) {
  GraphDef graph;
  NodeDef* ph = AddSimpleNode("In", "Placeholder", {}, &graph);
  ph->mutable_attr()->at("dtype").set_type(DT_BOOL);
  NodeDef* cast = AddSimpleNode("Cast1", "Cast", {"In"}, &graph);
  cast->mutable_attr()->at("SrcT").set_type(DT_BOOL);
  AddSimpleNode("W1", "MatMul", {"Cast1", "Cast1"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size());
  EXPECT_EQ(output_view.GetNode("Cast1")->attr().at("DstT").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
}

TEST_F(AutoMixedPrecisionTest, StackV2) {
  GraphDef graph;
  AddSimpleNode("Handle1", "Const", {}, &graph);
  AddSimpleNode("Stack1", "StackV2", {"Handle1"}, &graph);
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("Push1", "StackPushV2", {"Stack1", "In"}, &graph);
  AddSimpleNode("W1", "MatMul", {"In", "In"}, &graph);
  AddSimpleNode("Push2", "StackPushV2", {"Stack1", "W1"}, &graph);
  AddSimpleNode("Pop1", "StackPopV2", {"Stack1"}, &graph);
  AddSimpleNode("G1", "Tanh", {"Pop1"}, &graph);
  AddSimpleNode("W2", "MatMul", {"G1", "G1"}, &graph);
  AddSimpleNode("Push3", "StackPushV2", {"Stack1", "W2"}, &graph);
  AddSimpleNode("Handle2", "Const", {}, &graph);
  AddSimpleNode("Stack2", "StackV2", {"Handle2"}, &graph);
  AddSimpleNode("Push1-2", "StackPushV2", {"Stack2", "In"}, &graph);
  AddSimpleNode("Pop1-2", "StackPopV2", {"Stack2"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 1);
  EXPECT_EQ(output_view.GetNode("Stack1")->attr().at("elem_type").type(),
            DT_HALF);
  EXPECT_EQ(output_view.GetNode("Push1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("Push2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("Pop1")->attr().at("elem_type").type(),
            DT_HALF);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("Push3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("Stack2")->attr().at("elem_type").type(),
            DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("Push1-2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("Pop1-2")->attr().at("elem_type").type(),
            DT_FLOAT);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
