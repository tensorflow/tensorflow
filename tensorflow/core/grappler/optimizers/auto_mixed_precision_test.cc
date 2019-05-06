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

// Currently, this test only passes when TensorFlow passes with CUDA, because
// otherwise the optimizer will not turn clearlist nodes to float16. When
// looking at clearlist nodes, this optimizer checks if the nodes have a float16
// GPU OpKernel, but without CUDA there are no GPU OpKernels at all.
#if GOOGLE_CUDA

#include "tensorflow/core/grappler/optimizers/auto_mixed_precision.h"

#include <utility>
#include <vector>

#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

// TODO(benbarsdell): Improve the numerical checks in these tests. The tests
// were originally written only to check the graph coloring, so the graphs do
// not have particularly realistic numerical behavior.

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
      for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
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
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.234f, {32});
  Output blk1 = ops::Exp(s.WithOpName("blk1"), input);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), blk1);
  Output gry1 = ops::Sqrt(s.WithOpName("gry1"), clr1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), gry1);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr2);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  VerifyGraphsEquivalent(item.graph, output, __FUNCTION__);

  GraphView output_view(&output);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("blk1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("gry1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionTest, AlreadyFp16) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f, {32, 32});
  Output cst1 = ops::Cast(s.WithOpName("cst1"), input, DT_HALF);
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), cst1, cst1);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), wht1);
  Output cst2 = ops::Cast(s.WithOpName("cst2"), clr1, DT_FLOAT);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), cst2);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr2);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));
  VLOG(1) << output.DebugString();

  VerifyGraphsEquivalent(item.graph, output, __FUNCTION__);
  GraphView output_view(&output);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("cst1")->attr().at("DstT").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("cst2")->attr().at("SrcT").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("cst2")->attr().at("DstT").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionTest, Simple) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output blk1 = ops::Exp(s.WithOpName("blk1"), input);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), blk1);
  Output gry1 = ops::Sqrt(s.WithOpName("gry1"), clr1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), gry1);
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), clr2, clr2);
  Output clr3 = ops::Relu(s.WithOpName("clr3"), wht1);
  Output blk2 = ops::Log(s.WithOpName("blk2"), clr3);
  Output clr4 = ops::Relu(s.WithOpName("clr4"), blk2);
  Output blk3 = ops::SparseMatMul(s.WithOpName("blk3"), clr4, clr4);
  Output clr5 = ops::Relu(s.WithOpName("clr5"), blk3);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr5);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("blk1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("gry1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("clr3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("blk2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr4")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("blk3")->attr().at("Ta").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("blk3")->attr().at("Tb").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr5")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-4);
  }
}

TEST_F(AutoMixedPrecisionTest, BidirectionalClearChain) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output clr1 = ops::Relu(s.WithOpName("clr1"), input);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), input);
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), clr1, clr1);
  auto clr3 = ops::ShapeN(s.WithOpName("clr3"), {clr1, clr2});
  Output clr4 = ops::Relu(s.WithOpName("clr4"), clr2);
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), wht1);
  Output fetch2 = ops::Identity(s.WithOpName("fetch2"), clr4);

  GrapplerItem item;
  item.fetch = {"fetch1", "fetch2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 3);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("clr3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("clr4")->attr().at("T").type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionTest, PreserveFetches) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), input, input);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), wht1);
  Output gry1 = ops::Sqrt(s.WithOpName("gry1"), clr1);
  Output blk1 = ops::Exp(s.WithOpName("blk1"), gry1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), blk1);
  Output wht2 = ops::MatMul(s.WithOpName("wht2"), clr2, clr2);
  Output clr3 = ops::Relu(s.WithOpName("clr3"), wht2);
  Output blk2 = ops::Exp(s.WithOpName("blk2"), clr3);
  Output clr4 = ops::Relu(s.WithOpName("clr4"), blk2);

  GrapplerItem item;
  item.fetch = {"wht1", "clr2", "clr3"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("gry1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("blk1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("wht2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("clr3")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("blk2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr4")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-3);
  }
}

TEST_F(AutoMixedPrecisionTest, PreserveCPUNodes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output clr1 = ops::Relu(s.WithOpName("clr1"), input);
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), clr1, clr1);
  Output gry1 = ops::Tanh(s.WithOpName("gry1"), wht1);
  Output wht2 = ops::MatMul(s.WithOpName("wht2").WithDevice(
                                "/job:localhost/replica:0/task:0/device:CPU:0"),
                            gry1, gry1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), wht2);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr2);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("gry1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("wht2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionTest, PreserveIdentityAfterVariable) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output var1 = ops::Variable(s.WithOpName("var1"), {32, 32}, DT_FLOAT);
  Output clr1 = ops::Identity(s.WithOpName("clr1"), var1);
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), input, clr1);
  Output input2 = ops::Const(s.WithOpName("input2"), 1.f / 32, {32, 32});
  Output clr2 = ops::Identity(s.WithOpName("clr2"), input2);
  Output wht2 = ops::MatMul(s.WithOpName("wht2"), input, clr2);
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), wht1);
  Output fetch2 = ops::Identity(s.WithOpName("fetch2"), wht2);

  GrapplerItem item;
  item.fetch = {"fetch1", "fetch2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto var1_tensor =
      GenerateConstantTensor<DT_FLOAT>(TensorShape({32, 32}), 3.141593f);
  std::vector<std::pair<string, Tensor>> feed = {{"var1", var1_tensor}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 5);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("var1")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("input2")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht2")->attr().at("T").type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-3);
  }
}

TEST_F(AutoMixedPrecisionTest, FusedBatchNorm) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  // Uses NHWC data format because non-GPU execution does not support NCHW.
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {8, 56, 56, 16});
  Output weight = ops::Const(s.WithOpName("weight"), 2.f, {3, 3, 16, 16});
  Output scale = ops::Const(s.WithOpName("scale"), 3.f, {16});
  Output offset = ops::Const(s.WithOpName("offset"), 4.f, {16});
  Output mean = ops::Const(s.WithOpName("mean"), 5.f, {0});
  Output variance = ops::Const(s.WithOpName("variance"), 6.f, {0});
  Output wht1 = ops::Conv2D(s.WithOpName("wht1"), input, weight, {1, 1, 1, 1},
                            "SAME", ops::Conv2D::DataFormat("NHWC"));
  auto fbn1_op =
      ops::FusedBatchNorm(s.WithOpName("fbn1"), wht1, scale, offset, mean,
                          variance, ops::FusedBatchNorm::DataFormat("NHWC"));
  Output fbn1 = fbn1_op.y;
  Output fbn1_rs1 = fbn1_op.reserve_space_1;
  Output fbn1_rs2 = fbn1_op.reserve_space_2;
  Output bng1 = ops::FusedBatchNormGrad(
                    s.WithOpName("bng1"), fbn1, wht1, scale, fbn1_rs1, fbn1_rs2,
                    ops::FusedBatchNormGrad::DataFormat("NHWC"))
                    .x_backprop;
  Output gry1 = ops::Add(s.WithOpName("gry1"), fbn1, bng1);
  Output wht2 = ops::Conv2D(s.WithOpName("wht2"), gry1, weight, {1, 1, 1, 1},
                            "SAME", ops::Conv2D::DataFormat("NHWC"));
  Output fetch = ops::Identity(s.WithOpName("fetch"), wht2);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 3);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("fbn1")->op(), "FusedBatchNormV2");
  EXPECT_EQ(output_view.GetNode("fbn1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("fbn1")->attr().at("U").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("bng1")->op(), "FusedBatchNormGradV2");
  EXPECT_EQ(output_view.GetNode("bng1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("bng1")->attr().at("U").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("gry1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht2")->attr().at("T").type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 1e-3);
  }
}

TEST_F(AutoMixedPrecisionTest, RepeatedAndListTypeAttrs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), input, input);
  auto clr1_op = ops::IdentityN(s.WithOpName("clr1"), {wht1, wht1, wht1});
  Output gry1 =
      ops::AddN(s.WithOpName("gry1"),
                {clr1_op.output[0], clr1_op.output[1], clr1_op.output[2]});
  Output wht2 = ops::MatMul(s.WithOpName("wht2"), gry1, gry1);
  Output fetch = ops::Identity(s.WithOpName("fetch"), wht2);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  for (auto type : output_view.GetNode("clr1")->attr().at("T").list().type()) {
    EXPECT_EQ(type, DT_HALF);
  }
  EXPECT_EQ(output_view.GetNode("gry1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht2")->attr().at("T").type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionTest, ExistingCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), true, {32, 32});
  Output cst1 = ops::Cast(s.WithOpName("cst1"), input, DT_FLOAT);
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), cst1, cst1);
  Output fetch = ops::Identity(s.WithOpName("fetch"), wht1);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 1);
  EXPECT_EQ(output_view.GetNode("cst1")->attr().at("SrcT").type(), DT_BOOL);
  EXPECT_EQ(output_view.GetNode("cst1")->attr().at("DstT").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionTest, TensorArray) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto ta1 = ops::TensorArray(s.WithOpName("ta1"), 8, DT_FLOAT);
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output idx1 = ops::Const(s.WithOpName("idx1"), 1);
  Output idx2 = ops::Const(s.WithOpName("idx2"), 2);
  Output idx3 = ops::Const(s.WithOpName("idx3"), 3);
  Output ta1w1 = ops::TensorArrayWrite(s.WithOpName("ta1w1"), ta1.handle, idx1,
                                       input, ta1.flow)
                     .flow_out;
  Output wht1 = ops::MatMul(s.WithOpName("wht1"), input, input);
  Output ta1w2 = ops::TensorArrayWrite(s.WithOpName("ta1w2"), ta1.handle, idx2,
                                       wht1, ta1.flow)
                     .flow_out;
  Output ta1r1 = ops::TensorArrayRead(s.WithOpName("ta1r1"), ta1.handle, idx2,
                                      ta1w2, DT_FLOAT)
                     .value;
  Output gry1 = ops::Tanh(s.WithOpName("gry1"), ta1r1);
  Output wht2 = ops::MatMul(s.WithOpName("wht2"), gry1, gry1);
  Output ta1w3 = ops::TensorArrayWrite(s.WithOpName("ta1w3"), ta1.handle, idx3,
                                       wht2, ta1.flow)
                     .flow_out;
  Output ta1r2 = ops::TensorArrayRead(s.WithOpName("ta1r2"), ta1.handle, idx3,
                                      ta1w3, DT_FLOAT)
                     .value;
  auto ta2 = ops::TensorArray(s.WithOpName("ta2"), 8, DT_FLOAT);
  Output ta2w1 = ops::TensorArrayWrite(s.WithOpName("ta2w1"), ta2.handle, idx1,
                                       input, ta2.flow)
                     .flow_out;
  Output ta2r1 = ops::TensorArrayRead(s.WithOpName("ta2r1"), ta2.handle, idx1,
                                      ta2w1, DT_FLOAT)
                     .value;
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), ta1r2);
  Output fetch2 = ops::Identity(s.WithOpName("fetch2"), ta2r1);

  GrapplerItem item;
  item.fetch = {"fetch1", "fetch2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("ta1")->attr().at("dtype").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("ta1w1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("ta1w2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("ta1r1")->attr().at("dtype").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("gry1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("ta1w3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("ta2")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("ta2w1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("ta2r1")->attr().at("dtype").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-4);
  }
}

TEST_F(AutoMixedPrecisionTest, StackV2) {
  // TODO(benbarsdell): Add execution and numerical checks to this test
  // (difficult because there is currently no C API for creating Stack ops).
  GraphDef graph;
  AddSimpleNode("handle1", "Const", {}, &graph);
  AddSimpleNode("stack1", "StackV2", {"handle1"}, &graph);
  AddSimpleNode("input", "Placeholder", {}, &graph);
  AddSimpleNode("psh1", "StackPushV2", {"stack1", "input"}, &graph);
  AddSimpleNode("wht1", "MatMul", {"input", "input"}, &graph);
  AddSimpleNode("psh2", "StackPushV2", {"stack1", "wht1"}, &graph);
  AddSimpleNode("pop1", "StackPopV2", {"stack1"}, &graph);
  AddSimpleNode("gry1", "Tanh", {"pop1"}, &graph);
  AddSimpleNode("wht2", "MatMul", {"gry1", "gry1"}, &graph);
  AddSimpleNode("psh3", "StackPushV2", {"stack1", "wht2"}, &graph);
  AddSimpleNode("handle2", "Const", {}, &graph);
  AddSimpleNode("stack2", "StackV2", {"handle2"}, &graph);
  AddSimpleNode("psh1-2", "StackPushV2", {"stack2", "input"}, &graph);
  AddSimpleNode("pop1-2", "StackPopV2", {"stack2"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 1);
  EXPECT_EQ(output_view.GetNode("stack1")->attr().at("elem_type").type(),
            DT_HALF);
  EXPECT_EQ(output_view.GetNode("psh1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("psh2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("pop1")->attr().at("elem_type").type(),
            DT_HALF);
  EXPECT_EQ(output_view.GetNode("gry1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("wht2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("psh3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("stack2")->attr().at("elem_type").type(),
            DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("psh1-2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("pop1-2")->attr().at("elem_type").type(),
            DT_FLOAT);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
