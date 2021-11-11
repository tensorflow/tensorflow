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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM || INTEL_MKL

#include "tensorflow/core/grappler/optimizers/auto_mixed_precision.h"

#include <utility>
#include <vector>

#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/util.h"

// TODO(benbarsdell): Improve the numerical checks in these tests. The tests
// were originally written only to check the graph coloring, so the graphs do
// not have particularly realistic numerical behavior.

namespace tensorflow {
namespace grappler {
namespace {

template <DataType DTYPE>
Tensor GenerateIdentityMatrix(int64_t height, int64_t width) {
  typedef typename EnumToDataType<DTYPE>::Type T;
  Tensor tensor(DTYPE, TensorShape{height, width});
  for (int64_t i = 0; i < height; ++i) {
    for (int64_t j = 0; j < width; ++j) {
      tensor.matrix<T>()(i, j) = i == j;
    }
  }
  return tensor;
}

template <DataType DTYPE>
Tensor GenerateRandomTensorInRange(const TensorShape& shape, double minval,
                                   double maxval) {
  typedef typename EnumToDataType<DTYPE>::Type T;
  Tensor tensor(DTYPE, shape);
  for (auto i = 0; i < tensor.NumElements(); i++)
    tensor.flat<T>()(i) =
        (random::New64() % 65536 / 65536.0) * (maxval - minval) + minval;
  return tensor;
}

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

// Currently, this test suite only passes when TensorFlow passes with CUDA/HIP,
// because otherwise the optimizer will not turn clearlist nodes to float16.
// When looking at clearlist nodes, this optimizer checks if the nodes have a
// float16 GPU OpKernel, but without CUDA/HIP there are no GPU OpKernels at all.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

const std::pair<int, int> kMinGPUArch = {7, 0};

class AutoMixedPrecisionTest : public GrapplerTest {
 protected:
  void SetUp() override {
    int num_gpus = GetNumAvailableGPUs();
    // If GPUs are available, require that they all satisfy the min arch.
    gpu_available_ = (num_gpus > 0);
#if GOOGLE_CUDA
    gpu_available_ =
        gpu_available_ && (num_gpus == GetNumAvailableGPUs(kMinGPUArch));
#else  // Here we force Tensorflow to use the virtual GFX906
    gpu_available_ = false;
#endif
    if (gpu_available_) {
      virtual_cluster_.reset(new SingleMachine(/* timeout_s = */ 10, 1, 1));
    } else {
      DeviceProperties device_properties;
      device_properties.set_type("GPU");
#if GOOGLE_CUDA
      device_properties.mutable_environment()->insert({"architecture", "7"});
      device_properties.mutable_environment()->insert({"cuda", "9010"});
#else
      device_properties.mutable_environment()->insert(
          {"architecture", "gfx906"});
#endif
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

  void TestSimpleUnaryInferOp(
      double input_min, double input_max, double atol, double rtol,
      const std::function<Output(const tensorflow::Scope&, Output)>&
          test_op_factory) {
    int size = 128;
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output eye = ops::Const(s.WithOpName("eye"),
                            GenerateIdentityMatrix<DT_FLOAT>(size, size));
    Output input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT);
    Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, eye);
    Output infer1 = test_op_factory(s.WithOpName("infer1"), allow1);
    Output allow2 = ops::MatMul(s.WithOpName("allow2"), infer1, eye);
    Output fetch1 = ops::Identity(s.WithOpName("fetch1"), allow2);
    GrapplerItem item;
    item.fetch = {"fetch1"};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    auto input_tensor = GenerateRandomTensorInRange<DT_FLOAT>(
        TensorShape({size, size}), input_min, input_max);
    std::vector<std::pair<string, Tensor>> feed = {{"input", input_tensor}};
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);

    AutoMixedPrecision optimizer;
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

    VLOG(1) << output.DebugString();

    GraphView output_view(&output);
    EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(),
              DT_FLOAT);
    EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
    EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_HALF);
    EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);

    auto tensors = EvaluateNodes(output, item.fetch, feed);
    EXPECT_EQ(tensors.size(), tensors_expected.size());
    EXPECT_EQ(tensors.size(), item.fetch.size());
    for (int i = 0; i < item.fetch.size(); ++i) {
      test::ExpectClose(tensors_expected[i], tensors[i], atol, rtol);
    }
  }

  std::unique_ptr<Cluster> virtual_cluster_;
  bool gpu_available_;
};

TEST_F(AutoMixedPrecisionTest, NoOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.234f, {32});
  Output deny1 = ops::Exp(s.WithOpName("deny1"), input);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), deny1);
  Output infer1 = ops::Sqrt(s.WithOpName("infer1"), clr1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), infer1);
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
  EXPECT_EQ(output_view.GetNode("deny1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_FLOAT);
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
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), cst1, cst1);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), allow1);
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
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
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
  Output deny1 = ops::Exp(s.WithOpName("deny1"), input);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), deny1);
  Output infer1 = ops::Sqrt(s.WithOpName("infer1"), clr1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), infer1);
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), clr2, clr2);
  Output clr3 = ops::Relu(s.WithOpName("clr3"), allow1);
  Output infer2 = ops::Log(s.WithOpName("infer2"), clr3);
  Output clr4 = ops::Relu(s.WithOpName("clr4"), infer2);
  Output deny2 = ops::SparseMatMul(s.WithOpName("deny2"), clr4, clr4);
  Output clr5 = ops::Relu(s.WithOpName("clr5"), deny2);
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
  EXPECT_EQ(output_view.GetNode("deny1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("clr3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("infer2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr4")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("deny2")->attr().at("Ta").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("deny2")->attr().at("Tb").type(), DT_FLOAT);
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
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), clr1, clr1);
  auto clr3 = ops::ShapeN(s.WithOpName("clr3"), {clr1, clr2});
  Output clr4 = ops::Relu(s.WithOpName("clr4"), clr2);
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), allow1);
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
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
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
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, input);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), allow1);
  Output infer1 = ops::Sqrt(s.WithOpName("infer1"), clr1);
  Output deny1 = ops::Exp(s.WithOpName("deny1"), infer1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), deny1);
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), clr2, clr2);
  Output clr3 = ops::Relu(s.WithOpName("clr3"), allow2);
  Output deny2 = ops::Exp(s.WithOpName("deny2"), clr3);
  Output clr4 = ops::Relu(s.WithOpName("clr4"), deny2);

  GrapplerItem item;
  item.fetch = {"allow1", "clr2", "clr3"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("deny1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("clr3")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("deny2")->attr().at("T").type(), DT_FLOAT);
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
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), clr1, clr1);
  Output infer1 = ops::Tanh(s.WithOpName("infer1"), allow1);
  Output allow2 =
      ops::MatMul(s.WithOpName("allow2").WithDevice(
                      "/job:localhost/replica:0/task:0/device:CPU:0"),
                  infer1, infer1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), allow2);
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
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_FLOAT);
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
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, clr1);
  Output input2 = ops::Const(s.WithOpName("input2"), 1.f / 32, {32, 32});
  Output clr2 = ops::Identity(s.WithOpName("clr2"), input2);
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), input, clr2);
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), allow1);
  Output fetch2 = ops::Identity(s.WithOpName("fetch2"), allow2);

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
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("input2")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);

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
  Output allow1 =
      ops::Conv2D(s.WithOpName("allow1"), input, weight, {1, 1, 1, 1}, "SAME",
                  ops::Conv2D::DataFormat("NHWC"));
  auto fbn1_op =
      ops::FusedBatchNorm(s.WithOpName("fbn1"), allow1, scale, offset, mean,
                          variance, ops::FusedBatchNorm::DataFormat("NHWC"));
  Output fbn1 = fbn1_op.y;
  Output fbn1_rs1 = fbn1_op.reserve_space_1;
  Output fbn1_rs2 = fbn1_op.reserve_space_2;
  Output bng1 = ops::FusedBatchNormGrad(
                    s.WithOpName("bng1"), fbn1, allow1, scale, fbn1_rs1,
                    fbn1_rs2, ops::FusedBatchNormGrad::DataFormat("NHWC"))
                    .x_backprop;
  Output infer1 = ops::Add(s.WithOpName("infer1"), fbn1, bng1);
  Output allow2 =
      ops::Conv2D(s.WithOpName("allow2"), infer1, weight, {1, 1, 1, 1}, "SAME",
                  ops::Conv2D::DataFormat("NHWC"));
  Output fetch = ops::Identity(s.WithOpName("fetch"), allow2);

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
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("fbn1")->op(), "FusedBatchNormV2");
  EXPECT_EQ(output_view.GetNode("fbn1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("fbn1")->attr().at("U").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("bng1")->op(), "FusedBatchNormGradV2");
  EXPECT_EQ(output_view.GetNode("bng1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("bng1")->attr().at("U").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 1e-2);
  }
}

TEST_F(AutoMixedPrecisionTest, RepeatedAndListTypeAttrs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, input);
  auto clr1_op = ops::IdentityN(s.WithOpName("clr1"), {allow1, allow1, allow1});
  Output infer1 =
      ops::AddN(s.WithOpName("infer1"),
                {clr1_op.output[0], clr1_op.output[1], clr1_op.output[2]});
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), infer1, infer1);
  Output fetch = ops::Identity(s.WithOpName("fetch"), allow2);

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
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  for (auto type : output_view.GetNode("clr1")->attr().at("T").list().type()) {
    EXPECT_EQ(type, DT_HALF);
  }
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);

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
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), cst1, cst1);
  Output fetch = ops::Identity(s.WithOpName("fetch"), allow1);

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
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionTest, RecurrentEdgeColorMismatch) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output deny1 = ops::Exp(s.WithOpName("deny1"), input);
  Output ent1 =
      ops::internal::Enter(s.WithOpName("ent1"), deny1, "loop1").output;
  // Note that the second input is later replaced with "nxt1".
  Output mrg1 = ops::Merge(s.WithOpName("mrg1"), {ent1, ent1}).output;
  // For simplicity, the loop condition is constant false.
  Output con1 = ops::Const(s.WithOpName("con1"), false, {});
  Output lpc1 = ops::LoopCond(s.WithOpName("lpc1"), con1).output;
  auto swt1 = ops::Switch(s.WithOpName("swt1"), mrg1, lpc1);
  Output infer1 = ops::Sqrt(s.WithOpName("infer1"), swt1.output_true);
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), infer1, infer1);
  Output nxt1 = ops::NextIteration(s.WithOpName("nxt1"), allow1);
  Output ext1 = ops::internal::Exit(s.WithOpName("ext1"), swt1.output_false);
  Output fetch = ops::Identity(s.WithOpName("fetch"), ext1);
  // Add a second merge node from the same NextIteration node. This case arises
  // during graph optimization of some models.
  auto mrg2 = ops::Merge(s.WithOpName("mrg2"), {ent1, nxt1});

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  NodeMap node_map_original(&item.graph);
  auto merge_node = node_map_original.GetNode("mrg1");
  // Modify the graph to create a loop.
  merge_node->set_input(1, "nxt1");
  // Add a control edge to ensure the loop condition is inside the frame.
  auto const_node = node_map_original.GetNode("con1");
  const_node->add_input("^mrg1");
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  // Note that mrg1 gets painted deny because it is between deny1 and infer1.
  // This forces nxt1 and mrg2 to be painted deny as well (they would otherwise
  // be painted allow because they are clear and have a direct path to allow1).
  EXPECT_EQ(output_view.GetNode("deny1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("ent1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("mrg1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("swt1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("nxt1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("ext1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("mrg2")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionTest, TensorListSetGet) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  tensorflow::Input shape = {32, 32};
  auto tl1 = ops::TensorListReserve(s.WithOpName("tl1"), {32, 32}, 8, DT_FLOAT);
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output idx1 = ops::Const(s.WithOpName("idx1"), 1);
  Output idx2 = ops::Const(s.WithOpName("idx2"), 2);
  Output idx3 = ops::Const(s.WithOpName("idx3"), 3);
  auto tl1w1 =
      ops::TensorListSetItem(s.WithOpName("tl1w1"), tl1.handle, idx1, input);
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, input);
  auto tl1w2 =
      ops::TensorListSetItem(s.WithOpName("tl1w2"), tl1.handle, idx2, allow1);
  // Ensure that TensorListResize doesn't cause any problems.
  Output tl1rs =
      ops::TensorListResize(s.WithOpName("tl1rs"), tl1w2.output_handle, 6);
  Output tl1r1 = ops::TensorListGetItem(s.WithOpName("tl1r1"), tl1rs, idx2,
                                        shape, DT_FLOAT)
                     .item;
  Output infer1 = ops::Tanh(s.WithOpName("infer1"), tl1r1);
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), infer1, infer1);
  auto tl1w3 =
      ops::TensorListSetItem(s.WithOpName("tl1w3"), tl1.handle, idx3, allow2);
  Output tl1r2 =
      ops::TensorListGetItem(s.WithOpName("tl1r2"), tl1w3.output_handle, idx3,
                             shape, DT_FLOAT)
          .item;
  auto tl2 = ops::TensorListReserve(s.WithOpName("tl2"), shape, 8, DT_FLOAT);
  auto tl2w1 =
      ops::TensorListSetItem(s.WithOpName("tl2w1"), tl2.handle, idx1, input);
  Output tl2r1 =
      ops::TensorListGetItem(s.WithOpName("tl2r1"), tl2w1.output_handle, idx1,
                             shape, DT_FLOAT)
          .item;
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), tl1r2);
  Output fetch2 = ops::Identity(s.WithOpName("fetch2"), tl2r1);

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
  const char* type_key = "element_dtype";
  EXPECT_EQ(output_view.GetNode("tl1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1w1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1w2")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1r1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1w3")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl2")->attr().at(type_key).type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("tl2w1")->attr().at(type_key).type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("tl2r1")->attr().at(type_key).type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-4);
  }
}

TEST_F(AutoMixedPrecisionTest, TensorListPushPop) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  tensorflow::Input shape = {32, 32};
  auto tl1 = ops::EmptyTensorList(s.WithOpName("tl1"), {32, 32}, 8, DT_FLOAT);
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  auto tl1w1 =
      ops::TensorListPushBack(s.WithOpName("tl1w1"), tl1.handle, input);
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, input);
  auto tl1w2 = ops::TensorListPushBack(s.WithOpName("tl1w2"),
                                       tl1w1.output_handle, allow1);
  Output tl1r1 = ops::TensorListPopBack(s.WithOpName("tl1r1"),
                                        tl1w2.output_handle, shape, DT_FLOAT)
                     .tensor;
  Output infer1 = ops::Tanh(s.WithOpName("infer1"), tl1r1);
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), infer1, infer1);
  auto tl1w3 =
      ops::TensorListPushBack(s.WithOpName("tl1w3"), tl1.handle, allow2);
  Output tl1r2 = ops::TensorListPopBack(s.WithOpName("tl1r2"),
                                        tl1w3.output_handle, shape, DT_FLOAT)
                     .tensor;
  auto tl2 = ops::EmptyTensorList(s.WithOpName("tl2"), shape, 8, DT_FLOAT);
  auto tl2w1 =
      ops::TensorListPushBack(s.WithOpName("tl2w1"), tl2.handle, input);
  Output tl2r1 = ops::TensorListPopBack(s.WithOpName("tl2r1"),
                                        tl2w1.output_handle, shape, DT_FLOAT)
                     .tensor;
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), tl1r2);
  Output fetch2 = ops::Identity(s.WithOpName("fetch2"), tl2r1);

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
  const char* type_key = "element_dtype";
  EXPECT_EQ(output_view.GetNode("tl1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1w1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1w2")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1r1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1w3")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl2")->attr().at(type_key).type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("tl2w1")->attr().at(type_key).type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("tl2r1")->attr().at(type_key).type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-4);
  }
}

TEST_F(AutoMixedPrecisionTest, TensorListFromTensor) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  tensorflow::Input shape = {32};
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, input);
  auto tl1 = ops::TensorListFromTensor(s.WithOpName("tl1"), allow1, shape);
  Output tl1r1 = ops::TensorListStack(s.WithOpName("tl1r1"), tl1.output_handle,
                                      shape, DT_FLOAT)
                     .tensor;
  Output infer1 = ops::Tanh(s.WithOpName("infer1"), tl1r1);
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), infer1, infer1);
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), allow2);

  // This tests that a allow-painted object node (tl2) will force an unpainted
  // client node (tl2w1) to be painted allow as well. (Without the force, tl2w1
  // would remain unpainted, producing an invalid graph).
  auto tl2 = ops::TensorListFromTensor(s.WithOpName("tl2"), allow1, shape);
  auto tl2w1 =
      ops::TensorListPushBack(s.WithOpName("tl2w1"), tl2.output_handle, input);

  GrapplerItem item;
  item.fetch = {"fetch1"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  const char* type_key = "element_dtype";
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1r1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl2")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl2w1")->attr().at(type_key).type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 2e-4);
  }
}

TEST_F(AutoMixedPrecisionTest, TensorListPushBackBatchAndConcatLists) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  tensorflow::Input shape = {32, 32};
  auto tl1 = ops::EmptyTensorList(s.WithOpName("tl1"), {32, 32}, 8, DT_FLOAT);
  auto tl2 = ops::EmptyTensorList(s.WithOpName("tl2"), {32, 32}, 8, DT_FLOAT);
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, input);
  Output tl1_tl2 =
      ops::Stack(s.WithOpName("tl1_tl2"), {tl1.handle, tl2.handle});
  Output allow1_allow1 =
      ops::Stack(s.WithOpName("allow1_allow1"), {allow1, allow1});
  auto tl12w1 = ops::TensorListPushBackBatch(s.WithOpName("tl12w1"), tl1_tl2,
                                             allow1_allow1);
  OutputList tl12w1_outputs =
      ops::Split(s.WithOpName("tl12w1_outputs"), 0, tl12w1.output_handles, 2)
          .output;
  Output scalar_shape = ops::Const(s.WithOpName("scalar_shape"), 0, {0});
  Output tl12w1_output0 = ops::Reshape(s.WithOpName("tl12w1_output0"),
                                       tl12w1_outputs[0], scalar_shape);
  Output tl12w1_output1 = ops::Reshape(s.WithOpName("tl12w1_output1"),
                                       tl12w1_outputs[1], scalar_shape);
  Output tl3 = ops::TensorListConcatLists(s.WithOpName("tl3"), tl12w1_output0,
                                          tl12w1_output1, DT_FLOAT);
  Output tl3r1 =
      ops::TensorListPopBack(s.WithOpName("tl3r1"), tl3, shape, DT_FLOAT)
          .tensor;
  Output infer1 = ops::Tanh(s.WithOpName("infer1"), tl3r1);
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), infer1, infer1);
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), allow2);

  GrapplerItem item;
  item.fetch = {"fetch1"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  const char* type_key = "element_dtype";
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl2")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl3")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl3r1")->attr().at(type_key).type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-4);
  }
}

TEST_F(AutoMixedPrecisionTest, TensorListThroughFunction) {
  // This test passes a tensor list handle through a function with its own
  // Tensor List ops inside to test that the types are not changed to a
  // conflicting state.
  // A separate Tensor List cluster is added to test that it is still changed to
  // DT_HALF.
  FunctionDefLibrary function_lib;
  const Tensor kShape = test::AsTensor<int32>({32, 32});
  FunctionDef func1 = FunctionDefHelper::Define(
      "Func1", {"ihandle: variant", "x: float"},
      {"ohandle: variant", "y: float"}, {},
      {
          {{"tl1w1_handle"},
           "TensorListPushBack",
           {"ihandle", "x"},
           {{"element_dtype", DT_FLOAT}}},
          {{"shape"}, "Const", {}, {{"value", kShape}, {"dtype", DT_INT32}}},
          {{"tl1r1_handle", "tl1r1_data"},
           "TensorListPopBack",
           {"tl1w1_handle", "shape"},
           {{"element_dtype", DT_FLOAT}}},
          {{"ohandle"}, "Identity", {"tl1r1_handle"}, {{"T", DT_VARIANT}}},
          {{"y"}, "Identity", {"tl1r1_data"}, {{"T", DT_FLOAT}}},
      });
  function_lib.add_function()->Swap(&func1);

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  TF_CHECK_OK(s.graph()->AddFunctionLibrary(function_lib));
  tensorflow::Input shape = {32, 32};
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, input);
  Output infer1 = ops::Tanh(s.WithOpName("infer1"), allow1);
  auto tl1 = ops::EmptyTensorList(s.WithOpName("tl1"), {32, 32}, 8, DT_FLOAT);
  auto tl1w1 =
      ops::TensorListPushBack(s.WithOpName("tl1w1"), tl1.handle, infer1);
  auto _infer1 = tensorflow::ops::AsNodeOut(s, infer1);
  auto _tl1w1_handle = tensorflow::ops::AsNodeOut(s, tl1w1.output_handle);
  auto builder =
      tensorflow::NodeBuilder("Func1", "Func1", s.graph()->op_registry());
  tensorflow::Node* func1_op;
  TF_CHECK_OK(builder.Input(_tl1w1_handle)
                  .Input(_infer1)
                  .Finalize(s.graph(), &func1_op));
  Output func1_handle(func1_op, 0);
  Output tl1r1 = ops::TensorListPopBack(s.WithOpName("tl1r1"), func1_handle,
                                        shape, DT_FLOAT)
                     .tensor;
  auto tl2 = ops::EmptyTensorList(s.WithOpName("tl2"), {32, 32}, 8, DT_FLOAT);
  auto tl2w1 =
      ops::TensorListPushBack(s.WithOpName("tl2w1"), tl2.handle, infer1);
  Output tl2r1 = ops::TensorListPopBack(s.WithOpName("tl2r1"),
                                        tl2w1.output_handle, shape, DT_FLOAT)
                     .tensor;
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), tl1r1, tl2r1);
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), allow2);

  GrapplerItem item;
  item.fetch = {"fetch1"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  const char* type_key = "element_dtype";
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl2")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl2w1")->attr().at(type_key).type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("tl2r1")->attr().at(type_key).type(), DT_HALF);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-4);
  }
}

int GetCudaVersion(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  for (const auto& device : devices) {
    const DeviceProperties& device_properties = device.second;
    if (device_properties.type() == "GPU") {
      const auto& device_env = device_properties.environment();
      auto it = device_env.find("cuda");
      if (it != device_env.end()) {
        string cuda_version_str = it->second;
        return std::stoi(cuda_version_str);
      }
    }
  }
  return 0;
}

bool IsSupportedGPU(const Cluster& cluster) {
#ifdef GOOGLE_CUDA
  return GetCudaVersion(cluster) >= 9010;
#else
  return true;
#endif
}

TEST_F(AutoMixedPrecisionTest, BatchMatMul) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Const(s.WithOpName("input"), 1.f / 33, {64, 32, 32});
  Output allow1 = ops::BatchMatMul(s.WithOpName("allow1"), input, input);
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), allow1);

  GrapplerItem item;
  item.fetch = {"fetch1"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  if (IsSupportedGPU(*virtual_cluster_.get())) {
    EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
    EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_HALF);
  } else {
    EXPECT_EQ(output.node_size(), item.graph.node_size());
    EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_FLOAT);
  }

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 3.0e-3);
  }
}

TEST_F(AutoMixedPrecisionTest, EluOp) {
  TestSimpleUnaryInferOp(
      -5, 5, 1.0e-3, 1.0e-3,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Elu(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, ErfOp) {
  TestSimpleUnaryInferOp(
      -5, 5, 1.0e-3, -1,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Erf(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, ErfcOp) {
  TestSimpleUnaryInferOp(
      -5, 5, 1.0e-3, -1,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Erfc(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, InvOp) {
  TestSimpleUnaryInferOp(
      0.01, 10, -1, 1.0e-3,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Inv(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, LogOp) {
  TestSimpleUnaryInferOp(
      0.01, 10, 1.0e-3, 2.0e-3,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Log(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, Log1pOp) {
  TestSimpleUnaryInferOp(
      -0.99, 9, 1.0e-3, 5.0e-3,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Log1p(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, LogSoftmaxOp) {
  TestSimpleUnaryInferOp(
      -8, 8, -1, 1.0e-2,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::LogSoftmax(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, ReciprocalOp) {
  TestSimpleUnaryInferOp(
      0.01, 10, -1, 1.0e-3,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Reciprocal(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, SigmoidOp) {
  TestSimpleUnaryInferOp(
      -5, 5, 1.0e-3, -1,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Sigmoid(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, SoftmaxOp) {
  TestSimpleUnaryInferOp(
      -8, 8, 2.0e-3, -1,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Softmax(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, SoftplusOp) {
  TestSimpleUnaryInferOp(
      -5, 5, 1.0e-3, 1.0e-3,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Softplus(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, SqrtOp) {
  TestSimpleUnaryInferOp(
      0, 10, 1.0e-3, 1.0e-3,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Sqrt(scope, input);
      });
}

TEST_F(AutoMixedPrecisionTest, TanhOp) {
  TestSimpleUnaryInferOp(
      -5, 5, 1.0e-3, -1,
      [](const tensorflow::Scope& scope, Output input) -> Output {
        return ops::Tanh(scope, input);
      });
}

class AutoMixedPrecisionCpuTest : public GrapplerTest {
 protected:
  void SetUp() override {
    virtual_cluster_.reset(new SingleMachine(/* timeout_s = */ 10, 1, 0));
    TF_CHECK_OK(virtual_cluster_->Provision());
  }
  void TearDown() override { TF_CHECK_OK(virtual_cluster_->Shutdown()); }

  std::unique_ptr<Cluster> virtual_cluster_;
};

TEST_F(AutoMixedPrecisionCpuTest, Simple) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output deny1 = ops::Exp(s.WithOpName("deny1"), input);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), deny1);
  Output infer1 = ops::Sqrt(s.WithOpName("infer1"), clr1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), infer1);
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), clr2, clr2);
  Output clr3 = ops::Relu(s.WithOpName("clr3"), allow1);
  Output infer2 = ops::Log(s.WithOpName("infer2"), clr3);
  Output clr4 = ops::Relu(s.WithOpName("clr4"), infer2);
  Output deny2 = ops::SparseMatMul(s.WithOpName("deny2"), clr4, clr4);
  Output clr5 = ops::Relu(s.WithOpName("clr5"), deny2);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr5);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer{AutoMixedPrecisionMode::CPU};
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  const int expected_cast_ops = 9;
  EXPECT_EQ(output.node_size(), item.graph.node_size() + expected_cast_ops);

  GraphView output_view(&output);
  // Matmul is a FP32 op now
  auto matmul_op = output_view.GetNode("allow1");
  EXPECT_EQ(matmul_op->attr().at("T").type(), DT_FLOAT);
  for (auto edge : output_view.GetFaninEdges(*matmul_op, false)) {
    EXPECT_EQ(edge.src.node->op(), "Cast");
    EXPECT_EQ(edge.src.node->attr().at("SrcT").type(), DT_HALF);
    EXPECT_EQ(edge.src.node->attr().at("DstT").type(), DT_FLOAT);
  }
  for (auto edge : output_view.GetFanoutEdges(*matmul_op, false)) {
    EXPECT_EQ(edge.dst.node->op(), "Cast");
    EXPECT_EQ(edge.dst.node->attr().at("SrcT").type(), DT_FLOAT);
    EXPECT_EQ(edge.dst.node->attr().at("DstT").type(), DT_HALF);
  }
}

TEST_F(AutoMixedPrecisionCpuTest, MixedFanout) {
  // Test when an FP16 allowed node has a mixed fanout of FP16 allowed node and
  // FP32 node.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  Output input1 = ops::Const(s.WithOpName("input1"), 1.f / 32, {32, 32});
  Output input2 = ops::Const(s.WithOpName("input2"), 2.f / 32, {32, 32});
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input1, input2);
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), allow1, input2);
  Output deny = ops::Exp(s.WithOpName("deny"), allow1);
  Output infer = ops::Add(s.WithOpName("infer"), deny, allow2);
  Output fetch = ops::Identity(s.WithOpName("fetch"), infer);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer{AutoMixedPrecisionMode::CPU};
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  const int expected_cast_ops = 10;
  EXPECT_EQ(output.node_size(), item.graph.node_size() + expected_cast_ops);

  GraphView output_view(&output);
  auto allow1_op = output_view.GetNode("allow1");
  for (auto edge : output_view.GetFaninEdges(*allow1_op, false)) {
    EXPECT_EQ(edge.src.node->op(), "Cast");
    EXPECT_EQ(edge.src.node->attr().at("SrcT").type(), DT_HALF);
    EXPECT_EQ(edge.src.node->attr().at("DstT").type(), DT_FLOAT);
  }
  for (auto edge : output_view.GetFanoutEdges(*allow1_op, false)) {
    EXPECT_EQ(edge.dst.node->op(), "Cast");
    EXPECT_EQ(edge.dst.node->attr().at("SrcT").type(), DT_FLOAT);
    EXPECT_EQ(edge.dst.node->attr().at("DstT").type(), DT_HALF);
  }
  auto deny_op = output_view.GetNode("deny");
  for (auto edge : output_view.GetFaninEdges(*deny_op, false)) {
    EXPECT_EQ(edge.src.node->op(), "Cast");
    EXPECT_EQ(edge.src.node->attr().at("SrcT").type(), DT_HALF);
    EXPECT_EQ(edge.src.node->attr().at("DstT").type(), DT_FLOAT);
  }
  for (auto edge : output_view.GetFanoutEdges(*deny_op, false)) {
    EXPECT_NE(edge.dst.node->op(), "Cast");
  }
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if INTEL_MKL

class AutoMixedPrecisionMklTest : public GrapplerTest {
 protected:
  void SetUp() override {
    virtual_cluster_.reset(new SingleMachine(/* timeout_s = */ 10, 1, 0));
    TF_CHECK_OK(virtual_cluster_->Provision());
  }
  void TearDown() override { TF_CHECK_OK(virtual_cluster_->Shutdown()); }

  std::unique_ptr<Cluster> virtual_cluster_;
};

TEST_F(AutoMixedPrecisionMklTest, AlreadyBf16) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  Output input = ops::Const(s.WithOpName("input"), 1.f, {32, 32});
  Output cst1 = ops::Cast(s.WithOpName("cst1"), input, DT_BFLOAT16);
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), cst1, cst1);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), allow1);
  Output cst2 = ops::Cast(s.WithOpName("cst2"), clr1, DT_FLOAT);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), cst2);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr2);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer{AutoMixedPrecisionMode::MKL};
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));
  VLOG(1) << output.DebugString();

  VerifyGraphsEquivalent(item.graph, output, __FUNCTION__);
  GraphView output_view(&output);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("cst1")->attr().at("DstT").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("cst2")->attr().at("SrcT").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("cst2")->attr().at("DstT").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
  }
}

TEST_F(AutoMixedPrecisionMklTest, Simple) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output deny1 = ops::Exp(s.WithOpName("deny1"), input);
  Output clr1 = ops::Relu(s.WithOpName("clr1"), deny1);
  Output infer1 = ops::Sqrt(s.WithOpName("infer1"), clr1);
  Output clr2 = ops::Relu(s.WithOpName("clr2"), infer1);
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), clr2, clr2);
  Output clr3 = ops::Relu(s.WithOpName("clr3"), allow1);
  Output deny2 = ops::Log(s.WithOpName("deny2"), clr3);
  Output clr4 = ops::Relu(s.WithOpName("clr4"), deny2);
  Output deny3 = ops::SparseMatMul(s.WithOpName("deny3"), clr4, clr4);
  Output clr5 = ops::Relu(s.WithOpName("clr5"), deny3);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr5);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer{AutoMixedPrecisionMode::MKL};
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("input")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("deny1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr2")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("clr3")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("deny2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr4")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("deny3")->attr().at("Ta").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("deny3")->attr().at("Tb").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr5")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 5e-4);
  }
}

TEST_F(AutoMixedPrecisionMklTest, TensorListSetGet) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  tensorflow::Input shape = {32, 32};
  auto tl1 = ops::TensorListReserve(s.WithOpName("tl1"), {32, 32}, 8, DT_FLOAT);
  Output input = ops::Const(s.WithOpName("input"), 1.f / 32, {32, 32});
  Output idx1 = ops::Const(s.WithOpName("idx1"), 1);
  Output idx2 = ops::Const(s.WithOpName("idx2"), 2);
  Output idx3 = ops::Const(s.WithOpName("idx3"), 3);
  auto tl1w1 =
      ops::TensorListSetItem(s.WithOpName("tl1w1"), tl1.handle, idx1, input);
  Output allow1 = ops::MatMul(s.WithOpName("allow1"), input, input);
  auto tl1w2 =
      ops::TensorListSetItem(s.WithOpName("tl1w2"), tl1.handle, idx2, allow1);
  // Ensure that TensorListResize doesn't cause any problems.
  Output tl1rs =
      ops::TensorListResize(s.WithOpName("tl1rs"), tl1w2.output_handle, 6);
  Output tl1r1 = ops::TensorListGetItem(s.WithOpName("tl1r1"), tl1rs, idx2,
                                        shape, DT_FLOAT)
                     .item;
  Output infer1 = ops::Mul(s.WithOpName("infer1"), tl1r1, tl1r1);
  Output allow2 = ops::MatMul(s.WithOpName("allow2"), infer1, infer1);
  auto tl1w3 =
      ops::TensorListSetItem(s.WithOpName("tl1w3"), tl1.handle, idx3, allow2);
  Output tl1r2 =
      ops::TensorListGetItem(s.WithOpName("tl1r2"), tl1w3.output_handle, idx3,
                             shape, DT_FLOAT)
          .item;
  auto tl2 = ops::TensorListReserve(s.WithOpName("tl2"), shape, 8, DT_FLOAT);
  auto tl2w1 =
      ops::TensorListSetItem(s.WithOpName("tl2w1"), tl2.handle, idx1, input);
  Output tl2r1 =
      ops::TensorListGetItem(s.WithOpName("tl2r1"), tl2w1.output_handle, idx1,
                             shape, DT_FLOAT)
          .item;
  Output fetch1 = ops::Identity(s.WithOpName("fetch1"), tl1r2);
  Output fetch2 = ops::Identity(s.WithOpName("fetch2"), tl2r1);

  GrapplerItem item;
  item.fetch = {"fetch1", "fetch2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer{AutoMixedPrecisionMode::MKL};
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 2);
  const char* type_key = "element_dtype";
  EXPECT_EQ(output_view.GetNode("tl1")->attr().at(type_key).type(),
            DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("tl1w1")->attr().at(type_key).type(),
            DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("allow1")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("tl1w2")->attr().at(type_key).type(),
            DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("tl1r1")->attr().at(type_key).type(),
            DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("infer1")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("allow2")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("tl1w3")->attr().at(type_key).type(),
            DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("tl2")->attr().at(type_key).type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("tl2w1")->attr().at(type_key).type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("tl2r1")->attr().at(type_key).type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 1e-2);
  }
}

TEST_F(AutoMixedPrecisionMklTest, InferFollowUpStreamAllow) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Test only applicable to MKL auto-mixed precision.";
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  Output input1 = ops::Const(s.WithOpName("input1"), 1.f / 32, {8, 56, 56, 16});
  Output weight = ops::Const(s.WithOpName("weight"), 2.f, {3, 3, 16, 16});
  Output allow =
      ops::Conv2D(s.WithOpName("allow"), input1, weight, {1, 1, 1, 1}, "SAME",
                  ops::Conv2D::DataFormat("NHWC"));
  Output input2 = ops::Const(s.WithOpName("input2"), 1.f / 32, {16});
  Output infer = ops::BiasAdd(s.WithOpName("infer"), allow, input2);
  Output clr = ops::Relu(s.WithOpName("clr"), infer);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer{AutoMixedPrecisionMode::MKL};
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size() + 4);
  EXPECT_EQ(output_view.GetNode("input1")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("weight")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("input2")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("allow")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("infer")->attr().at("T").type(), DT_BFLOAT16);
  EXPECT_EQ(output_view.GetNode("clr")->attr().at("T").type(), DT_BFLOAT16);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i], -1, 1e-2);
  }
}

TEST_F(AutoMixedPrecisionMklTest, InferFollowUpStreamDeny) {
  if (!IsMKLEnabled())
    GTEST_SKIP() << "Test only applicable to MKL auto-mixed precision.";
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  Output input1 = ops::Const(s.WithOpName("input1"), 1.f / 32, {8, 56, 56, 16});
  Output input2 = ops::Const(s.WithOpName("input2"), 1.f, {16});
  Output input3 = ops::Const(s.WithOpName("input3"), 1.f / 32, {16});
  Output deny = ops::Pow(s.WithOpName("deny"), input1, input2);
  Output infer = ops::BiasAdd(s.WithOpName("infer"), deny, input3);
  Output clr = ops::Relu(s.WithOpName("clr"), infer);
  Output fetch = ops::Identity(s.WithOpName("fetch"), clr);

  GrapplerItem item;
  item.fetch = {"fetch"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  AutoMixedPrecision optimizer{AutoMixedPrecisionMode::MKL};
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  EXPECT_EQ(output_view.GetNode("input1")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("input2")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("input3")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("deny")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("infer")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("clr")->attr().at("T").type(), DT_FLOAT);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectClose(tensors_expected[i], tensors[i]);
  }
}
#endif  // INTEL_MKL

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM || INTEL_MKL
