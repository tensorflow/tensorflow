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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer.h"

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

using ::tensorflow::Scope;
using ::tensorflow::ops::Conv2D;
using ::tensorflow::ops::Identity;
using ::tensorflow::ops::RandomUniform;

constexpr int kBatchSize = 32;
constexpr int kWidth = 10;
constexpr int kHeight = 10;
constexpr int kDepthIn = 8;
constexpr int kKernel = 3;
constexpr int kDepthOut = 16;

Output SimpleConv2D(tensorflow::Scope* s, int input_size, int filter_size,
                    const string& padding, const string& device) {
  int batch_size = 8;
  int input_height = input_size;
  int input_width = input_size;
  int input_depth = 3;
  int filter_count = 2;
  int stride = 1;
  TensorShape input_shape({batch_size, input_height, input_width, input_depth});
  Tensor input_data(DT_FLOAT, input_shape);
  test::FillIota<float>(&input_data, 1.0f);
  Output input =
      ops::Const(s->WithOpName("Input"), Input::Initializer(input_data));

  TensorShape filter_shape(
      {filter_size, filter_size, input_depth, filter_count});
  Tensor filter_data(DT_FLOAT, filter_shape);
  test::FillIota<float>(&filter_data, 1.0f);
  Output filter =
      ops::Const(s->WithOpName("Filter"), Input::Initializer(filter_data));

  Output conv = ops::Conv2D(s->WithOpName("Conv2D").WithDevice(device), input,
                            filter, {1, stride, stride, 1}, padding);
  return conv;
}

Output SimpleConv2DBackpropInput(tensorflow::Scope* s, int input_size,
                                 int filter_size, const string& padding,
                                 bool dilated) {
  int batch_size = 128;
  int input_height = input_size;
  int input_width = input_size;
  int input_depth = 3;
  int filter_count = 2;
  int stride = 1;
  TensorShape input_sizes_shape({4});
  Tensor input_data(DT_INT32, input_sizes_shape);
  test::FillValues<int>(&input_data,
                        {batch_size, input_height, input_width, input_depth});
  Output input_sizes =
      ops::Const(s->WithOpName("InputSizes"), Input::Initializer(input_data));

  TensorShape filter_shape(
      {filter_size, filter_size, input_depth, filter_count});
  Output filter =
      ops::Variable(s->WithOpName("Filter"), filter_shape, DT_FLOAT);

  int output_height = input_height;
  int output_width = input_width;
  TensorShape output_shape(
      {batch_size, output_height, output_width, filter_count});
  Tensor output_data(DT_FLOAT, output_shape);
  test::FillIota<float>(&output_data, 1.0f);
  Output output =
      ops::Const(s->WithOpName("Output"), Input::Initializer(output_data));

  Output conv_backprop_input;
  Output input_sizes_i =
      ops::Identity(s->WithOpName("InputSizesIdentity"), input_sizes);
  ops::Conv2DBackpropInput::Attrs attrs;
  if (dilated) {
    attrs = attrs.Dilations({1, 2, 2, 1});
  }
  conv_backprop_input = ops::Conv2DBackpropInput(
      s->WithOpName("Conv2DBackpropInput"), input_sizes_i, filter, output,
      {1, stride, stride, 1}, padding, attrs);

  return conv_backprop_input;
}

class GenericLayoutOptimizerTest : public GrapplerTest {
 protected:
  void SetUp() override {
    bool gpu_available = GetNumAvailableGPUs() > 0;

    if (gpu_available) {
      virtual_cluster_ =
          absl::make_unique<SingleMachine>(/*timeout_s=*/10, 1, 1);
    } else {
      DeviceProperties cpu_device;
      cpu_device.set_type("CPU");
      cpu_device.set_frequency(1000);
      cpu_device.set_num_cores(4);
      cpu_device.set_bandwidth(32);
      cpu_device.set_l1_cache_size(32 * 1024);
      cpu_device.set_l2_cache_size(256 * 1024);
      cpu_device.set_l3_cache_size(4 * 1024 * 1024);
      cpu_device.set_memory_size(1024 * 1024);
      DeviceProperties gpu_device;
      gpu_device.set_type("GPU");
      gpu_device.mutable_environment()->insert({"architecture", "6"});
      virtual_cluster_ = absl::WrapUnique(
          new VirtualCluster({{"/CPU:0", cpu_device}, {"/GPU:1", gpu_device}}));
    }
    TF_ASSERT_OK(virtual_cluster_->Provision());
  }

  void TearDown() override { TF_ASSERT_OK(virtual_cluster_->Shutdown()); }

  std::unique_ptr<Cluster> virtual_cluster_;
};

void VerifyRegularFaninMatch(const utils::NodeView* node, int port,
                             absl::string_view fanin_name, int fanin_port) {
  ASSERT_GE(node->NumRegularFanins(), port);
  const auto& fanin = node->GetRegularFanin(port);
  EXPECT_EQ(fanin.node_view()->GetName(), fanin_name);
  EXPECT_EQ(fanin.index(), fanin_port);
}

void VerifyRegularFanoutMatch(const utils::NodeView* node, int port,
                              absl::string_view fanout_name, int fanout_port) {
  bool found = false;
  for (const auto& regular_fanout : node->GetRegularFanout(port)) {
    if (regular_fanout.node_view()->GetName() == fanout_name &&
        regular_fanout.index() == fanout_port) {
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

void VerifyDataFormatAttributeMatch(const utils::NodeView* node,
                                    absl::string_view attr_value) {
  const auto* attr = node->GetAttr("data_format");
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->s(), attr_value);
}

TEST_F(GenericLayoutOptimizerTest, OptimizeSimpleConv2DGraph) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // A simple graph contains 1 "NHWC" Conv2D node, 2 input and 1 output nodes.
  Scope scope = Scope::NewRootScope();

  auto conv2d = SimpleConv2D(&scope, 4, 2, "VALID", "");
  auto identity = Identity(scope.WithOpName("Output"), conv2d);
  GrapplerItem item;
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  // The expected optimized graph contains 2 extra sets of Transpose nodes and
  // has the Conv2D's data_format set to "NCHW".
  auto* input_transpose_node =
      graph_view.GetNode("Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0, "Input", 0);

  auto* conv2d_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv2d_node, nullptr);
  ASSERT_EQ(conv2d_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(conv2d_node, 0, input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(conv2d_node, 1, "Filter", 0);
  VerifyDataFormatAttributeMatch(conv2d_node, "NCHW");

  auto* output_transpose_node =
      graph_view.GetNode("Conv2D-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, conv2d_node->GetName(), 0);

  auto* output_node = graph_view.GetNode("Output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_transpose_node->GetName(), 0);
}

TEST_F(GenericLayoutOptimizerTest, PreserveFetch) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID", "");
  auto i = ops::Identity(s.WithOpName("i"), conv);
  GrapplerItem item;
  item.fetch.push_back("Conv2D");
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  VerifyDataFormatAttributeMatch(conv_node, "NHWC");
}

TEST_F(GenericLayoutOptimizerTest, EmptyDevice) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID", "");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  VerifyDataFormatAttributeMatch(conv_node, "NCHW");
}

TEST_F(GenericLayoutOptimizerTest, GPUDevice) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv =
      SimpleConv2D(&s, 4, 2, "VALID", "/job:w/replica:0/task:0/device:GPU:0");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  VerifyDataFormatAttributeMatch(conv_node, "NCHW");
}

TEST_F(GenericLayoutOptimizerTest, CPUDevice) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID", "/CPU:0");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  VerifyDataFormatAttributeMatch(conv_node, "NHWC");
}

TEST_F(GenericLayoutOptimizerTest, Connectivity) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  Scope scope = Scope::NewRootScope();
  auto conv = SimpleConv2D(&scope, 4, 2, "VALID", "/device:GPU:0");
  auto i1 = ops::Identity(scope.WithOpName("i1"), conv);
  auto i2 = ops::Identity(scope.WithOpName("i2"), i1);
  auto i3 = ops::Identity(scope.WithOpName("i3"), i2);
  GrapplerItem item;
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  // Make the graph not in topological order to test the handling of multi-hop
  // connectivity (here we say two nodes are connected if all nodes in the
  // middle are layout agnostic). If the graph is already in topological order,
  // the problem is easier, where layout optimizer only needs to check
  // single-hop connectivity.
  Status status;
  utils::GraphView graph_view_original(&item.graph, &status);
  const int i1_index = graph_view_original.GetNode("i1")->node_index();
  const int i2_index = graph_view_original.GetNode("i2")->node_index();
  item.graph.mutable_node()->SwapElements(i1_index, i2_index);

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* node_i2_output = graph_view.GetNode("i2");
  ASSERT_NE(node_i2_output, nullptr);
  // Layout optimizer should process i2, as it detects i2 is connected with the
  // Conv2D node two hops away. Similarly i1 is processed as well, as i1 is
  // directly connected to the Conv2D node.
  ASSERT_EQ(node_i2_output->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(node_i2_output, 0, "i1", 0);
}

TEST_F(GenericLayoutOptimizerTest, Conv2DBackpropInputNonConstInputSizes) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  Scope s = Scope::NewRootScope();
  auto conv = SimpleConv2DBackpropInput(&s, 7, 2, "SAME", /*dilated=*/false);
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv2d_backprop_node = graph_view.GetNode("Conv2DBackpropInput");
  ASSERT_NE(conv2d_backprop_node, nullptr);
  ASSERT_EQ(conv2d_backprop_node->NumRegularFanins(), 3);
  VerifyRegularFaninMatch(
      conv2d_backprop_node, 0,
      "Conv2DBackpropInput-0-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer",
      0);
  auto* input_sizes_node = graph_view.GetNode(
      "Conv2DBackpropInput-0-DataFormatVecPermuteNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(input_sizes_node, nullptr);
  EXPECT_EQ(input_sizes_node->GetOp(), "DataFormatVecPermute");
  ASSERT_EQ(input_sizes_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(input_sizes_node, 0, "InputSizesIdentity", 0);
}

TEST_F(GenericLayoutOptimizerTest, Conv2DDataFormatVecPermuteCollapse) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  Scope scope = Scope::NewRootScope().WithDevice("/device:GPU:0");
  auto conv = SimpleConv2D(&scope, 4, 2, "VALID", "/device:GPU:0");
  auto shape = ops::Shape(scope.WithOpName("shape"), conv);
  auto value = ops::Const(scope.WithOpName("value"), 0, {});
  auto fill = ops::Fill(scope.WithOpName("fill"), shape, value);
  auto i = ops::Identity(scope.WithOpName("i"), fill);
  GrapplerItem item;
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  // Graph before optimization:
  // input -> conv2d -> shape -> fill -> output
  //
  // Graph after expansion:
  // input -> T -> conv2d -> T' -> T -> shape -> D' -> D -> fill -> T' -> output
  //
  // Graph after collapsion:
  // input -> T -> conv2d -> shape -> fill -> T' -> output
  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv2d_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv2d_node, nullptr);
  ASSERT_EQ(conv2d_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(conv2d_node, 0,
                          "Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer", 0);

  auto* shape_node = graph_view.GetNode("shape");
  ASSERT_NE(shape_node, nullptr);
  ASSERT_EQ(shape_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(shape_node, 0, conv2d_node->GetName(), 0);

  auto* fill_node = graph_view.GetNode("fill");
  ASSERT_NE(fill_node, nullptr);
  ASSERT_EQ(fill_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(fill_node, 0, shape_node->GetName(), 0);
  VerifyRegularFanoutMatch(fill_node, 0,
                           "fill-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);

  auto* graph_output = graph_view.GetNode("i");
  ASSERT_NE(graph_output, nullptr);
  ASSERT_EQ(graph_output->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(graph_output, 0,
                          "fill-0-0-TransposeNCHWToNHWC-LayoutOptimizer", 0);
}

TEST_F(GenericLayoutOptimizerTest, DoNotPruneNonAddedCancellableTransposes) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GrapplerItem item;
  {
    Scope scope = Scope::NewRootScope().WithDevice("/device:GPU:0");
    auto input =
        ops::RandomUniform(scope.WithOpName("input"),
                           {kBatchSize, kHeight, kWidth, kDepthIn}, DT_FLOAT);
    // NHWC -> NCHW: {0, 3, 1, 2}
    auto input_in_transpose =
        ops::Transpose(scope.WithOpName("input_in_transpose"), input,
                       ops::Const(scope, {0, 3, 1, 2}, {4}));
    // NCHW -> NHWC: {0, 2, 3, 1}
    auto input_out_transpose = ops::Transpose(
        scope.WithOpName("input_out_transpose"), input_in_transpose,
        ops::Const(scope, {0, 2, 3, 1}, {4}));
    Tensor bias_data(DT_FLOAT, TensorShape({kDepthIn}));
    test::FillIota<float>(&bias_data, 1.0f);
    auto bias_add = ops::BiasAdd(scope.WithOpName("bias_add"),
                                 input_out_transpose, bias_data);
    auto output_in_transpose =
        ops::Transpose(scope.WithOpName("output_in_transpose"), bias_add,
                       ops::Const(scope, {0, 3, 1, 2}, {4}));
    auto output_out_transpose = ops::Transpose(
        scope.WithOpName("output_out_transpose"), output_in_transpose,
        ops::Const(scope, {0, 2, 3, 1}, {4}));
    auto output =
        ops::Identity(scope.WithOpName("output"), output_out_transpose);
    TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  }

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);

  auto* input_node = graph_view.GetNode("input");
  ASSERT_NE(input_node, nullptr);

  auto* input_in_transpose_node = graph_view.GetNode("input_in_transpose");
  ASSERT_NE(input_in_transpose_node, nullptr);
  ASSERT_EQ(input_in_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_in_transpose_node, 0, input_node->GetName(), 0);

  auto* input_out_transpose_node = graph_view.GetNode("input_out_transpose");
  ASSERT_NE(input_out_transpose_node, nullptr);
  ASSERT_EQ(input_out_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_out_transpose_node, 0,
                          input_in_transpose_node->GetName(), 0);

  auto* bias_add_in_transpose_node =
      graph_view.GetNode("bias_add-0-TransposeNHWCToNCHW-LayoutOptimizer");
  ASSERT_NE(bias_add_in_transpose_node, nullptr);
  ASSERT_EQ(bias_add_in_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(bias_add_in_transpose_node, 0,
                          input_out_transpose_node->GetName(), 0);

  auto* bias_add_node = graph_view.GetNode("bias_add");
  ASSERT_NE(bias_add_node, nullptr);
  ASSERT_EQ(bias_add_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(bias_add_node, 0,
                          bias_add_in_transpose_node->GetName(), 0);

  auto* bias_add_out_transpose_node =
      graph_view.GetNode("bias_add-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  ASSERT_NE(bias_add_out_transpose_node, nullptr);
  ASSERT_EQ(bias_add_out_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(bias_add_out_transpose_node, 0,
                          bias_add_node->GetName(), 0);

  auto* output_in_transpose_node = graph_view.GetNode("output_in_transpose");
  ASSERT_NE(output_in_transpose_node, nullptr);
  ASSERT_EQ(output_in_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_in_transpose_node, 0,
                          bias_add_out_transpose_node->GetName(), 0);

  auto* output_out_transpose_node = graph_view.GetNode("output_out_transpose");
  ASSERT_NE(output_out_transpose_node, nullptr);
  ASSERT_EQ(output_out_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_out_transpose_node, 0,
                          output_in_transpose_node->GetName(), 0);

  auto* output_node = graph_view.GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_out_transpose_node->GetName(),
                          0);
}

TEST_F(GenericLayoutOptimizerTest, CancelTransposeAroundPad) {
  using test::function::NDef;

  GenericLayoutOptimizer optimizer(RewriterConfig::AGGRESSIVE);

  const Tensor kPermuteNhwcToNchw = test::AsTensor<int32>({0, 3, 1, 2});
  const Tensor kPermuteNchwToNhwc = test::AsTensor<int32>({0, 2, 3, 1});
  const Tensor kPad = test::AsTensor<int32>({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});

  GrapplerItem item;
  item.graph = test::function::GDef({
      NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}),

      NDef("paddings", "Const", {}, {{"dtype", DT_INT32}, {"value", kPad}}),
      NDef("perm_nhwc_to_nchw", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermuteNhwcToNchw}}),
      NDef("perm_nchw_to_nhwc", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermuteNchwToNhwc}}),

      NDef("transpose_0", "Transpose", {"x", "perm_nhwc_to_nchw"},
           {{"T", DT_FLOAT}, {"Tperm", DT_INT32}}),
      NDef("pad", "Pad", {"transpose_0", "paddings"},
           {{"T", DT_FLOAT}, {"Tpaddings", DT_INT32}}),
      NDef("transpose_1", "Transpose", {"pad", "perm_nchw_to_nhwc"},
           {{"T", DT_FLOAT}, {"Tperm", DT_INT32}}),
      NDef("transpose_2", "Transpose", {"pad", "perm_nchw_to_nhwc"},
           {{"T", DT_FLOAT}, {"Tperm", DT_INT32}}),
  });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  const Tensor kPermutedPaddings =
      test::AsTensor<int32>({1, 2, 5, 6, 7, 8, 3, 4}, {4, 2});

  GraphDef expected = test::function::GDef({
      NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}),

      NDef("paddings", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermutedPaddings}}),
      NDef("perm_nhwc_to_nchw", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermuteNhwcToNchw}}),
      NDef("perm_nchw_to_nhwc", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermuteNchwToNhwc}}),

      // Transpose nodes replaced by Identity nodes.
      NDef("transpose_0", "Identity", {"x"}, {{"T", DT_FLOAT}}),
      NDef("pad", "Pad", {"transpose_0", "paddings"},
           {{"T", DT_FLOAT}, {"Tpaddings", DT_INT32}}),
      NDef("transpose_1", "Identity", {"pad"}, {{"T", DT_FLOAT}}),
      NDef("transpose_2", "Identity", {"pad"}, {{"T", DT_FLOAT}}),
  });

  CompareGraphs(expected, output);

  Tensor x = GenerateRandomTensor<DT_FLOAT>({2, 6, 6, 8});
  item.fetch = {"transpose_1", "transpose_2"};
  item.feed.emplace_back("x", x);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  ASSERT_EQ(tensors.size(), 2);
  ASSERT_EQ(tensors_expected.size(), 2);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
}

// TODO(yanzha): Add more complex Graph for test.

}  // namespace grappler
}  // namespace tensorflow
