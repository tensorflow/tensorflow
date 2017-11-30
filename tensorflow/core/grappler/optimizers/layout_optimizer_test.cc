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

#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class LayoutOptimizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DeviceProperties device_properties;
    device_properties.set_type("GPU");
    device_properties.mutable_environment()->insert({"architecture", "6"});
    virtual_cluster_.reset(new VirtualCluster({{"/GPU:0", device_properties}}));
  }

  Output SimpleConv2D(tensorflow::Scope* s, int input_size, int filter_size,
                      const string& padding) {
    return SimpleConv2D(s, input_size, filter_size, padding, "");
  }

  Output SimpleConv2D(tensorflow::Scope* s, int input_size, int filter_size,
                      const string& padding, const string& device) {
    int batch_size = 128;
    int input_height = input_size;
    int input_width = input_size;
    int input_depth = 3;
    int filter_count = 2;
    int stride = 1;
    TensorShape input_shape(
        {batch_size, input_height, input_width, input_depth});
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
                                   int filter_size, const string& padding) {
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
    Tensor filter_data(DT_FLOAT, filter_shape);
    test::FillIota<float>(&filter_data, 1.0f);
    Output filter =
        ops::Const(s->WithOpName("Filter"), Input::Initializer(filter_data));

    int output_height = input_height;
    int output_width = input_width;
    TensorShape output_shape(
        {batch_size, output_height, output_width, filter_count});
    Tensor output_data(DT_FLOAT, output_shape);
    test::FillIota<float>(&output_data, 1.0f);
    Output output =
        ops::Const(s->WithOpName("Output"), Input::Initializer(output_data));

    Output conv_backprop_input = ops::Conv2DBackpropInput(
        s->WithOpName("Conv2DBackpropInput"), input_sizes, filter, output,
        {1, stride, stride, 1}, padding);
    TensorShape input_shape(
        {batch_size, input_height, input_width, input_depth});
    return conv_backprop_input;
  }

  Tensor GetAttrValue(const NodeDef& node) {
    Tensor tensor;
    CHECK(tensor.FromProto(node.attr().at({"value"}).tensor()));
    return tensor;
  }

  Output SimpleFusedBatchNormGrad(tensorflow::Scope* s, bool is_training) {
    int batch_size = 16;
    int input_height = 8;
    int input_width = 8;
    int input_channels = 3;
    TensorShape shape({batch_size, input_height, input_width, input_channels});
    Tensor data(DT_FLOAT, shape);
    test::FillIota<float>(&data, 1.0f);
    Output x = ops::Const(s->WithOpName("Input"), Input::Initializer(data));
    Output y_backprop =
        ops::Const(s->WithOpName("YBackprop"), Input::Initializer(data));

    TensorShape shape_vector({input_channels});
    Tensor data_vector(DT_FLOAT, shape_vector);
    test::FillIota<float>(&data_vector, 2.0f);
    Output scale =
        ops::Const(s->WithOpName("Scale"), Input::Initializer(data_vector));
    Output reserve1 =
        ops::Const(s->WithOpName("Reserve1"), Input::Initializer(data_vector));
    Output reserve2 =
        ops::Const(s->WithOpName("Reserve2"), Input::Initializer(data_vector));

    ops::FusedBatchNormGrad::Attrs attrs;
    attrs.is_training_ = is_training;
    auto output =
        ops::FusedBatchNormGrad(s->WithOpName("FusedBatchNormGrad"), y_backprop,
                                x, scale, reserve1, reserve2, attrs);
    return output.x_backprop;
  }

  std::unique_ptr<VirtualCluster> virtual_cluster_;
};

TEST_F(LayoutOptimizerTest, Conv2DBackpropInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2DBackpropInput(&s, 7, 2, "SAME");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;

  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  string input_name = AddPrefixToNodeName("Conv2DBackpropInput-InputSizes",
                                          "LayoutOptimizer", "-");
  auto input_sizes_node = node_map.GetNode(input_name);
  CHECK(input_sizes_node);
  auto conv2d_backprop_node = node_map.GetNode("Conv2DBackpropInput");
  CHECK(conv2d_backprop_node);
  EXPECT_EQ(input_name, conv2d_backprop_node->input(0));
  auto input_sizes = GetAttrValue(*input_sizes_node);
  Tensor input_sizes_expected(DT_INT32, {4});
  test::FillValues<int>(&input_sizes_expected, {128, 3, 7, 7});
  test::ExpectTensorEqual<int>(input_sizes_expected, input_sizes);
}

TEST_F(LayoutOptimizerTest, FilterSizeIsOne) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 2, 1, "SAME");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  EXPECT_FALSE(
      node_map.GetNode("LayoutOptimizerTransposeNHWCToNCHW-Conv2D-Input"));
}

TEST_F(LayoutOptimizerTest, FilterSizeNotOne) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 2, 1, "SAME");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  EXPECT_FALSE(
      node_map.GetNode("LayoutOptimizerTransposeNHWCToNCHW-Conv2D-Input"));
}

TEST_F(LayoutOptimizerTest, EqualSizeWithValidPadding) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 2, 2, "VALID");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  EXPECT_FALSE(
      node_map.GetNode("LayoutOptimizerTransposeNHWCToNCHW-Conv2D-Input"));
}

TEST_F(LayoutOptimizerTest, EqualSizeWithSamePadding) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 2, 2, "SAME");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  EXPECT_TRUE(
      node_map.GetNode("LayoutOptimizerTransposeNHWCToNCHW-Conv2D-Input-0"));
}

TEST_F(LayoutOptimizerTest, NotEqualSizeWithValidPadding) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  EXPECT_TRUE(
      node_map.GetNode("LayoutOptimizerTransposeNHWCToNCHW-Conv2D-Input-0"));
}

TEST_F(LayoutOptimizerTest, Pad) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  auto c = ops::Const(s.WithOpName("c"), {1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
  auto p = ops::Pad(s.WithOpName("p"), conv, c);
  auto o = ops::Identity(s.WithOpName("o"), p);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);

  auto pad = node_map.GetNode("p");
  EXPECT_EQ(pad->input(0), "Conv2D");

  auto pad_const = node_map.GetNode("LayoutOptimizer-p-c");
  EXPECT_TRUE(pad_const);
  EXPECT_TRUE(pad_const->attr().find("value") != pad_const->attr().end());
  Tensor tensor;
  EXPECT_TRUE(
      tensor.FromProto(pad_const->mutable_attr()->at({"value"}).tensor()));
  Tensor tensor_expected(DT_INT32, {4, 2});
  test::FillValues<int>(&tensor_expected, {1, 2, 7, 8, 3, 4, 5, 6});
  test::ExpectTensorEqual<int>(tensor_expected, tensor);
}

TEST_F(LayoutOptimizerTest, Connectivity) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  auto i1 = ops::Identity(s.WithOpName("i1"), conv);
  auto i2 = ops::Identity(s.WithOpName("i2"), i1);
  auto i3 = ops::Identity(s.WithOpName("i3"), i2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  // Make the graph not in topological order to test the handling of multi-hop
  // connectivity (here we say two nodes are connected if all nodes in the
  // middle are layout agnostic). If the graph is already in topological order,
  // the problem is easier, where layout optimizer only needs to check
  // single-hop connectivity.
  NodeMap node_map_original(&item.graph);
  auto node_i1 = node_map_original.GetNode("i1");
  auto node_i2 = node_map_original.GetNode("i2");
  node_i2->Swap(node_i1);
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map_output(&output);
  auto node_i2_output = node_map_output.GetNode("i2");
  // Layout optimizer should process i2, as it detects i2 is connected with the
  // Conv2D node two hops away. Similarly i1 is processed as well, as i1 is
  // directly connected to the Conv2D node. The two added transposes between
  // i1 and i2 should cancel each other, and as a result i2 is directly
  // connected to i1.
  EXPECT_EQ(node_i2_output->input(0), "i1");
}

TEST_F(LayoutOptimizerTest, PreserveFetch) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  auto i = ops::Identity(s.WithOpName("i"), conv);
  GrapplerItem item;
  item.fetch.push_back("Conv2D");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv_node = node_map.GetNode("Conv2D");
  EXPECT_EQ(conv_node->attr().at({"data_format"}).s(), "NHWC");
}

TEST_F(LayoutOptimizerTest, EmptyDevice) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv_node = node_map.GetNode("Conv2D");
  EXPECT_EQ(conv_node->attr().at({"data_format"}).s(), "NCHW");
}

TEST_F(LayoutOptimizerTest, GPUDevice) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv =
      SimpleConv2D(&s, 3, 2, "VALID", "/job:w/replica:0/task:0/device:gpu:0");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv_node = node_map.GetNode("Conv2D");
  EXPECT_EQ(conv_node->attr().at({"data_format"}).s(), "NCHW");
}

TEST_F(LayoutOptimizerTest, CPUDeviceLowercase) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv =
      SimpleConv2D(&s, 3, 2, "VALID", "/job:w/replica:0/task:0/device:cpu:0");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv_node = node_map.GetNode("Conv2D");
  EXPECT_EQ(conv_node->attr().at({"data_format"}).s(), "NHWC");
}

TEST_F(LayoutOptimizerTest, CPUDeviceUppercase) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID", "/CPU:0");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv_node = node_map.GetNode("Conv2D");
  EXPECT_EQ(conv_node->attr().at({"data_format"}).s(), "NHWC");
}

TEST_F(LayoutOptimizerTest, FusedBatchNormGradTrainingTrue) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x_backprop = SimpleFusedBatchNormGrad(&s, true);
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {x_backprop});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv_node = node_map.GetNode("FusedBatchNormGrad");
  EXPECT_EQ(conv_node->attr().at({"data_format"}).s(), "NCHW");
}

TEST_F(LayoutOptimizerTest, FusedBatchNormGradTrainingFalse) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x_backprop = SimpleFusedBatchNormGrad(&s, false);
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {x_backprop});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv_node = node_map.GetNode("FusedBatchNormGrad");
  EXPECT_EQ(conv_node->attr().at({"data_format"}).s(), "NHWC");
}

TEST_F(LayoutOptimizerTest, SplitDimC) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  auto c = ops::Const(s.WithOpName("c"), 3, {});
  auto split = ops::Split(s.WithOpName("split"), c, conv, 2);
  auto i = ops::Identity(s.WithOpName("i"), split[0]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto split_node = node_map.GetNode("split");
  EXPECT_EQ(split_node->input(0), "LayoutOptimizerSplitConst-split");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto split_const = node_map.GetNode("LayoutOptimizerSplitConst-split");
  EXPECT_EQ(split_const->op(), "Const");
  EXPECT_EQ(split_const->attr().at({"value"}).tensor().int_val(0), 1);
}

TEST_F(LayoutOptimizerTest, SplitDimH) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  auto c = ops::Const(s.WithOpName("c"), 1, {});
  auto split = ops::Split(s.WithOpName("split"), c, conv, 2);
  auto i = ops::Identity(s.WithOpName("i"), split[0]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto split_node = node_map.GetNode("split");
  EXPECT_EQ(split_node->input(0), "LayoutOptimizerSplitConst-split");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto split_const = node_map.GetNode("LayoutOptimizerSplitConst-split");
  EXPECT_EQ(split_const->op(), "Const");
  EXPECT_EQ(split_const->attr().at({"value"}).tensor().int_val(0), 2);
}

TEST_F(LayoutOptimizerTest, SplitDimW) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  auto c = ops::Const(s.WithOpName("c"), 2, {});
  auto split = ops::Split(s.WithOpName("split"), c, conv, 2);
  auto i = ops::Identity(s.WithOpName("i"), split[0]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto split_node = node_map.GetNode("split");
  EXPECT_EQ(split_node->input(0), "LayoutOptimizerSplitConst-split");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto split_const = node_map.GetNode("LayoutOptimizerSplitConst-split");
  EXPECT_EQ(split_const->op(), "Const");
  EXPECT_EQ(split_const->attr().at({"value"}).tensor().int_val(0), 3);
}

TEST_F(LayoutOptimizerTest, SplitDimN) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  auto c = ops::Const(s.WithOpName("c"), 0, {});
  auto split = ops::Split(s.WithOpName("split"), c, conv, 2);
  auto i = ops::Identity(s.WithOpName("i"), split[0]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto split_node = node_map.GetNode("split");
  EXPECT_EQ(split_node->input(0), "LayoutOptimizerSplitConst-split");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto split_const = node_map.GetNode("LayoutOptimizerSplitConst-split");
  EXPECT_EQ(split_const->op(), "Const");
  EXPECT_EQ(split_const->attr().at({"value"}).tensor().int_val(0), 0);
}

TEST_F(LayoutOptimizerTest, SplitNonConstDim) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 3, 2, "VALID");
  auto c = ops::Const(s.WithOpName("c"), 0, {});
  auto i1 = ops::Identity(s.WithOpName("i1"), c);
  auto split = ops::Split(s.WithOpName("split"), i1, conv, 2);
  auto i2 = ops::Identity(s.WithOpName("i"), split[0]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto split_node = node_map.GetNode("split");
  EXPECT_EQ(split_node->input(0), "i1");
  EXPECT_EQ(split_node->input(1),
            "LayoutOptimizerTransposeNCHWToNHWC-Conv2D-split");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
