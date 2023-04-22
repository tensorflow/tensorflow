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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class LayoutOptimizerTest : public GrapplerTest {
 protected:
  void SetUp() override {
    gpu_available_ = GetNumAvailableGPUs() > 0;

    if (gpu_available_) {
      virtual_cluster_.reset(new SingleMachine(/* timeout_s = */ 10, 1, 1));
    } else {
      DeviceProperties device_properties;
      device_properties.set_type("GPU");
      device_properties.mutable_environment()->insert({"architecture", "6"});
      virtual_cluster_.reset(
          new VirtualCluster({{"/GPU:1", device_properties}}));
    }
    TF_CHECK_OK(virtual_cluster_->Provision());
  }

  void TearDown() override { TF_CHECK_OK(virtual_cluster_->Shutdown()); }

  template <typename T = float>
  Output SimpleConv2D(tensorflow::Scope* s, int input_size, int filter_size,
                      const string& padding) {
    return SimpleConv2D<T>(s, input_size, filter_size, padding, "");
  }

  template <typename T = float>
  Output SimpleConv2D(tensorflow::Scope* s, int input_size, int filter_size,
                      const string& padding, const string& device) {
    int batch_size = 8;
    int input_height = input_size;
    int input_width = input_size;
    int input_depth = 3;
    int filter_count = 2;
    int stride = 1;
    TensorShape input_shape(
        {batch_size, input_height, input_width, input_depth});
    Tensor input_data(DataTypeToEnum<T>::value, input_shape);
    test::FillIota<T>(&input_data, static_cast<T>(1));
    Output input =
        ops::Const(s->WithOpName("Input"), Input::Initializer(input_data));

    TensorShape filter_shape(
        {filter_size, filter_size, input_depth, filter_count});
    Tensor filter_data(DataTypeToEnum<T>::value, filter_shape);
    test::FillIota<T>(&filter_data, static_cast<T>(1));
    Output filter =
        ops::Const(s->WithOpName("Filter"), Input::Initializer(filter_data));

    ops::Conv2D::Attrs attrs;
    const int kExplicitPaddings[] = {0, 0, 1, 2, 3, 4, 0, 0};
    if (padding == "EXPLICIT") {
      attrs = attrs.ExplicitPaddings(kExplicitPaddings);
    }

    Output conv = ops::Conv2D(s->WithOpName("Conv2D").WithDevice(device), input,
                              filter, {1, stride, stride, 1}, padding, attrs);
    return conv;
  }

  Output SimpleConv2DBackpropInput(tensorflow::Scope* s, int input_size,
                                   int filter_size, const string& padding) {
    return SimpleConv2DBackpropInput(s, input_size, filter_size, padding, true,
                                     true);
  }

  Output SimpleConv2DBackpropInput(tensorflow::Scope* s, int input_size,
                                   int filter_size, const string& padding,
                                   bool const_input_size, bool dilated) {
    int batch_size = 128;
    int input_height = input_size;
    int input_width = input_size;
    int input_depth = 3;
    int filter_count = 2;
    int stride = 1;
    int dilation = dilated ? 2 : 1;
    int64_t padding_top = 1;
    int64_t padding_bottom = 2;
    int64_t padding_left = 3;
    int64_t padding_right = 4;
    int64_t output_height;
    int64_t output_width;
    Padding padding_enum;
    if (padding == "SAME") {
      padding_enum = SAME;
    } else if (padding == "VALID") {
      padding_enum = VALID;
    } else {
      CHECK_EQ(padding, "EXPLICIT");
      padding_enum = EXPLICIT;
    }
    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        input_height, filter_size, dilation, stride, padding_enum,
        &output_height, &padding_top, &padding_bottom));
    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        input_width, filter_size, dilation, stride, padding_enum, &output_width,
        &padding_left, &padding_right));
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

    TensorShape output_shape(
        {batch_size, output_height, output_width, filter_count});
    Tensor output_data(DT_FLOAT, output_shape);
    test::FillIota<float>(&output_data, 1.0f);
    Output output =
        ops::Const(s->WithOpName("Output"), Input::Initializer(output_data));

    Output conv_backprop_input;
    Output input_sizes_i =
        ops::Identity(s->WithOpName("InputSizesIdentity"), input_sizes);
    std::vector<int> dilations{1, dilation, dilation, 1};
    std::vector<int> explicit_paddings;
    if (padding == "EXPLICIT") {
      explicit_paddings = {0,
                           0,
                           static_cast<int>(padding_top),
                           static_cast<int>(padding_bottom),
                           static_cast<int>(padding_left),
                           static_cast<int>(padding_right),
                           0,
                           0};
    }
    auto attrs =
        ops::Conv2DBackpropInput::Attrs().Dilations(dilations).ExplicitPaddings(
            explicit_paddings);
    if (const_input_size) {
      conv_backprop_input = ops::Conv2DBackpropInput(
          s->WithOpName("Conv2DBackpropInput"), input_sizes, filter, output,
          {1, stride, stride, 1}, padding, attrs);
    } else {
      conv_backprop_input = ops::Conv2DBackpropInput(
          s->WithOpName("Conv2DBackpropInput"), input_sizes_i, filter, output,
          {1, stride, stride, 1}, padding, attrs);
    }
    return conv_backprop_input;
  }

  Tensor GetAttrValue(const NodeDef& node) {
    Tensor tensor;
    CHECK(tensor.FromProto(node.attr().at({"value"}).tensor()));
    return tensor;
  }

  TensorShape GetAttrShape(const NodeDef& node) {
    return TensorShape(node.attr().at({"shape"}).shape());
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

  std::unique_ptr<Cluster> virtual_cluster_;
  bool gpu_available_;
};

TEST_F(LayoutOptimizerTest, Conv2DBackpropInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2DBackpropInput(&s, 7, 2, "EXPLICIT");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;

  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  string input_name = "Conv2DBackpropInput-0-LayoutOptimizer";
  auto input_sizes_node = node_map.GetNode(input_name);
  CHECK(input_sizes_node);
  auto conv2d_backprop_node = node_map.GetNode("Conv2DBackpropInput");
  CHECK(conv2d_backprop_node);
  EXPECT_EQ(input_name, conv2d_backprop_node->input(0));
  auto input_sizes = GetAttrValue(*input_sizes_node);
  Tensor input_sizes_expected(DT_INT32, {4});
  test::FillValues<int>(&input_sizes_expected, {128, 3, 7, 7});
  test::ExpectTensorEqual<int>(input_sizes_expected, input_sizes);

  if (gpu_available_) {
    TensorShape filter_shape = GetAttrShape(*node_map.GetNode("Filter"));
    Tensor filter_data = GenerateRandomTensor<DT_FLOAT>(filter_shape);
    std::vector<string> fetch = {"Fetch"};
    auto tensors_expected =
        EvaluateNodes(item.graph, fetch, {{"Filter", filter_data}});
    auto tensors = EvaluateNodes(output, fetch, {{"Filter", filter_data}});
    EXPECT_EQ(1, tensors_expected.size());
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  }
}

TEST_F(LayoutOptimizerTest, Conv2DBackpropInputNonConstInputSizes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2DBackpropInput(&s, 7, 2, "SAME", false, false);
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;

  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv2d_backprop_node = node_map.GetNode("Conv2DBackpropInput");
  CHECK(conv2d_backprop_node);
  EXPECT_EQ(conv2d_backprop_node->input(0),
            "Conv2DBackpropInput-0-VecPermuteNHWCToNCHW-LayoutOptimizer");
  auto input_sizes_node = node_map.GetNode(
      "Conv2DBackpropInput-0-VecPermuteNHWCToNCHW-LayoutOptimizer");
  CHECK(input_sizes_node);
  EXPECT_EQ(input_sizes_node->input(0), "InputSizesIdentity");
  EXPECT_EQ(input_sizes_node->op(), "DataFormatVecPermute");
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
  EXPECT_TRUE(node_map.GetNode("Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer"));
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
  EXPECT_TRUE(node_map.GetNode("Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer"));
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
  EXPECT_TRUE(node_map.GetNode("Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer"));
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
  EXPECT_TRUE(node_map.GetNode("Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer"));
}

TEST_F(LayoutOptimizerTest, NotEqualSizeWithValidPadding) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  EXPECT_TRUE(node_map.GetNode("Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer"));
}

TEST_F(LayoutOptimizerTest, ExplicitPadding) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "EXPLICIT");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  EXPECT_TRUE(node_map.GetNode("Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer"));
}

TEST_F(LayoutOptimizerTest, DataTypeIsInt32) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D<int32>(&s, 4, 2, "EXPLICIT");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  EXPECT_FALSE(
      node_map.GetNode("Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer"));
}

TEST_F(LayoutOptimizerTest, Pad) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
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

  auto pad_const = node_map.GetNode("p-1-LayoutOptimizer");
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
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
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

TEST_F(LayoutOptimizerTest, ConnectivityBinaryOpWithInputScalarAnd4D) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto i1 = ops::Identity(s.WithOpName("i1"), conv);
  auto i2 = ops::Identity(s.WithOpName("i2"), i1);
  auto scalar_sub = ops::Const(s.WithOpName("scalar_sub"), 3.0f, {});
  auto sub = ops::Sub(s.WithOpName("sub"), scalar_sub, i2);
  auto i3 = ops::Identity(s.WithOpName("i3"), sub);
  auto i4 = ops::Identity(s.WithOpName("i4"), i3);
  auto i5 = ops::Identity(s.WithOpName("i5"), i4);
  auto scalar_mul = ops::Const(s.WithOpName("scalar_mul"), 3.0f, {});
  auto mul = ops::Mul(s.WithOpName("mul"), scalar_mul, i5);
  auto i6 = ops::Identity(s.WithOpName("i6"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  // Make the graph not in topological order to test the handling of multi-hop
  // connectivity (here we say two nodes are connected if all nodes in the
  // middle are layout agnostic). If the graph is already in topological order,
  // the problem is easier, where layout optimizer only needs to check
  // single-hop connectivity.
  NodeMap node_map_original(&item.graph);
  auto node_i1 = node_map_original.GetNode("i1");
  auto node_mul = node_map_original.GetNode("mul");
  node_mul->Swap(node_i1);
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map_output(&output);
  auto mul_node = node_map_output.GetNode("mul");
  EXPECT_EQ(mul_node->input(0), "scalar_mul");
  EXPECT_EQ(mul_node->input(1), "i5");
}

TEST_F(LayoutOptimizerTest, PreserveFetch) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
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
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
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
      SimpleConv2D(&s, 4, 2, "VALID", "/job:w/replica:0/task:0/device:gpu:0");
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
      SimpleConv2D(&s, 4, 2, "VALID", "/job:w/replica:0/task:0/device:cpu:0");
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
  auto conv = SimpleConv2D(&s, 4, 2, "VALID", "/CPU:0");
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
  auto conv = SimpleConv2D(&s, 5, 2, "VALID");
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
  EXPECT_EQ(split_node->input(0), "split-0-LayoutOptimizer");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto split_const = node_map.GetNode("split-0-LayoutOptimizer");
  EXPECT_EQ(split_const->op(), "Const");
  EXPECT_EQ(split_const->attr().at({"value"}).tensor().int_val(0), 1);
}

TEST_F(LayoutOptimizerTest, SplitDimH) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 6, 2, "SAME");
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
  EXPECT_EQ(split_node->input(0), "split-0-LayoutOptimizer");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto split_const = node_map.GetNode("split-0-LayoutOptimizer");
  EXPECT_EQ(split_const->op(), "Const");
  EXPECT_EQ(split_const->attr().at({"value"}).tensor().int_val(0), 2);
}

TEST_F(LayoutOptimizerTest, SplitDimW) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 5, 2, "VALID");
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
  EXPECT_EQ(split_node->input(0), "split-0-LayoutOptimizer");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto split_const = node_map.GetNode("split-0-LayoutOptimizer");
  EXPECT_EQ(split_const->op(), "Const");
  EXPECT_EQ(split_const->attr().at({"value"}).tensor().int_val(0), 3);
}

TEST_F(LayoutOptimizerTest, SplitDimN) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 5, 2, "VALID");
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
  EXPECT_EQ(split_node->input(0), "split-0-LayoutOptimizer");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto split_const = node_map.GetNode("split-0-LayoutOptimizer");
  EXPECT_EQ(split_const->op(), "Const");
  EXPECT_EQ(split_const->attr().at({"value"}).tensor().int_val(0), 0);
}

TEST_F(LayoutOptimizerTest, SplitNonConstDim) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 5, 2, "VALID");
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
  EXPECT_EQ(split_node->input(0), "split-0-DimMapNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(split_node->input(1), "Conv2D");
  auto map_node = node_map.GetNode("split-0-DimMapNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(map_node->op(), "DataFormatDimMap");
  EXPECT_EQ(map_node->input(0), "i1");
}

TEST_F(LayoutOptimizerTest, SplitSamePortToMultipleInputsOfSameNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 5, 2, "VALID");
  auto axis = ops::Const(s.WithOpName("axis"), 3);
  auto split = ops::Split(s.WithOpName("split"), axis, conv, 2);
  auto concat =
      ops::Concat(s.WithOpName("concat"), {split[1], split[1], split[1]}, axis);
  auto o = ops::Identity(s.WithOpName("o"), concat);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto concat_node = node_map.GetNode("concat");
  EXPECT_EQ(concat_node->input(0), "split:1");
  EXPECT_EQ(concat_node->input(1), "split:1");
  EXPECT_EQ(concat_node->input(2), "split:1");
  EXPECT_EQ(concat_node->input(3), "concat-3-LayoutOptimizer");
  auto concat_dim = node_map.GetNode("concat-3-LayoutOptimizer");
  EXPECT_EQ(concat_dim->attr().at({"value"}).tensor().int_val(0), 1);
}

TEST_F(LayoutOptimizerTest, ConcatDimH) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "SAME");
  auto axis = ops::Const(s.WithOpName("axis"), 1);
  auto split = ops::Split(s.WithOpName("split"), axis, conv, 2);
  auto concat = ops::Concat(s.WithOpName("concat"), {split[0], split[1]}, axis);
  auto o = ops::Identity(s.WithOpName("o"), concat);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto concat_node = node_map.GetNode("concat");
  EXPECT_EQ(concat_node->input(0), "split");
  EXPECT_EQ(concat_node->input(1), "split:1");
  EXPECT_EQ(concat_node->input(2), "concat-2-LayoutOptimizer");
  auto concat_dim = node_map.GetNode("concat-2-LayoutOptimizer");
  EXPECT_EQ(concat_dim->attr().at({"value"}).tensor().int_val(0), 2);
}

TEST_F(LayoutOptimizerTest, ConcatNonConst) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "SAME");
  auto axis = ops::Const(s.WithOpName("axis"), 1);
  auto i = ops::Identity(s.WithOpName("i"), axis);
  auto split = ops::Split(s.WithOpName("split"), axis, conv, 2);
  auto concat = ops::Concat(s.WithOpName("concat"), {split[0], split[1]}, i);
  auto o = ops::Identity(s.WithOpName("o"), concat);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto concat_node = node_map.GetNode("concat");
  EXPECT_EQ(concat_node->input(0), "split");
  EXPECT_EQ(concat_node->input(1), "split:1");
  EXPECT_EQ(concat_node->input(2), "concat-2-DimMapNHWCToNCHW-LayoutOptimizer");
  auto concat_dim =
      node_map.GetNode("concat-2-DimMapNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(concat_dim->op(), "DataFormatDimMap");
  EXPECT_EQ(concat_dim->input(0), "i");
}

TEST_F(LayoutOptimizerTest, ConcatDimW) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "SAME");
  auto axis = ops::Const(s.WithOpName("axis"), 2);
  auto split = ops::Split(s.WithOpName("split"), axis, conv, 2);
  auto concat = ops::Concat(s.WithOpName("concat"), {split[0], split[1]}, axis);
  auto o = ops::Identity(s.WithOpName("o"), concat);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto concat_node = node_map.GetNode("concat");
  EXPECT_EQ(concat_node->input(0), "split");
  EXPECT_EQ(concat_node->input(1), "split:1");
  EXPECT_EQ(concat_node->input(2), "concat-2-LayoutOptimizer");
  auto concat_dim = node_map.GetNode("concat-2-LayoutOptimizer");
  EXPECT_EQ(concat_dim->attr().at({"value"}).tensor().int_val(0), 3);
}

TEST_F(LayoutOptimizerTest, ConcatDimN) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto axis = ops::Const(s.WithOpName("axis"), 0);
  auto split = ops::Split(s.WithOpName("split"), axis, conv, 2);
  auto concat = ops::Concat(s.WithOpName("concat"), {split[0], split[1]}, axis);
  auto o = ops::Identity(s.WithOpName("o"), concat);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto concat_node = node_map.GetNode("concat");
  EXPECT_EQ(concat_node->input(0), "split");
  EXPECT_EQ(concat_node->input(1), "split:1");
  EXPECT_EQ(concat_node->input(2), "concat-2-LayoutOptimizer");
  auto concat_dim = node_map.GetNode("concat-2-LayoutOptimizer");
  EXPECT_EQ(concat_dim->attr().at({"value"}).tensor().int_val(0), 0);
}

TEST_F(LayoutOptimizerTest, ConcatDimC) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto axis = ops::Const(s.WithOpName("axis"), 3);
  auto split = ops::Split(s.WithOpName("split"), axis, conv, 2);
  auto concat = ops::Concat(s.WithOpName("concat"), {split[0], split[1]}, axis);
  auto o = ops::Identity(s.WithOpName("o"), concat);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto concat_node = node_map.GetNode("concat");
  EXPECT_EQ(concat_node->input(0), "split");
  EXPECT_EQ(concat_node->input(1), "split:1");
  EXPECT_EQ(concat_node->input(2), "concat-2-LayoutOptimizer");
  auto concat_dim = node_map.GetNode("concat-2-LayoutOptimizer");
  EXPECT_EQ(concat_dim->attr().at({"value"}).tensor().int_val(0), 1);
}

TEST_F(LayoutOptimizerTest, Sum) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto reduction_indices =
      ops::Const(s.WithOpName("reduction_indices"), {0, 1, 2}, {3});
  auto sum = ops::Sum(s.WithOpName("sum"), conv, reduction_indices);
  auto o = ops::Identity(s.WithOpName("o"), sum);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  // TODO(yaozhang): enable SumProcessor with auto-tuning. Currently disabled
  // because of the worse performance in some cases.
  /*
  NodeMap node_map(&output);
  auto sum_node = node_map.GetNode("sum");
  EXPECT_EQ(sum_node->input(0), "Conv2D");
  EXPECT_EQ(sum_node->input(1), "LayoutOptimizer-sum-reduction_indices");
  auto sum_const = node_map.GetNode("LayoutOptimizer-sum-reduction_indices");
  Tensor tensor;
  EXPECT_TRUE(
      tensor.FromProto(sum_const->mutable_attr()->at({"value"}).tensor()));
  Tensor tensor_expected(DT_INT32, {3});
  test::FillValues<int>(&tensor_expected, {0, 2, 3});
  test::ExpectTensorEqual<int>(tensor_expected, tensor);
  */
}

TEST_F(LayoutOptimizerTest, MulScalarAnd4D) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto scalar = ops::Const(s.WithOpName("scalar"), 3.0f, {});
  auto mul = ops::Mul(s.WithOpName("mul"), scalar, conv);
  auto o = ops::Identity(s.WithOpName("o"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto mul_node = node_map.GetNode("mul");
  EXPECT_EQ(mul_node->input(0), "scalar");
  EXPECT_EQ(mul_node->input(1), "Conv2D");
}

TEST_F(LayoutOptimizerTest, Mul4DAndScalar) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto scalar = ops::Const(s.WithOpName("scalar"), 3.0f, {});
  auto mul = ops::Mul(s.WithOpName("mul"), conv, scalar);
  auto o = ops::Identity(s.WithOpName("o"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto mul_node = node_map.GetNode("mul");
  EXPECT_EQ(mul_node->input(0), "Conv2D");
  EXPECT_EQ(mul_node->input(1), "scalar");
}

TEST_F(LayoutOptimizerTest, Mul4DAndUnknownRank) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto unknown_rank =
      ops::Placeholder(s.WithOpName("unknown"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape()));
  Output c = ops::Const(s.WithOpName("c"), 3.0f, {8, 2, 2, 2});
  Output mul = ops::Mul(s.WithOpName("mul"), conv, unknown_rank);
  auto o = ops::AddN(s.WithOpName("o"), {mul, c});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto mul_node = node_map.GetNode("mul");
  // Node mul should not be processed by layout optimizer, because one of its
  // inputs is of unknown rank.
  EXPECT_EQ(mul_node->input(0),
            "Conv2D-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(mul_node->input(1), "unknown");
}

TEST_F(LayoutOptimizerTest, Mul4DAnd4D) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto i = ops::Identity(s.WithOpName("i"), conv);
  auto mul = ops::Mul(s.WithOpName("mul"), conv, i);
  auto o = ops::Identity(s.WithOpName("o"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto mul_node = node_map.GetNode("mul");
  EXPECT_EQ(mul_node->input(0), "Conv2D");
  EXPECT_EQ(mul_node->input(1), "i");
}

TEST_F(LayoutOptimizerTest, Mul4DAndVector) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto vector = ops::Const(s.WithOpName("vector"), {3.0f, 7.0f}, {2});
  auto mul = ops::Mul(s.WithOpName("mul"), conv, vector);
  auto o = ops::Identity(s.WithOpName("o"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto mul_node = node_map.GetNode("mul");
  EXPECT_EQ(mul_node->input(0), "Conv2D");
  EXPECT_EQ(mul_node->input(1), "mul-1-ReshapeNHWCToNCHW-LayoutOptimizer");
  auto mul_const = node_map.GetNode("mul-1-ReshapeConst-LayoutOptimizer");
  Tensor tensor;
  EXPECT_TRUE(
      tensor.FromProto(mul_const->mutable_attr()->at({"value"}).tensor()));
  Tensor tensor_expected(DT_INT32, {4});
  test::FillValues<int>(&tensor_expected, {1, 2, 1, 1});
  test::ExpectTensorEqual<int>(tensor_expected, tensor);
}

TEST_F(LayoutOptimizerTest, MulVectorAnd4D) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto vector = ops::Const(s.WithOpName("vector"), {3.0f, 7.0f}, {2});
  auto mul = ops::Mul(s.WithOpName("mul"), vector, conv);
  auto o = ops::Identity(s.WithOpName("o"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto mul_node = node_map.GetNode("mul");
  EXPECT_EQ(mul_node->input(0), "mul-0-ReshapeNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(mul_node->input(1), "Conv2D");
  auto mul_const = node_map.GetNode("mul-0-ReshapeConst-LayoutOptimizer");
  Tensor tensor;
  EXPECT_TRUE(
      tensor.FromProto(mul_const->mutable_attr()->at({"value"}).tensor()));
  Tensor tensor_expected(DT_INT32, {4});
  test::FillValues<int>(&tensor_expected, {1, 2, 1, 1});
  test::ExpectTensorEqual<int>(tensor_expected, tensor);
}

TEST_F(LayoutOptimizerTest, SliceConst) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 5, 2, "VALID");
  auto begin = ops::Const(s.WithOpName("begin"), {0, 2, 3, 1}, {4});
  auto size = ops::Const(s.WithOpName("size"), {4, 1, 2, 4}, {4});
  auto slice = ops::Slice(s.WithOpName("slice"), conv, begin, size);
  auto o = ops::Identity(s.WithOpName("o"), slice);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto slice_node = node_map.GetNode("slice");
  EXPECT_EQ(slice_node->input(0), "Conv2D");
  EXPECT_EQ(slice_node->input(1), "slice-1-LayoutOptimizer");
  EXPECT_EQ(slice_node->input(2), "slice-2-LayoutOptimizer");

  auto begin_const = node_map.GetNode("slice-1-LayoutOptimizer");
  Tensor begin_tensor;
  EXPECT_TRUE(begin_tensor.FromProto(
      begin_const->mutable_attr()->at({"value"}).tensor()));
  Tensor begin_tensor_expected(DT_INT32, {4});
  test::FillValues<int>(&begin_tensor_expected, {0, 1, 2, 3});
  test::ExpectTensorEqual<int>(begin_tensor_expected, begin_tensor);

  auto size_const = node_map.GetNode("slice-2-LayoutOptimizer");
  Tensor size_tensor;
  EXPECT_TRUE(size_tensor.FromProto(
      size_const->mutable_attr()->at({"value"}).tensor()));
  Tensor size_tensor_expected(DT_INT32, {4});
  test::FillValues<int>(&size_tensor_expected, {4, 4, 1, 2});
  test::ExpectTensorEqual<int>(size_tensor_expected, size_tensor);
}

TEST_F(LayoutOptimizerTest, SliceNonConst) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 5, 2, "VALID");
  auto begin = ops::Const(s.WithOpName("begin"), {0, 2, 3, 1}, {4});
  auto ibegin = ops::Identity(s.WithOpName("ibegin"), begin);
  auto size = ops::Const(s.WithOpName("size"), {4, 1, 2, 4}, {4});
  auto isize = ops::Identity(s.WithOpName("isize"), size);
  auto slice = ops::Slice(s.WithOpName("slice"), conv, ibegin, isize);
  auto o = ops::Identity(s.WithOpName("o"), slice);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto slice_node = node_map.GetNode("slice");
  EXPECT_EQ(slice_node->input(0), "Conv2D");
  EXPECT_EQ(slice_node->input(1),
            "slice-1-VecPermuteNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(slice_node->input(2),
            "slice-2-VecPermuteNHWCToNCHW-LayoutOptimizer");
  auto perm1 = node_map.GetNode("slice-1-VecPermuteNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(perm1->op(), "DataFormatVecPermute");
  EXPECT_EQ(perm1->input(0), "ibegin");
  auto perm2 = node_map.GetNode("slice-2-VecPermuteNHWCToNCHW-LayoutOptimizer");
  EXPECT_EQ(perm1->op(), "DataFormatVecPermute");
  EXPECT_EQ(perm2->input(0), "isize");
}

TEST_F(LayoutOptimizerTest, DoNotApplyOptimizerTwice) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto scalar =
      ops::Const(s.WithOpName("AlreadyApplied-LayoutOptimizer"), 3.0f, {});
  auto mul = ops::Mul(s.WithOpName("mul"), scalar, scalar);
  auto o = ops::Identity(s.WithOpName("o"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
}

TEST_F(LayoutOptimizerTest, ShapeNWithInputs4DAnd4D) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto shapen = ops::ShapeN(s.WithOpName("shapen"), {conv, conv});
  auto add = ops::Add(s.WithOpName("add"), shapen[0], shapen[1]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto shapen_node = node_map.GetNode("shapen");
  EXPECT_EQ(shapen_node->input(0), "Conv2D");
  EXPECT_EQ(shapen_node->input(1), "Conv2D");
  auto add_node = node_map.GetNode("add");
  EXPECT_EQ(add_node->input(0),
            "shapen-0-0-VecPermuteNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(add_node->input(1),
            "shapen-0-1-VecPermuteNCHWToNHWC-LayoutOptimizer");
  auto vec_permute1 =
      node_map.GetNode("shapen-0-0-VecPermuteNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(vec_permute1->input(0), "shapen");
  EXPECT_EQ(vec_permute1->op(), "DataFormatVecPermute");
  auto vec_permute2 =
      node_map.GetNode("shapen-0-1-VecPermuteNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(vec_permute2->input(0), "shapen:1");
  EXPECT_EQ(vec_permute2->op(), "DataFormatVecPermute");
}

TEST_F(LayoutOptimizerTest, ShapeNWithInputsVectorAnd4D) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto vector = ops::Const(s.WithOpName("vector"), 3.0f, {7});
  auto shapen = ops::ShapeN(s.WithOpName("shapen"), {vector, conv});
  auto add = ops::Add(s.WithOpName("add"), shapen[0], shapen[1]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto shapen_node = node_map.GetNode("shapen");
  EXPECT_EQ(shapen_node->input(0), "vector");
  EXPECT_EQ(shapen_node->input(1), "Conv2D");
  auto add_node = node_map.GetNode("add");
  EXPECT_EQ(add_node->input(0), "shapen");
  EXPECT_EQ(add_node->input(1),
            "shapen-0-1-VecPermuteNCHWToNHWC-LayoutOptimizer");
  auto vec_permute =
      node_map.GetNode("shapen-0-1-VecPermuteNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(vec_permute->input(0), "shapen:1");
  EXPECT_EQ(vec_permute->op(), "DataFormatVecPermute");
}

TEST_F(LayoutOptimizerTest, ShapeNWithInputs4DAndNoNeedToTransform4D) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto tensor_4d = ops::Const(s.WithOpName("tensor_4d"), 3.0f, {1, 1, 1, 3});
  auto i1 = ops::Identity(s.WithOpName("i1"), tensor_4d);
  Output i2 = ops::Identity(s.WithOpName("i2"), i1);
  auto shapen = ops::ShapeN(s.WithOpName("shapen"), {conv, i2});
  auto add = ops::Add(s.WithOpName("add"), shapen[0], shapen[1]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto shapen_node = node_map.GetNode("shapen");
  EXPECT_EQ(shapen_node->input(0), "Conv2D");
  EXPECT_EQ(shapen_node->input(1), "i2");
}

TEST_F(LayoutOptimizerTest, Switch) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  ops::Variable ctrl(s.WithOpName("ctrl"), {}, DT_BOOL);
  auto sw = ops::Switch(s.WithOpName("switch"), conv, ctrl);
  auto i1 = ops::Identity(s.WithOpName("i1"), sw.output_true);
  auto i2 = ops::Identity(s.WithOpName("i2"), sw.output_false);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto switch_node = node_map.GetNode("switch");
  EXPECT_EQ(switch_node->input(0), "Conv2D");
  EXPECT_EQ(switch_node->input(1), "ctrl");
  auto i1_node = node_map.GetNode("i1");
  auto i2_node = node_map.GetNode("i2");
  auto trans1 = node_map.GetNode(i1_node->input(0));
  EXPECT_EQ(trans1->input(0), "switch:1");
  auto trans2 = node_map.GetNode(i2_node->input(0));
  EXPECT_EQ(trans2->input(0), "switch");
}

TEST_F(LayoutOptimizerTest, MergeBothInputsConvertible) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  Output i1 = ops::Identity(s.WithOpName("i1"), conv);
  auto merge = ops::Merge(s.WithOpName("merge"), {conv, i1});
  auto i2 = ops::Identity(s.WithOpName("i2"), merge.output);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto merge_node = node_map.GetNode("merge");
  EXPECT_EQ(merge_node->input(0), "Conv2D");
  EXPECT_EQ(merge_node->input(1), "i1");
  auto i2_node = node_map.GetNode("i2");
  EXPECT_EQ(i2_node->input(0), "merge-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  auto transpose =
      node_map.GetNode("merge-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(transpose->input(0), "merge");
}

TEST_F(LayoutOptimizerTest, MergeOneInputNotConvertible) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto tensor_4d = ops::Const(s.WithOpName("tensor_4d"), 3.0f, {1, 1, 1, 3});
  auto merge = ops::Merge(s.WithOpName("merge"), {tensor_4d, conv});
  auto i2 = ops::Identity(s.WithOpName("i2"), merge.output);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto merge_node = node_map.GetNode("merge");
  EXPECT_EQ(merge_node->input(0), "tensor_4d");
  EXPECT_EQ(merge_node->input(1),
            "Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer");
}

TEST_F(LayoutOptimizerTest, Complex) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto comp = ops::Complex(s.WithOpName("complex"), conv, conv);
  auto i = ops::Identity(s.WithOpName("i"), comp);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto merge_node = node_map.GetNode("complex");
  EXPECT_EQ(merge_node->input(0), "Conv2D");
  EXPECT_EQ(merge_node->input(1), "Conv2D");
  auto trans =
      node_map.GetNode("complex-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(trans->attr().at("T").type(), DT_COMPLEX64);
}

TEST_F(LayoutOptimizerTest, IdentityNWithInputsVectorAnd4D) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto vector = ops::Const(s.WithOpName("vector"), 3.0f, {2});
  auto identity_n = ops::IdentityN(s.WithOpName("identity_n"), {vector, conv});
  auto add = ops::Add(s.WithOpName("add"), identity_n[0], identity_n[1]);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto i = node_map.GetNode("identity_n");
  EXPECT_EQ(i->input(0), "vector");
  EXPECT_EQ(i->input(1), "Conv2D");
  auto trans =
      node_map.GetNode("identity_n-0-1-TransposeNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(trans->input(0), "identity_n:1");
  auto add_node = node_map.GetNode("add");
  EXPECT_EQ(add_node->input(0), "identity_n");
  EXPECT_EQ(add_node->input(1),
            "identity_n-0-1-TransposeNCHWToNHWC-LayoutOptimizer");
}

TEST_F(LayoutOptimizerTest, LoopNoLiveLock) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto c = ops::Const(s.WithOpName("const"), 3.0f, {8, 3, 3, 2});
  auto merge = ops::Merge(s.WithOpName("merge"), {c, c});
  auto i0 = ops::Identity(s.WithOpName("i0"), merge.output);
  ops::Variable v_ctrl(s.WithOpName("v_ctrl"), {}, DT_BOOL);
  auto sw = ops::Switch(s.WithOpName("switch"), i0, v_ctrl);
  auto next = ops::NextIteration(s.WithOpName("next"), sw.output_true);
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto mul = ops::Mul(s.WithOpName("mul"), conv, sw.output_false);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  NodeMap node_map_original(&item.graph);
  auto merge_node = node_map_original.GetNode("merge");
  // Modify the graph to create a loop
  merge_node->set_input(1, "next");
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto conv_node = node_map.GetNode("Conv2D");
  EXPECT_EQ(conv_node->input(0),
            "Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer");
  auto mul_node = node_map.GetNode("mul");
  EXPECT_EQ(mul_node->input(0),
            "Conv2D-0-0-TransposeNCHWToNHWC-LayoutOptimizer");
}

TEST_F(LayoutOptimizerTest, DevicePlacement) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID");
  auto shape = ops::Shape(s.WithOpName("s"), conv);
  auto i = ops::Identity(s.WithOpName("i"), shape);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  VirtualPlacer virtual_placer(virtual_cluster_->GetDevices());
  for (auto& node : *item.graph.mutable_node()) {
    string device = virtual_placer.get_canonical_device_name(node);
    node.set_device(device);
  }
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto vec_permute =
      node_map.GetNode("s-0-0-VecPermuteNCHWToNHWC-LayoutOptimizer");
  EXPECT_EQ(vec_permute->attr().at("_kernel").s(), "host");
}

TEST_F(LayoutOptimizerTest, PermConstWithDevice) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  const string worker0_gpu0 = "/job:w/replica:0/task:0/device:gpu:0";
  const string worker1_gpu1 = "/job:w/replica:0/task:1/device:gpu:1";
  const string worker0_node_prefix = "job_w_replica_0_task_0_device_gpu_0-";
  const string worker1_node_prefix = "job_w_replica_0_task_1_device_gpu_1-";
  const string perm_nchw2nhwc_str = "PermConstNCHWToNHWC-LayoutOptimizer";
  const string perm_nhwc2nchw_str = "PermConstNHWCToNCHW-LayoutOptimizer";
  auto conv_0 = SimpleConv2D(&s, 4, 2, "VALID", worker0_gpu0);
  auto shape_0 = ops::Shape(s.WithOpName("s"), conv_0);
  auto i_0 = ops::Identity(s.WithOpName("i"), shape_0);
  auto conv_1 = SimpleConv2D(&s, 4, 2, "VALID", worker1_gpu1);
  auto shape_1 = ops::Shape(s.WithOpName("s"), conv_1);
  auto i_1 = ops::Identity(s.WithOpName("i"), shape_1);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(virtual_cluster_.get(), item, &output);
  NodeMap node_map(&output);
  auto const_permute_0_0 =
      node_map.GetNode(worker0_node_prefix + perm_nchw2nhwc_str);
  auto const_permute_0_1 =
      node_map.GetNode(worker0_node_prefix + perm_nhwc2nchw_str);
  EXPECT_EQ(const_permute_0_0->device(), worker0_gpu0);
  EXPECT_EQ(const_permute_0_1->device(), worker0_gpu0);
  auto const_permute_1_0 =
      node_map.GetNode(worker1_node_prefix + perm_nchw2nhwc_str);
  auto const_permute_1_1 =
      node_map.GetNode(worker1_node_prefix + perm_nhwc2nchw_str);
  EXPECT_EQ(const_permute_1_0->device(), worker1_gpu1);
  EXPECT_EQ(const_permute_1_1->device(), worker1_gpu1);
  EXPECT_FALSE(node_map.GetNode(perm_nchw2nhwc_str));
  EXPECT_FALSE(node_map.GetNode(perm_nhwc2nchw_str));
}
}  // namespace
}  // namespace grappler
}  // namespace tensorflow
