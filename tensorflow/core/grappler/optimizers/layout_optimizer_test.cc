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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class LayoutOptimizerTest : public ::testing::Test {
 protected:
  Output SimpleConv2D(tensorflow::Scope* s, int input_size, int filter_size,
                      const string& padding) {
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

    Output conv = ops::Conv2D(s->WithOpName("Conv2D"), input, filter,
                              {1, stride, stride, 1}, padding);
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
};

TEST_F(LayoutOptimizerTest, Conv2DBackpropInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2DBackpropInput(&s, 7, 2, "SAME");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
