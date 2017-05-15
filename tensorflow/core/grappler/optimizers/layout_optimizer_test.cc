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

void AddOutputShape(Node* node, const TensorShape& shape) {
  std::vector<TensorShapeProto> output_shapes;
  TensorShapeProto shape_proto;
  shape.AsProto(&shape_proto);
  output_shapes.push_back(shape_proto);
  node->AddAttr("_output_shapes", output_shapes);
}

class LayoutOptimizerTest : public ::testing::Test {
 protected:
  Output SimpleConv(tensorflow::Scope* s, int input_size, int filter_size,
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
    AddOutputShape(input.node(), input_shape);

    TensorShape filter_shape(
        {filter_size, filter_size, input_depth, filter_count});
    Tensor filter_data(DT_FLOAT, filter_shape);
    test::FillIota<float>(&filter_data, 1.0f);
    Output filter =
        ops::Const(s->WithOpName("Filter"), Input::Initializer(filter_data));
    AddOutputShape(filter.node(), filter_shape);

    Output conv = ops::Conv2D(s->WithOpName("Conv2D"), input, filter,
                              {1, stride, stride, 1}, padding);
    AddOutputShape(conv.node(), input_shape);
    return conv;
  }
};

TEST_F(LayoutOptimizerTest, FilterSizeIsOne) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv(&s, 2, 1, "SAME");
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
  auto conv = SimpleConv(&s, 2, 1, "SAME");
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
  auto conv = SimpleConv(&s, 2, 2, "VALID");
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
  auto conv = SimpleConv(&s, 2, 2, "SAME");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  NodeMap node_map(&output);
  EXPECT_TRUE(
      node_map.GetNode("LayoutOptimizerTransposeNHWCToNCHW-Conv2D-Input"));
}

TEST_F(LayoutOptimizerTest, NotEqualSizeWithValidPadding) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv(&s, 2, 3, "VALID");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  LayoutOptimizer optimizer;
  optimizer.set_num_gpus(1);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  NodeMap node_map(&output);
  EXPECT_TRUE(
      node_map.GetNode("LayoutOptimizerTransposeNHWCToNCHW-Conv2D-Input"));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
