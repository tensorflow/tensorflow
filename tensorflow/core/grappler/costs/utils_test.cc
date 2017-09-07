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

#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class UtilsTest : public ::testing::Test {
 public:
  void CreateConstOp(const string& name, std::initializer_list<int64> dims,
                     NodeDef* node) {
    Tensor tensor(DT_FLOAT, TensorShape(dims));
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      tensor.flat<float>()(i) = i / 10.0f;
    }
    TF_CHECK_OK(NodeDefBuilder(name, "Const")
                    .Attr("dtype", DT_FLOAT)
                    .Attr("value", tensor)
                    .Finalize(node));
  }

  void CreateConstSizesOp(const string& name, const std::vector<int32>& sizes,
                          NodeDef* node) {
    TensorShape shape;
    shape.AddDim(sizes.size());
    Tensor tensor(DT_INT32, shape);
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      tensor.flat<int32>()(i) = sizes[i];
    }
    TF_CHECK_OK(NodeDefBuilder(name, "Const")
                    .Attr("dtype", DT_INT32)
                    .Attr("value", tensor)
                    .Finalize(node));
  }
};

TEST_F(UtilsTest, ConvOpInfo) {
  int batch = 32;
  int rows = 7;
  int cols = 9;
  int filter_rows = 3;
  int filter_cols = 3;
  int out_rows = 7;
  int out_cols = 9;
  int in_depth = 3;
  int out_depth = 5;
  int stride = 1;

  std::unordered_map<string, const NodeDef*> name_to_node;
  GraphDef graph;
  NodeDef* input = graph.add_node();
  name_to_node["input"] = input;
  CreateConstOp("input", {batch, rows, cols, in_depth}, input);
  NodeDef* filter = graph.add_node();
  name_to_node["filter"] = filter;
  CreateConstOp("filter", {filter_rows, filter_cols, in_depth, out_depth},
                filter);
  NodeDef* output_backprop = graph.add_node();
  name_to_node["output_backprop"] = output_backprop;
  CreateConstOp("output_backprop", {batch, out_rows, out_cols, out_depth},
                output_backprop);
  NodeDef* input_sizes = graph.add_node();
  name_to_node["input_sizes"] = input;
  CreateConstSizesOp("input_sizes",
                     std::vector<int32>({batch, rows, cols, in_depth}),
                     input_sizes);
  NodeDef* filter_sizes = graph.add_node();
  name_to_node["filter_sizes"] = filter_sizes;
  CreateConstSizesOp(
      "filter_sizes",
      std::vector<int32>({filter_rows, filter_cols, in_depth, out_depth}),
      filter_sizes);

  TensorShape paddings_shape({4, 2});
  Tensor paddings_tensor(DT_INT32, paddings_shape);
  for (int64 i = 0; i < paddings_tensor.NumElements(); ++i) {
    paddings_tensor.flat<int32>()(i) = 0;
  }
  TF_CHECK_OK(NodeDefBuilder("paddings", "Const")
                  .Attr("dtype", DT_INT32)
                  .Attr("value", paddings_tensor)
                  .Finalize(graph.add_node()));

  // Now add the convolution op
  NodeDef* conv = graph.add_node();
  TF_CHECK_OK(NodeDefBuilder("conv2d", "Conv2D")
                  .Input("input", 0, DT_FLOAT)
                  .Input("filter", 0, DT_FLOAT)
                  .Attr("strides", {1, stride, stride, 1})
                  .Attr("padding", "SAME")
                  .Finalize(conv));

  NodeDef* conv_bp_in = graph.add_node();
  TF_CHECK_OK(NodeDefBuilder("conv2d_bp_in", "Conv2DBackpropInput")
                  .Input("input_sizes", 0, DT_INT32)
                  .Input("filter", 0, DT_FLOAT)
                  .Input("output_backprop", 0, DT_FLOAT)
                  .Attr("strides", {1, stride, stride, 1})
                  .Attr("padding", "SAME")
                  .Finalize(conv_bp_in));

  NodeDef* conv_bp_filter = graph.add_node();
  TF_CHECK_OK(NodeDefBuilder("conv2d_bp_filter", "Conv2DBackpropFilter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("filter_sizes", 0, DT_INT32)
                  .Input("output_backprop", 0, DT_FLOAT)
                  .Attr("strides", {1, stride, stride, 1})
                  .Attr("padding", "SAME")
                  .Finalize(conv_bp_filter));

  for (const auto& node : graph.node()) {
    if (node.name().find("conv2d") != 0) {
      continue;
    }
    std::vector<OpInfo::TensorProperties> inputs;
    inputs.resize(node.input_size());
    OpInfo info = BuildOpInfoWithoutDevice(node, name_to_node, inputs);
    if (node.name() == "conv2d") {
      EXPECT_EQ(2, info.inputs_size());
    } else if (node.name() == "conv2dbp_in") {
      EXPECT_EQ(3, info.inputs_size());
    } else if (node.name() == "conv2d_bp_filter") {
      EXPECT_EQ(3, info.inputs_size());
    }
  }
}

TEST_F(UtilsTest, TestSkipControlInput) {
  GraphDef graph;
  TF_CHECK_OK(NodeDefBuilder("constant", "Const")
                  .Attr("dtype", DT_INT32)
                  .Finalize(graph.add_node()));
  TF_CHECK_OK(NodeDefBuilder("constfold", "NoOp")
                  .ControlInput("constant")
                  .Finalize(graph.add_node()));

  std::unordered_map<string, const NodeDef*> name_to_node;
  for (const auto& node : graph.node()) {
    name_to_node[node.name()] = &node;
  }

  bool node_found = false;
  for (const auto& node : graph.node()) {
    if (node.name() == "constfold") {
      std::vector<OpInfo::TensorProperties> inputs;
      OpInfo info = BuildOpInfoWithoutDevice(node, name_to_node, inputs);
      node_found = true;
      EXPECT_EQ(0, info.inputs_size());
    }
  }
  EXPECT_TRUE(node_found);
}

}  // end namespace grappler
}  // end namespace tensorflow
