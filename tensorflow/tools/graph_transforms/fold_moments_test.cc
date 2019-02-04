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

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status FoldMoments(const GraphDef& input_graph_def,
                   const TransformFuncContext& context,
                   GraphDef* output_graph_def);

class FoldMomentsTest : public ::testing::Test {
 protected:
  void TestFoldMoments() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));
    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    // sub graph
    Tensor reduction_indices_mean_data(DT_INT32, TensorShape{2});
    test::FillValues<int>(&reduction_indices_mean_data, {1, 2});
    Output reduction_indices_mean_op =
        Const(root.WithOpName("mean/reduction_indices"),
              Input::Initializer(reduction_indices_mean_data));
    Output reduction_indices_variance_op =
        Const(root.WithOpName("variance/reduction_indices"),
              Input::Initializer(reduction_indices_mean_data));
    Output mean_op =
        Mean(root.WithOpName("mean_op"), conv_op, reduction_indices_mean_op,
             Mean::Attrs().KeepDims(true));
    Output sub_op = Sub(root.WithOpName("sub_op"), conv_op, mean_op);
    Output mul_op = Mul(root.WithOpName("mul_op"), sub_op, sub_op);
    Output variance_op =
        Mean(root.WithOpName("varianve_op"), mul_op,
             reduction_indices_variance_op, Mean::Attrs().KeepDims(true));

    // nodes that use mean and variance
    Output mul_mean_op = Mul(root.WithOpName("mul_mean_op"), mean_op, mean_op);
    Tensor epsilon_data(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&epsilon_data, {0.0001});
    Output epsilon_op =
        Const(root.WithOpName("y"), Input::Initializer(epsilon_data));
    Output add_variance_op =
        Add(root.WithOpName("add_op"), variance_op, epsilon_op);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef fused_graph_def;
    TF_ASSERT_OK(FoldMoments(original_graph_def, {{}, {}}, &fused_graph_def));

    const int original_node_size = original_graph_def.node_size();
    const int fused_node_size = fused_graph_def.node_size();

    EXPECT_EQ(original_node_size - fused_node_size, 4);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(fused_graph_def, &node_map);

    EXPECT_EQ(1, node_map.count("mean_op__moments"));
    EXPECT_EQ(1, node_map.count("mean_op_axes"));
    EXPECT_EQ(1, node_map.count("conv_op"));
    EXPECT_EQ(1, node_map.count("mul_mean_op"));
    EXPECT_EQ(1, node_map.count("add_op"));
    auto moments_node = node_map.at("mean_op__moments");
    EXPECT_EQ(DT_FLOAT, moments_node->attr().at("T").type());
    EXPECT_EQ(DT_INT32, moments_node->attr().at("Tidx").type());
    EXPECT_EQ(true, moments_node->attr().at("keep_dims").b());
    EXPECT_EQ(2, moments_node->input_size());
    EXPECT_EQ("conv_op", moments_node->input(0));
    EXPECT_EQ("mean_op_axes", moments_node->input(1));
    EXPECT_EQ("mean_op__moments:0", node_map.at("mul_mean_op")->input(0));
    EXPECT_EQ("mean_op__moments:1", node_map.at("add_op")->input(0));
  }
};

TEST_F(FoldMomentsTest, TestFoldMoments) { TestFoldMoments(); }

}  // namespace graph_transforms
}  // namespace tensorflow