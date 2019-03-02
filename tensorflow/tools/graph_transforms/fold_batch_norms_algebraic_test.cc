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
Status FoldBatchNormsAlgebraic(const GraphDef& input_graph_def,
                               const TransformFuncContext& context,
                               GraphDef* output_graph_def);
Status FoldMoments(const GraphDef& input_graph_def,
                   const TransformFuncContext& context,
                   GraphDef* output_graph_def);

class FoldBatchNormsAlgebraicTest : public ::testing::Test {
 protected:
  void TestFoldBatchNormsAlgebraic() {
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
    Tensor variance_data(DT_FLOAT, TensorShape({1, 1, 1, 2}));
    test::FillValues<float>(&variance_data, {0.2, 0.5});
    Output variance_op = Const(root.WithOpName("variance_op"),
                               Input::Initializer(variance_data));
    Tensor mean_data(DT_FLOAT, TensorShape({1, 1, 1, 2}));
    test::FillValues<float>(&mean_data, {-1.0, 2.0});
    Output mean_op =
        Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));
    // sub graph
    Tensor epsilon_data(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&epsilon_data, {0.01});
    Output epsilon_op =
        Const(root.WithOpName("y"), Input::Initializer(epsilon_data));
    Output add_variance_op =
        Add(root.WithOpName("add_op"), variance_op, epsilon_op);
    Output rsqrt_op = Rsqrt(root.WithOpName("rsqrt_op"), add_variance_op);
    Tensor gamma_data(DT_FLOAT, TensorShape({1, 1, 1, 2}));
    test::FillValues<float>(&gamma_data, {1.5, 2.5});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));
    Output mul_op = Mul(root.WithOpName("mul_op"), rsqrt_op, gamma_op);
    Output mul_1_op = Mul(root.WithOpName("mul_1_op"), conv_op, mul_op);
    Output mul_2_op = Mul(root.WithOpName("mul_2_op"), mean_op, mul_op);
    Tensor beta_data(DT_FLOAT, TensorShape({1, 1, 1, 2}));
    test::FillValues<float>(&beta_data, {0.1, 0.2});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));
    Output sub_op = Sub(root.WithOpName("sub_op"), beta_op, mul_2_op);
    Output add_output_op =
        Add(root.WithOpName("add_output_op"), mul_1_op, sub_op);

    // the node that use sub-graph's output
    Output relu_op = Relu(root.WithOpName("relu_op"), add_output_op);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef fused_graph_def;
    TF_ASSERT_OK(FoldBatchNormsAlgebraic(original_graph_def, {{}, {}},
                                         &fused_graph_def));

    const int original_node_size = original_graph_def.node_size();
    const int fused_node_size = fused_graph_def.node_size();
    EXPECT_EQ(original_node_size - fused_node_size, 7);

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(fused_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("add_output_op__InstanceNorm"));
    EXPECT_EQ(1, node_map.count("gamma_op"));
    EXPECT_EQ(1, node_map.count("beta_op"));
    EXPECT_EQ(1, node_map.count("conv_op"));
    EXPECT_EQ(1, node_map.count("relu_op"));
    auto instance_norms_node = node_map.at("add_output_op__InstanceNorm");
    EXPECT_EQ("InstanceNorm", instance_norms_node->op());
    EXPECT_EQ(5, instance_norms_node->input_size());
    EXPECT_EQ(DT_FLOAT, instance_norms_node->attr().at("T").type());
  }

  void TestFoldBatchNormsAlgebraicAndFoldMoments() {
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
    // moments sub graph
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
    Output moments_sub_op =
        Sub(root.WithOpName("moments_sub_op"), conv_op, mean_op);
    Output moments_mul_op =
        Mul(root.WithOpName("moments_mul_op"), moments_sub_op, moments_sub_op);
    Output variance_op =
        Mean(root.WithOpName("varianve_op"), moments_mul_op,
             reduction_indices_variance_op, Mean::Attrs().KeepDims(true));

    // norms sub graph
    Tensor epsilon_data(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&epsilon_data, {0.01});
    Output epsilon_op =
        Const(root.WithOpName("y"), Input::Initializer(epsilon_data));
    Output add_variance_op =
        Add(root.WithOpName("add_op"), variance_op, epsilon_op);
    Output rsqrt_op = Rsqrt(root.WithOpName("rsqrt_op"), add_variance_op);
    Tensor gamma_data(DT_FLOAT, TensorShape({1, 1, 1, 2}));
    test::FillValues<float>(&gamma_data, {1.5, 2.5});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));
    Output mul_op = Mul(root.WithOpName("mul_op"), rsqrt_op, gamma_op);
    Output mul_1_op = Mul(root.WithOpName("mul_1_op"), conv_op, mul_op);
    Output mul_2_op = Mul(root.WithOpName("mul_2_op"), mean_op, mul_op);
    Tensor beta_data(DT_FLOAT, TensorShape({1, 1, 1, 2}));
    test::FillValues<float>(&beta_data, {0.1, 0.2});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));
    Output sub_op = Sub(root.WithOpName("sub_op"), beta_op, mul_2_op);
    Output add_output_op =
        Add(root.WithOpName("add_output_op"), mul_1_op, sub_op);

    // the node that use sub-graph's output
    Output relu_op = Relu(root.WithOpName("relu_op"), add_output_op);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    // fold moments batch norms firstly
    GraphDef fused_batch_norms_graph_def;
    TF_ASSERT_OK(FoldBatchNormsAlgebraic(original_graph_def, {{}, {}},
                                         &fused_batch_norms_graph_def));
    GraphDef fused_moments_graph_def;
    TF_ASSERT_OK(FoldMoments(fused_batch_norms_graph_def, {{}, {}},
                             &fused_moments_graph_def));

    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(fused_moments_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("add_output_op__InstanceNorm"));
    EXPECT_EQ(1, node_map.count("gamma_op"));
    EXPECT_EQ(1, node_map.count("beta_op"));
    EXPECT_EQ(1, node_map.count("conv_op"));
    EXPECT_EQ(1, node_map.count("relu_op"));
    auto instance_norms_node = node_map.at("add_output_op__InstanceNorm");
    EXPECT_EQ("InstanceNorm", instance_norms_node->op());
    EXPECT_EQ(DT_FLOAT, instance_norms_node->attr().at("T").type());

    EXPECT_EQ(1, node_map.count("mean_op__moments"));
    EXPECT_EQ(1, node_map.count("mean_op_axes"));
    auto moments_node = node_map.at("mean_op__moments");
    EXPECT_EQ(DT_FLOAT, moments_node->attr().at("T").type());
    EXPECT_EQ(DT_INT32, moments_node->attr().at("Tidx").type());
    EXPECT_EQ(true, moments_node->attr().at("keep_dims").b());
    EXPECT_EQ(2, moments_node->input_size());
    EXPECT_EQ("conv_op", moments_node->input(0));
    EXPECT_EQ("mean_op_axes", moments_node->input(1));
    EXPECT_EQ(5, instance_norms_node->input_size());
    EXPECT_EQ("conv_op", instance_norms_node->input(0));
    EXPECT_EQ("gamma_op", instance_norms_node->input(1));
    EXPECT_EQ("beta_op", instance_norms_node->input(2));
    EXPECT_EQ("mean_op__moments:0", instance_norms_node->input(3));
    EXPECT_EQ("mean_op__moments:1", instance_norms_node->input(4));
  }
};

TEST_F(FoldBatchNormsAlgebraicTest, TestFoldBatchNormsAlgebraicOnly) {
  TestFoldBatchNormsAlgebraic();
}

TEST_F(FoldBatchNormsAlgebraicTest, TestFoldBatchNormsAlgebraicAndFlodMoments) {
  TestFoldBatchNormsAlgebraicAndFoldMoments();
}

}  // namespace graph_transforms
}  // namespace tensorflow