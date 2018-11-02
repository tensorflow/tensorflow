/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status FoldSubDivMulBatchNorms(const GraphDef& input_graph_def,
                               const TransformFuncContext& context,
                               GraphDef* output_graph_def);

class FoldSubDivMulBatchNormsTest : public ::testing::Test {
 protected:
  void TestFoldSubDivMulFusedBatchNorms() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    // Create the original graph as shown below:
    // ------------------------------------------------------------
    // Input-->Conv2d-->Sub-->RealDiv-->Mul-->BiasAdd-->
    //           ^       ^       ^       ^       ^
    //         Weights  Mean  Variance Gamma    Beta
    // ------------------------------------------------------------

    // Create the Conv Op with inputs input_op and weights_op.
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

    // Create the Sub op with inputs conv_op and mean_op.
    Tensor mean_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mean_data, {10.0f, 20.0f});
    Output mean_op =
        Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));

    Output sub_op = Sub(root.WithOpName("sub_op"), conv_op, mean_op);

    // Create the realdiv op with inputs sub_op and variance_op.
    Tensor variance_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&variance_data, {0.25f, 0.5f});
    Output variance_op = Const(root.WithOpName("variance_op"),
                               Input::Initializer(variance_data));

    Output realdiv_op =
        RealDiv(root.WithOpName("realdiv_op"), sub_op, variance_op);

    // Create the mul op with inputs realdiv_op and gamma_op.
    Tensor gamma_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&gamma_data, {1.0f, 2.0f});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));

    Output mul_op = Mul(root.WithOpName("mul_op"), realdiv_op, gamma_op);

    // Create the biasadd op with inputs mul_op and beta_op.
    // Since the tensor at the output of biasadd_op needs to be verified
    // we also make this is the output op and named it appropriately as output.
    Tensor beta_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&beta_data, {0.1f, 0.6f});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));

    Output biasadd_op = BiasAdd(root.WithOpName("output"), mul_op, beta_op);

    // Create the original graph def from as above.
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    // create and run a session on original graph to get output tensor values.
    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    // Create folded graph from original graph using FoldSubDivMulBatchNorms.
    // -----------------------------------------------------
    // Input-->Conv2d------------------------->BiasAdd-->
    //           ^                                ^
    //       NewWeights                        NewBeta
    // -----------------------------------------------------
    GraphDef folded_graph_def;
    TF_ASSERT_OK(FoldSubDivMulBatchNorms(original_graph_def, {{}, {"output"}},
                                         &folded_graph_def));

    // create and run a session on folded graph to get output tensor values.
    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(folded_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    // Verify both output tensor values are same (within epsilon difference).
    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 2e-5);

    // Verify the folded graph has Sub, RealDiv and Mul nodes removed.
    for (const NodeDef& node : folded_graph_def.node()) {
      EXPECT_NE("Sub", node.op());
      EXPECT_NE("RealDiv", node.op());
      EXPECT_NE("Mul", node.op());
    }
  }
};

TEST_F(FoldSubDivMulBatchNormsTest, TestFoldSubDivMulFusedBatchNorms) {
  TestFoldSubDivMulFusedBatchNorms();
}

}  // namespace graph_transforms
}  // namespace tensorflow
