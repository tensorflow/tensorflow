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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
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
Status FoldBatchNorms(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def);

class FoldBatchNormsTest : public ::testing::Test {
 protected:
  void TestFoldBatchNormsConv2D() {
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

    Tensor mul_values_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mul_values_data, {2.0f, 3.0f});
    Output mul_values_op = Const(root.WithOpName("mul_values"),
                                 Input::Initializer(mul_values_data));

    Output mul_op = Mul(root.WithOpName("output"), conv_op, mul_values_op);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(
        FoldBatchNorms(original_graph_def, {{}, {"output"}}, &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("Mul", node.op());
    }
  }

  void TestFoldBatchNormsMatMul() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&weights_data, {1.0f, 2.0f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output matmul_op =
        MatMul(root.WithOpName("matmul_op"), input_op, weights_op);

    Tensor mul_values_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mul_values_data, {2.0f, 3.0f});
    Output mul_values_op = Const(root.WithOpName("mul_values"),
                                 Input::Initializer(mul_values_data));

    Output mul_op = Mul(root.WithOpName("output"), matmul_op, mul_values_op);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(
        FoldBatchNorms(original_graph_def, {{}, {"output"}}, &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("Mul", node.op());
    }
  }
};

TEST_F(FoldBatchNormsTest, TestFoldBatchNormsConv2D) {
  TestFoldBatchNormsConv2D();
}
TEST_F(FoldBatchNormsTest, TestFoldBatchNormsMatMul) {
  TestFoldBatchNormsMatMul();
}

}  // namespace graph_transforms
}  // namespace tensorflow
