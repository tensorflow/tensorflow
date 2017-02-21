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
Status FoldOldBatchNorms(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def);

class FoldOldBatchNormsTest : public ::testing::Test {
 protected:
  void TestFoldOldBatchNorms() {
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

    Tensor mean_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mean_data, {10.0f, 20.0f});
    Output mean_op =
        Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));

    Tensor variance_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&variance_data, {0.25f, 0.5f});
    Output variance_op = Const(root.WithOpName("variance_op"),
                               Input::Initializer(variance_data));

    Tensor beta_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&beta_data, {0.1f, 0.6f});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));

    Tensor gamma_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&gamma_data, {1.0f, 2.0f});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    // This is needed because we're trying to convert over a deprecated op which
    // should only be present in older GraphDef files. Without this we see a
    // deprecation error.
    // This is justified because we're trying to test a tool that is expected to
    // run on legacy files, to help users convert over to less problematic
    // versions.
    NodeDef batch_norm_node;
    batch_norm_node.set_op("BatchNormWithGlobalNormalization");
    batch_norm_node.set_name("output");
    AddNodeInput("conv_op", &batch_norm_node);
    AddNodeInput("mean_op", &batch_norm_node);
    AddNodeInput("variance_op", &batch_norm_node);
    AddNodeInput("beta_op", &batch_norm_node);
    AddNodeInput("gamma_op", &batch_norm_node);
    SetNodeAttr("T", DT_FLOAT, &batch_norm_node);
    SetNodeAttr("variance_epsilon", 0.00001f, &batch_norm_node);
    SetNodeAttr("scale_after_normalization", false, &batch_norm_node);
    *(original_graph_def.mutable_node()->Add()) = batch_norm_node;
    original_graph_def.mutable_versions()->set_producer(8);

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(FoldOldBatchNorms(original_graph_def, {{}, {"output"}},
                                   &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("BatchNormWithGlobalNormalization", node.op());
    }
  }
};

TEST_F(FoldOldBatchNormsTest, TestFoldOldBatchNorms) {
  TestFoldOldBatchNorms();
}

}  // namespace graph_transforms
}  // namespace tensorflow
