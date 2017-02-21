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
Status QuantizeWeights(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def);

class QuantizeWeightsTest : public ::testing::Test {
 protected:
  void TestQuantizeWeights() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 10}));
    test::FillValues<float>(
        &weights_data,
        {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f,
         3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f,
         0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f,
         0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("output"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef quantized_graph_def;
    TF_ASSERT_OK(QuantizeWeights(original_graph_def, {{}, {"output"}},
                                 &quantized_graph_def));

    std::unique_ptr<Session> quantized_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(quantized_session->Create(quantized_graph_def));
    std::vector<Tensor> quantized_outputs;
    TF_ASSERT_OK(
        quantized_session->Run({}, {"output"}, {}, &quantized_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], quantized_outputs[0],
                                  0.5);

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(quantized_graph_def, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("input_op"));
    const NodeDef* q_input_op = node_lookup.at("input_op");
    EXPECT_EQ(DT_FLOAT, q_input_op->attr().at("dtype").type());
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    const NodeDef* q_weights_op = node_lookup.at("weights_op");
    EXPECT_EQ("Dequantize", q_weights_op->op());
    const string& weights_const_name =
        NodeNameFromInput(q_weights_op->input(0));
    EXPECT_EQ(1, node_lookup.count(weights_const_name));
    const NodeDef* q_weights_const = node_lookup.at(weights_const_name);
    EXPECT_EQ("Const", q_weights_const->op());
    EXPECT_EQ(DT_QUINT8, q_weights_const->attr().at("dtype").type());
  }
};

TEST_F(QuantizeWeightsTest, TestQuantizeWeights) { TestQuantizeWeights(); }

}  // namespace graph_transforms
}  // namespace tensorflow
