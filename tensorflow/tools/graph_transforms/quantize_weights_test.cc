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
  void BuildGraphDef(const TensorShape& input_shape,
                     std::initializer_list<float> input_values,
                     const TensorShape& weight_shape,
                     std::initializer_list<float> weight_values,
                     GraphDef* original_graph_def) {
    auto root = tensorflow::Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, input_shape);
    test::FillValues<float>(&input_data, input_values);
    Output input_op =
        ops::Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, weight_shape);
    test::FillValues<float>(&weights_data, weight_values);
    Output weights_op = ops::Const(root.WithOpName("weights_op"),
                                   Input::Initializer(weights_data));

    Output conv_op = ops::Conv2D(root.WithOpName("output"), input_op,
                                 weights_op, {1, 1, 1, 1}, "VALID");

    TF_ASSERT_OK(root.ToGraphDef(original_graph_def));
  }

  void TestQuantizeWeights() {
    GraphDef original_graph_def;
    BuildGraphDef({1, 1, 6, 2},
                  {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                   -5.0f, -3.0f, -6.0f},
                  {1, 2, 2, 10},
                  {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f,
                   3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f,
                   0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f,
                   0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f},
                  &original_graph_def);

    GraphDef quantized_graph_def;
    TF_ASSERT_OK(QuantizeWeights(original_graph_def, {{}, {"output"}},
                                 &quantized_graph_def));

    // Verify the structure of the quantized graph.
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

    // Run the the original graph.
    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    // Run the the quantized graph.
    std::unique_ptr<Session> quantized_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(quantized_session->Create(quantized_graph_def));
    std::vector<Tensor> quantized_outputs;
    TF_ASSERT_OK(
        quantized_session->Run({}, {"output"}, {}, &quantized_outputs));

    // Compare the results
    test::ExpectTensorNear<float>(original_outputs[0], quantized_outputs[0],
                                  0.5);
  }
};

TEST_F(QuantizeWeightsTest, TestQuantizeWeights) { TestQuantizeWeights(); }

TEST_F(QuantizeWeightsTest, RangesAlwaysIncludeZero) {
  GraphDef original_graph_def;
  BuildGraphDef({1, 1, 4, 4},
                {-1.0f, -4.0f, -2.0f, -5.0f, -1.0f, -4.0f, -2.0f, -5.0f, -1.0f,
                 -4.0f, -2.0f, -5.0f, -1.0f, -4.0f, -2.0f, -5.0f},
                {1, 2, 2, 10},
                {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f,
                 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f,
                 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f,
                 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f},
                &original_graph_def);
  GraphDef quantized_graph_def;
  TF_ASSERT_OK(QuantizeWeights(original_graph_def, {{}, {"output"}},
                               &quantized_graph_def));

  std::map<string, const NodeDef*> node_lookup;
  MapNamesToNodes(quantized_graph_def, &node_lookup);

  auto expected_tensor = [](float value) {
    Tensor tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&tensor, {value});
    return tensor;
  };
  auto existing_tensor = [&node_lookup](string op) {
    const NodeDef* node_def = node_lookup.at(op);
    CHECK(node_def);
    return GetNodeTensorAttr(*node_def, "value");
  };

  // The max of input_op is moved from -1.0 to 0.0.
  test::ExpectTensorNear<float>(
      expected_tensor(-5.0), existing_tensor("input_op_quantized_min"), 1e-5);
  test::ExpectTensorNear<float>(
      expected_tensor(0.0), existing_tensor("input_op_quantized_max"), 1e-5);

  // The min of weights_op is moved from 0.1 to 0.0.
  test::ExpectTensorNear<float>(
      expected_tensor(0.0), existing_tensor("weights_op_quantized_min"), 1e-5);
  test::ExpectTensorNear<float>(
      expected_tensor(4.0), existing_tensor("weights_op_quantized_max"), 1e-5);
}

}  // namespace graph_transforms
}  // namespace tensorflow
