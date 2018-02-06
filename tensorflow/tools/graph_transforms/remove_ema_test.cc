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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare transformations here, so we don't need a public header.
Status FakeQuantizeTraining(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def);

Status RemoveEMA(const GraphDef& input_graph_def,
                 const TransformFuncContext& context,
                 GraphDef* output_graph_def);

Status QuantizeNodes(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def);

class RemoveEMATest : public ::testing::Test {};

TEST_F(RemoveEMATest, FakeQuant_RemoveEMA_QuantizeTraining) {
  // Build a small graph.
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  Tensor a_data(DT_FLOAT, TensorShape({1, 1}));
  test::FillIota<float>(&a_data, 1.0f);
  Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

  Tensor b_data(DT_FLOAT, TensorShape({1, 1}));
  test::FillIota<float>(&b_data, 1.0f);
  Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

  Output matmul = MatMul(root.WithOpName("matmul"), a_const, b_const);
  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

  // (1) FakeQuantize the graph.
  GraphDef fake_quantized_graph_def;
  TransformFuncContext context;
  TF_ASSERT_OK(
      FakeQuantizeTraining(graph_def, context, &fake_quantized_graph_def));

  // Test that the transformation resulted in a graph with more nodes.
  EXPECT_GT(fake_quantized_graph_def.node_size(), graph_def.node_size());

  // (2) Run the graph to initialize the newly added variables.
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_ASSERT_OK(session->Create(fake_quantized_graph_def));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {"matmul"}, {}, &outputs));

  // (3) Freeze the graph. Create a "frozen graph" that matches what we would
  // expect if we actually froze the above graph.
  // TODO(suharshs): Use a c++ freeze graph alternative, when one is available.
  GraphDef frozen_graph_def;
  for (const NodeDef& node : fake_quantized_graph_def.node()) {
    if (node.op() == "Variable" || node.op() == "VariableV2") {
      NodeDef const_node;
      const_node.set_op("Const");
      const_node.set_name(node.name());
      SetNodeAttr("dtype", DT_FLOAT, &const_node);
      Tensor tensor(DT_FLOAT, {});
      tensor.flat<float>()(0) = 1.0f;
      SetNodeTensorAttr<float>("value", tensor, &const_node);
      *(frozen_graph_def.mutable_node()->Add()) = const_node;
    } else {
      *(frozen_graph_def.mutable_node()->Add()) = node;
    }
  }

  // Test that freezing the graph resulted in a graph with the same number of
  // nodes.
  EXPECT_EQ(frozen_graph_def.node_size(), fake_quantized_graph_def.node_size());

  // (4) RemoveEMA on the graph to make it compatible with QuantizeNodes.
  GraphDef removed_ema_graph_def;
  TF_ASSERT_OK(RemoveEMA(frozen_graph_def, context, &removed_ema_graph_def));

  // Test that the transformation resulted in a graph with less nodes.
  EXPECT_LT(removed_ema_graph_def.node_size(), frozen_graph_def.node_size());

  // (5) QuantizeNodes and inspect the final graph.
  // TODO(suharshs): Add a more thorough inspection of the structure of
  // the output graph.
  GraphDef quantized_graph_def;
  TF_ASSERT_OK(
      QuantizeNodes(removed_ema_graph_def, context, &quantized_graph_def));

  // Test that the transformation resulted in a graph with more nodes.
  EXPECT_GT(quantized_graph_def.node_size(), removed_ema_graph_def.node_size());

  // Make sure that the FakeQuantizeWithMinMaxVars op has been removed.
  for (const NodeDef& node : quantized_graph_def.node()) {
    EXPECT_NE(node.op(), "FakeQuantWithMinMaxVars");
  }
}

}  // namespace graph_transforms
}  // namespace tensorflow
