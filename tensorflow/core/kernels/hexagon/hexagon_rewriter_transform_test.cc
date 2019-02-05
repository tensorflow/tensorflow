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
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declared here so we don't have to put it in a public header.
Status RewriteQuantizedStrippedModelForHexagon(
    const GraphDef& input_graph_def, const TransformFuncContext& context,
    GraphDef* output_graph_def);

namespace {

TEST(HexagonRewriteTransformTest, BasicRun) {
  Scope root = tensorflow::Scope::NewRootScope();

  // Create a simple graph that calculates (a + b) * placeholder.
  Tensor a_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&a_data, 1.0f);
  Output a_const = ops::Const(root.WithOpName("a"), Input::Initializer(a_data));

  Tensor b_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&b_data, 1.0f);
  Output b_const = ops::Const(root.WithOpName("b"), Input::Initializer(b_data));

  Output add = ops::Add(root.WithOpName("add"), a_const, b_const);

  Output placeholder =
      ops::Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

  Output mul = ops::Mul(root.WithOpName("output"), add, placeholder);

  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

  GraphDef result;
  TransformFuncContext context;
  context.input_names = {"placeholder"};
  context.output_names = {"output"};
  context.params.insert(std::pair<string, std::vector<string>>(
      {"input_shape0", {string("1,1,1,1")}}));
  context.params.insert(std::pair<string, std::vector<string>>(
      {"input_type0", {string("float")}}));
  TF_ASSERT_OK(
      RewriteQuantizedStrippedModelForHexagon(graph_def, context, &result));

  // Node in the input graph is fused to
  // 1 input placeholder node + 1 fused output node
  EXPECT_EQ(2, result.node_size());
}

}  // namespace
}  // namespace graph_transforms
}  // namespace tensorflow
