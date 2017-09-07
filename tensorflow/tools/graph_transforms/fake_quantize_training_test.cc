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
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status FakeQuantizeTraining(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def);

class FakeQuantizeTrainingTest : public ::testing::Test {};

// For now, since the fake_quantize_training transform just calls the
// quantize_training rewrite from tensorflow/core/graph/quantize_training.h,
// we just test that the graph has been changed by the transform.
// TODO(suharshs): Once we implement the fake_quantize_training transform
// using the GTT, write proper tests of the transform here.
TEST_F(FakeQuantizeTrainingTest, TransformOccurred) {
  auto root = tensorflow::Scope::DisabledShapeInferenceScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  Tensor a_data(DT_FLOAT, TensorShape());
  test::FillIota<float>(&a_data, 1.0f);
  Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

  Tensor b_data(DT_FLOAT, TensorShape());
  test::FillIota<float>(&b_data, 1.0f);
  Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

  Output matmul = MatMul(root.WithOpName("matmul"), a_const, b_const);
  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

  GraphDef result;
  TransformFuncContext context;
  TF_ASSERT_OK(FakeQuantizeTraining(graph_def, context, &result));

  // Test that the transformation resulted in a graph with more nodes.
  EXPECT_GT(result.node_size(), graph_def.node_size());
}

}  // namespace graph_transforms
}  // namespace tensorflow
