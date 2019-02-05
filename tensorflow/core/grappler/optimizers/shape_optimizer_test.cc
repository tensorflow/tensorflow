/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/shape_optimizer.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class ShapeOptimizerTest : public GrapplerTest {};

TEST_F(ShapeOptimizerTest, OptimizeShapeProduct) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 3.14f, {32, 16});
  Output c = ops::Shape(s.WithOpName("c"), a);
  Output d = ops::Const(s.WithOpName("d"), 0, {1});
  ops::ReduceProd::Attrs attrs;
  Output e = ops::ReduceProd(s.WithOpName("e"), c, d, attrs.KeepDims(false));
  Output f = ops::ReduceProd(s.WithOpName("f"), c, d, attrs.KeepDims(true));

  GrapplerItem item;
  item.fetch = {"e", "f"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ShapeOptimizer optimizer;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "e") {
      found++;
      EXPECT_EQ("Size", node.op());
      EXPECT_EQ("a", node.input(0));
    } else if (node.name() == "f") {
      found++;
      EXPECT_EQ("Prod", node.op());
      EXPECT_EQ("c", node.input(0));
    }
  }
  EXPECT_EQ(2, found);

  auto tensors_actual = EvaluateNodes(output, item.fetch);
  EXPECT_NEAR(tensors_expected[0].scalar<int>()(),
              tensors_actual[0].scalar<int>()(), 0);
  EXPECT_NEAR(tensors_expected[1].scalar<int>()(),
              tensors_actual[1].scalar<int>()(), 0);
}

TEST_F(ShapeOptimizerTest, OptimizeShapeRatio) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 3.14f, {32, 32});
  Output b = ops::Const(s.WithOpName("b"), 3.14f, {32, 16});
  Output c = ops::Size(s.WithOpName("c"), a);
  Output d = ops::Size(s.WithOpName("d"), b);
  Output e = ops::Div(s.WithOpName("e"), c, d);

  GrapplerItem item;
  item.fetch = {"e"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ShapeOptimizer optimizer;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "e") {
      found++;
      EXPECT_EQ("Const", node.op());
    }
  }
  EXPECT_EQ(1, found);

  auto tensors_actual = EvaluateNodes(output, item.fetch);
  EXPECT_NEAR(tensors_expected[0].scalar<int>()(),
              tensors_actual[0].scalar<int>()(), 0);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
