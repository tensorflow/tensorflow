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

// Tests for ShapeInference.

#include "tensorflow/compiler/jit/shape_inference.h"

#include <map>
#include <vector>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(ShapeInferenceTest, Basics) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(root.WithOpName("A"), DT_FLOAT,
                            ops::Placeholder::Shape({2, 3}));
  auto b = ops::Placeholder(root.WithOpName("B"), DT_FLOAT,
                            ops::Placeholder::Shape({3}));
  auto c = ops::Placeholder(root.WithOpName("C"), DT_FLOAT);
  auto d = ops::Add(root.WithOpName("D"), a, b);
  auto e = ops::Add(root.WithOpName("E"), d, c);
  auto f = ops::Neg(root.WithOpName("F"), e);
  auto g = ops::AddN(root.WithOpName("G"), std::initializer_list<Output>{e, f});

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(root.ToGraph(graph.get()));

  GraphShapeInfo shape_info;
  TF_ASSERT_OK(InferShapes(graph.get(), /*arg_shapes=*/{},
                           /*fnlib_def=*/nullptr, &shape_info));

  std::map<string, std::vector<PartialTensorShape>> expected = {
      {"A", {PartialTensorShape({2, 3})}}, {"B", {PartialTensorShape({3})}},
      {"C", {PartialTensorShape()}},       {"D", {PartialTensorShape({2, 3})}},
      {"E", {PartialTensorShape()}},       {"F", {PartialTensorShape()}},
      {"G", {PartialTensorShape()}},
  };
  TF_EXPECT_OK(ShapeAnnotationsMatch(*graph, shape_info, expected));
}

TEST(ShapeInferenceTest, WhileLoop) {
  // Graph:
  // x = array_ops.placeholder(dtypes.int32)
  // y = control_flow_ops.while_loop(lambda i: i < 10, lambda i: i + 1, [x])
  Graph graph(OpRegistry::Global());
  {
    Scope scope = Scope::NewRootScope().ExitOnError();

    auto dummy = ops::Placeholder(scope.WithOpName("Dummy"), DT_INT32,
                                  ops::Placeholder::Shape({}));

    auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32,
                                   ops::Placeholder::Shape({}));
    auto enter =
        ops::internal::Enter(scope.WithOpName("while/Enter"), source, "aloop");
    // Add an unused Enter node. These should be ignored.
    auto enter2 =
        ops::internal::Enter(scope.WithOpName("while/Enter2"), source, "aloop");
    auto merge = ops::Merge(scope.WithOpName("while/Merge"),
                            std::initializer_list<Input>{enter, dummy});
    auto ten = ops::Const<int32>(
        scope.WithOpName("while/Less/y").WithControlDependencies(merge.output),
        10);
    auto less = ops::Less(scope.WithOpName("while/Less"), merge.output, ten);
    auto loop_cond = ops::LoopCond(scope.WithOpName("while/LoopCond"), less);
    auto switch_node =
        ops::Switch(scope.WithOpName("while/Switch"), merge.output, loop_cond);
    auto exit = ops::internal::Exit(scope.WithOpName("while/Exit"),
                                    switch_node.output_false);
    auto identity = ops::Identity(scope.WithOpName("while/Identity"),
                                  switch_node.output_true);
    auto identity_shape =
        ops::Const<int32>(scope.WithOpName("while/Identity/shape"), {});
    auto identity_reshaped = ops::Reshape(
        scope.WithOpName("while/Identity/reshaped"), identity, identity_shape);

    auto one = ops::Const<int32>(
        scope.WithOpName("while/add/y").WithControlDependencies(identity), 1);
    auto add = ops::Add(scope.WithOpName("while/add"), identity_reshaped, one);
    auto next_iteration =
        ops::NextIteration(scope.WithOpName("while/NextIteration"), add);

    auto sink = ops::Identity(scope.WithOpName("sink"), exit);

    // Remove the dummy node and add the loop backedge.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(next_iteration.node(), 0, merge.output.node(), 1);

    TF_EXPECT_OK(scope.ToGraph(&graph));
  }

  GraphShapeInfo shape_info;
  TF_ASSERT_OK(InferShapes(&graph, /*arg_shapes=*/{}, /*fnlib_def=*/nullptr,
                           &shape_info));
  std::map<string, std::vector<PartialTensorShape>> expected = {
      {"while/Identity", {PartialTensorShape()}},
      {"while/add", {PartialTensorShape({})}},
  };
  TF_EXPECT_OK(ShapeAnnotationsMatch(graph, shape_info, expected));
}

TEST(ShapeInferenceTest, WhileLoopWithResource) {
  // Graph:
  // x = resource_variable_ops.var_handle_op(dtype=dtypes.float32, shape=[2, 3])
  // y = control_flow_ops.while_loop(lambda _: true, lambda x: x, [x])
  Graph graph(OpRegistry::Global());
  {
    Scope scope = Scope::NewRootScope().ExitOnError();

    auto x =
        ops::VarHandleOp(scope.WithOpName("x"), DT_FLOAT, TensorShape({2, 3}));
    auto enter =
        ops::internal::Enter(scope.WithOpName("while/Enter"), x, "aloop");
    auto dummy = ops::Placeholder(scope.WithOpName("dummy"), DT_RESOURCE);
    auto merge = ops::Merge(scope.WithOpName("while/Merge"),
                            std::initializer_list<Input>{enter, dummy});
    auto false_value = ops::Const<bool>(scope.WithOpName("false"), false);
    auto loop_cond =
        ops::LoopCond(scope.WithOpName("while/LoopCond"), false_value);
    auto switch_node =
        ops::Switch(scope.WithOpName("while/Switch"), merge.output, loop_cond);
    auto exit = ops::internal::Exit(scope.WithOpName("while/Exit"),
                                    switch_node.output_false);
    auto identity = ops::Identity(scope.WithOpName("while/Identity"),
                                  switch_node.output_true);
    auto next_iteration =
        ops::NextIteration(scope.WithOpName("while/NextIteration"), identity);
    auto sink = ops::Identity(scope.WithOpName("sink"), exit);

    // Remove the dummy node and add the loop backedge.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(next_iteration.node(), 0, merge.output.node(), 1);

    TF_EXPECT_OK(scope.ToGraph(&graph));
  }

  // Check that we can infer shape for "sink" node (Merge node output).
  GraphShapeInfo shape_info;
  TF_ASSERT_OK(InferShapes(&graph, /*arg_shapes=*/{}, /*fnlib_def=*/nullptr,
                           &shape_info));
  auto iter = shape_info.find("sink");
  EXPECT_NE(iter, shape_info.end());
  EXPECT_EQ(iter->second.size(), 1);
  EXPECT_EQ(iter->second.at(0).handle_type, DT_FLOAT);
  TensorShape resource_shape;
  EXPECT_TRUE(iter->second.at(0).handle_shape.AsTensorShape(&resource_shape));
  EXPECT_EQ(resource_shape, TensorShape({2, 3}));
}

}  // namespace
}  // namespace tensorflow
