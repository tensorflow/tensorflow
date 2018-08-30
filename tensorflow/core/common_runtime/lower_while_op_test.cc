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

#include "tensorflow/core/common_runtime/lower_while_op.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Status Rewrite(std::unique_ptr<Graph>* graph) {
  FunctionDefLibrary flib;
  FunctionLibraryDefinition flib_def((*graph)->op_registry(), flib);

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;
  opt_options.flib_def = &flib_def;
  LowerWhileOpPass pass;
  return pass.Run(opt_options);
}

TEST(LowerWhileOpTest, Simple) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;
  *f_lib_proto.add_function() = test::function::XTimesTwo();
  *f_lib_proto.add_function() = test::function::LessThanOrEqualToN(8);
  FunctionLibraryDefinition f_lib(OpRegistry::Global(), f_lib_proto);

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::_Arg(root.WithOpName("A"), DT_INT32, 0);
  Node* while_node;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("LessThanOrEqualToN");
  AttrValue body_func;
  body_func.mutable_func()->set_name("XTimesTwo");
  TF_ASSERT_OK(NodeBuilder("while", "While", &f_lib)
                   .Input(inputs)
                   .Attr("T", {DT_INT32})
                   .Attr("cond", cond_func)
                   .Attr("body", body_func)
                   .Attr(LowerIfOpPass::kLowerUsingSwitchMergeAttr, true)
                   .Finalize(root.graph(), &while_node));
  TF_ASSERT_OK(root.DoShapeInference(while_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // The input graph has no lower level control flow primitives.
  int node_called_while_count = 0;
  for (const auto* op : graph->op_nodes()) {
    ASSERT_FALSE(op->IsEnter());
    ASSERT_FALSE(op->IsExit());
    ASSERT_FALSE(op->IsSwitch());
    ASSERT_FALSE(op->IsMerge());
    ASSERT_FALSE(op->IsNextIteration());
    ASSERT_FALSE(op->IsLoopCond());
    if (op->name() == "while") {
      node_called_while_count++;
    }
  }
  ASSERT_EQ(node_called_while_count, 1);

  TF_ASSERT_OK(Rewrite(&graph));

  int enter_count = 0;
  int exit_count = 0;
  int switch_count = 0;
  int merge_count = 0;
  int next_iteration_count = 0;
  node_called_while_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsEnter()) {
      ++enter_count;
    }
    if (op->IsExit()) {
      ++exit_count;
    }
    if (op->IsSwitch()) {
      ++switch_count;
    }
    if (op->IsMerge()) {
      ++merge_count;
    }
    if (op->IsNextIteration()) {
      ++next_iteration_count;
    }
    if (op->name() == "while") {
      node_called_while_count++;
    }
    ASSERT_NE(op->type_string(), "While");
  }
  // One node per loop input.
  ASSERT_EQ(enter_count, 1);
  ASSERT_EQ(exit_count, 1);
  ASSERT_EQ(switch_count, 1);
  ASSERT_EQ(merge_count, 1);
  ASSERT_EQ(next_iteration_count, 1);
  ASSERT_EQ(node_called_while_count, 1);

  // Verify execution.
  ClientSession session(root);
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(while_node)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 16);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(3));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(while_node)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 12);
  }
}

TEST(LowerWhileOpTest, MultipleInputs) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XPlusOneXTimesY();
  *(f_lib_proto.add_function()) = test::function::XYXLessThanOrEqualToN(4);
  FunctionLibraryDefinition f_lib(OpRegistry::Global(), f_lib_proto);

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::_Arg(root.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(root.WithOpName("B"), DT_INT32, 1);
  Node* while_node;
  std::vector<NodeBuilder::NodeOut> inputs(
      {NodeBuilder::NodeOut(a.node()), NodeBuilder::NodeOut(b.node())});
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("XYXLessThanOrEqualToN");
  AttrValue body_func;
  body_func.mutable_func()->set_name("XPlusOneXTimesY");
  TF_ASSERT_OK(NodeBuilder("while", "While", &f_lib)
                   .Input(inputs)
                   .Attr("T", {DT_INT32, DT_INT32})
                   .Attr("cond", cond_func)
                   .Attr("body", body_func)
                   .Attr(LowerIfOpPass::kLowerUsingSwitchMergeAttr, true)
                   .Finalize(root.graph(), &while_node));
  TF_ASSERT_OK(root.DoShapeInference(while_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // The input graph has no lower level control flow primitives.
  for (const auto* op : graph->op_nodes()) {
    ASSERT_FALSE(op->IsEnter());
    ASSERT_FALSE(op->IsExit());
    ASSERT_FALSE(op->IsSwitch());
    ASSERT_FALSE(op->IsMerge());
    ASSERT_FALSE(op->IsNextIteration());
    ASSERT_FALSE(op->IsLoopCond());
  }

  TF_ASSERT_OK(Rewrite(&graph));

  int enter_count = 0;
  int exit_count = 0;
  int switch_count = 0;
  int merge_count = 0;
  int next_iteration_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsEnter()) {
      ++enter_count;
    }
    if (op->IsExit()) {
      ++exit_count;
    }
    if (op->IsSwitch()) {
      ++switch_count;
    }
    if (op->IsMerge()) {
      ++merge_count;
    }
    if (op->IsNextIteration()) {
      ++next_iteration_count;
    }
    ASSERT_NE(op->type_string(), "While");
  }
  // Two nodes per loop input.
  ASSERT_EQ(enter_count, 2);
  ASSERT_EQ(exit_count, 2);
  ASSERT_EQ(switch_count, 2);
  ASSERT_EQ(merge_count, 2);
  ASSERT_EQ(next_iteration_count, 2);

  // Verify execution.
  ClientSession session(root);
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    feeds.emplace(Output(b.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(
        feeds, {Output(while_node, 0), Output(while_node, 1)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 2);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 5);
    EXPECT_EQ(out_tensors[1].scalar<int>()(), 24);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(3));
    feeds.emplace(Output(b.node()), Input::Initializer(5));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(
        feeds, {Output(while_node, 0), Output(while_node, 1)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 2);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 5);
    EXPECT_EQ(out_tensors[1].scalar<int>()(), 60);
  }
}

}  // namespace
}  // namespace tensorflow
