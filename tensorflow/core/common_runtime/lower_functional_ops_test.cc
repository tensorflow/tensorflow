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

#include "tensorflow/core/common_runtime/lower_functional_ops.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

typedef FunctionDefHelper FDH;

constexpr const char* const kLowerUsingSwitchMergeAttr =
    LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr;

static void AssertHasSubstr(StringPiece s, StringPiece expected) {
  ASSERT_TRUE(absl::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

SessionOptions SessionOptionsWithInlining() {
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  return session_options;
}

Status Rewrite(std::unique_ptr<Graph>* graph) {
  FunctionLibraryDefinition flib_def((*graph)->flib_def());
  GraphOptimizationPassOptions opt_options;
  SessionOptions session_options = SessionOptionsWithInlining();
  opt_options.session_options = &session_options;
  opt_options.graph = graph;
  opt_options.flib_def = &flib_def;
  LowerFunctionalOpsPass pass;
  return pass.Run(opt_options);
}

// (counter:int32, pred:bool, x:int32) -> counter < N
FunctionDef WhileWithIfCond(int32_t N) {
  const Tensor kN = test::AsScalar<int32>(N);
  return FDH::Define(
      // Name
      "WhileWithIfCond",
      // Args
      {"counter: int32", "pred: bool", "x: int32"},
      // Return values
      {"z: bool"},
      // Attr def
      {},
      // Nodes
      {
          {{"N"}, "Const", {}, {{"value", kN}, {"dtype", DT_INT32}}},
          {{"z"}, "Less", {"counter", "N"}, {{"T", DT_INT32}}},
      });
}

// (counter:int32, pred:bool, x:int32) ->
//   (counter+1, pred, if pred: x * 2 else: x * 4)
FunctionDef WhileWithIfBody() {
  NameAttrList then_func;
  then_func.set_name("XTimesTwo");
  NameAttrList else_func;
  else_func.set_name("XTimesFour");
  const Tensor kOne = test::AsScalar<int32>(1);
  std::vector<DataType> input_types = {DT_INT32};
  std::vector<DataType> output_types = {DT_INT32};
  return FDH::Define(
      // Name
      "WhileWithIfBody",
      // Args
      {"counter: int32", "pred: bool", "x: int32"},
      // Return values
      {"updated_counter: int32", "pred: bool", "if: int32"},
      // Attr def
      {},
      // Nodes
      {
          {{"if"},
           "If",
           {"pred", "x"},
           {{"then_branch", then_func},
            {"else_branch", else_func},
            {"Tcond", DT_BOOL},
            {"Tin", input_types},
            {"Tout", output_types},
            {kLowerUsingSwitchMergeAttr, true}}},
          {{"one"}, "Const", {}, {{"value", kOne}, {"dtype", DT_INT32}}},
          {{"updated_counter"}, "Add", {"counter", "one"}, {{"T", DT_INT32}}},
      });
}

TEST(LowerIfWhileTest, CondInWhile) {
  // Tests the value of `a` for different values of args after the following
  // program:
  //
  // Args:
  // counter = Placeholder(type = int32)
  // pred = Placeholder(type = bool)
  // a = Placeholder(type = int32)
  // N = 3
  // while (counter < N) {
  //   counter += 1;
  //   if (pred) {
  //     a *= 2;
  //   } else {
  //     a *= 4;
  //   }
  // }

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  FunctionDefLibrary f_lib_proto;
  // Cond functions.
  *f_lib_proto.add_function() = test::function::XTimesTwo();
  *f_lib_proto.add_function() = test::function::XTimesFour();
  // While functions.
  *f_lib_proto.add_function() = WhileWithIfCond(3);
  *f_lib_proto.add_function() = WhileWithIfBody();

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto counter = ops::Placeholder(root.WithOpName("counter"), DT_INT32);
  auto pred = ops::Placeholder(root.WithOpName("pred"), DT_BOOL);
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  std::vector<NodeBuilder::NodeOut> inputs(
      {NodeBuilder::NodeOut(counter.node()), NodeBuilder::NodeOut(pred.node()),
       NodeBuilder::NodeOut(a.node())});
  Node* while_node;
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("WhileWithIfCond");
  AttrValue body_func;
  body_func.mutable_func()->set_name("WhileWithIfBody");
  TF_ASSERT_OK(NodeBuilder("while", "While", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("T", {DT_INT32, DT_BOOL, DT_INT32})
                   .Attr("cond", cond_func)
                   .Attr("body", body_func)
                   .Attr(kLowerUsingSwitchMergeAttr, true)
                   .Finalize(root.graph(), &while_node));
  TF_ASSERT_OK(root.DoShapeInference(while_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  // Lowered graph has no While and If ops.
  for (const auto* op : graph->op_nodes()) {
    ASSERT_NE(op->type_string(), "While");
    ASSERT_NE(op->type_string(), "If");
  }

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(counter.node()), Input::Initializer(0));
    feeds.emplace(Output(pred.node()), Input::Initializer(true));
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(while_node, 2)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 8);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(counter.node()), Input::Initializer(0));
    feeds.emplace(Output(pred.node()), Input::Initializer(false));
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(while_node, 2)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 64);  // a
  }
}

// x:int32 ->
//    while x <= N:
//      x*=2;
//    return x;
FunctionDef IfWithWhileThen() {
  NameAttrList cond_func;
  cond_func.set_name("LessThanOrEqualToN");
  NameAttrList body_func;
  body_func.set_name("XTimesTwo");
  std::vector<DataType> input_and_output_types = {DT_INT32};
  std::vector<TensorShape> output_shapes = {TensorShape()};
  return FDH::Define(
      // Name
      "IfWithWhileThen",
      // Args
      {"x: int32"},
      // Return values
      {"while: int32"},
      // Attr def
      {},
      // Nodes
      {
          {{"while"},
           "While",
           {"x"},
           {{"cond", cond_func},
            {"body", body_func},
            {"T", input_and_output_types},
            {"output_shapes", output_shapes},
            {kLowerUsingSwitchMergeAttr, true}}},
      });
}

TEST(LowerIfWhileTest, WhileInCond) {
  // Tests the value of `a` for different values of args after the following
  // program:
  //
  // Args:
  // pred = Placeholder(dtype = bool)
  // a = Placeholder(dtype = int32)
  // N = 8
  // if (pred) {
  //   while (a <= N) {
  //     a *= 2;
  //   }
  // }
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;
  *f_lib_proto.add_function() = test::function::XTimesTwo();
  *f_lib_proto.add_function() = test::function::LessThanOrEqualToN(8);
  *f_lib_proto.add_function() = IfWithWhileThen();

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto pred = ops::Placeholder(root.WithOpName("pred"), DT_BOOL);
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});
  AttrValue then_func;
  then_func.mutable_func()->set_name("IfWithWhileThen");
  AttrValue else_func;
  else_func.mutable_func()->set_name("XTimesTwo");
  Node* if_node;
  TF_ASSERT_OK(NodeBuilder("if", "If", &root.graph()->flib_def())
                   .Input(pred.node())
                   .Input(inputs)
                   .Attr("then_branch", then_func)
                   .Attr("else_branch", else_func)
                   .Attr("Tout", {DT_INT32})
                   .Attr(kLowerUsingSwitchMergeAttr, true)
                   .Finalize(root.graph(), &if_node));
  TF_ASSERT_OK(root.DoShapeInference(if_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // The input graph has no lower level control flow primitives.
  int node_called_if_count = 0;
  for (const auto* op : graph->op_nodes()) {
    ASSERT_FALSE(op->IsEnter());
    ASSERT_FALSE(op->IsExit());
    ASSERT_FALSE(op->IsSwitch());
    ASSERT_FALSE(op->IsMerge());
    ASSERT_FALSE(op->IsNextIteration());
    ASSERT_FALSE(op->IsLoopCond());
    if (op->name() == "if") {
      node_called_if_count++;
    }
  }
  ASSERT_EQ(node_called_if_count, 1);

  TF_ASSERT_OK(Rewrite(&graph));

  node_called_if_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->name() == "if") {
      node_called_if_count++;
    }
    ASSERT_NE(op->type_string(), "While");
    ASSERT_NE(op->type_string(), "If");
  }
  // One node per loop input.
  ASSERT_EQ(node_called_if_count, 1);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(pred.node()), Input::Initializer(true));
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(if_node)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 16);
  }
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(pred.node()), Input::Initializer(false));
    feeds.emplace(Output(a.node()), Input::Initializer(1));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(if_node)}, &out_tensors));
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 2);
  }
}

}  // namespace
}  // namespace tensorflow
