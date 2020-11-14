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

#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_ops.h"
#include "tensorflow/compiler/tf2xla/test_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

// Returns the names of the "then" and "else" functions for the If node in a
// graph.
Status FindIfThenAndElse(const GraphDef& graph, string* op_name,
                         NameAttrList* then_fn, NameAttrList* else_fn) {
  for (const NodeDef& node : graph.node()) {
    if (node.op() == "If") {
      *op_name = node.name();
      const NameAttrList* result;
      TF_RETURN_IF_ERROR(GetNodeAttr(node, "then_branch", &result));
      *then_fn = *result;
      TF_RETURN_IF_ERROR(GetNodeAttr(node, "else_branch", &result));
      *else_fn = *result;
      return Status::OK();
    }
  }
  return errors::NotFound("No If node found in graph");
}

// Graph:
// x = array_ops.placeholder(dtypes.int32)
// y = array_ops.placeholder(dtypes.int32)
// z = control_flow_ops.cond(
//     math_ops.less(y, x), lambda: math_ops.multiply(y, 17),
//     lambda: math_ops.add(x, 23))
//
// Tests different node filters and functionalization inside of a function.
class ConditionalTestFixture
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    restrict_to_tpu_nodes_ = std::get<0>(GetParam());
    wrap_condition_in_function_ = std::get<1>(GetParam());
  }
  void RunTest();

 private:
  void BuildCondGraph(Graph* cond_graph);
  void CheckGraphDef(const GraphDef& graph_def,
                     const FunctionLibraryDefinition& library);

  bool restrict_to_tpu_nodes_ = false;
  bool wrap_condition_in_function_ = false;
};

TEST_P(ConditionalTestFixture, ConditionalTests) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(
    FunctionalizeControlFlow, ConditionalTestFixture,
    ::testing::Combine(::testing::Bool(), ::testing::Bool()),
    [](const ::testing::TestParamInfo<ConditionalTestFixture::ParamType>&
           info) {
      bool restrict_to_tpu_nodes = std::get<0>(info.param);
      bool wrap_cond_in_function = std::get<1>(info.param);
      string name =
          absl::StrCat(restrict_to_tpu_nodes ? "with_filter" : "without_filter",
                       wrap_cond_in_function ? "_in_function" : "_in_graph");
      return name;
    });

void ConditionalTestFixture::BuildCondGraph(Graph* cond_graph) {
  {
    Scope scope = Scope::NewRootScope().ExitOnError();

    auto x = ops::Placeholder(scope.WithOpName("x"), DT_INT32);
    auto y = ops::Placeholder(scope.WithOpName("y"), DT_INT32);
    auto less = ops::Less(scope.WithOpName("cond/Less"), y, x);
    auto switch_1 = ops::Switch(scope.WithOpName("cond/Switch"), less, less);

    auto identity_t =
        ops::Identity(scope.WithOpName("cond/Identity"), switch_1.output_true);
    auto seventeen = ops::Const<int32>(
        scope.WithOpName("cond").WithControlDependencies(identity_t), 17);
    auto switch_2 = ops::Switch(scope.WithOpName("cond/Switch"), y, less);
    auto mul = ops::Multiply(scope.WithOpName("cond/Mul"), switch_2.output_true,
                             seventeen);

    auto identity_f =
        ops::Identity(scope.WithOpName("cond/Identity"), switch_1.output_false);
    auto twenty_three = ops::Const<int32>(
        scope.WithOpName("cond").WithControlDependencies(identity_f), 23);
    auto switch_3 = ops::Switch(scope.WithOpName("cond/Switch"), x, less);
    auto add = ops::Add(scope.WithOpName("cond/false/add"),
                        switch_3.output_false, twenty_three);

    auto merge = ops::Merge(scope.WithOpName("cond/Merge"),
                            std::initializer_list<Input>{add, mul});

    TF_EXPECT_OK(scope.ToGraph(cond_graph));

    // Set `_tpu_replicate` attribute for all nodes.
    for (Node* n : cond_graph->nodes()) {
      n->AddAttr("_tpu_replicate", "cluster");
    }
  }
}

void ConditionalTestFixture::CheckGraphDef(
    const GraphDef& graph_def, const FunctionLibraryDefinition& library) {
  string op_name;
  NameAttrList then_fn;
  NameAttrList else_fn;
  TF_EXPECT_OK(FindIfThenAndElse(graph_def, &op_name, &then_fn, &else_fn));
  InstantiationResultForTest else_result;
  TF_EXPECT_OK(
      InstantiateFunctionForTest(else_fn.name(), library, &else_result));

  // Outer graph
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto y = ops::Placeholder(scope.WithOpName("y"), DT_INT32);
    auto x = ops::Placeholder(scope.WithOpName("x"), DT_INT32);
    auto less = ops::Less(scope.WithOpName("cond/Less"), y, x);
    auto if_op =
        ops::If(scope.WithOpName(op_name), less,
                std::initializer_list<Input>{less, y, x}, {DT_INT32}, then_fn,
                else_fn, ops::If::OutputShapes({PartialTensorShape()}));
    auto id = ops::Identity(scope.WithOpName("cond/Merge"), if_op.output[0]);
    GraphDef expected;
    TF_EXPECT_OK(scope.ToGraphDef(&expected));
    TF_EXPECT_GRAPH_EQ(expected, graph_def);
  }

  // then body.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto arg_0 = ops::_Arg(scope.WithOpName("arg0"), DT_BOOL, 0);
    auto arg_1 = ops::_Arg(scope.WithOpName("arg1"), DT_INT32, 1);
    auto arg_2 = ops::_Arg(scope.WithOpName("arg2"), DT_INT32, 2);
    auto identity = ops::Identity(scope.WithOpName("cond/Identity"), arg_0);
    auto cond = ops::Const(
        scope.WithOpName("cond").WithControlDependencies(identity), 17);
    auto mul = ops::Mul(scope.WithOpName("cond/Mul"), arg_1, cond);
    auto retval0 = ops::_Retval(scope.WithOpName("retval0_RetVal"), mul, 0);

    GraphDef expected;
    TF_EXPECT_OK(scope.ToGraphDef(&expected));

    InstantiationResultForTest result;
    TF_EXPECT_OK(InstantiateFunctionForTest(then_fn.name(), library, &result));

    EXPECT_EQ(DataTypeVector{DT_INT32}, result.ret_types);
    EXPECT_EQ((DataTypeVector{DT_BOOL, DT_INT32, DT_INT32}), result.arg_types);
    TF_EXPECT_GRAPH_EQ(expected, result.gdef);
  }

  // else body.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto arg_0 = ops::_Arg(scope.WithOpName("arg0"), DT_BOOL, 0);
    auto arg_1 = ops::_Arg(scope.WithOpName("arg1"), DT_INT32, 1);
    auto arg_2 = ops::_Arg(scope.WithOpName("arg2"), DT_INT32, 2);
    auto identity = ops::Identity(scope.WithOpName("cond/Identity_1"), arg_0);
    auto cond_1 = ops::Const(
        scope.WithOpName("cond_1").WithControlDependencies(identity), 23);
    auto add = ops::Add(scope.WithOpName("cond/false/add"), arg_2, cond_1);
    auto retval0 = ops::_Retval(scope.WithOpName("retval0_RetVal"), add, 0);

    GraphDef expected;
    TF_EXPECT_OK(scope.ToGraphDef(&expected));

    InstantiationResultForTest result;
    TF_EXPECT_OK(InstantiateFunctionForTest(else_fn.name(), library, &result));

    EXPECT_EQ(DataTypeVector{DT_INT32}, result.ret_types);
    EXPECT_EQ((DataTypeVector{DT_BOOL, DT_INT32, DT_INT32}), result.arg_types);
    TF_EXPECT_GRAPH_EQ(expected, result.gdef);
  }
}

void ConditionalTestFixture::RunTest() {
  Graph graph(OpRegistry::Global());
  if (wrap_condition_in_function_) {
    // Wrap condition in a function which is called from `graph`.
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32);

    Graph cond_graph(OpRegistry::Global());
    BuildCondGraph(&cond_graph);

    FunctionDef cond_fdef;
    TF_ASSERT_OK(GraphToFunctionDef(cond_graph, "cond_fn", &cond_fdef));

    FunctionDefLibrary fdef_lib;
    *(fdef_lib.add_function()) = cond_fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));
    NodeDef cond_fn;
    cond_fn.set_name("cond_node");
    cond_fn.set_op("cond_fn");
    *(cond_fn.add_input()) = "source";
    Status status;
    scope.graph()->AddNode(cond_fn, &status);
    TF_ASSERT_OK(status);
    TF_ASSERT_OK(scope.ToGraph(&graph));
  } else {
    // Build condition in `graph`.
    BuildCondGraph(&graph);
  }
  FunctionLibraryDefinition library(graph.flib_def());
  // If `restrict_to_tpu_nodes_` is true let filter function return true for
  // `_tpu_replicate` nodes.
  NodeFilter node_filter =
      restrict_to_tpu_nodes_
          ? [](const Node* n) { return n->attrs().Find("_tpu_replicate"); }
          : NodeFilter{};

  GraphDef optimized_graph_def;
  graph.ToGraphDef(&optimized_graph_def);
  TF_ASSERT_OK(FunctionalizeControlFlowForGraphDef(
      &optimized_graph_def, &library, node_filter,
      /*include_functions=*/wrap_condition_in_function_));
  TF_ASSERT_OK(FunctionalizeControlFlow(
      &graph, &library, node_filter,
      /*include_functions=*/wrap_condition_in_function_));

  if (wrap_condition_in_function_) {
    // Check if function body was functionalized.
    auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
        /*device_mgr=*/nullptr, tensorflow::Env::Default(),
        /*config=*/nullptr, TF_GRAPH_DEF_VERSION, &library,
        tensorflow::OptimizerOptions());
    FunctionLibraryRuntime* flr =
        pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
    FunctionLibraryRuntime::Handle handle;

    // Functionalized function name is the type string of `cond_node`.
    string func_name;
    for (Node* n : graph.nodes()) {
      if (n->name() == "cond_node") {
        func_name = n->type_string();
        break;
      }
    }
    TF_ASSERT_OK(flr->Instantiate(func_name, AttrSlice(), &handle));
    const FunctionBody* body = flr->GetFunctionBody(handle);
    GraphDef graph_def;
    body->graph->ToGraphDef(&graph_def);
    CheckGraphDef(graph_def, library);
  } else {
    // Check if graphs were functionalized.
    CheckGraphDef(optimized_graph_def, library);
    GraphDef converted_graph_def;
    graph.ToGraphDef(&converted_graph_def);
    CheckGraphDef(converted_graph_def, library);
  }
}

// Returns the names of the "cond" and "body" functions for the While node
// in a graph.
Status FindWhileCondAndBody(const GraphDef& graph, NameAttrList* cond,
                            NameAttrList* body) {
  for (const NodeDef& node : graph.node()) {
    if (node.op() == "While") {
      const NameAttrList* result;
      TF_RETURN_IF_ERROR(GetNodeAttr(node, "cond", &result));
      *cond = *result;
      TF_RETURN_IF_ERROR(GetNodeAttr(node, "body", &result));
      *body = *result;
      return Status::OK();
    }
  }
  return errors::NotFound("No While node found in graph");
}

// Graph:
// x = array_ops.placeholder(dtypes.int32)
// y = control_flow_ops.while_loop(lambda i: i < 10, lambda i: i + 1, [x])
TEST(FunctionalizeControlFlow, OneLoopVar) {
  Graph graph(OpRegistry::Global());
  {
    Scope scope = Scope::NewRootScope().ExitOnError();

    auto dummy = ops::Placeholder(scope.WithOpName("Dummy"), DT_INT32);

    auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32);
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
    auto switch_ =
        ops::Switch(scope.WithOpName("while/Switch"), merge.output, loop_cond);
    auto exit = ops::internal::Exit(scope.WithOpName("while/Exit"),
                                    switch_.output_false);
    auto identity =
        ops::Identity(scope.WithOpName("while/Identity"), switch_.output_true);
    auto one = ops::Const<int32>(
        scope.WithOpName("while/add/y").WithControlDependencies(identity), 1);
    auto add = ops::Add(scope.WithOpName("while/add"), identity, one);
    auto next_iteration =
        ops::NextIteration(scope.WithOpName("while/NextIteration"), add);

    auto sink = ops::Identity(scope.WithOpName("sink"), exit);

    // Remove the dummy node and add the loop backedge.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(next_iteration.node(), 0, merge.output.node(), 1);

    TF_EXPECT_OK(scope.ToGraph(&graph));
  }

  // Regression test: control edges from an Enter node to the graph sink should
  // be ignored.
  for (Node* n : graph.nodes()) {
    if (n->name() == "while/Enter") {
      graph.AddControlEdge(n, graph.sink_node());
    }
  }

  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  GraphDef optimized_graph_def;
  graph.ToGraphDef(&optimized_graph_def);
  TF_ASSERT_OK(
      FunctionalizeControlFlowForGraphDef(&optimized_graph_def, &library));
  TF_ASSERT_OK(FunctionalizeControlFlow(&graph, &library));
  GraphDef converted_graph_def;
  graph.ToGraphDef(&converted_graph_def);

  for (const GraphDef& graph_def : {optimized_graph_def, converted_graph_def}) {
    NameAttrList cond_fn, body_fn;
    TF_EXPECT_OK(FindWhileCondAndBody(graph_def, &cond_fn, &body_fn));

    // Outer graph
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32);
      auto while_op =
          ops::While(scope.WithOpName("while/LoopCond"),
                     std::initializer_list<Input>{source}, cond_fn, body_fn);
      auto sink = ops::Identity(scope.WithOpName("sink"), while_op[0]);
      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));
      TF_EXPECT_GRAPH_EQ(expected, graph_def);
    }

    // Condition graph
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto arg = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
      auto ten = ops::Const<int32>(
          scope.WithOpName("while/Less/y").WithControlDependencies(arg), 10);
      auto less = ops::Less(scope.WithOpName("while/Less"), arg, ten);
      auto retval = ops::_Retval(scope.WithOpName("retval0_RetVal"), less, 0);

      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));

      InstantiationResultForTest result;
      TF_EXPECT_OK(
          InstantiateFunctionForTest(cond_fn.name(), library, &result));

      EXPECT_EQ(DataTypeVector{DT_INT32}, result.arg_types);
      EXPECT_EQ(DataTypeVector{DT_BOOL}, result.ret_types);
      TF_EXPECT_GRAPH_EQ(expected, result.gdef);
    }

    // Body graph.
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto arg = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
      auto identity = ops::Identity(scope.WithOpName("while/Identity"), arg);
      auto one = ops::Const<int32>(
          scope.WithOpName("while/add/y").WithControlDependencies(identity), 1);
      auto add = ops::Add(scope.WithOpName("while/add"), identity, one);
      auto retval = ops::_Retval(scope.WithOpName("retval0_RetVal"), add, 0);

      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));

      InstantiationResultForTest result;
      TF_EXPECT_OK(
          InstantiateFunctionForTest(body_fn.name(), library, &result));

      EXPECT_EQ(DataTypeVector{DT_INT32}, result.arg_types);
      EXPECT_EQ(DataTypeVector{DT_INT32}, result.ret_types);
      TF_EXPECT_GRAPH_EQ(expected, result.gdef);
    }
  }
}

FunctionDef GetNoinlineFunctionDef() {
  FunctionDef fdef = FunctionDefHelper::Create(
      "increment_fn", {"x:int32"}, {"add:int32"}, {},
      {
          {{"add/y"}, "Const", {}, {{"dtype", DT_INT32}}},
          {{"add_0"}, "Add", {"x", "add/y:output:0"}, {{"T", DT_INT32}}},
      },
      {{"add", "add_0:z:0"}});
  (*fdef.mutable_attr())["_noinline"].set_b(true);
  return fdef;
}

// @function.Defun(noinline=True)
// def increment_fn(x):
//   return [x + 1]
// Define the above function, and add it to the given graph. It's used as the
// while loop body in NoinlineLoopBody test.
Status AddNoinlineFunctionToGraph(const string& node_name, Graph* graph) {
  FunctionDefLibrary fdef_lib;
  *(fdef_lib.add_function()) = GetNoinlineFunctionDef();
  TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(fdef_lib));
  NodeDef increment_fn;
  increment_fn.set_name(node_name);
  increment_fn.set_op("increment_fn");
  *increment_fn.add_input() = "while/Identity";
  *increment_fn.add_input() = "^while/Identity";
  Status status;
  graph->AddNode(increment_fn, &status);
  return status;
}

// Graph:
// x = array_ops.placeholder(dtypes.int32)
// y = control_flow_ops.while_loop(lambda i: i < 10, increment_fn, [x])
TEST(FunctionalizeControlFlow, NoinlineLoopBody) {
  const string& noinline_node_name = "while/increment_fn";
  Graph graph(OpRegistry::Global());
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto dummy = ops::Placeholder(scope.WithOpName("Dummy"), DT_INT32);
    auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32);
    auto enter = ops::internal::Enter(scope.WithOpName("while/Enter"), source,
                                      "while/while_context");
    auto merge = ops::Merge(scope.WithOpName("while/Merge"),
                            std::initializer_list<Input>{enter, dummy});
    auto ten = ops::Const<int32>(
        scope.WithOpName("while/Less/y").WithControlDependencies(merge.output),
        10);
    auto less = ops::Less(scope.WithOpName("while/Less"), merge.output, ten);
    auto loop_cond = ops::LoopCond(scope.WithOpName("while/LoopCond"), less);
    auto switch_ =
        ops::Switch(scope.WithOpName("while/Switch"), merge.output, loop_cond);
    auto exit = ops::internal::Exit(scope.WithOpName("while/Exit"),
                                    switch_.output_false);
    auto identity =
        ops::Identity(scope.WithOpName("while/Identity"), switch_.output_true);

    TF_ASSERT_OK(AddNoinlineFunctionToGraph(noinline_node_name, scope.graph()));

    NodeDef next_iter;
    next_iter.set_name("while/NextIteration");
    next_iter.set_op("NextIteration");
    *next_iter.add_input() = noinline_node_name;
    (*next_iter.mutable_attr())["T"].set_type(DT_INT32);

    Status status;
    Node* n = scope.graph()->AddNode(next_iter, &status);
    TF_ASSERT_OK(status);

    // Remove the dummy node and add the loop backedge.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(n, 0, merge.output.node(), 1);
    TF_ASSERT_OK(scope.ToGraph(&graph));
  }

  FunctionLibraryDefinition library(graph.flib_def());
  GraphDef optimized_graph_def;
  graph.ToGraphDef(&optimized_graph_def);

  *(optimized_graph_def.mutable_library()->add_function()) =
      GetNoinlineFunctionDef();

  TF_ASSERT_OK(
      FunctionalizeControlFlowForGraphDef(&optimized_graph_def, &library));
  TF_ASSERT_OK(FunctionalizeControlFlow(&graph, &library));
  GraphDef converted_graph_def;
  graph.ToGraphDef(&converted_graph_def);

  for (const GraphDef& graph_def : {optimized_graph_def, converted_graph_def}) {
    NameAttrList cond_fn, body_fn;
    TF_ASSERT_OK(FindWhileCondAndBody(graph_def, &cond_fn, &body_fn));

    // Outer graph
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32);
      auto while_op =
          ops::While(scope.WithOpName("while/LoopCond"),
                     std::initializer_list<Input>{source}, cond_fn, body_fn);
      GraphDef expected;
      TF_ASSERT_OK(scope.ToGraphDef(&expected));
      TF_EXPECT_GRAPH_EQ(expected, graph_def);
    }

    // Body graph.
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto arg = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
      TF_ASSERT_OK(
          AddNoinlineFunctionToGraph(noinline_node_name, scope.graph()));
      auto identity = ops::Identity(scope.WithOpName("while/Identity"), arg);
      NodeDef retval;
      retval.set_name("retval0_RetVal");
      retval.set_op(FunctionLibraryDefinition::kRetOp);
      *retval.add_input() = noinline_node_name;
      (*retval.mutable_attr())["T"].set_type(DT_INT32);
      (*retval.mutable_attr())["index"].set_i(0);
      Status status;
      scope.graph()->AddNode(retval, &status);
      TF_ASSERT_OK(status);

      GraphDef expected;
      TF_ASSERT_OK(scope.ToGraphDef(&expected));

      InstantiationResultForTest result;
      // Verify that increment_fn has been copied to library.
      TF_EXPECT_OK(
          InstantiateFunctionForTest(body_fn.name(), library, &result));

      EXPECT_EQ(DataTypeVector{DT_INT32}, result.arg_types);
      EXPECT_EQ(DataTypeVector{DT_INT32}, result.ret_types);
      // Ignore the function library when comparing the graphs.
      expected.clear_library();
      TF_EXPECT_GRAPH_EQ(expected, result.gdef);
    }
  }
}

TEST(FunctionalizeControlFlow, MissingFunctionDefInLibrary) {
  const string& noinline_node_name = "while/increment_fn";
  Graph graph(OpRegistry::Global());
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32);
    auto identity = ops::Identity(scope.WithOpName("while/Identity"), source);
    TF_ASSERT_OK(AddNoinlineFunctionToGraph(noinline_node_name, scope.graph()));
    TF_ASSERT_OK(scope.ToGraph(&graph));
  }

  FunctionLibraryDefinition library(graph.flib_def());
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  graph_def.clear_library();

  Status status = FunctionalizeControlFlowForGraphDef(&graph_def, &library);
  EXPECT_EQ(tensorflow::error::NOT_FOUND, status.code());
}

// Tests functionalizing OneLoopVar where the loop value is not used post the
// loop.
// Graph:
// x = array_ops.placeholder(dtypes.int32)
// control_flow_ops.while_loop(lambda i: i < 10, lambda i: i + 1, [x])
TEST(FunctionalizeControlFlow, OneLoopVarWithoutExit) {
  Graph graph(OpRegistry::Global());
  {
    Scope scope = Scope::NewRootScope().ExitOnError();

    auto dummy = ops::Placeholder(scope.WithOpName("Dummy"), DT_INT32);

    auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32);
    auto enter =
        ops::internal::Enter(scope.WithOpName("while/Enter"), source, "aloop");
    auto merge = ops::Merge(scope.WithOpName("while/Merge"),
                            std::initializer_list<Input>{enter, dummy});
    auto ten = ops::Const<int32>(
        scope.WithOpName("while/Less/y").WithControlDependencies(merge.output),
        10);
    auto less = ops::Less(scope.WithOpName("while/Less"), merge.output, ten);
    auto loop_cond = ops::LoopCond(scope.WithOpName("while/LoopCond"), less);
    auto switch_ =
        ops::Switch(scope.WithOpName("while/Switch"), merge.output, loop_cond);
    auto identity =
        ops::Identity(scope.WithOpName("while/Identity"), switch_.output_true);
    auto one = ops::Const<int32>(
        scope.WithOpName("while/add/y").WithControlDependencies(identity), 1);
    auto add = ops::Add(scope.WithOpName("while/add"), identity, one);
    auto next_iteration =
        ops::NextIteration(scope.WithOpName("while/NextIteration"), add);

    // Remove the dummy node and add the loop backedge.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(next_iteration.node(), 0, merge.output.node(), 1);

    TF_EXPECT_OK(scope.ToGraph(&graph));
  }

  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  GraphDef optimized_graph_def;
  graph.ToGraphDef(&optimized_graph_def);
  TF_ASSERT_OK(
      FunctionalizeControlFlowForGraphDef(&optimized_graph_def, &library));
  TF_ASSERT_OK(FunctionalizeControlFlow(&graph, &library));
  GraphDef converted_graph_def;
  graph.ToGraphDef(&converted_graph_def);

  for (const GraphDef& graph_def : {optimized_graph_def, converted_graph_def}) {
    NameAttrList cond_fn, body_fn;
    TF_EXPECT_OK(FindWhileCondAndBody(graph_def, &cond_fn, &body_fn));

    // Outer graph
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto source = ops::Placeholder(scope.WithOpName("source"), DT_INT32);
      auto while_op =
          ops::While(scope.WithOpName("while/LoopCond"),
                     std::initializer_list<Input>{source}, cond_fn, body_fn);
      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));
      TF_EXPECT_GRAPH_EQ(expected, graph_def);
    }

    // Condition graph
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto arg = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
      auto ten = ops::Const<int32>(
          scope.WithOpName("while/Less/y").WithControlDependencies(arg), 10);
      auto less = ops::Less(scope.WithOpName("while/Less"), arg, ten);
      auto retval = ops::_Retval(scope.WithOpName("retval0_RetVal"), less, 0);

      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));

      InstantiationResultForTest result;
      TF_EXPECT_OK(
          InstantiateFunctionForTest(cond_fn.name(), library, &result));

      EXPECT_EQ(DataTypeVector{DT_INT32}, result.arg_types);
      EXPECT_EQ(DataTypeVector{DT_BOOL}, result.ret_types);
      TF_EXPECT_GRAPH_EQ(expected, result.gdef);
    }

    // Body graph.
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto arg = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
      auto identity = ops::Identity(scope.WithOpName("while/Identity"), arg);
      auto one = ops::Const<int32>(
          scope.WithOpName("while/add/y").WithControlDependencies(identity), 1);
      auto add = ops::Add(scope.WithOpName("while/add"), identity, one);
      auto retval = ops::_Retval(scope.WithOpName("retval0_RetVal"), add, 0);

      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));

      InstantiationResultForTest result;
      TF_EXPECT_OK(
          InstantiateFunctionForTest(body_fn.name(), library, &result));

      EXPECT_EQ(DataTypeVector{DT_INT32}, result.arg_types);
      EXPECT_EQ(DataTypeVector{DT_INT32}, result.ret_types);
      TF_EXPECT_GRAPH_EQ(expected, result.gdef);
    }
  }
}

// Graph:
// x = array_ops.placeholder(dtypes.int32)
// y = array_ops.placeholder(dtypes.int32)
// cond = lambda (i, j): i + 3 < 10
// body = lambda (i, j): (i < 10, j * 2)
// z = control_flow_ops.while_loop(cond, body, [x, y])
TEST(FunctionalizeControlFlow, TwoLoopVars) {
  Graph graph(OpRegistry::Global());
  {
    Scope scope = Scope::NewRootScope().ExitOnError();

    auto dummy = ops::Placeholder(scope.WithOpName("Dummy"), DT_INT32);

    auto x = ops::Placeholder(scope.WithOpName("Placeholder/x"), DT_INT32);
    auto y = ops::Placeholder(scope.WithOpName("Placeholder/y"), DT_INT32);
    auto enter_x =
        ops::internal::Enter(scope.WithOpName("while/Enter/x"), x, "aloop");
    auto enter_y =
        ops::internal::Enter(scope.WithOpName("while/Enter/y"), y, "aloop");
    auto merge_x = ops::Merge(scope.WithOpName("while/Merge/x"),
                              std::initializer_list<Input>{enter_x, dummy});
    auto merge_y = ops::Merge(scope.WithOpName("while/Merge/y"),
                              std::initializer_list<Input>{enter_y, dummy});

    // Loop condition
    auto three = ops::Const<int32>(scope.WithOpName("while/cond/three")
                                       .WithControlDependencies(merge_x.output),
                                   3);
    auto cond_add =
        ops::Add(scope.WithOpName("while/cond/Add"), merge_x.output, three);
    auto ten = ops::Const<int32>(scope.WithOpName("while/cond/ten")
                                     .WithControlDependencies(merge_x.output),
                                 10);
    auto less = ops::Less(scope.WithOpName("while/cond/Less"), cond_add, ten);
    auto loop_cond = ops::LoopCond(scope.WithOpName("while/LoopCond"), less);

    auto switch_x = ops::Switch(scope.WithOpName("while/Switch/x"),
                                merge_x.output, loop_cond);
    auto switch_y = ops::Switch(scope.WithOpName("while/Switch/y"),
                                merge_y.output, loop_cond);

    auto exit_x = ops::internal::Exit(scope.WithOpName("while/Exit/x"),
                                      switch_x.output_false);
    auto exit_y = ops::internal::Exit(scope.WithOpName("while/Exit/y"),
                                      switch_y.output_false);

    auto identity_x = ops::Identity(scope.WithOpName("while/Identity/x"),
                                    switch_x.output_true);
    auto identity_y = ops::Identity(scope.WithOpName("while/Identity/y"),
                                    switch_y.output_true);

    auto one = ops::Const<int32>(
        scope.WithOpName("while/add/one").WithControlDependencies(identity_x),
        1);
    auto two = ops::Const<int32>(
        scope.WithOpName("while/mul/two").WithControlDependencies(identity_x),
        2);

    auto add = ops::Add(scope.WithOpName("while/add"), identity_x, one);
    auto mul = ops::Add(scope.WithOpName("while/mul"), identity_y, two);
    auto next_iteration_x =
        ops::NextIteration(scope.WithOpName("while/NextIteration/x"), add);
    auto next_iteration_y =
        ops::NextIteration(scope.WithOpName("while/NextIteration/y"), mul);

    auto sink_x = ops::Identity(scope.WithOpName("sink_x"), exit_x);
    auto sink_y = ops::Identity(scope.WithOpName("sink_y"), exit_y);

    // Remove the dummy node and add the loop backedges.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(next_iteration_x.node(), 0, merge_x.output.node(),
                           1);
    scope.graph()->AddEdge(next_iteration_y.node(), 0, merge_y.output.node(),
                           1);

    TF_EXPECT_OK(scope.ToGraph(&graph));
  }

  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  GraphDef optimized_graph_def;
  graph.ToGraphDef(&optimized_graph_def);
  TF_ASSERT_OK(
      FunctionalizeControlFlowForGraphDef(&optimized_graph_def, &library));
  TF_ASSERT_OK(FunctionalizeControlFlow(&graph, &library));
  GraphDef converted_graph_def;
  graph.ToGraphDef(&converted_graph_def);

  for (const GraphDef& graph_def : {optimized_graph_def, converted_graph_def}) {
    NameAttrList cond_fn, body_fn;
    TF_EXPECT_OK(FindWhileCondAndBody(graph_def, &cond_fn, &body_fn));

    // Outer graph.
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto x = ops::Placeholder(scope.WithOpName("Placeholder/x"), DT_INT32);
      auto y = ops::Placeholder(scope.WithOpName("Placeholder/y"), DT_INT32);
      auto while_op =
          ops::While(scope.WithOpName("while/LoopCond"),
                     std::initializer_list<Input>{x, y}, cond_fn, body_fn);
      auto sink_x = ops::Identity(scope.WithOpName("sink_x"), while_op[0]);
      auto sink_y = ops::Identity(scope.WithOpName("sink_y"), while_op[1]);
      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));
      TF_EXPECT_GRAPH_EQ(expected, graph_def);
    }

    // Condition graph.
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
      auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_INT32, 1);
      auto three = ops::Const<int32>(scope.WithOpName("while/cond/three")
                                         .WithControlDependencies(arg0.output),
                                     3);
      auto cond_add =
          ops::Add(scope.WithOpName("while/cond/Add"), arg0.output, three);
      auto ten = ops::Const<int32>(scope.WithOpName("while/cond/ten")
                                       .WithControlDependencies(arg0.output),
                                   10);
      auto less = ops::Less(scope.WithOpName("while/cond/Less"), cond_add, ten);
      auto retval = ops::_Retval(scope.WithOpName("retval0_RetVal"), less, 0);

      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));

      InstantiationResultForTest result;
      TF_EXPECT_OK(
          InstantiateFunctionForTest(cond_fn.name(), library, &result));

      EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32}), result.arg_types);
      EXPECT_EQ(DataTypeVector{DT_BOOL}, result.ret_types);
      TF_EXPECT_GRAPH_EQ(expected, result.gdef);
    }

    // Body graph.
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
      auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_INT32, 1);

      auto identity_x =
          ops::Identity(scope.WithOpName("while/Identity/x"), arg0);
      auto identity_y =
          ops::Identity(scope.WithOpName("while/Identity/y"), arg1);

      auto one = ops::Const<int32>(
          scope.WithOpName("while/add/one").WithControlDependencies(identity_x),
          1);
      auto two = ops::Const<int32>(
          scope.WithOpName("while/mul/two").WithControlDependencies(identity_x),
          2);

      auto add = ops::Add(scope.WithOpName("while/add"), identity_x, one);
      auto mul = ops::Add(scope.WithOpName("while/mul"), identity_y, two);
      auto retval0 = ops::_Retval(scope.WithOpName("retval0_RetVal"), add, 0);
      auto retval1 = ops::_Retval(scope.WithOpName("retval1_RetVal"), mul, 1);

      GraphDef expected;
      TF_EXPECT_OK(scope.ToGraphDef(&expected));

      InstantiationResultForTest result;
      TF_EXPECT_OK(
          InstantiateFunctionForTest(body_fn.name(), library, &result));

      EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32}), result.arg_types);
      EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32}), result.ret_types);
      TF_EXPECT_GRAPH_EQ(expected, result.gdef);
    }
  }
}

// More complex example with nesting, loop-invariant arguments, and resource
// variables. Used for multiple tests with different node filters.
class ComplexTestFixture
    : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {
 protected:
  void SetUp() override {
    restrict_to_tpu_nodes_ = std::get<0>(GetParam());
    mark_inner_loop_tpu_ = std::get<1>(GetParam());
    mark_outer_loop_tpu_ = std::get<2>(GetParam());
  }
  void RunTest();

 private:
  void CheckOuterNodesFunctionalized(const GraphDef& graph_def,
                                     const FunctionLibraryDefinition& library,
                                     NameAttrList& inner_cond_fn,
                                     NameAttrList& inner_body_fn);
  void CheckInnerNodesFunctionalized(const GraphDef& graph_def,
                                     const FunctionLibraryDefinition& library,
                                     const NameAttrList& inner_cond_fn,
                                     const NameAttrList& inner_body_fn);

  bool restrict_to_tpu_nodes_ = false;
  bool mark_inner_loop_tpu_ = false;
  bool mark_outer_loop_tpu_ = false;
};

TEST_P(ComplexTestFixture, ComplexTests) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(
    FunctionalizeControlFlow, ComplexTestFixture,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool()),
    [](const ::testing::TestParamInfo<ComplexTestFixture::ParamType>& info) {
      bool restrict_to_tpu_nodes = std::get<0>(info.param);
      bool mark_inner_loop_tpu = std::get<1>(info.param);
      bool mark_outer_loop_tpu = std::get<2>(info.param);

      string node_string;
      if (mark_inner_loop_tpu && mark_outer_loop_tpu)
        node_string = "both_loops_tpu";
      else if (!mark_inner_loop_tpu && !mark_outer_loop_tpu)
        node_string = "no_loop_tpu";
      else
        node_string = mark_inner_loop_tpu ? "inner_loop_tpu" : "outer_loop_tpu";

      string name = absl::StrCat(
          restrict_to_tpu_nodes ? "restricted_" : "unrestricted_", node_string);
      return name;
    });

void ComplexTestFixture::RunTest() {
  // Graph:
  //
  // accum = resource_variable_ops.ResourceVariable(1)
  // x = array_ops.placeholder(2, dtype=dtypes.int32)
  // y = 3 + x
  //
  // def inner_body(j, k):
  //   add = state_ops.assign_add(accum, k * j + x)
  //   with ops.control_dependencies([add]):
  //     return [j + 1, k]
  //
  // def body(i):
  //   m = control_flow_ops.while_loop(lambda j, k: j < 5, inner_body,
  //                                   [1, y], name="inner")
  //   with ops.control_dependencies(m):
  //     return [i + 1]
  //
  // z = control_flow_ops.while_loop(lambda i: i < 10, body, [0], name="outer")
  Graph graph(OpRegistry::Global());
  {
    Scope scope = Scope::NewRootScope().ExitOnError();

    auto dummy = ops::Placeholder(scope.WithOpName("Dummy"), DT_INT32);

    auto x = ops::Placeholder(scope.WithOpName("x"), DT_INT32);
    auto three = ops::Const<int32>(scope.WithOpName("three"), 3);
    auto y = ops::Add(scope.WithOpName("y"), x, three);

    auto var = ops::VarHandleOp(scope.WithOpName("Variable"), DT_INT32,
                                TensorShape({}));

    // Outer loop
    auto zero = ops::Const<int32>(scope.WithOpName("outer/Const"), 0);
    auto enter_i =
        ops::internal::Enter(scope.WithOpName("outer/Enter_i"), zero, "outer");
    auto merge_i = ops::Merge(scope.WithOpName("outer/Merge_i"),
                              std::initializer_list<Input>{enter_i, dummy});
    auto ten = ops::Const<int32>(scope.WithOpName("outer/Less/y")
                                     .WithControlDependencies(merge_i.output),
                                 10);
    auto less_i =
        ops::Less(scope.WithOpName("outer/Less_i"), merge_i.output, ten);
    auto outer_loop_cond =
        ops::LoopCond(scope.WithOpName("outer/LoopCond"), less_i);
    auto switch_i = ops::Switch(scope.WithOpName("outer/Switch"),
                                merge_i.output, outer_loop_cond);
    auto exit_i = ops::internal::Exit(scope.WithOpName("outer/Exit"),
                                      switch_i.output_false);
    auto identity_i =
        ops::Identity(scope.WithOpName("outer/Identity"), switch_i.output_true);

    auto enter_x_outer =
        ops::internal::Enter(scope.WithOpName("outer/Enter_x"), x, "outer",
                             ops::internal::Enter::Attrs().IsConstant(true));
    auto enter_k_outer =
        ops::internal::Enter(scope.WithOpName("outer/Enter_k"), y, "outer",
                             ops::internal::Enter::Attrs().IsConstant(true));
    auto enter_var_outer =
        ops::internal::Enter(scope.WithOpName("outer/Enter_var"), var, "outer",
                             ops::internal::Enter::Attrs().IsConstant(true));

    // Inner loop
    auto one_j = ops::Const<int32>(
        scope.WithOpName("outer/j").WithControlDependencies(identity_i), 1);
    auto enter_j = ops::internal::Enter(scope.WithOpName("outer/inner/Enter_j"),
                                        one_j, "inner");
    auto enter_k =
        ops::internal::Enter(scope.WithOpName("outer/inner/Enter_k")
                                 .WithControlDependencies(identity_i),
                             enter_k_outer, "inner");
    auto enter_x = ops::internal::Enter(
        scope.WithOpName("outer/inner/Enter_x"), enter_x_outer, "inner",
        ops::internal::Enter::Attrs().IsConstant(true));
    auto enter_var = ops::internal::Enter(
        scope.WithOpName("outer/inner/Enter_var"), enter_var_outer, "inner",
        ops::internal::Enter::Attrs().IsConstant(true));

    auto merge_j = ops::Merge(scope.WithOpName("outer/inner/Merge_j"),
                              std::initializer_list<Input>{enter_j, dummy});
    auto merge_k = ops::Merge(scope.WithOpName("outer/inner/Merge_k"),
                              std::initializer_list<Input>{enter_k, dummy});

    auto five = ops::Const<int32>(scope.WithOpName("outer/inner/Five")
                                      .WithControlDependencies(merge_j.output),
                                  5);
    auto less_j =
        ops::Less(scope.WithOpName("outer/inner/Less_j"), merge_j.output, five);
    auto loop_cond =
        ops::LoopCond(scope.WithOpName("outer/inner/LoopCond"), less_j);

    auto switch_j = ops::Switch(scope.WithOpName("outer/inner/Switch_j"),
                                merge_j.output, loop_cond);
    auto switch_k = ops::Switch(scope.WithOpName("outer/inner/Switch_k"),
                                merge_k.output, loop_cond);
    auto exit_j = ops::internal::Exit(scope.WithOpName("outer/inner/Exit_j"),
                                      switch_j.output_false);
    auto exit_k = ops::internal::Exit(scope.WithOpName("outer/inner/Exit_k"),
                                      switch_k.output_false);
    auto identity_j = ops::Identity(scope.WithOpName("outer/inner/Identity_j"),
                                    switch_j.output_true);
    auto identity_k = ops::Identity(scope.WithOpName("outer/inner/Identity_k"),
                                    switch_k.output_true);

    // Variable update
    auto mul_jk =
        ops::Mul(scope.WithOpName("outer/inner/mul"), identity_j, identity_k);
    auto add_jkx =
        ops::Add(scope.WithOpName("outer/inner/add"), mul_jk, enter_x);
    auto assign = ops::AssignAddVariableOp(
        scope.WithOpName("outer/inner/assign_add"), enter_var, add_jkx);

    auto one = ops::Const<int32>(
        scope.WithOpName("outer/inner/One")
            .WithControlDependencies(
                absl::Span<const Operation>{assign.operation}),
        1);
    auto add_j =
        ops::Add(scope.WithOpName("outer/inner/add_j"), identity_j, one);

    auto next_iteration_j = ops::NextIteration(
        scope.WithOpName("outer/inner/NextIteration_j"), add_j);
    auto next_iteration_k = ops::NextIteration(
        scope.WithOpName("outer/inner/NextIteration_k"), identity_k);

    // Body and backedge for outer loop.
    auto one_outer = ops::Const<int32>(
        scope.WithOpName("outer/add/y").WithControlDependencies(identity_i), 1);
    auto add_i =
        ops::Add(scope.WithOpName("outer/add")
                     .WithControlDependencies(absl::Span<const Operation>{
                         exit_j.output.op(), exit_k.output.op()}),
                 identity_i, one_outer);
    auto next_iteration_i =
        ops::NextIteration(scope.WithOpName("outer/NextIteration"), add_i);

    auto sink = ops::Identity(scope.WithOpName("sink"), exit_i);

    // Remove the dummy node and add the loop backedge.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(next_iteration_i.node(), 0, merge_i.output.node(),
                           1);
    scope.graph()->AddEdge(next_iteration_j.node(), 0, merge_j.output.node(),
                           1);
    scope.graph()->AddEdge(next_iteration_k.node(), 0, merge_k.output.node(),
                           1);

    TF_EXPECT_OK(scope.ToGraph(&graph));
  }
  // Add '_tpu_replicate' attributes as specified.
  for (Node* n : graph.nodes()) {
    string name = n->name();
    bool is_inner_node = name.find("outer/inner/") != string::npos;
    bool is_outer_node = !is_inner_node && name.find("outer/") != string::npos;
    if ((is_inner_node && mark_inner_loop_tpu_) ||
        (is_outer_node && mark_outer_loop_tpu_)) {
      n->AddAttr("_tpu_replicate", "cluster");
    }
  }

  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  GraphDef orig_graph_def, optimized_graph_def;
  graph.ToGraphDef(&orig_graph_def);
  optimized_graph_def = orig_graph_def;
  // If `restrict_to_tpu_nodes_` is true let filter function return true for
  // `_tpu_replicate` nodes, otherwise don't set filter.
  NodeFilter node_filter =
      restrict_to_tpu_nodes_
          ? [](const Node* n) { return n->attrs().Find("_tpu_replicate"); }
          : NodeFilter{};

  Status status1 = FunctionalizeControlFlowForGraphDef(&optimized_graph_def,
                                                       &library, node_filter);
  Status status2 = FunctionalizeControlFlow(&graph, &library, node_filter);
  ASSERT_EQ(status1, status2);
  if (restrict_to_tpu_nodes_ && mark_outer_loop_tpu_ && !mark_inner_loop_tpu_) {
    // This case violates the precondition of `FunctionalizeControlFlow`, we
    // expect an internal error.
    ASSERT_EQ(errors::IsInternal(status1), true);
    return;
  } else {
    // Supported cases, no error expected.
    TF_ASSERT_OK(status1);
  }

  GraphDef optimized_converted_graph_def;
  graph.ToGraphDef(&optimized_converted_graph_def);
  for (const GraphDef& graph_def :
       {optimized_graph_def, optimized_converted_graph_def}) {
    NameAttrList inner_cond_fn, inner_body_fn;
    if (!restrict_to_tpu_nodes_ ||
        (restrict_to_tpu_nodes_ && mark_outer_loop_tpu_ &&
         mark_inner_loop_tpu_)) {
      // We expect that both inner and outer nodes have been functionalized.
      CheckOuterNodesFunctionalized(graph_def, library, inner_cond_fn,
                                    inner_body_fn);
      CheckInnerNodesFunctionalized(graph_def, library, inner_cond_fn,
                                    inner_body_fn);
    } else /*restrict_to_tpu_nodes_ == true*/ {
      if (!mark_outer_loop_tpu_ && !mark_inner_loop_tpu_) {
        // Graph has no TPU nodes so we expect no functionalization.
        TF_EXPECT_GRAPH_EQ(orig_graph_def, graph_def);
      } else if (!mark_outer_loop_tpu_ && mark_inner_loop_tpu_) {
        // We expect that only inner nodes have been functionalized.
        TF_EXPECT_OK(
            FindWhileCondAndBody(graph_def, &inner_cond_fn, &inner_body_fn));
        CheckInnerNodesFunctionalized(graph_def, library, inner_cond_fn,
                                      inner_body_fn);
      }
    }
  }
}

void ComplexTestFixture::CheckOuterNodesFunctionalized(
    const GraphDef& graph_def, const FunctionLibraryDefinition& library,
    NameAttrList& inner_cond_fn, NameAttrList& inner_body_fn) {
  NameAttrList outer_cond_fn, outer_body_fn;
  TF_EXPECT_OK(FindWhileCondAndBody(graph_def, &outer_cond_fn, &outer_body_fn));

  // Outer graph.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto x = ops::Placeholder(scope.WithOpName("x"), DT_INT32);
    auto three = ops::Const<int32>(scope.WithOpName("three"), 3);
    auto y = ops::Add(scope.WithOpName("y"), x, three);

    auto var = ops::VarHandleOp(scope.WithOpName("Variable"), DT_INT32,
                                TensorShape({}));

    auto zero = ops::Const<int32>(scope.WithOpName("outer/Const"), 0);

    auto while_op = ops::While(scope.WithOpName("outer/LoopCond"),
                               std::initializer_list<Input>{zero, y, x, var},
                               outer_cond_fn, outer_body_fn);
    auto sink = ops::Identity(scope.WithOpName("sink"), while_op[0]);
    GraphDef expected;
    TF_EXPECT_OK(scope.ToGraphDef(&expected));
    TF_EXPECT_GRAPH_EQ(expected, graph_def);
  }

  // Outer condition graph.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_INT32, 1);
    auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_INT32, 2);
    auto arg3 = ops::_Arg(scope.WithOpName("arg3"), DT_RESOURCE, 3);

    auto ten = ops::Const<int32>(
        scope.WithOpName("outer/Less/y").WithControlDependencies(arg0.output),
        10);
    auto less = ops::Less(scope.WithOpName("outer/Less_i"), arg0, ten);
    auto retval = ops::_Retval(scope.WithOpName("retval0_RetVal"), less, 0);

    GraphDef expected;
    TF_EXPECT_OK(scope.ToGraphDef(&expected));

    InstantiationResultForTest result;
    TF_EXPECT_OK(
        InstantiateFunctionForTest(outer_cond_fn.name(), library, &result));

    EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32, DT_INT32, DT_RESOURCE}),
              result.arg_types);
    EXPECT_EQ(DataTypeVector{DT_BOOL}, result.ret_types);
    TF_EXPECT_GRAPH_EQ(expected, result.gdef);
  }

  // Outer body graph.
  {
    InstantiationResultForTest result;
    TF_EXPECT_OK(
        InstantiateFunctionForTest(outer_body_fn.name(), library, &result));

    // Find the inner condition and body names.
    TF_EXPECT_OK(
        FindWhileCondAndBody(result.gdef, &inner_cond_fn, &inner_body_fn));

    Scope scope = Scope::NewRootScope().ExitOnError();
    auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_INT32, 1);
    auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_INT32, 2);
    auto arg3 = ops::_Arg(scope.WithOpName("arg3"), DT_RESOURCE, 3);

    auto identity_i = ops::Identity(scope.WithOpName("outer/Identity"), arg0);
    auto one_j = ops::Const<int32>(
        scope.WithOpName("outer/j").WithControlDependencies(identity_i), 1);
    auto while_op =
        ops::While(scope.WithOpName("outer/inner/LoopCond"),
                   std::initializer_list<Input>{one_j, arg1, arg2, arg3},
                   inner_cond_fn, inner_body_fn);

    auto one_outer = ops::Const<int32>(
        scope.WithOpName("outer/add/y").WithControlDependencies(identity_i), 1);
    auto add_i =
        ops::Add(scope.WithOpName("outer/add")
                     .WithControlDependencies(absl::Span<const Operation>{
                         while_op[0].op(), while_op[1].op()}),
                 identity_i, one_outer);

    auto retval0 = ops::_Retval(scope.WithOpName("retval0_RetVal"), add_i, 0);
    auto retval1 = ops::_Retval(scope.WithOpName("retval1_RetVal"), arg1, 1);
    auto retval2 = ops::_Retval(scope.WithOpName("retval2_RetVal"), arg2, 2);
    auto retval3 = ops::_Retval(scope.WithOpName("retval3_RetVal"), arg3, 3);

    GraphDef expected;
    TF_EXPECT_OK(scope.ToGraphDef(&expected));

    EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32, DT_INT32, DT_RESOURCE}),
              result.arg_types);
    EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32, DT_INT32, DT_RESOURCE}),
              result.ret_types);
    TF_EXPECT_GRAPH_EQ(expected, result.gdef);
  }
}

void ComplexTestFixture::CheckInnerNodesFunctionalized(
    const GraphDef& graph_def, const FunctionLibraryDefinition& library,
    const NameAttrList& inner_cond_fn, const NameAttrList& inner_body_fn) {
  // Inner condition graph.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_INT32, 1);
    auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_INT32, 2);
    auto arg3 = ops::_Arg(scope.WithOpName("arg3"), DT_RESOURCE, 3);

    auto five = ops::Const<int32>(
        scope.WithOpName("outer/inner/Five").WithControlDependencies(arg0), 5);
    auto less_j = ops::Less(scope.WithOpName("outer/inner/Less_j"), arg0, five);
    auto retval = ops::_Retval(scope.WithOpName("retval0_RetVal"), less_j, 0);

    GraphDef expected;
    TF_EXPECT_OK(scope.ToGraphDef(&expected));

    InstantiationResultForTest result;
    TF_EXPECT_OK(
        InstantiateFunctionForTest(inner_cond_fn.name(), library, &result));

    EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32, DT_INT32, DT_RESOURCE}),
              result.arg_types);
    EXPECT_EQ(DataTypeVector{DT_BOOL}, result.ret_types);
    TF_EXPECT_GRAPH_EQ(expected, result.gdef);
  }

  // Inner body graph.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_INT32, 1);
    auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_INT32, 2);
    auto arg3 = ops::_Arg(scope.WithOpName("arg3"), DT_RESOURCE, 3);

    auto identity_j =
        ops::Identity(scope.WithOpName("outer/inner/Identity_j"), arg0);
    auto identity_k =
        ops::Identity(scope.WithOpName("outer/inner/Identity_k"), arg1);

    auto mul_jk =
        ops::Mul(scope.WithOpName("outer/inner/mul"), identity_j, identity_k);
    auto add_jkx = ops::Add(scope.WithOpName("outer/inner/add"), mul_jk, arg2);
    auto assign = ops::AssignAddVariableOp(
        scope.WithOpName("outer/inner/assign_add"), arg3, add_jkx);

    auto one = ops::Const<int32>(
        scope.WithOpName("outer/inner/One")
            .WithControlDependencies(
                absl::Span<const Operation>{assign.operation}),
        1);
    auto add_j =
        ops::Add(scope.WithOpName("outer/inner/add_j"), identity_j, one);

    auto retval0 = ops::_Retval(scope.WithOpName("retval0_RetVal"), add_j, 0);
    auto retval1 =
        ops::_Retval(scope.WithOpName("retval1_RetVal"), identity_k, 1);
    auto retval2 = ops::_Retval(scope.WithOpName("retval2_RetVal"), arg2, 2);
    auto retval3 = ops::_Retval(scope.WithOpName("retval3_RetVal"), arg3, 3);

    GraphDef expected;
    TF_EXPECT_OK(scope.ToGraphDef(&expected));

    InstantiationResultForTest result;
    TF_EXPECT_OK(
        InstantiateFunctionForTest(inner_body_fn.name(), library, &result));

    EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32, DT_INT32, DT_RESOURCE}),
              result.arg_types);
    EXPECT_EQ((DataTypeVector{DT_INT32, DT_INT32, DT_INT32, DT_RESOURCE}),
              result.ret_types);
    TF_EXPECT_GRAPH_EQ(expected, result.gdef);
  }
}

}  // namespace
}  // namespace tensorflow
