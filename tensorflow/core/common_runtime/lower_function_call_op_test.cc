/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/config/flag_defs.h"
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

AttrValue FuncAttr(const string& name) {
  AttrValue attr;
  attr.mutable_func()->set_name(name);
  return attr;
}

AttrValue FuncAttr(const string& name, const DataType type) {
  AttrValue attr;
  attr.mutable_func()->set_name(name);
  (*attr.mutable_func()->mutable_attr())["T"].set_type(type);
  return attr;
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

TEST(LowerFunctionCallTest, InlineFunctionCall) {
  using FDH = FunctionDefHelper;

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  FunctionDefLibrary f_lib_proto;

  // `add` node is not required to compute regular output `o`, but it must
  // execute because it is in `control_ret`.
  *(f_lib_proto.add_function()) =
      FDH::Create("AddAndMul", {"i: int32"}, {"o: int32"}, {},
                  {{{"add"}, "Add", {"i", "i"}, {{"T", DT_INT32}}},
                   {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_INT32}}}},
                  /*ret_def=*/{{"o", "ret:z:0"}},
                  /*control_ret_def=*/{{"must_execute", "add"}});

  // Construct a graph:
  //   A = Placeholder[dtype=int32]
  //   F = PartitionedCall[f=AddAndMul](a)
  //   B = Identity(func, ^func)
  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  Node* function_call;
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});
  TF_ASSERT_OK(NodeBuilder("F", "PartitionedCall", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("Tin", {DT_INT32})
                   .Attr("Tout", {DT_INT32})
                   .Attr("f", FuncAttr("AddAndMul"))
                   .Finalize(root.graph(), &function_call));
  TF_ASSERT_OK(root.DoShapeInference(function_call));

  auto b = ops::Identity(root.WithOpName("B"), Output(function_call, 0));
  root.graph()->AddControlEdge(function_call, b.node());

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  // Verify the resultant graph has no PartitionedCall ops and function body was
  // inlined into the main graph.
  int partitioned_call_count = 0;
  int add_count = 0;
  int mul_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsPartitionedCall()) partitioned_call_count++;
    if (op->type_string() == "Add") add_count++;
    if (op->type_string() == "Mul") mul_count++;
  }

  ASSERT_EQ(partitioned_call_count, 0);
  ASSERT_EQ(add_count, 1);
  ASSERT_EQ(mul_count, 1);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(b)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 100);
  }
}

TEST(LowerFunctionCallTest, InlineFunctionCallAfterPruning) {
  flags::Global().enable_function_pruning_before_inlining.reset(true);
  using FDH = FunctionDefHelper;

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  FunctionDefLibrary f_lib_proto;

  // `add` node is not required to compute regular output `o`, but it must
  // execute because it is in `control_ret`.
  // The `div` node and the unused arguments `j` and `k` should be pruned.
  *(f_lib_proto.add_function()) = FDH::Create(
      "AddAndMul", {"i: int32", "j: int32", "k: int32", "r: resource"},
      {"o: int32"}, {},
      {{{"add"}, "Add", {"i", "i"}, {{"T", DT_INT32}}},
       {{"div"}, "FloorDiv", {"i", "i"}, {{"T", DT_INT32}}},
       {{"gather"},
        "ResourceGather",
        {"r", "i"},
        {{"Tindices", DT_INT32}, {"dtype", DT_FLOAT}}},
       {{"ret"}, "Mul", {"i", "i"}, {{"T", DT_INT32}}}},
      /*ret_def=*/{{"o", "ret:z:0"}},
      /*control_ret_def=*/{{"must_execute", "add"}});

  // Construct a graph:
  //   X = Placeholder[dtype=int32]
  //   Y = Placeholder[dtype=int32]
  //   Z = Placeholder[dtype=int32]
  //   R = Placeholder[dtype=resource]
  //   F = PartitionedCall[f=AddAndMul](a)
  //   B = Identity(func, ^func)
  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto x = ops::Placeholder(root.WithOpName("X"), DT_INT32);
  auto y = ops::Placeholder(root.WithOpName("Y"), DT_INT32);
  auto z = ops::Placeholder(root.WithOpName("Z"), DT_INT32);
  auto r = ops::Placeholder(root.WithOpName("R"), DT_RESOURCE);
  Node* function_call;
  std::vector<NodeBuilder::NodeOut> inputs(
      {NodeBuilder::NodeOut(x.node()), NodeBuilder::NodeOut(y.node()),
       NodeBuilder::NodeOut(z.node()), NodeBuilder::NodeOut(r.node())});
  TF_ASSERT_OK(NodeBuilder("F", "PartitionedCall", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("Tin", {DT_INT32, DT_INT32, DT_INT32, DT_RESOURCE})
                   .Attr("Tout", {DT_INT32})
                   .Attr("f", FuncAttr("AddAndMul"))
                   .Finalize(root.graph(), &function_call));
  TF_ASSERT_OK(root.DoShapeInference(function_call));

  auto b = ops::Identity(root.WithOpName("B"), Output(function_call, 0));
  root.graph()->AddControlEdge(function_call, b.node());

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  // Verify the resultant graph has no PartitionedCall ops and function body was
  // inlined into the main graph.
  int partitioned_call_count = 0;
  int add_count = 0;
  int mul_count = 0;
  int floor_div_count = 0;
  int resource_gather_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsPartitionedCall()) partitioned_call_count++;
    if (op->type_string() == "Add") add_count++;
    if (op->type_string() == "Mul") mul_count++;
    if (op->type_string() == "FloorDiv") floor_div_count++;
    if (op->type_string() == "ResourceGather") resource_gather_count++;
  }

  ASSERT_EQ(partitioned_call_count, 0);
  ASSERT_EQ(add_count, 1);
  ASSERT_EQ(mul_count, 1);
  ASSERT_EQ(floor_div_count, 0);
  ASSERT_EQ(resource_gather_count, 0);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(x.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(b)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 100);
  }
  flags::Global().enable_function_pruning_before_inlining.reset(false);
}

TEST(LowerFunctionCallTest, DoNotInlineTpuOrXlaFunctions) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  FunctionDef tpu_func = test::function::XTimesTwo();
  tpu_func.mutable_signature()->set_name("TpuXTimesTwo");
  (*tpu_func.mutable_attr())["_tpu_replicate"].set_b(true);

  FunctionDef xla_func = test::function::XTimesTwo();
  xla_func.mutable_signature()->set_name("XlaXTimesTwo");
  (*xla_func.mutable_attr())["_xla_compile_id"].set_s("cluster_0");

  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XTimesTwo();

  // Construct a graph:
  //   A = Placeholder[dtype=int32]
  //   B = XTimesTwo[_tpu_replicate="cluster"](A)
  //   C = XTimesTwo[_xla_compile_id="cluster"](A)
  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_INT32);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* tpu_call;
  TF_ASSERT_OK(NodeBuilder("B", "PartitionedCall", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("Tin", {DT_INT32})
                   .Attr("Tout", {DT_INT32})
                   .Attr("f", FuncAttr("XTimesTwo", DT_INT32))
                   .Attr("_tpu_replicate", "cluster")
                   .Finalize(root.graph(), &tpu_call));

  Node* xla_call;
  TF_ASSERT_OK(NodeBuilder("C", "PartitionedCall", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("Tin", {DT_INT32})
                   .Attr("Tout", {DT_INT32})
                   .Attr("f", FuncAttr("XTimesTwo", DT_INT32))
                   .Attr("_xla_compile_id", "cluster")
                   .Finalize(root.graph(), &xla_call));

  TF_ASSERT_OK(root.DoShapeInference(tpu_call));
  TF_ASSERT_OK(root.DoShapeInference(xla_call));
  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(Rewrite(&graph));

  // Verify that we do not inline any of the special function call nodes.
  int partitioned_call_count = 0;
  for (const auto* op : graph->op_nodes()) {
    if (op->IsPartitionedCall()) partitioned_call_count++;
  }
  ASSERT_EQ(partitioned_call_count, 2);

  // Verify execution.
  ClientSession session(root, SessionOptionsWithInlining());
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(10));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(
        session.Run(feeds, {Output(tpu_call), Output(xla_call)}, &out_tensors));
    EXPECT_EQ(out_tensors.size(), 2);
    EXPECT_EQ(out_tensors[0].scalar<int>()(), 20);
    EXPECT_EQ(out_tensors[1].scalar<int>()(), 20);
  }
}

}  // namespace
}  // namespace tensorflow
