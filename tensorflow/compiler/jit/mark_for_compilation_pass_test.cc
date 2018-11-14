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

#include "tensorflow/compiler/jit/mark_for_compilation_pass_test_helper.h"

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

REGISTER_OP("UncompilableNullary").Output("o: float");
REGISTER_OP("UncompilableUnary").Input("a: float").Output("o: float");

std::unordered_map<string, string> GetClusters(const Graph& graph) {
  std::unordered_map<string, string> ids;
  for (Node* node : graph.nodes()) {
    string cluster;
    if (GetNodeAttr(node->attrs(), kXlaClusterAttr, &cluster).ok()) {
      CHECK(!cluster.empty());
      ids[node->name()] = cluster;
    }
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Clusters:";
    for (const auto& p : ids) {
      VLOG(2) << " " << p.first << " -> " << p.second;
    }
  }
  return ids;
}

absl::flat_hash_map<string, std::vector<string>> GetClusterSets(
    const Graph& g, std::vector<string>* cluster_names = nullptr) {
  CHECK(cluster_names == nullptr || cluster_names->empty());
  absl::flat_hash_map<string, std::vector<string>> cluster_sets;
  for (const auto& p : GetClusters(g)) {
    cluster_sets[p.second].push_back(p.first);
  }
  for (auto& p : cluster_sets) {
    if (cluster_names != nullptr) {
      cluster_names->push_back(p.first);
    }
    std::sort(p.second.begin(), p.second.end());
  }
  if (cluster_names != nullptr) {
    std::sort(cluster_names->begin(), cluster_names->end());
  }
  return cluster_sets;
}

TEST(XlaCompilationTest, Chains) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a =
        ops::SourceOp("UncompilableNullary", builder.opts().WithName("A"));
    Node* b = ops::UnaryOp("Relu", a, builder.opts().WithName("B"));
    Node* c = ops::UnaryOp("Relu", b, builder.opts().WithName("C"));
    Node* d =
        ops::UnaryOp("UncompilableUnary", c, builder.opts().WithName("D"));
    Node* e = ops::UnaryOp("Relu", d, builder.opts().WithName("E"));
    ops::UnaryOp("Relu", e, builder.opts().WithName("F"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_EQ(4, clusters.size());
  EXPECT_EQ(clusters["B"], clusters["C"]);
  EXPECT_EQ(clusters["E"], clusters["F"]);
  EXPECT_NE(clusters["B"], clusters["E"]);
  EXPECT_TRUE(clusters.find("A") == clusters.cend());
  EXPECT_TRUE(clusters.find("D") == clusters.cend());
}

TEST(XlaCompilationTest, UncompilableCycles) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b =
        ops::UnaryOp("UncompilableUnary", a, builder.opts().WithName("B"));
    ops::BinaryOp("MatMul", a, b, builder.opts().WithName("C"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  EXPECT_TRUE(clusters.empty());
}

TEST(XlaCompilationTest, CompilableCycles) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::UnaryOp("Relu", a, builder.opts().WithName("B"));
    ops::BinaryOp("MatMul", a, b, builder.opts().WithName("C"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  EXPECT_EQ(3, clusters.size());
  EXPECT_EQ(clusters["A"], clusters["B"]);
  EXPECT_EQ(clusters["A"], clusters["C"]);
}

TEST(XlaCompilationTest, Complex128Unsupported) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp(
        "Const", builder.opts()
                     .WithName("A")
                     .WithAttr("dtype", DT_COMPLEX128)
                     .WithAttr("value", Tensor(DT_COMPLEX128, TensorShape())));
    Node* b = ops::UnaryOp("Neg", a, builder.opts().WithName("B"));
    ops::BinaryOp("MatMul", a, b, builder.opts().WithName("C"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_TRUE(clusters.empty());
}

TEST(XlaCompilationTest, HalfSupported) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Tensor t(DT_HALF, TensorShape());
    t.scalar<Eigen::half>()() = static_cast<Eigen::half>(0.0f);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_HALF)
                                         .WithAttr("value", t));
    Node* b = ops::UnaryOp("Neg", a, builder.opts().WithName("B"));
    ops::BinaryOp("MatMul", a, b, builder.opts().WithName("C"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_FALSE(clusters.empty());
}

TEST(XlaCompilationTest, ConcatWithConstArg) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    Tensor t(DT_INT32, TensorShape());
    t.scalar<int32>()() = 0;
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* dim = ops::SourceOp("Const", builder.opts()
                                           .WithName("Dim")
                                           .WithAttr("dtype", DT_INT32)
                                           .WithAttr("value", t));
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", t));

    NodeBuilder concat_builder("Concat", "Concat",
                               builder.opts().op_registry());
    concat_builder.Input(dim).Input({a, a}).Attr("N", 2);
    builder.opts().FinalizeBuilder(&concat_builder);

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_EQ(3, clusters.size());  // Everything should be compiled.
}

TEST(XlaCompilationTest, FunctionCalls) {
  FunctionDef compilable = FunctionDefHelper::Define(
      "CompilableFn", {"n_a:float", "n_b:float"}, {"n_c:float"}, {},
      {{{"n_c"}, "Add", {"n_a", "n_b"}, {{"T", DT_FLOAT}}}});
  FunctionDef uncompilable =
      FunctionDefHelper::Define("UncompilableFn", {"n_a:float"}, {"n_c:float"},
                                {}, {{{"n_c"}, "UncompilableUnary", {"n_a"}}});
  FunctionDef noinline = compilable;
  noinline.mutable_signature()->set_name("NoInlineFn");
  AddAttr("_noinline", static_cast<bool>(true), noinline.mutable_attr());

  FunctionDefLibrary flib;
  *flib.add_function() = compilable;
  *flib.add_function() = uncompilable;
  *flib.add_function() = noinline;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  std::unique_ptr<Graph> graph(new Graph(&flib_def));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
    Node* a =
        ops::SourceOp("UncompilableNullary", builder.opts().WithName("A"));
    Node* b = ops::BinaryOp("CompilableFn", a, a, builder.opts().WithName("B"));
    Node* c = ops::UnaryOp("Relu", b, builder.opts().WithName("C"));
    ops::UnaryOp("UncompilableFn", c, builder.opts().WithName("D"));
    ops::BinaryOp("NoInlineFn", c, c, builder.opts().WithName("E"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph, &flib_def));
  auto clusters = GetClusters(*graph);

  EXPECT_EQ(2, clusters.size());
  EXPECT_FALSE(clusters["B"].empty());
  EXPECT_EQ(clusters["B"], clusters["C"]);
  EXPECT_TRUE(clusters.find("A") == clusters.cend());
  EXPECT_TRUE(clusters.find("D") == clusters.cend());
  EXPECT_TRUE(clusters.find("E") == clusters.cend());
}

// Metadata-only operators such as Shape/Rank/Size may not be the root of a
// cluster. This is partially to work around b/26800664, and partially because
// we should probably prefer to compile metadata operators with their producers
// wherever possible, rather than their consumers.
TEST(XlaCompilationTest, MetadataOpsDontStartClusters) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a =
        ops::SourceOp("UncompilableNullary", builder.opts().WithName("A"));
    // While all of the following ops are notionally compilable, none is
    // permitted
    // to start a cluster. So nothing should be compiled.
    Node* b = ops::UnaryOp("Shape", a, builder.opts().WithName("B"));
    Node* c = ops::UnaryOp("Rank", b, builder.opts().WithName("C"));
    Node* d = ops::UnaryOp("Size", c, builder.opts().WithName("D"));
    ops::UnaryOp("Shape", d, builder.opts().WithName("E"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_EQ(0, clusters.size());  // Nothing should be compiled.
}

static Status GradForUnaryCwise(FunctionDef* g,
                                std::vector<FunctionDefHelper::Node> nodes) {
  for (auto& n : nodes) {
    if (n.attr.empty()) {
      n.attr = {{"T", DT_FLOAT}};
    }
  }
  *g = FunctionDefHelper::Define(
      // Arg defs
      {"x: float", "dy: float"},
      // Ret val defs
      {"dx: float"},
      // Attr defs
      {},
      // Nodes
      nodes);
  return Status::OK();
}

// A gradient containing only supported operators
Status SupportedGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Tanh", {"x"}},
      {{"y2"}, "Square", {"y"}, {}, {"dy"}},
      FunctionDefHelper::Const("one", 1.0f),
      {{"a"}, "Sub", {"one", "y2"}},
      {{"dx"}, "Mul", {"dy", "a"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Supported", SupportedGrad);

// A gradient containing an unsupported operator.
Status UnsupportedGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Tanh", {"x"}},
      {{"y2"}, "UncompilableUnary", {"y"}, {}, {"dy"}},
      FunctionDefHelper::Const("one", 1.0f),
      {{"a"}, "Sub", {"one", "y2"}},
      {{"dx"}, "Mul", {"dy", "a"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Unsupported", UnsupportedGrad);

TEST(XlaCompilationTest, SymbolicGradients) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a =
        ops::SourceOp("UncompilableNullary", builder.opts().WithName("A"));

    // Builds a Symbolic gradient for Supported
    NodeBuilder b_builder("B", "SymbolicGradient",
                          builder.opts().op_registry());
    NameAttrList b_name_attr;
    b_name_attr.set_name("Supported");
    b_builder.Attr("f", b_name_attr);
    b_builder.Attr("Tin", {DT_FLOAT, DT_FLOAT});
    b_builder.Attr("Tout", {DT_FLOAT});
    b_builder.Input({a, a});
    Node* b = builder.opts().FinalizeBuilder(&b_builder);

    Node* c = ops::UnaryOp("Relu", b, builder.opts().WithName("C"));

    // Builds a Symbolic gradient for Unsupported
    NodeBuilder d_builder("D", "SymbolicGradient",
                          builder.opts().op_registry());
    NameAttrList d_name_attr;
    d_name_attr.set_name("Unsupported");
    d_builder.Attr("f", d_name_attr);
    d_builder.Attr("Tin", {DT_FLOAT, DT_FLOAT});
    d_builder.Attr("Tout", {DT_FLOAT});
    d_builder.Input({c, c});
    builder.opts().FinalizeBuilder(&d_builder);

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  EXPECT_EQ(2, clusters.size());
  EXPECT_FALSE(clusters["B"].empty());
  EXPECT_EQ(clusters["B"], clusters["C"]);
  EXPECT_TRUE(clusters.find("A") == clusters.cend());
  EXPECT_TRUE(clusters.find("D") == clusters.cend());
}

TEST(XlaCompilationTest, Loops) {
  // Regression test for b/32350199, where the autoclustering code introduced a
  // deadlock in a graph containing a while loop.
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(root.WithOpName("A"), DT_FLOAT);
  auto b = ops::Placeholder(root.WithOpName("B"), DT_FLOAT);
  auto c = ops::Add(root.WithOpName("C"), a, b);
  auto enter = ops::internal::Enter(root, c, "aframe");
  auto next_iter = ops::NextIteration(root, enter);
  auto exit = ops::internal::Exit(root, next_iter);
  auto d = ops::Add(root.WithOpName("D"), c, exit);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_EXPECT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  // Nothing should be compiled. In particular, 'd' and 'c' must not be
  // compiled.
  EXPECT_EQ(0, clusters.size());
}

TEST(XlaCompilationTest, CyclesWithAllDifferentScopesGlobalJitOverridden) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor())
                                         .WithAttr(kXlaScopeAttr, "ScopeA"));
    Node* b = ops::UnaryOp(
        "Relu", a,
        builder.opts().WithName("B").WithAttr(kXlaScopeAttr, "ScopeB"));
    ops::BinaryOp(
        "MatMul", a, b,
        builder.opts().WithName("C").WithAttr(kXlaScopeAttr, "ScopeC"));
    TF_CHECK_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  FunctionDefLibrary flib;
  FunctionLibraryDefinition flib_def(graph->op_registry(), flib);
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(
      &graph, &flib_def, &session_options));
  auto clusters = GetClusters(*graph);

  // The computation is: C = A + relu(A)
  // where A sits in ScopeA, relu(A) sits in ScopeB, and C sits in ScopeC.
  // In this case, the GlobalJitLevel overrides the scopes to cluster while
  // ignoring scopes.
  EXPECT_EQ(3, clusters.size());
  EXPECT_EQ(clusters["A"], clusters["B"]);
  EXPECT_EQ(clusters["A"], clusters["C"]);
}

TEST(XlaCompilationTest, CyclesWithAllDifferentScopes) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor())
                                         .WithAttr(kXlaScopeAttr, "ScopeA"));
    Node* b = ops::UnaryOp(
        "Relu", a,
        builder.opts().WithName("B").WithAttr(kXlaScopeAttr, "ScopeB"));
    ops::BinaryOp(
        "MatMul", a, b,
        builder.opts().WithName("C").WithAttr(kXlaScopeAttr, "ScopeC"));
    TF_CHECK_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  // The computation is: C = A + relu(A)
  // where A sits in ScopeA, relu(A) sits in ScopeB, and C sits in ScopeC.
  // In this case, we cannot fuse anything, and there are no clusters.
  EXPECT_EQ(0, clusters.size());
}

TEST(XlaCompilationTest, CyclesWithSplittingScopes) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor())
                                         .WithAttr(kXlaScopeAttr, "Scope1"));
    Node* b = ops::UnaryOp(
        "Relu", a,
        builder.opts().WithName("B").WithAttr(kXlaScopeAttr, "Scope1"));
    Node* c = ops::BinaryOp(
        "MatMul", a, b,
        builder.opts().WithName("C").WithAttr(kXlaScopeAttr, "Scope2"));
    ops::BinaryOp(
        "Add", b, c,
        builder.opts().WithName("D").WithAttr(kXlaScopeAttr, "Scope2"));
    TF_CHECK_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  // The computation is: D = relu(A) + (A @ relu(A))
  // where A and relu(A) are in Scope1, and the @, + ops are in Scope2.
  // In this case, we can fuse the A and relu(A), and we can fuse the
  // second half of the operations; there are two clusters.
  EXPECT_EQ(4, clusters.size());
  EXPECT_EQ(clusters["A"], clusters["B"]);
  EXPECT_NE(clusters["A"], clusters["C"]);
  EXPECT_EQ(clusters["C"], clusters["D"]);
}

TEST(XlaCompilationTest, CyclesWithDifferentScopesAndBridge) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor())
                                         .WithAttr(kXlaScopeAttr, "ScopeA"));
    Node* b = ops::UnaryOp(
        "Relu", a,
        builder.opts().WithName("B").WithAttr(kXlaScopeAttr, "ScopeB"));
    ops::BinaryOp("MatMul", a, b, builder.opts().WithName("C"));
    TF_CHECK_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  // The computation is: C = A @ relu(A)
  // where A sits in ScopeA, relu(A) sits in ScopeB, and C sits in ScopeC.
  // In this case, we cannot fuse anything.
  EXPECT_EQ(2, clusters.size());
  EXPECT_NE(clusters["A"], clusters["B"]);
  EXPECT_EQ(clusters["B"], clusters["C"]);
}

namespace {
Node* MakeRead(const Scope& scope, const string& id) {
  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output read =
      ops::ReadVariableOp(scope.WithOpName("Read" + id), var_handle, DT_FLOAT);
  return read.node();
}

Node* MakeWrite(const Scope& scope, const string& id) {
  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output value_to_write =
      ops::Const(scope.WithOpName("ValueToAssign" + id), 1.0f);
  ops::AssignVariableOp assign_op(scope.WithOpName("Assignment" + id),
                                  var_handle, value_to_write);
  return assign_op.operation.node();
}

Node* MakeNeutral(const Scope& scope, const string& id) {
  return ops::Const(scope.WithOpName("Const" + id), 42.0f).node();
}
}  // namespace

TEST(XlaCompilationTest, ResourcesClusteringAllowed) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(read, write);

  FixupSourceAndSinkEdges(root.graph());
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_EXPECT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  absl::flat_hash_map<string, std::vector<string>> cluster_sets =
      GetClusterSets(*graph);
  ASSERT_EQ(cluster_sets.size(), 1);
  std::vector<string> expected_clustered_nodes = {"AssignmentW", "ReadR",
                                                  "ValueToAssignW"};
  ASSERT_EQ(cluster_sets.begin()->second, expected_clustered_nodes);
}

TEST(XlaCompilationTest, ResourcesClusteringDisallowed) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, read);

  FixupSourceAndSinkEdges(root.graph());
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_EXPECT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  absl::flat_hash_map<string, std::vector<string>> cluster_sets =
      GetClusterSets(*graph);
  ASSERT_EQ(cluster_sets.size(), 1);
  std::vector<string> expected_clustered_nodes = {"AssignmentW",
                                                  "ValueToAssignW"};
  ASSERT_EQ(cluster_sets.begin()->second, expected_clustered_nodes);
}

TEST(XlaCompilationTest, ChainOfOps) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* write_0 = MakeWrite(root, "W0");
  Node* neutral_0 = MakeNeutral(root, "N0");
  Node* read_0 = MakeRead(root, "R0");
  Node* write_1 = MakeWrite(root, "W1");
  Node* neutral_1 = MakeNeutral(root, "N1");
  Node* read_1 = MakeRead(root, "R1");

  root.graph()->AddControlEdge(write_0, neutral_0);
  root.graph()->AddControlEdge(neutral_0, read_0);
  root.graph()->AddControlEdge(read_0, write_1);
  root.graph()->AddControlEdge(write_1, neutral_1);
  root.graph()->AddControlEdge(neutral_1, read_1);

  FixupSourceAndSinkEdges(root.graph());
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_EXPECT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::vector<string> cluster_names;
  absl::flat_hash_map<string, std::vector<string>> cluster_sets =
      GetClusterSets(*graph, &cluster_names);

  ASSERT_EQ(cluster_sets.size(), 2);

  std::vector<string> expected_clustered_nodes_a = {"AssignmentW0", "ConstN0",
                                                    "ValueToAssignW0"};
  ASSERT_EQ(cluster_sets[cluster_names[0]], expected_clustered_nodes_a);

  std::vector<string> expected_clustered_nodes_b = {
      "AssignmentW1", "ConstN1", "ReadR0", "ValueToAssignW1"};
  ASSERT_EQ(cluster_sets[cluster_names[1]], expected_clustered_nodes_b);
}

TEST(XlaCompilationTest, IllegalCycle_UsefulErrorMessage) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope root = Scope::NewRootScope().ExitOnError();
  {
    auto BuildNoopNode = [](absl::string_view name, Graph* graph) {
      NodeDefBuilder builder(name, "NoOp");
      NodeDef def;
      TF_CHECK_OK(builder.Finalize(&def));

      Status status;
      Node* node = graph->AddNode(def, &status);
      TF_CHECK_OK(status);
      return node;
    };

    Node* a = BuildNoopNode("a", graph.get());
    Node* b = BuildNoopNode("b", graph.get());
    Node* c = BuildNoopNode("c", graph.get());
    graph->AddControlEdge(a, b);
    graph->AddControlEdge(b, c);
    graph->AddControlEdge(c, a);
  }

  TF_EXPECT_OK(root.ToGraph(graph.get()));

  Status status = MarkForCompilationPassTestHelper::MarkForCompilation(&graph);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.ToString(),
                                "Edge from c to a would create a cycle.\n"
                                "+-> a\n"
                                "|   b\n"
                                "+-- c\n"));
}

TEST(XlaCompilationTest, Retval) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::UnaryOp("Relu", a, builder.opts().WithName("B"));
    ops::UnaryOp("_Retval", b,
                 builder.opts()
                     .WithName("R")
                     .WithAttr("T", DT_FLOAT)
                     .WithAttr("index", 0));

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  EXPECT_EQ(2, clusters.size());
  EXPECT_TRUE(clusters.find("R") == clusters.cend());
  EXPECT_EQ(clusters["A"], clusters["B"]);
}

TEST(XlaCompilationTest, DontCountIdentityOps) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope root = Scope::NewRootScope().ExitOnError();
  {
    auto a = ops::_Arg(root.WithOpName("A"), DT_INT32, 0);
    auto b = ops::Identity(root.WithOpName("B"), a);
    auto c = ops::Identity(root.WithOpName("C"), b);
    auto r = ops::_Retval(root.WithOpName("R"), c, 0);
  }
  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  EXPECT_TRUE(clusters.empty());
}

TEST(XlaCompilationTest, DontCountIdentityOpsWithLocalJit) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope root = Scope::NewRootScope().ExitOnError();
  {
    auto a = ops::_Arg(root.WithOpName("A"), DT_INT32, 0);
    auto b = ops::Identity(root.WithOpName("B"), a);
    b.node()->AddAttr(kXlaCompileAttr, true);
    auto r = ops::_Retval(root.WithOpName("R"), b, 0);
  }
  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  EXPECT_TRUE(clusters.empty());
}

TEST(XlaCompilationTest, ConstOp) {
  // valid data type
  {
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    Scope root = Scope::NewRootScope().ExitOnError();
    auto c = ops::Const(root.WithOpName("const"), 0.5f);
    c.node()->AddAttr(kXlaCompileAttr, true);
    TF_ASSERT_OK(root.ToGraph(graph.get()));
    TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
    EXPECT_EQ(1, GetClusters(*graph).size());
  }

  // invalid data type
  {
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    Scope root = Scope::NewRootScope().ExitOnError();
    auto c = ops::Const(root.WithOpName("const"), string("string"));
    c.node()->AddAttr(kXlaCompileAttr, true);
    TF_ASSERT_OK(root.ToGraph(graph.get()));
    TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
    EXPECT_TRUE(GetClusters(*graph).empty());
  }
}

TEST(XlaCompilationTest, DontClusterIdentityWithRefInput) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output variable = ops::Variable(root.WithOpName("variable"),
                                  PartialTensorShape{}, DT_FLOAT);
  Output read = ops::Identity(root.WithOpName("read"), variable);
  Output neg = ops::Negate(root.WithOpName("negate"), read);
  Output add = ops::Add(root.WithOpName("add"), neg, neg);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  ASSERT_FALSE(clusters.empty());
  string cluster_name = clusters.begin()->second;

  std::unordered_map<string, string> expected_clusters(
      {{"negate", cluster_name}, {"add", cluster_name}});
  EXPECT_EQ(clusters, expected_clusters);
}

TEST(XlaCompilationTest, ClusterIdentityWithNonRefInput) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output variable = ops::Variable(root.WithOpName("variable"),
                                  PartialTensorShape{}, DT_FLOAT);
  Output read = ops::Identity(root.WithOpName("read"), variable);
  Output neg = ops::Negate(root.WithOpName("negate"), read);
  Output identity = ops::Negate(root.WithOpName("identity"), neg);
  Output add = ops::Add(root.WithOpName("add"), identity, neg);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  ASSERT_FALSE(clusters.empty());
  string cluster_name = clusters.begin()->second;

  std::unordered_map<string, string> expected_clusters(
      {{"negate", cluster_name},
       {"identity", cluster_name},
       {"add", cluster_name}});
  EXPECT_EQ(clusters, expected_clusters);
}

TEST(XlaCompilationTest, ClusterControlTrigger) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output recv_a = ops::_Recv(root.WithOpName("recv_a"), DT_BOOL, "tensor_a",
                             "sender", 0, "receiver");
  Output recv_b = ops::_Recv(root.WithOpName("recv_b"), DT_BOOL, "tensor_b",
                             "sender", 0, "receiver");
  Output const_a = ops::Const(root.WithOpName("const_a"), 42);

  ops::ControlTrigger ctrl_trigger_a(root.WithOpName("ctrl_trigger_a"));
  ops::ControlTrigger ctrl_trigger_b(root.WithOpName("ctrl_trigger_b"));
  root.graph()->AddControlEdge(recv_a.node(), ctrl_trigger_a.operation.node());
  root.graph()->AddControlEdge(recv_b.node(), ctrl_trigger_a.operation.node());
  root.graph()->AddControlEdge(ctrl_trigger_b.operation.node(), const_a.node());

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  // TODO(b/118970344): ctrl_trigger_a has inputs with mismatching deadness so
  // it won't be clustered.  ctrl_trigger_b is okay to cluster but we don't
  // cluster it because of b/118970344.
  EXPECT_TRUE(clusters.empty());
}

TEST(XlaCompilationTest, RandomShape) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output shape_shape = ops::Const(root.WithOpName("shape_shape"), {2}, {1});
  Output shape =
      ops::RandomUniformInt(root.WithOpName("shape"), shape_shape,
                            ops::Const(root.WithOpName("minval"), 1),
                            ops::Const(root.WithOpName("maxval"), 20));
  Output reshape_input =
      ops::Placeholder(root.WithOpName("reshape_input"), DT_FLOAT,
                       ops::Placeholder::Shape(TensorShape({500, 500})));
  Output reshape =
      ops::Reshape(root.WithOpName("reshape"), reshape_input, shape);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_EQ(clusters["shape"], "");
}

TEST(XlaCompilationTest, RandomShapeWithFunc) {
  Scope root = Scope::DisabledShapeInferenceScope().ExitOnError();

  FunctionDefLibrary flib_def;
  FunctionDef func = FunctionDefHelper::Create(
      /*function_name=*/"Stateful_func", /*in_def=*/{},
      /*out_def=*/{"out: int32"},
      /*attr_def*/
      {}, /*node_def=*/
      {FunctionDefHelper::Const("shape_shape", 2),
       FunctionDefHelper::Const("minval", 1),
       FunctionDefHelper::Const("maxval", 20),
       {{"shape"},
        "RandomUniformInt",
        {"shape_shape:output:0", "minval:output:0", "maxval:output:0"},
        {{"Tout", DataType::DT_INT32}, {"T", DataType::DT_INT32}}}},
      /*ret_def=*/{{"out", "shape:output:0"}});

  func.mutable_signature()->set_is_stateful(true);
  *flib_def.add_function() = std::move(func);
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));
  NodeDef call_node;
  call_node.set_name("fn_call");
  call_node.set_op("Stateful_func");
  Status status;
  Node* call = root.graph()->AddNode(call_node, &status);
  TF_ASSERT_OK(status);

  Output shape = Output(call, 0);
  Output reshape_input =
      ops::Placeholder(root.WithOpName("reshape_input"), DT_FLOAT,
                       ops::Placeholder::Shape(TensorShape({500, 500})));
  Output reshape =
      ops::Reshape(root.WithOpName("reshape"), reshape_input, shape);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));
  auto fld = absl::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(),
                                                          flib_def);
  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph, fld.get()));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_EQ(clusters["fn_call"], "");
}

TEST(XlaCompilationTest, RandomShapeOnXlaDevice) {
  absl::string_view xla_gpu_device =
      "/job:worker/replica:0/task:0/device:XLA_GPU:0";

  Scope root = Scope::NewRootScope().ExitOnError();
  Output shape_shape =
      ops::Const(root.WithOpName("test/shape_shape"), {2}, {1});
  Output shape =
      ops::RandomUniformInt(root.WithOpName("test/shape_rng"), shape_shape,
                            ops::Const(root.WithOpName("test/minval"), 1),
                            ops::Const(root.WithOpName("test/maxval"), 20));
  Output reshape_input =
      ops::Placeholder(root.WithOpName("test/reshape_input"), DT_FLOAT,
                       ops::Placeholder::Shape(TensorShape({500, 500})));
  Output reshape =
      ops::Reshape(root.WithOpName("test/reshape"), reshape_input, shape);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  for (Node* n : graph->nodes()) {
    if (absl::StartsWith(n->name(), /*prefix=*/"test/")) {
      n->set_assigned_device_name(string(xla_gpu_device));
    }
  }
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_EQ(clusters["test/shape_rng"], "");
  EXPECT_EQ(clusters["test/reshape"], "");
}

TEST(XlaCompilationTest, TensorArrayShapeOnXlaDevice) {
  absl::string_view xla_gpu_device =
      "/job:worker/replica:0/task:0/device:XLA_GPU:0";
  Scope root = Scope::NewRootScope().ExitOnError();
  ops::TensorArray tensor_array(root.WithOpName("test/tensor_array"), 1,
                                DT_INT32);
  Output zero = ops::Const(root.WithOpName("test/zero"), 0);
  ops::TensorArrayWrite tensor_array_write(
      root.WithOpName("test/write"), tensor_array.handle, zero,
      ops::Const(root.WithOpName("test/forty_two"), 42.0f), tensor_array.flow);
  Output tensor_array_read =
      ops::TensorArrayRead(root.WithOpName("test/read"), tensor_array.handle,
                           zero, tensor_array_write.flow_out, DT_INT32);
  Output reshape =
      ops::Reshape(root.WithOpName("test/reshape"),
                   ops::Placeholder(root.WithOpName("placeholder"), DT_FLOAT),
                   tensor_array_read);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  for (Node* n : graph->nodes()) {
    if (absl::StartsWith(n->name(), /*prefix=*/"test/")) {
      n->set_assigned_device_name(string(xla_gpu_device));
    }
  }
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_NE(clusters["test/read"], "");
  EXPECT_EQ(clusters["test/read"], clusters["test/reshape"]);
}

TEST(XlaCompilationTest, DontClusterMergingNodes) {
  // MatMulCombined below takes data from nodes on GPU0 and GPU1 and is placed
  // on GPU1. However, it should not be clustered with the previous node on
  // GPU1, because that will serialize production of its inputs that should be
  // done in parallel.
  //
  // This graph is:
  // (Const0, Const0) -> MatMul0
  // (Const1, Const1) -> MatMul1
  // (MatMul0, MatMul1) -> MatMulCombined
  //
  // Device0: [Const0, Const0, MatMul0]
  // Device1: [Const1, Const1, MatMul1, MatMulCombined]
  //
  // Cluster0: [Const0, Const0, MatMul0]
  // Cluster1: [Const1, Const1, MatMul1]
  // Cluster2: [MatMulCombined]
  Scope root = Scope::NewRootScope().ExitOnError();
  absl::string_view xla_gpu_dev0 =
      "/job:worker/replica:0/task:0/device:XLA_GPU:0";
  absl::string_view xla_gpu_dev1 =
      "/job:worker/replica:0/task:0/device:XLA_GPU:1";
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Output a = ops::Const(root.WithOpName("A_dev0"), 1.0f, {2, 2});
  Output b = ops::Const(root.WithOpName("B_dev1"), 1.0f, {2, 2});
  Output matmul0 = ops::MatMul(root.WithOpName("MatMul0_dev0"), a, a);
  Output matmul1 = ops::MatMul(root.WithOpName("MatMul1_dev1"), b, b);

  Output combined =
      ops::MatMul(root.WithOpName("MatMulCombined_dev1"), matmul0, matmul1);
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  for (Node* n : graph->nodes()) {
    if (absl::EndsWith(n->name(), /*suffix=*/"dev0")) {
      n->set_assigned_device_name(string(xla_gpu_dev0));
    } else if (absl::EndsWith(n->name(), /*suffix=*/"dev1")) {
      n->set_assigned_device_name(string(xla_gpu_dev1));
    }
  }
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  // Each of the MatMuls should be in a separate cluster.
  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_NE(clusters["MatMul0_dev0"], clusters["MatMul1_dev1"]);
  EXPECT_NE(clusters["MatMulCombined_dev1"], clusters["MatMul0_dev0"]);
  EXPECT_NE(clusters["MatMulCombined_dev1"], clusters["MatMul1_dev1"]);
  EXPECT_EQ(clusters["A_dev0"], clusters["MatMul0_dev0"]);
  EXPECT_EQ(clusters["B_dev1"], clusters["MatMul1_dev1"]);
}

// TODO(b/117085735): This form of clustering should be prevented.
TEST(XlaCompilationTest, NOT_DontClusterSpreadingNodes) {
  // MatMulSource below creates data for nodes on GPU0 and GPU1 and is placed
  // on GPU0. However, it should not be clustered with the next node on
  // GPU0, because that will prevent the node on GPU1 from beginning its work as
  // soon as the data has been produced.
  //
  // This graph is:
  // (Const0, Const0) -> MatMulSource
  // MatMulSource -> (MatMul0, MatMul1)
  //
  // Device0: [Const0, Const1, MatMulSource, MatMul0]
  // Device1: [MatMul1]
  //
  // Cluster0: [Const0, Const1, MatMulSource]
  // Cluster1: [MatMul0]
  // Cluster2: [MatMul1]
  Scope root = Scope::NewRootScope().ExitOnError();
  absl::string_view xla_gpu_dev0 =
      "/job:worker/replica:0/task:0/device:XLA_GPU:0";
  absl::string_view xla_gpu_dev1 =
      "/job:worker/replica:0/task:0/device:XLA_GPU:1";
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Output a = ops::Const(root.WithOpName("A_dev0"), 1.0f, {2, 2});
  Output matmul_source =
      ops::MatMul(root.WithOpName("MatMulSource_dev0"), a, a);

  Output matmul0 = ops::MatMul(root.WithOpName("MatMul0_dev0"), matmul_source,
                               matmul_source);
  Output matmul1 = ops::MatMul(root.WithOpName("MatMul1_dev1"), matmul_source,
                               matmul_source);

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  for (Node* n : graph->nodes()) {
    if (absl::EndsWith(n->name(), /*suffix=*/"dev0")) {
      n->set_assigned_device_name(string(xla_gpu_dev0));
    } else if (absl::EndsWith(n->name(), /*suffix=*/"dev1")) {
      n->set_assigned_device_name(string(xla_gpu_dev1));
    }
  }
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_EQ(clusters["A_dev0"], clusters["MatMulSource_dev0"]);
  EXPECT_NE(clusters["MatMul0_dev0"], clusters["MatMul1_dev1"]);
  EXPECT_NE(clusters["MatMulSource_dev0"], clusters["MatMul1_dev1"]);

  // Improved Heuristics should prevent this probably.
  EXPECT_EQ(clusters["MatMulSource_dev0"], clusters["MatMul0_dev0"]);
}

TEST(XlaCompilationTest, ClusterStatefulRandomOpOnXlaDevice) {
  absl::string_view xla_cpu_device =
      "/job:worker/replica:0/task:0/device:XLA_CPU:0";

  Scope root = Scope::NewRootScope().ExitOnError();
  Output shape = ops::Const(root.WithOpName("test/shape_shape"), {200, 200});
  Output a = ops::RandomUniform(root.WithOpName("test/a"), shape, DT_FLOAT);
  Output b = ops::RandomUniform(root.WithOpName("test/b"), shape, DT_FLOAT);
  Output c = ops::Add(root.WithOpName("test/c"), a, b);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  for (Node* n : graph->nodes()) {
    if (absl::StartsWith(n->name(), /*prefix=*/"test/")) {
      n->set_assigned_device_name(string(xla_cpu_device));
    }
  }
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_NE(clusters["test/a"], "");
  EXPECT_NE(clusters["test/b"], "");
  EXPECT_NE(clusters["test/c"], "");
}

TEST(XlaCompilationTest, DontAutoClusterStatefulRandomOp) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output shape = ops::Const(root.WithOpName("test/shape_shape"), {200, 200});
  Output a = ops::RandomUniform(root.WithOpName("test/a"), shape, DT_FLOAT);
  Output b = ops::RandomUniform(root.WithOpName("test/b"), shape, DT_FLOAT);
  Output c = ops::Add(root.WithOpName("test/c"), a, b);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_EQ(clusters["test/a"], "");
  EXPECT_EQ(clusters["test/b"], "");
}

TEST(XlaCompilationTest, ClusterDummyOpsOnXlaDevice) {
  absl::string_view xla_cpu_device =
      "/job:worker/replica:0/task:0/device:XLA_CPU:0";

  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);
  Output check =
      ops::CheckNumerics(root.WithOpName("test/check"), a, "test/check");
  Output ge = ops::GreaterEqual(root.WithOpName("test/greaterequal"), check, b);
  Operation assert = ops::Assert(root.WithOpName("test/assert"), ge, {a, b});

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  for (Node* n : graph->nodes()) {
    if (absl::StartsWith(n->name(), /*prefix=*/"test/")) {
      n->set_assigned_device_name(string(xla_cpu_device));
    }
  }
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_NE(clusters["test/check"], "");
  EXPECT_NE(clusters["test/greaterequal"], "");
  EXPECT_NE(clusters["test/assert"], "");
}

TEST(XlaCompilationTest, DontAutoClusterDummyOps) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);
  Output check =
      ops::CheckNumerics(root.WithOpName("test/check"), a, "test/check");
  Output ge = ops::GreaterEqual(root.WithOpName("test/greaterequal"), check, b);
  Operation assert = ops::Assert(root.WithOpName("test/assert"), ge, {a, b});

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_EQ(clusters["test/assert"], "");
  EXPECT_EQ(clusters["test/check"], "");
}

}  // namespace
}  // namespace tensorflow
