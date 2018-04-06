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

#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
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

REGISTER_OP("UncompilableNullary").Output("o: float");
REGISTER_OP("UncompilableUnary").Input("a: float").Output("o: float");

Status MarkForCompilation(std::unique_ptr<Graph>* graph,
                          FunctionLibraryDefinition* flib_def) {
  // Assign all nodes to the CPU device.
  static const char* kCpuDevice = "/job:localhost/replica:0/task:0/cpu:0";
  for (Node* n : (*graph)->nodes()) {
    n->set_assigned_device_name(kCpuDevice);
  }

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;
  opt_options.flib_def = flib_def;
  MarkForCompilationPass pass;
  return pass.RunImpl(opt_options);
}

Status MarkForCompilation(std::unique_ptr<Graph>* graph) {
  FunctionDefLibrary flib;
  FunctionLibraryDefinition flib_def((*graph)->op_registry(), flib);
  return MarkForCompilation(graph, &flib_def);
}

std::unordered_map<string, string> GetClusters(const Graph& graph) {
  std::unordered_map<string, string> ids;
  for (Node* node : graph.nodes()) {
    string cluster;
    if (GetNodeAttr(node->attrs(), kXlaClusterAttr, &cluster).ok()) {
      CHECK(!cluster.empty());
      ids[node->name()] = cluster;
    }
  }
  return ids;
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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
  AddAttr("_noinline", bool(true), noinline.mutable_attr());

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

  TF_ASSERT_OK(MarkForCompilation(&graph, &flib_def));
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
  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  // Nothing should be compiled. In particular, 'd' and 'c' must not be
  // compiled.
  EXPECT_EQ(0, clusters.size());
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  // The computation is: C = A @ relu(A)
  // where A sits in ScopeA, relu(A) sits in ScopeB, and C sits in ScopeC.
  // In this case, we cannot fuse anything.
  EXPECT_EQ(2, clusters.size());
  EXPECT_NE(clusters["A"], clusters["B"]);
  EXPECT_EQ(clusters["B"], clusters["C"]);
}

REGISTER_OP("ResourceInput").Input("a: resource").Output("o: float");
REGISTER_OP("ResourceOutput").Input("a: float").Output("o: resource");

namespace {

class DummyOp : public XlaOpKernel {
  using XlaOpKernel::XlaOpKernel;
  void Compile(XlaOpKernelContext* ctx) override {}
};

REGISTER_XLA_OP(Name("ResourceInput"), DummyOp);
REGISTER_XLA_OP(Name("ResourceOutput"), DummyOp);

}  // namespace

TEST(XlaCompilationTest, Resources) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDef graphdef;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a =
        ops::SourceOp("UncompilableNullary", builder.opts().WithName("A"));
    Node* b = ops::UnaryOp("Relu", a, builder.opts().WithName("B"));
    // We should not form clusters with resource ops by default.
    Node* c = ops::UnaryOp("ResourceOutput", b, builder.opts().WithName("C"));
    Node* d = ops::UnaryOp("ResourceInput", c, builder.opts().WithName("D"));
    ops::UnaryOp("Relu", d, builder.opts().WithName("E"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }
  TF_ASSERT_OK(MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_EQ(0, clusters.size());  // Nothing should be compiled.
}

TEST(XlaCompilationTest, IllegalCycle_UsefulErrorMessage) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope root = Scope::NewRootScope().ExitOnError();
  {
    auto BuildNoopNode = [](StringPiece name, Graph* graph) {
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

  Status status = MarkForCompilation(&graph);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(status.ToString(),
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

  TF_ASSERT_OK(MarkForCompilation(&graph));
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
  TF_ASSERT_OK(MarkForCompilation(&graph));
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
  TF_ASSERT_OK(MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  EXPECT_TRUE(clusters.empty());
}

}  // namespace
}  // namespace tensorflow
