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

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass_test_helper.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

using ::tensorflow::testing::FindNodeByName;

namespace tensorflow {
namespace {

static bool Initialized = [] {
  tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  return true;
}();

REGISTER_OP("UncompilableNullary").Output("o: float");
REGISTER_OP("UncompilableUnary").Input("a: float").Output("o: float");

std::unordered_map<string, string> GetClusters(const Graph& graph) {
  std::unordered_map<string, string> ids;
  for (Node* node : graph.nodes()) {
    string cluster;
    if (TryGetNodeAttr(node->attrs(), kXlaClusterAttr, &cluster)) {
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

std::set<string> GetClusterNames(const Graph& graph) {
  std::set<string> names;
  for (Node* node : graph.nodes()) {
    string cluster;
    if (TryGetNodeAttr(node->attrs(), kXlaClusterAttr, &cluster)) {
      CHECK(!cluster.empty());
      names.insert(cluster);
    }
  }
  return names;
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

TEST(XlaCompilationTest, StringUnsupported) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp(
        "Const", builder.opts()
                     .WithName("A")
                     .WithAttr("dtype", DT_STRING)
                     .WithAttr("value", Tensor(DT_STRING, TensorShape())));
    Node* b = ops::UnaryOp("EncodeBase64", a, builder.opts().WithName("B"));
    ops::BinaryOp("StringSplit", a, b, builder.opts().WithName("C"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_TRUE(clusters.empty());
}

TEST(XlaCompilationTest, WhereUnsupported) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_INT32)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::UnaryOp("Where", a, builder.opts().WithName("B"));
    ops::BinaryOp("Gather", b, a, builder.opts().WithName("C"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_TRUE(!clusters.empty());
}

TEST(XlaCompilationTest, HalfSupported) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
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

// Tests that PartitionedCalls are only marked for compilation if every node
// inside the function can be compiled.
TEST(XlaCompilationTest, PartitionedCallUnsupported) {
  FunctionDef compilable = FunctionDefHelper::Define(
      "CompilableFn", {"n_a:float", "n_b:float"}, {"n_c:float"}, {},
      {{{"n_c"}, "Add", {"n_a", "n_b"}, {{"T", DT_FLOAT}}}});
  FunctionDef uncompilable =
      FunctionDefHelper::Define("UncompilableFn", {"n_a:float"}, {"n_c:float"},
                                {}, {{{"n_c"}, "UncompilableUnary", {"n_a"}}});

  FunctionDefLibrary flib;
  *flib.add_function() = compilable;
  *flib.add_function() = uncompilable;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  std::unique_ptr<Graph> graph(new Graph(&flib_def));
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("A"), DT_FLOAT);

  NameAttrList b_name_attr;
  b_name_attr.set_name("CompilableFn");
  ops::PartitionedCall b(root.WithOpName("B"), {a, a}, {DT_FLOAT}, b_name_attr);
  NameAttrList c_name_attr;
  c_name_attr.set_name("UncompilableFn");

  ops::PartitionedCall c(root.WithOpName("C"), {a}, {DT_FLOAT}, c_name_attr);
  Output d = ops::Add(root.WithOpName("D"), b.output.front(), c.output.front());

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph, &flib_def));
  auto clusters = GetClusters(*graph);

  EXPECT_EQ(2, clusters.size());
  EXPECT_FALSE(clusters["B"].empty());
  EXPECT_TRUE(clusters["C"].empty());
  EXPECT_EQ(clusters["B"], clusters["D"]);
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
  EXPECT_FALSE(clusters["C"].empty());
  EXPECT_EQ(clusters["C"], clusters["E"]);
  EXPECT_TRUE(clusters.find("A") == clusters.cend());
  EXPECT_TRUE(clusters.find("B") == clusters.cend());
  EXPECT_TRUE(clusters.find("D") == clusters.cend());
}

TEST(XlaCompilationTest, CallXlaDeviceFuncWithResourceOp) {
  FunctionDef compilable = FunctionDefHelper::Define(
      "FnWithResourceOp", {"var:resource", "val:float"}, {"retval:float"}, {},
      {{{"assign_op"},
        "AssignVariableOp",
        {"var", "val"},
        {{"dtype", DT_FLOAT}}},
       {{"retval"}, "Identity", {"val"}, {{"T", DT_FLOAT}}, {"assign_op"}}});

  FunctionDefLibrary flib;
  *flib.add_function() = compilable;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  std::unique_ptr<Graph> graph(new Graph(&flib_def));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
    Node* resource =
        ops::SourceOp("VarHandleOp", builder.opts()
                                         .WithName("varhandle")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("shape", TensorShape({})));

    Tensor const_tensor(DT_FLOAT, TensorShape({}));
    const_tensor.scalar<float>()() = 42.0f;
    Node* value = ops::SourceOp("Const", builder.opts()
                                             .WithName("const")
                                             .WithAttr("value", const_tensor)
                                             .WithAttr("dtype", DT_FLOAT));

    Node* call = ops::BinaryOp("FnWithResourceOp", resource, value,
                               builder.opts().WithName("A"));
    Node* tanh0 = ops::UnaryOp("Tanh", call, builder.opts().WithName("tanh0"));
    Node* tanh1 = ops::UnaryOp("Tanh", tanh0, builder.opts().WithName("tanh1"));
    ops::UnaryOp("Tanh", tanh1, builder.opts().WithName("tanh2"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  string xla_cpu_device = "/job:worker/replica:0/task:0/device:XLA_CPU:0";
  testing::FindNodeByName(graph.get(), "A")
      ->set_assigned_device_name(xla_cpu_device);
  testing::FindNodeByName(graph.get(), "tanh0")
      ->set_assigned_device_name(xla_cpu_device);
  testing::FindNodeByName(graph.get(), "tanh1")
      ->set_assigned_device_name(xla_cpu_device);
  testing::FindNodeByName(graph.get(), "tanh2")
      ->set_assigned_device_name(xla_cpu_device);

  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph, &flib_def));
  auto clusters = GetClusters(*graph);

  EXPECT_NE(clusters["A"], "");
}

static absl::Status GradForUnaryCwise(
    FunctionDef* g, std::vector<FunctionDefHelper::Node> nodes) {
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
  return absl::OkStatus();
}

// A gradient containing only supported operators
absl::Status SupportedGrad(const AttrSlice& attrs, FunctionDef* g) {
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
absl::Status UnsupportedGrad(const AttrSlice& attrs, FunctionDef* g) {
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
  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph, &flib_def));
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

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(
      &graph, MarkForCompilationPassTestHelper::Options().WithNoGlobalJit()));
  auto clusters = GetClusters(*graph);

  // The computation is: C = A + relu(A)
  // where A sits in ScopeA, relu(A) sits in ScopeB, and C sits in ScopeC.
  // In this case, we cannot fuse anything, and there are no clusters.
  EXPECT_EQ(0, clusters.size());
}

TEST(XlaCompilationTest, CyclesWithSplittingScopes) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor())
                                         .WithAttr(kXlaCompileAttr, true)
                                         .WithAttr(kXlaScopeAttr, "Scope1"));
    Node* b = ops::UnaryOp("Relu", a,
                           builder.opts()
                               .WithName("B")
                               .WithAttr(kXlaCompileAttr, true)
                               .WithAttr(kXlaScopeAttr, "Scope1"));
    Node* c = ops::BinaryOp("MatMul", a, b,
                            builder.opts()
                                .WithName("C")
                                .WithAttr(kXlaCompileAttr, true)
                                .WithAttr(kXlaScopeAttr, "Scope2"));
    ops::BinaryOp("Add", b, c,
                  builder.opts()
                      .WithName("D")
                      .WithAttr(kXlaCompileAttr, true)
                      .WithAttr(kXlaScopeAttr, "Scope2"));
    TF_CHECK_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(
      &graph, MarkForCompilationPassTestHelper::Options().WithNoGlobalJit()));
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
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor())
                                         .WithAttr(kXlaCompileAttr, true)
                                         .WithAttr(kXlaScopeAttr, "ScopeA"));
    Node* b = ops::UnaryOp("Relu", a,
                           builder.opts()
                               .WithName("B")
                               .WithAttr(kXlaCompileAttr, true)
                               .WithAttr(kXlaScopeAttr, "ScopeB"));
    ops::BinaryOp("MatMul", a, b, builder.opts().WithName("C"));
    TF_CHECK_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(
      &graph, MarkForCompilationPassTestHelper::Options().WithNoGlobalJit()));
  auto clusters = GetClusters(*graph);

  // The computation is: C = A @ relu(A)
  // where A sits in ScopeA, relu(A) sits in ScopeB, and C sits in ScopeC.
  // In this case, we cannot fuse anything.
  EXPECT_EQ(2, clusters.size());
  EXPECT_NE(clusters["A"], clusters["B"]);
  EXPECT_NE(clusters["B"], clusters["C"]);
}

TEST(XlaCompilationTest, DontClusterNodesWithMismatchingDeadness) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);

  ops::Switch switch_a(root.WithOpName("switch_a"), value, cond_a);
  ops::Switch switch_b(root.WithOpName("switch_b"), value, cond_b);

  Output tanh_a0 = ops::Tanh(root.WithOpName("tan_a0"), switch_a.output_true);
  Output tanh_a1 = ops::Tanh(root.WithOpName("tan_a1"), tanh_a0);

  Output tanh_b0 = ops::Tanh(root.WithOpName("tan_b0"), switch_b.output_true);
  Output tanh_b1 = ops::Tanh(root.WithOpName("tan_b1"), tanh_b0);

  Output add = ops::Add(root.WithOpName("add"), tanh_a1, tanh_b1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_EXPECT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(
      &graph,
      MarkForCompilationPassTestHelper::Options().WithDeadnessAnalysis()));
  auto clusters = GetClusters(*graph);

  EXPECT_NE(clusters["tan_a0"], "");
  EXPECT_NE(clusters["tan_a1"], "");
  EXPECT_NE(clusters["tan_b0"], "");
  EXPECT_NE(clusters["tan_b1"], "");

  EXPECT_EQ(clusters["tan_a0"], clusters["tan_a1"]);
  EXPECT_EQ(clusters["tan_b0"], clusters["tan_b1"]);

  EXPECT_NE(clusters["tan_a0"], clusters["tan_b0"]);
}

TEST(XlaCompilationTest, ClusterNodesWithMismatchingInputDeadness) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);

  ops::Switch switch_a(root.WithOpName("switch_a"), value, cond_a);
  ops::Switch switch_b(root.WithOpName("switch_b"), value, cond_b);

  Output add_a = ops::Add(root.WithOpName("add_a"), switch_a.output_true,
                          switch_b.output_true);
  Output add_b = ops::Add(root.WithOpName("add_b"), switch_a.output_true,
                          switch_b.output_true);
  Output add = ops::Add(root.WithOpName("add_c"), add_a, add_b);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_EXPECT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(
      &graph,
      MarkForCompilationPassTestHelper::Options().WithDeadnessAnalysis()));
  auto clusters = GetClusters(*graph);

  EXPECT_NE(clusters["add_a"], "");
  EXPECT_NE(clusters["add_b"], "");
  EXPECT_NE(clusters["add_c"], "");

  EXPECT_EQ(clusters["add_a"], clusters["add_b"]);
  EXPECT_EQ(clusters["add_b"], clusters["add_c"]);
}

namespace {
Node* MakeRead(const Scope& scope, const string& id,
               Node** var_handle_op = nullptr) {
  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output read =
      ops::ReadVariableOp(scope.WithOpName("Read" + id), var_handle, DT_FLOAT);
  if (var_handle_op) {
    *var_handle_op = var_handle.node();
  }
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
  ASSERT_EQ(cluster_sets.size(), 0);
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

  ASSERT_EQ(cluster_sets.size(), 1);

  std::vector<string> expected_clustered_nodes_a = {
      "AssignmentW1", "ConstN0", "ReadR0", "ValueToAssignW1"};
  ASSERT_EQ(cluster_sets[cluster_names[0]], expected_clustered_nodes_a);
}

TEST(XlaCompilationTest, IllegalCycle_UsefulErrorMessage) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope root = Scope::NewRootScope().ExitOnError();
  {
    auto BuildNoopNode = [](absl::string_view name, Graph* graph) {
      NodeDefBuilder builder(name, "NoOp");
      NodeDef def;
      TF_CHECK_OK(builder.Finalize(&def));

      absl::Status status;
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

  absl::Status status =
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.ToString(),
                                "Edge from c to a would create a cycle.\n"
                                "+-> a\n"
                                "|   b\n"
                                "+-- c\n"));
}

TEST(XlaCompilationTest, Retval) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
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

  EXPECT_TRUE(clusters.empty());
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
  absl::Status status;
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
  auto fld = std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(),
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
  Output a = ops::Tanh(root.WithOpName("tanh_A_dev0"),
                       ops::Const(root.WithOpName("A_dev0"), 1.0f, {2, 2}));
  Output b = ops::Tanh(root.WithOpName("tanh_B_dev1"),
                       ops::Const(root.WithOpName("B_dev1"), 1.0f, {2, 2}));
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

TEST(XlaCompilationTest, DontClusterMergingNodesOnCPU) {
  // This is similar to the 'DontClusterMergingNodes' above, except
  // MatMulCombined is placed on the CPU.
  Scope root = Scope::NewRootScope().ExitOnError();
  absl::string_view xla_gpu_dev0 = "/job:worker/replica:0/task:0/device:GPU:0";
  absl::string_view xla_gpu_dev1 = "/job:worker/replica:0/task:0/device:GPU:1";
  absl::string_view xla_cpu_dev0 = "/job:worker/replica:0/task:0/device:CPU:0";
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Output a = ops::Tanh(root.WithOpName("tanh_A_dev0"),
                       ops::Const(root.WithOpName("A_dev0"), 1.0f, {2, 2}));
  Output b = ops::Tanh(root.WithOpName("tanh_B_dev1"),
                       ops::Const(root.WithOpName("B_dev1"), 1.0f, {2, 2}));
  Output matmul0 = ops::MatMul(root.WithOpName("MatMul0_dev0"), a, a);
  Output matmul1 = ops::MatMul(root.WithOpName("MatMul1_dev1"), b, b);

  Output combined =
      ops::MatMul(root.WithOpName("MatMulCombined_cpu"), matmul0, matmul1);
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  for (Node* n : graph->nodes()) {
    if (absl::EndsWith(n->name(), /*suffix=*/"cpu")) {
      n->set_assigned_device_name(string(xla_cpu_dev0));
    } else if (absl::EndsWith(n->name(), /*suffix=*/"dev0")) {
      n->set_assigned_device_name(string(xla_gpu_dev0));
    } else if (absl::EndsWith(n->name(), /*suffix=*/"dev1")) {
      n->set_assigned_device_name(string(xla_gpu_dev1));
    }
  }
  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  // Each of the MatMuls should be in a separate cluster.
  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_NE(clusters["MatMul0_dev0"], clusters["MatMul1_dev1"]);
  EXPECT_NE(clusters["MatMulCombined_cpu"], clusters["MatMul0_dev0"]);
  EXPECT_NE(clusters["MatMulCombined_cpu"], clusters["MatMul1_dev1"]);
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

TEST(XlaCompilationTest, DontAutoClusterOpsProducingVariant) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_INT64);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_INT64);

  Output cast_a = ops::Cast(root.WithOpName("test/cast_a"), a, DT_INT32);
  Output cast_b = ops::Cast(root.WithOpName("test/cast_b"), b, DT_INT32);

  Output tensor_list_reserve = ops::TensorListReserve(
      root.WithOpName("test/tensor_list_reserve"), cast_a, cast_b, DT_FLOAT);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_EQ(clusters["test/tensor_list_reserve"], "");
}

TEST(XlaCompilationTest, DontAutoClusterOpsConsumingVariant) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output dummy_input =
      ops::Placeholder(root.WithOpName("test/dummy_input"), DT_INT64);
  Output variant_input =
      ops::Placeholder(root.WithOpName("test/variant_input"), DT_VARIANT);

  // Create one more node so that we don't avoid creating a cluster solely
  // because it would be trivial.
  Output dummy_cast =
      ops::Cast(root.WithOpName("test/dummy_cast"), dummy_input, DT_INT32);

  Output tensor_list_element_shape = ops::TensorListElementShape(
      root.WithOpName("test/tensor_list_element_shape"), variant_input,
      DT_INT32);

  root.graph()->AddControlEdge(dummy_cast.node(),
                               tensor_list_element_shape.node());

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_EQ(clusters["test/tensor_list_element_shape"], "");
}

TEST(XlaCompilationTest, ClusterOpsProducingVariantIfOnXlaDevice) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_INT64);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_INT64);

  Output cast_a = ops::Cast(root.WithOpName("test/cast_a"), a, DT_INT32);
  Output cast_b = ops::Cast(root.WithOpName("test/cast_b"), b, DT_INT32);

  Output tensor_list_reserve = ops::TensorListReserve(
      root.WithOpName("test/tensor_list_reserve"), cast_a, cast_b, DT_FLOAT);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  string xla_cpu_device = "/job:worker/replica:0/task:0/device:XLA_CPU:0";
  for (Node* n : graph->nodes()) {
    if (absl::StartsWith(n->name(), /*prefix=*/"test/")) {
      n->set_assigned_device_name(xla_cpu_device);
    }
  }

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);
  EXPECT_NE(clusters["test/tensor_list_reserve"], "");
}

const char* kCPU0 = "/job:worker/replica:0/task:0/device:CPU:0";
const char* kGPU0 = "/job:worker/replica:0/task:0/device:GPU:0";
const char* kXLA_GPU0 = "/job:worker/replica:0/task:0/device:XLA_GPU:0";
const char* kGPU1 = "/job:worker/replica:0/task:0/device:GPU:1";

TEST(XlaCompilationTest, CreateCombinedCpuGpuClusters) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);

  Output x = ops::Add(root.WithOpName("test/x"), a, b);
  Output y = ops::MatMul(root.WithOpName("test/y"), a, b);
  Output z = ops::Add(root.WithOpName("test/z"), x, y);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  FindNodeByName(graph.get(), "test/x")->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), "test/y")->set_assigned_device_name(kCPU0);
  FindNodeByName(graph.get(), "test/z")->set_assigned_device_name(kGPU0);

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_NE(clusters["test/x"], "");

  EXPECT_EQ(clusters["test/x"], clusters["test/y"]);
  EXPECT_EQ(clusters["test/y"], clusters["test/z"]);
}

TEST(XlaCompilationTest, DontCreateGpu0AndGpu1Clusters) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);

  Output x = ops::Add(root.WithOpName("test/x"), a, b);
  Output y = ops::Add(root.WithOpName("test/y"), x, x);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  FindNodeByName(graph.get(), "test/x")->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), "test/y")->set_assigned_device_name(kGPU1);

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_EQ(clusters["test/x"], "");
  EXPECT_EQ(clusters["test/y"], "");
}

TEST(XlaCompilationTest, DontCreateCombinedCpuUnknownClusters) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);

  Output x = ops::Add(root.WithOpName("test/x"), a, b);
  Output y = ops::Add(root.WithOpName("test/y"), x, x);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  FindNodeByName(graph.get(), "test/x")->set_assigned_device_name(kCPU0);
  FindNodeByName(graph.get(), "test/y")->set_assigned_device_name(kXLA_GPU0);

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_EQ(clusters["test/x"], "");
  EXPECT_EQ(clusters["test/y"], "");
}

TEST(XlaCompilationTest, ClusterResourceOpsWhenSafe) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Node* var_handle;
  Node* resource_read = MakeRead(root, "read", &var_handle);
  Output b = ops::Add(root.WithOpName("test/b"), Output(resource_read, 0), a);

  string resource_read_name = resource_read->name();
  string var_handle_name = var_handle->name();

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  FindNodeByName(graph.get(), "test/b")->set_assigned_device_name(kCPU0);
  FindNodeByName(graph.get(), resource_read_name)
      ->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), var_handle_name)->set_assigned_device_name(kGPU0);

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_NE(clusters["test/b"], "");
  EXPECT_EQ(clusters["test/b"], clusters[resource_read_name]);
}

TEST(XlaCompilationTest, DontClusterResourceOpsWhenUnsafe) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Node* var_handle;
  Node* resource_read = MakeRead(root, "read", &var_handle);
  Output b = ops::Add(root.WithOpName("test/b"), Output(resource_read, 0), a);

  string resource_read_name = resource_read->name();
  string var_handle_name = var_handle->name();

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  FindNodeByName(graph.get(), "test/b")->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), resource_read_name)
      ->set_assigned_device_name(kCPU0);
  FindNodeByName(graph.get(), var_handle_name)->set_assigned_device_name(kCPU0);

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_EQ(clusters["test/b"], "");
  EXPECT_EQ(clusters[resource_read_name], "");
}

TEST(XlaCompilationTest, DontClusterNodesWithScopedAllocatorAttr) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);

  Output x = ops::Add(root.WithOpName("test/x"), a, b);
  Output y = ops::MatMul(root.WithOpName("test/y"), a, b);
  Output z = ops::Add(root.WithOpName("test/z"), x, y);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  FindNodeByName(graph.get(), "test/x")->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), "test/y")->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), "test/z")->set_assigned_device_name(kGPU0);

  std::vector<int> scoped_allocator_value;
  scoped_allocator_value.push_back(0);
  scoped_allocator_value.push_back(155);
  FindNodeByName(graph.get(), "test/z")
      ->AddAttr("_scoped_allocator", scoped_allocator_value);

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_EQ(clusters["test/z"], "");
}

TEST(XlaCompilationTest, DontClusterNodesWithForwardFromAttr) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);

  Output x = ops::Add(root.WithOpName("test/x"), a, b);
  Output y = ops::MatMul(root.WithOpName("test/y"), a, b);
  Output z = ops::Add(root.WithOpName("test/z"), x, y);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  FindNodeByName(graph.get(), "test/x")->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), "test/y")->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), "test/z")->set_assigned_device_name(kGPU0);

  FindNodeByName(graph.get(), "test/z")->AddAttr("_forward_from", 0);

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_EQ(clusters["test/z"], "");
}

// Note, this relies on other implementation details to test the
// specific heuristic we care about here, so other changes might be at fault if
// this CL breaks. What we care about is that if a ShapeConsumingOp can be
// connected with a producer or consumer and cannot be clustered with both, it
// should be clustered with the producer.
TEST(XlaCompilationTest, ClusterShapeConsumerWithProducer) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);

  Output x = ops::MatMul(root.WithOpName("test/x"), a, b);
  Output y = ops::Size(root.WithOpName("test/y"), x);
  Output z = ops::Add(root.WithOpName("test/z"), y, y);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // Ensure that the "Size" op can only be clustered with either the producer or
  // consumer by putting them on different devices.
  FindNodeByName(graph.get(), "test/x")->set_assigned_device_name(kGPU0);
  FindNodeByName(graph.get(), "test/y")->set_assigned_device_name(kCPU0);
  FindNodeByName(graph.get(), "test/z")->set_assigned_device_name(kGPU1);

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_NE(clusters["test/y"], "");
  EXPECT_EQ(clusters["test/x"], clusters["test/y"]);
  EXPECT_NE(clusters["test/z"], clusters["test/y"]);
}

// Test that ShapeConsuming ops are still fully clustered whenever possible.
TEST(XlaCompilationTest, ClusterShapeConsumerWithProducerAndConsumer) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output a = ops::Placeholder(root.WithOpName("test/a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("test/b"), DT_FLOAT);

  Output x = ops::MatMul(root.WithOpName("test/x"), a, b);
  Output y = ops::Size(root.WithOpName("test/y"), x);
  Output z = ops::Add(root.WithOpName("test/z"), y, y);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_NE(clusters["test/y"], "");
  EXPECT_EQ(clusters["test/y"], clusters["test/x"]);
  EXPECT_EQ(clusters["test/y"], clusters["test/z"]);
}

void AddCtrlEdge(const Scope& scope, Operation a, Operation b) {
  scope.graph()->AddControlEdge(a.node(), b.node());
}

void AddCtrlEdge(const Scope& scope, Output a, Operation b) {
  AddCtrlEdge(scope, a.op(), b);
}

void AddCtrlEdge(const Scope& scope, Operation a, Output b) {
  AddCtrlEdge(scope, a, b.op());
}

// Tests that we pick a good clustering for graphs that have an integer
// increment operation control dependent on gradient update operations.
TEST(XlaCompilationTest, IterationIncrementAndGroupDeps) {
  Scope scope = Scope::NewRootScope().ExitOnError();

  Output iter =
      ops::VarHandleOp(scope.WithOpName("iter"), DT_INT64, TensorShape({}));
  Output weights_0 = ops::VarHandleOp(scope.WithOpName("weights_0"), DT_FLOAT,
                                      TensorShape({1000}));
  Output weights_1 = ops::VarHandleOp(scope.WithOpName("weights_1"), DT_FLOAT,
                                      TensorShape({1000}));

  // We update the weights by adding delta to them (to "simulate" a
  // ResourceApplyGradientDescent and similar things).
  Output delta = ops::Placeholder(scope.WithOpName("delta"), DT_FLOAT);

  ops::AssignAddVariableOp increment_op(
      scope.WithOpName("IncrementIteration"), iter,
      ops::Const(scope.WithOpName("one"), static_cast<int64_t>(1)));

  ops::AssignAddVariableOp weights_0_update_op(
      scope.WithOpName("weights_0_update"), weights_0, delta);
  ops::AssignAddVariableOp weights_1_update_op(
      scope.WithOpName("weights_1_update"), weights_1, delta);

  ops::NoOp group_deps(scope.WithOpName("group_deps"));

  ops::NoOp some_ctrl_input(scope.WithOpName("some_ctrl_input"));

  Output matmul_input =
      ops::Placeholder(scope.WithOpName("matmul_input"), DT_FLOAT);
  Output matmul_0 =
      ops::MatMul(scope.WithOpName("matmul_0"), matmul_input, matmul_input);
  Output matmul_1 =
      ops::MatMul(scope.WithOpName("matmul_1"), matmul_input, matmul_input);

  AddCtrlEdge(scope, increment_op, group_deps);
  AddCtrlEdge(scope, weights_0_update_op, increment_op);
  AddCtrlEdge(scope, weights_1_update_op, increment_op);

  AddCtrlEdge(scope, some_ctrl_input, weights_0_update_op);
  AddCtrlEdge(scope, some_ctrl_input, weights_1_update_op);

  AddCtrlEdge(scope, matmul_0, group_deps);
  AddCtrlEdge(scope, matmul_1, group_deps);

  AddCtrlEdge(scope, weights_0_update_op, matmul_0);
  AddCtrlEdge(scope, weights_1_update_op, matmul_1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  EXPECT_NE(clusters["some_ctrl_input"], "");
  EXPECT_EQ(clusters["some_ctrl_input"], clusters["weights_0_update"]);
  EXPECT_EQ(clusters["some_ctrl_input"], clusters["weights_1_update"]);
  EXPECT_EQ(clusters["some_ctrl_input"], clusters["matmul_0"]);
  EXPECT_EQ(clusters["some_ctrl_input"], clusters["matmul_0"]);
}

// Test a pattern where a special Identity node is driving consts in a loop.
// Expect that the Identity node will not go into any clusters.  Note that we
// create an incomplete graph here (e.g., lacking Enter/Exit/NextIteration,
// etc.) just enough to test the pattern, as a complete graph may be too
// cumbersome and unnecessary.
TEST(XlaCompilationTest, DontClusterTheSpecialIdentityDrivingConstsInLoop) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond = ops::Placeholder(root.WithOpName("cond"), DT_BOOL);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  Output loop_cond = ops::LoopCond(root.WithOpName("loop_cond"), cond);
  ops::Switch switch_node(root.WithOpName("switch"), value, loop_cond);

  Output identity =
      ops::Identity(root.WithOpName("identity"), switch_node.output_true);
  Output const_node = ops::Const(root.WithOpName("const"), 1.0f);
  root.graph()->AddControlEdge(identity.node(), const_node.node());
  Output tanh0 = ops::Tanh(root.WithOpName("tanh0"), const_node);
  Output tanh1 = ops::Tanh(root.WithOpName("tanh1"), tanh0);
  Output add = ops::Add(root.WithOpName("add"), const_node, tanh1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_EXPECT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(
      &graph,
      MarkForCompilationPassTestHelper::Options().WithDeadnessAnalysis()));
  auto clusters = GetClusters(*graph);

  EXPECT_EQ(clusters["identity"], "");
}

TEST(XlaCompilationTest, UnsupportedEnterExitPattern) {
  // Regression test for b/32350199, where the autoclustering code introduced a
  // deadlock in a graph containing a while loop.
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::Placeholder(root.WithOpName("A"), DT_FLOAT);
  auto enter_0 = ops::internal::Enter(root.WithOpName("enter_a"), a, "frame");
  auto exit_0 = ops::internal::Exit(root.WithOpName("exit_a"), enter_0);
  auto tanh = ops::Tanh(root.WithOpName("tanh"), exit_0);
  auto enter_1 =
      ops::internal::Enter(root.WithOpName("enter_1"), tanh, "frame");
  auto exit_1 = ops::internal::Exit(root.WithOpName("exit_1"), enter_1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_EXPECT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);

  // Nothing should be compiled.
  EXPECT_EQ(0, clusters.size());
}

TEST(XlaCompilationTest, DeterministicClusterNames) {
  auto create_graph =
      [](absl::string_view output_name) -> std::unique_ptr<Graph> {
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Tensor t(DT_FLOAT, TensorShape());
    t.scalar<float>()() = 0.0f;
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", t));
    Node* b = ops::UnaryOp("Neg", a, builder.opts().WithName("B"));
    ops::BinaryOp("MatMul", a, b, builder.opts().WithName(output_name));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
    return graph;
  };

  // Checks if two cluster names match for all parts except their sequence
  // number. Names are expected as: cluster_fp_seq#
  auto cluster_names_match = [](absl::string_view lhs_cluster_name,
                                absl::string_view rhs_cluster_name) {
    std::vector<absl::string_view> lhs_cluster_name_parts =
        absl::StrSplit(lhs_cluster_name, '_');
    std::vector<absl::string_view> rhs_cluster_name_parts =
        absl::StrSplit(rhs_cluster_name, '_');

    if (lhs_cluster_name_parts.size() != 3) {
      return errors::FailedPrecondition("unexpected lhs cluster name: ",
                                        lhs_cluster_name);
    }

    if (rhs_cluster_name_parts.size() != 3) {
      return errors::FailedPrecondition("unexpected rhs cluster name: ",
                                        rhs_cluster_name);
    }

    if (lhs_cluster_name_parts[0] != rhs_cluster_name_parts[0] ||
        lhs_cluster_name_parts[1] != rhs_cluster_name_parts[1]) {
      return errors::FailedPrecondition(
          "Cluster names mismatch: lhs: ", lhs_cluster_name,
          " rhs: ", rhs_cluster_name);
    }

    if (lhs_cluster_name_parts[2] == rhs_cluster_name_parts[2]) {
      return errors::FailedPrecondition(
          "cluster sequence numbers are the same: lhs: ", lhs_cluster_name,
          " rhs: ", rhs_cluster_name);
    }

    return absl::OkStatus();
  };

  testing::ResetClusterSequenceNumber();
  auto options = MarkForCompilationPassTestHelper::Options()
                     .WithDeterministicClusterNames();

  // Cluster the same graphs twice so we can observe that the prefix contains
  // the stable fingerprint.
  auto graph0 = create_graph("out");
  auto graph1 = create_graph("differs");
  auto graph2 = create_graph("out");      // same as graph0
  auto graph3 = create_graph("differs");  // same as graph1

  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph0, options));
  auto clusters0 = GetClusterNames(*graph0);
  ASSERT_EQ(clusters0.size(), 1);

  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph1, options));
  auto clusters1 = GetClusterNames(*graph1);
  ASSERT_EQ(clusters1.size(), 1);

  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph2, options));
  auto clusters2 = GetClusterNames(*graph2);
  ASSERT_EQ(clusters2.size(), 1);

  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph3, options));
  auto clusters3 = GetClusterNames(*graph3);
  ASSERT_EQ(clusters3.size(), 1);

  // clusters0 and clusters2 should be the same
  TF_EXPECT_OK(cluster_names_match(*clusters0.begin(), *clusters2.begin()));

  // clusters1 and clusters3 should also be the same
  TF_EXPECT_OK(cluster_names_match(*clusters1.begin(), *clusters3.begin()));

  // clusters0/2 should differ from clusters1/3
}

TEST(XlaCompilationTest, ClusterSessionName) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output variable = ops::Variable(root.WithOpName("variable"),
                                  PartialTensorShape{}, DT_FLOAT);
  Output read = ops::Identity(root.WithOpName("read"), variable);
  Output neg = ops::Negate(root.WithOpName("negate"), read);
  Output add = ops::Add(root.WithOpName("add"), neg, neg);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  auto options = MarkForCompilationPassTestHelper::Options().WithSessionName(
      "test_session_name");
  TF_ASSERT_OK(
      MarkForCompilationPassTestHelper::MarkForCompilation(&graph, options));

  std::unordered_map<string, string> clusters = GetClusters(*graph);

  ASSERT_FALSE(clusters.empty());
  string cluster_name = clusters.begin()->second;

  std::unordered_map<string, string> expected_clusters(
      {{"negate", cluster_name}, {"add", cluster_name}});
  EXPECT_EQ(clusters, expected_clusters);
  EXPECT_THAT(cluster_name, ::testing::StartsWith("test_session_name"));
}

namespace {
Node* MakeStageNode(GraphDefBuilder& builder, string name,
                    std::initializer_list<DataType> dtypes,
                    absl::Span<const ops::NodeOut> values) {
  auto opts = builder.opts()
                  .WithName(std::move(name))
                  .WithAttr("dtypes", std::move(dtypes));
  if (opts.HaveError()) {
    return nullptr;
  }

  NodeBuilder node_builder(name, "Stage", opts.op_registry());
  node_builder.Input(values);
  return opts.FinalizeBuilder(&node_builder);
}
}  // namespace

TEST(XlaCompilationTest, StagePipelinePreservedByClusterScopingPass) {
  auto build_staged_graph = [](std::unique_ptr<Graph>* graph) -> absl::Status {
    // Construct a graph as below with two pipeline stages and test that nodes
    // in different stages will not be merged if ClusterScopingPass is on.
    //
    //       b
    //       |
    //       v
    // a -> add0 -> relu0 -> stage
    //
    //             b
    //             |
    //             v
    // unstage -> add1 -> relu1
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("a")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::SourceOp("Const", builder.opts()
                                         .WithName("b")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* unstage = ops::SourceOp(
        "Unstage",
        builder.opts().WithName("unstage").WithAttr("dtypes", {DT_FLOAT}));

    Node* add0 = ops::BinaryOp("Add", a, b, builder.opts().WithName("add0"));
    Node* add1 =
        ops::BinaryOp("Add", unstage, b, builder.opts().WithName("add1"));
    Node* relu0 = ops::UnaryOp("Relu", add0, builder.opts().WithName("relu0"));
    ops::UnaryOp("Relu", add1, builder.opts().WithName("relu1"));
    MakeStageNode(builder, "stage", {DT_FLOAT}, {relu0});

    return GraphDefBuilderToGraph(builder, graph->get());
  };

  // All nodes go into the same cluster if ClusterScopingPass is off.
  {
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    TF_ASSERT_OK(build_staged_graph(&graph));

    TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(
        &graph,
        MarkForCompilationPassTestHelper::Options().WithNoClusterScoping()));

    std::unordered_map<string, string> clusters = GetClusters(*graph);
    EXPECT_EQ(clusters["add0"], clusters["add1"]);
    EXPECT_EQ(clusters["add0"], clusters["relu1"]);
    EXPECT_EQ(clusters["relu0"], clusters["add1"]);
    EXPECT_EQ(clusters["relu0"], clusters["relu1"]);
  }

  // By default, ClusterScopingPass is on and different pipeline stages should
  // not be merged.
  {
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    TF_ASSERT_OK(build_staged_graph(&graph));

    TF_ASSERT_OK(MarkForCompilationPassTestHelper::MarkForCompilation(&graph));

    std::unordered_map<string, string> clusters = GetClusters(*graph);
    EXPECT_NE(clusters["add0"], clusters["add1"]);
    EXPECT_NE(clusters["add0"], clusters["relu1"]);
    EXPECT_NE(clusters["relu0"], clusters["add1"]);
    EXPECT_NE(clusters["relu0"], clusters["relu1"]);
  }
}
TEST(XlaCompilationTest, XLALiteAllowlist) {
  auto* allowlist_table = tensorflow::GetAllowlistTable();
  absl::flat_hash_set<string> hallowlist;
  std::vector<string> vall_ops = XlaOpRegistry::GetAllRegisteredOps();
  absl::flat_hash_set<string> all_ops(vall_ops.begin(), vall_ops.end());

  // Check that all the operations in the table are existing TF operations
  for (auto pair : *allowlist_table) {
    hallowlist.insert(pair.second.begin(), pair.second.end());
    for (auto op : pair.second) {
      ASSERT_TRUE(all_ops.contains(op));
    }
  }

  // Check that all registered XLA operation are in the allowlist
  // table or are known to not be in it.

  absl::flat_hash_set<string> known_not_in_list =
      tensorflow::testing::GetKnownXLAAllowlistOp();
  std::vector<string> unknow_op;
  for (string op : vall_ops) {
    if (!hallowlist.contains(op) && !known_not_in_list.contains(op)) {
      unknow_op.push_back(op);
    }
  }
  EXPECT_TRUE(unknow_op.empty())
      << "Someone added support for a new TF operations inside XLA. They must "
         "be included in the XLALite allowlist or denylist:\n"
      << absl::StrJoin(unknow_op, "\n");
}
}  // namespace
}  // namespace tensorflow
