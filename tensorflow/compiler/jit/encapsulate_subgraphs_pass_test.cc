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

#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/equal_graph_def.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

bool EqualFunctionDef(const FunctionDef& a, const FunctionDef& b,
                      string* diff) {
  // TODO(phawkins) use a more sophisticated equality test.
  if (a.DebugString() != b.DebugString()) {
    if (diff) {
      *diff = strings::StrCat("Definition mismatch for function ",
                              a.signature().name(), ", expected:\n",
                              a.DebugString());
    }
    return false;
  }
  return true;
}

bool EqualFunctionDefLibrary(const FunctionDefLibrary& expected,
                             const FunctionDefLibrary& actual, string* diff) {
  std::unordered_map<string, const FunctionDef*> actual_index;
  for (const FunctionDef& function : actual.function()) {
    actual_index[function.signature().name()] = &function;
  }

  for (const FunctionDef& expected_function : expected.function()) {
    auto it = actual_index.find(expected_function.signature().name());
    if (it == actual_index.end()) {
      if (diff) {
        *diff = strings::StrCat("Did not find expected function '",
                                expected_function.signature().name(), "'");
      }
      return false;
    }
    if (!EqualFunctionDef(expected_function, *it->second, diff)) return false;
    actual_index.erase(it);
  }

  if (!actual_index.empty()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Found unexpected function '",
                              actual_index.begin()->second->signature().name(),
                              "'");
    }
    return false;
  }

  return true;
}

#define TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(expected, actual)         \
  do {                                                            \
    string diff;                                                  \
    EXPECT_TRUE(EqualFunctionDefLibrary(actual, expected, &diff)) \
        << diff << "\nActual: " << actual.DebugString();          \
  } while (false)

REGISTER_OP("InputTest").Output("o: float");

REGISTER_OP("UnaryTest").Input("a: float").Output("o: float");
REGISTER_OP("BinaryTest")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float");

REGISTER_OP("AddNLikeTest")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetIsCommutative()
    .SetIsAggregate();

Node* Input(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("InputTest", opts);
}

Node* Unary(ops::NodeOut a, const GraphDefBuilder::Options& opts) {
  return ops::UnaryOp("UnaryTest", a, opts);
}

Node* Binary(ops::NodeOut a, ops::NodeOut b,
             const GraphDefBuilder::Options& opts) {
  return ops::BinaryOp("BinaryTest", a, b, opts);
}

Node* AddNLike(std::vector<ops::NodeOut> inputs,
               const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("AddN"), "AddNLikeTest",
                           opts.op_registry());
  node_builder.Input(inputs);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ArgOp(int index, DataType type, const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("_Arg",
                       opts.WithAttr("T", type).WithAttr("index", index));
}

Node* RetOp(int index, ops::NodeOut a, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("Retval"), "_Retval",
                           opts.op_registry());
  node_builder.Input(a).Attr("index", index);
  return opts.FinalizeBuilder(&node_builder);
}

Status Encapsulate(GraphDef* graphdef, FunctionDefLibrary* library) {
  Status s;
  // Convert the GraphDef to a Graph
  std::unique_ptr<FunctionLibraryDefinition> lib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(), *library));
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  std::unique_ptr<Graph> graph(new Graph(lib_def.get()));
  s = ConvertGraphDefToGraph(options, *graphdef, graph.get());
  if (!s.ok()) return s;

  std::unique_ptr<Graph> graph_out;
  s = EncapsulateSubgraphsInFunctions("_encapsulate", *graph,
                                      /* rewrite_subgraph_fn= */ {},
                                      /* parallel_checking= */ false,
                                      &graph_out, lib_def.get());
  if (!s.ok()) return s;

  GraphDef graphdef_out;
  graph_out->ToGraphDef(&graphdef_out);
  graphdef->Swap(&graphdef_out);

  *library = lib_def->ToProto();
  return s;
}

// If there are no marked nodes, funcification should be a no-op.
TEST(EncapsulateSubgraphsTest, NoFunctions) {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);

  Node* a = Input(builder.opts().WithName("A"));
  Node* b = Input(builder.opts().WithName("B"));
  Node* c = Unary(a, builder.opts().WithName("C"));
  Binary(b, c, builder.opts().WithName("D"));

  GraphDef graphdef_in;
  FunctionDefLibrary library_in;
  builder.ToGraphDef(&graphdef_in);
  *library_in.add_function() = test::function::XTimesTwo();

  GraphDef graphdef_out = graphdef_in;
  FunctionDefLibrary library_out = library_in;
  TF_EXPECT_OK(Encapsulate(&graphdef_out, &library_out));

  TF_EXPECT_GRAPH_EQ(graphdef_in, graphdef_out);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_in, library_out);
}

// Test with one function to transform.
TEST(EncapsulateSubgraphsTest, OneFunction) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    *library.add_function() = test::function::XTimesTwo();

    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    // Give nodes 'c' and 'd' names that collide after lowercasing.
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d = Binary(b, c, b1.opts().WithName("c").WithControlInput(c).WithAttr(
                               "_encapsulate", "F1"));
    Binary(a, d, b1.opts().WithName("E"));
    b1.ToGraphDef(&graphdef);
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"input__0:float", "input__1:float"}, {"output__2:float"}, {},
      {
          {{"C"}, "UnaryTest", {"input__0"}},
          {{"c"}, "BinaryTest", {"input__1", "C:o:0"}, {}, {"C"}},
      },
      {{"output__2", "c:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    NodeBuilder node_builder("F1", "F1", lib_def.get());
    node_builder.Input(a).Input(b);
    Node* call = b2.opts().FinalizeBuilder(&node_builder);

    Binary(a, call, b2.opts().WithName("E"));
    b2.ToGraphDef(&graphdef_expected);
  }

  // If there are no marked nodes, funcification should be a no-op.
  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with two functions to transform.
TEST(EncapsulateSubgraphsTest, TwoFunctions) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    *library.add_function() = test::function::XTimesTwo();

    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* control = Input(b1.opts().WithName("Control"));
    Node* c =
        Unary(a, b1.opts().WithName("C").WithControlInput(control).WithAttr(
                     "_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithControlInput(control).WithAttr(
                         "_encapsulate", "F2"));
    Binary(a, d, b1.opts().WithName("E"));
    b1.ToGraphDef(&graphdef);
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"input__0:float"}, {"output__1:float"}, {},
      {
          {{"C"}, "UnaryTest", {"input__0"}},
      },
      {{"output__1", "C:o:0"}});
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F2", {"input__0:float", "input__1:float"}, {"output__2:float"}, {},
      {
          {{"D"}, "BinaryTest", {"input__0", "input__1"}},
      },
      {{"output__2", "D:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));
    Node* control = Input(b2.opts().WithName("Control"));

    NodeBuilder nb("F1", "F1", lib_def.get());
    nb.Input(a).ControlInput(control);
    Node* call1 = b2.opts().FinalizeBuilder(&nb);

    NodeBuilder nb2("F2", "F2", lib_def.get());
    nb2.Input(b).Input(call1).ControlInput(control);
    Node* call2 = b2.opts().FinalizeBuilder(&nb2);

    Binary(a, call2, b2.opts().WithName("E"));
    b2.ToGraphDef(&graphdef_expected);
  }

  // If there are no marked nodes, funcification should be a no-op.
  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Returns a vector of node names in 'graph', sorted by name.
std::vector<string> GraphNodes(const Graph& graph) {
  std::vector<string> nodes;
  for (const auto& node : graph.nodes()) {
    if (!node->IsSource() && !node->IsSink()) {
      nodes.push_back(node->name());
    }
  }
  std::sort(nodes.begin(), nodes.end());
  return nodes;
}

// Returns a sorted vector of (src, dst) edges in 'graph'.
std::vector<std::pair<string, string>> GraphEdges(const Graph& graph) {
  std::vector<std::pair<string, string>> edges;
  for (const Edge* edge : graph.edges()) {
    if (edge->src()->IsSource() || edge->dst()->IsSink()) continue;
    edges.emplace_back(
        strings::StrCat(edge->src()->name(), ":", edge->src_output()),
        strings::StrCat(edge->dst()->name(), ":", edge->dst_input()));
  }
  std::sort(edges.begin(), edges.end());
  return edges;
}

TEST(EncapsulateSubgraphsTest, InputDeduplication) {
  Scope root = Scope::NewRootScope().ExitOnError().WithDevice(
      "/job:localhost/replica:0/task:0/cpu:0");
  auto x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  auto add1 = ops::Add(root.WithOpName("add1"), x, x);
  add1.node()->AddAttr("_cluster", "cluster1");
  auto add2 = ops::Add(root.WithOpName("add2"), add1, add1);
  add2.node()->AddAttr("_cluster", "cluster2");
  auto out = ops::Mul(root.WithOpName("mul"), add1, add2);

  Graph graph_before_encapsulation(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph_before_encapsulation));

  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(EncapsulateSubgraphsInFunctions(
      "_cluster", graph_before_encapsulation, /*rewrite_subgraph_fn=*/{},
      /*parallel_checking=*/false, &graph, &library));

  std::vector<string> expected_nodes = {"cluster1", "cluster2", "mul", "x"};
  EXPECT_EQ(expected_nodes, GraphNodes(*graph));

  std::vector<std::pair<string, string>> expected_edges = {
      {"cluster1:0", "cluster2:0"},
      {"cluster1:0", "mul:0"},
      {"cluster2:0", "mul:1"},
      {"x:0", "cluster1:0"}};
  EXPECT_EQ(expected_edges, GraphEdges(*graph));
}

TEST(EncapsulateSubgraphsTest, ParallelChecking) {
  Scope root = Scope::NewRootScope().ExitOnError().WithDevice(
      "/job:localhost/replica:0/task:0/cpu:0");
  auto x1 = ops::Placeholder(root.WithOpName("x1"), DT_FLOAT);
  auto x2 = ops::Placeholder(root.WithOpName("x2"), DT_FLOAT);
  auto add1 = ops::Add(root.WithOpName("add1"), x1, x2);
  add1.node()->AddAttr("_cluster", "cluster1");
  auto add2 = ops::Add(root.WithOpName("add2"), add1, x2);
  add2.node()->AddAttr("_cluster", "cluster1");
  auto out = ops::Mul(root.WithOpName("mul"), x1, add2);

  Graph graph_before_encapsulation(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph_before_encapsulation));

  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(EncapsulateSubgraphsInFunctions(
      "_cluster", graph_before_encapsulation, /*rewrite_subgraph_fn=*/{},
      /*parallel_checking=*/true, &graph, &library));

  std::vector<string> expected_nodes = {
      "add1", "add2", "cluster1", "cluster1_parallel_check/_0",
      "mul",  "x1",   "x2"};
  EXPECT_EQ(expected_nodes, GraphNodes(*graph));

  std::vector<std::pair<string, string>> expected_edges = {
      {"add1:0", "add2:0"},
      {"add2:0", "cluster1_parallel_check/_0:0"},
      {"cluster1:0", "cluster1_parallel_check/_0:1"},
      {"cluster1_parallel_check/_0:0", "mul:1"},
      {"x1:0", "add1:0"},
      {"x1:0", "cluster1:0"},
      {"x1:0", "mul:0"},
      {"x2:0", "add1:1"},
      {"x2:0", "add2:1"},
      {"x2:0", "cluster1:1"},
  };
  EXPECT_EQ(expected_edges, GraphEdges(*graph));
}

}  // namespace
}  // namespace tensorflow
