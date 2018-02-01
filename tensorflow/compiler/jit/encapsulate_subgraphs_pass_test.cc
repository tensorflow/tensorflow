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

#include <utility>

#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

bool EqualFunctionDef(const FunctionDef& a, const FunctionDef& b,
                      string* diff) {
  // TODO(phawkins) use a more sophisticated equality test.
  if (a.DebugString() != b.DebugString()) {
    if (diff) {
      *diff = strings::StrCat("Definition mismatch for function ",
                              a.signature().name(), ", expected:\n",
                              a.DebugString(), "\ngot:\n", b.DebugString());
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
    EXPECT_TRUE(EqualFunctionDefLibrary(expected, actual, &diff)) \
        << diff << "\nActual: " << actual.DebugString();          \
  } while (false)

// TODO(misard): remove these fake registrations once there are real Ops to be
// compiled.
REGISTER_OP("_XlaSendToHost")
    .Input("input: dtypes")
    .Attr("dtypes: list(type) >= 0");

REGISTER_OP("_XlaRecvFromHost")
    .Output("output: dtypes")
    .Attr("dtypes: list(type) >= 0");

REGISTER_OP("_XlaSendFromHost")
    .Input("input: dtypes")
    .Attr("dtypes: list(type) >= 0");

REGISTER_OP("_XlaRecvAtHost")
    .Output("output: dtypes")
    .Attr("dtypes: list(type) >= 0");

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

Node* NoOp(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("NoOp", opts);
}

Node* Input(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("InputTest", opts);
}

Node* RecvAtHost(const gtl::ArraySlice<DataType>& dtypes,
                 const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("_XlaRecvAtHost"),
                           "_XlaRecvAtHost", opts.op_registry());
  return opts.WithAttr("dtypes", dtypes).FinalizeBuilder(&node_builder);
}

Node* SendFromHost(const std::vector<ops::NodeOut>& inputs,
                   const gtl::ArraySlice<DataType>& dtypes,
                   const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("_XlaSendFromHost"),
                           "_XlaSendFromHost", opts.op_registry());
  node_builder.Input(inputs);
  return opts.WithAttr("dtypes", dtypes).FinalizeBuilder(&node_builder);
}

Node* Unary(ops::NodeOut a, const GraphDefBuilder::Options& opts) {
  return ops::UnaryOp("UnaryTest", std::move(a), opts);
}

Node* Binary(ops::NodeOut a, ops::NodeOut b,
             const GraphDefBuilder::Options& opts) {
  return ops::BinaryOp("BinaryTest", std::move(a), std::move(b), opts);
}

Node* AddNLike(const std::vector<ops::NodeOut>& inputs,
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
  node_builder.Input(std::move(a)).Attr("index", index);
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
  s = EncapsulateSubgraphsInFunctions("_encapsulate", "_outside", *graph,
                                      /*rewrite_subgraph_fn=*/{},
                                      /*parallel_checking=*/false,
                                      /*reuse_existing_functions=*/false,
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
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef_in));
  *library_in.add_function() = test::function::XTimesTwo();

  GraphDef graphdef_out = graphdef_in;
  FunctionDefLibrary library_out = library_in;
  TF_EXPECT_OK(Encapsulate(&graphdef_out, &library_out));

  // If there are no marked nodes, funcification should be a no-op.
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
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"c_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"c"}, "BinaryTest", {"b_0_arg", "C:o:0"}, {}, {"C"}},
      },
      {{"c_0_retval", "c:o:0"}});

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
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

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
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float"}, {"c_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
      },
      {{"c_0_retval", "C:o:0"}});
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F2", {"b_0_arg:float", "c_0_arg:float"}, {"d_0_retval:float"}, {},
      {
          {{"D"}, "BinaryTest", {"b_0_arg", "c_0_arg"}},
      },
      {{"d_0_retval", "D:o:0"}});

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
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
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
      "_cluster", "_outside", graph_before_encapsulation,
      /*rewrite_subgraph_fn=*/{}, /*parallel_checking=*/false,
      /*reuse_existing_functions=*/false, &graph, &library));

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
      "_cluster", "_outside", graph_before_encapsulation,
      /*rewrite_subgraph_fn=*/{}, /*parallel_checking=*/true,
      /*reuse_existing_functions=*/false, &graph, &library));

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

const Node* FindNodeByName(const Graph& graph, const string& name) {
  for (const Node* node : graph.nodes()) {
    if (node->name() == name) return node;
  }
  return nullptr;
}

bool HasGuaranteeConstAttr(const Node& n) {
  bool is_guaranteed_constant = false;
  if (!GetNodeAttr(n.attrs(), "_is_guaranteed_constant",
                   &is_guaranteed_constant)
           .ok()) {
    return false;
  }
  return is_guaranteed_constant;
}

TEST(EncapsulateSubgraphsWithGuaranteeConstOpTest, Simple) {
  Scope root = Scope::NewRootScope().ExitOnError().WithDevice(
      "/job:localhost/replica:0/task:0/cpu:0");
  auto x1 = ops::Placeholder(root.WithOpName("x1"), DT_FLOAT);
  auto const_x2 = ops::Const(root.WithOpName("const_x2"), 10.0f);
  auto const_guarantee_x1 =
      ops::GuaranteeConst(root.WithOpName("const_guarantee_x1"), x1);
  auto add1 = ops::Add(root.WithOpName("add1"), const_guarantee_x1, const_x2);
  add1.node()->AddAttr("_encapsulate", "encapsulate1");

  Graph graph_before(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph_before));

  std::unique_ptr<Graph> graph_after;
  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  int guaranteed_consts = 0;
  TF_ASSERT_OK(EncapsulateSubgraphsInFunctions(
      "_encapsulate", "_outside", graph_before,
      /*rewrite_subgraph_fn=*/
      [&guaranteed_consts](std::unique_ptr<Graph>* graph_ptr,
                           std::vector<int>* input_permutation,
                           std::vector<int>* output_permutation,
                           NodeDef* call_def) {
        Graph* graph = graph_ptr->get();
        for (const Node* n : graph->nodes()) {
          if (n->type_string() == "_Arg" &&
              StringPiece(n->name()).starts_with("const")) {
            ++guaranteed_consts;
            EXPECT_TRUE(HasGuaranteeConstAttr(*n));
          } else {
            EXPECT_FALSE(HasGuaranteeConstAttr(*n));
          }
        }
        return Status::OK();
      },
      /*parallel_checking=*/false,
      /*reuse_existing_functions=*/false, &graph_after, &library));
  EXPECT_EQ(2, guaranteed_consts);
}

TEST(EncapsulateSubgraphsWithGuaranteeConstOpTest, Add) {
  Scope root = Scope::NewRootScope().ExitOnError().WithDevice(
      "/job:localhost/replica:0/task:0/cpu:0");
  auto x1 = ops::Placeholder(root.WithOpName("x1"), DT_FLOAT);
  auto x2 = ops::Placeholder(root.WithOpName("x2"), DT_FLOAT);
  auto const_guarantee_x1 =
      ops::GuaranteeConst(root.WithOpName("const_guarantee_x1"), x1);
  auto const_guarantee_x2 =
      ops::GuaranteeConst(root.WithOpName("const_guarantee_x2"), x2);
  auto const_guarantee_add1 = ops::Add(root.WithOpName("const_guarantee_add1"),
                                       const_guarantee_x1, const_guarantee_x2);
  auto add2 = ops::Add(root.WithOpName("add2"), const_guarantee_x1, x2);
  auto mul1 = ops::Mul(root.WithOpName("mul1"), const_guarantee_add1, add2);
  mul1.node()->AddAttr("_encapsulate", "encapsulate1");

  Graph graph_before(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph_before));

  std::unique_ptr<Graph> graph_after;
  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  int guaranteed_consts = 0;
  TF_ASSERT_OK(EncapsulateSubgraphsInFunctions(
      "_encapsulate", "_outside", graph_before,
      /*rewrite_subgraph_fn=*/
      [&guaranteed_consts](std::unique_ptr<Graph>* graph_ptr,
                           std::vector<int>* input_permutation,
                           std::vector<int>* output_permutation,
                           NodeDef* call_def) {
        Graph* graph = graph_ptr->get();
        for (const Node* n : graph->nodes()) {
          if (n->type_string() == "_Arg" &&
              StringPiece(n->name()).starts_with("const")) {
            ++guaranteed_consts;
            EXPECT_TRUE(HasGuaranteeConstAttr(*n));
          } else {
            EXPECT_FALSE(HasGuaranteeConstAttr(*n));
          }
        }
        return Status::OK();
      },
      /*parallel_checking=*/false,
      /*reuse_existing_functions=*/false, &graph_after, &library));
  // Only 1 runtime const, which is const_guarantee_add1. Add2 has one const
  // and another non-const, so overall non-const.
  EXPECT_EQ(1, guaranteed_consts);
}

// Test with one function to transform and one outside_compilation cluster.
TEST(EncapsulateSubgraphsTest, OneFunctionOneOutside) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    *library.add_function() = test::function::XTimesTwo();

    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    // Give nodes 'c' and 'd' names that collide after lowercasing.
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d = Binary(b, c,
                     b1.opts().WithName("c").WithControlInput(c).WithAttr(
                         "_encapsulate", "F1"));
    Node* e = Binary(c, d,
                     b1.opts()
                         .WithName("E")
                         .WithControlInputs({b, d})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* f = Binary(c, e,
                     b1.opts().WithName("F").WithControlInput(e).WithAttr(
                         "_encapsulate", "F1"));
    Binary(a, f, b1.opts().WithName("G").WithControlInput(e));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"c"}, "BinaryTest", {"b_0_arg", "C:o:0"}, {}, {"C"}},
          {{"F"},
           "BinaryTest",
           {"C:o:0", "outside_compilation_O1_recv:output:0"},
           {},
           {"outside_compilation_O1_recv"}},
          {{"outside_compilation_O1_send"},
           "_XlaSendToHost",
           {"C:o:0", "c:o:0"},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT, DT_FLOAT})}},
           {"c"}},
          {{"outside_compilation_O1_recv"},
           "_XlaRecvFromHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}},
           {"outside_compilation_O1_send"}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    NodeBuilder node_builder("F1", "F1", lib_def.get());
    node_builder.Input(a).Input(b);
    Node* call = b2.opts().FinalizeBuilder(&node_builder);

    Node* recv =
        RecvAtHost({DT_FLOAT, DT_FLOAT},
                   b2.opts().WithName("outside_compilation_F1_O1_recv"));
    Node* e = Binary(ops::NodeOut(recv, 0), ops::NodeOut(recv, 1),
                     b2.opts().WithName("E").WithControlInputs({recv, b}));
    Node* send = SendFromHost({e}, {DT_FLOAT},
                              b2.opts()
                                  .WithName("outside_compilation_F1_O1_send")
                                  .WithControlInput(e));

    Node* s = NoOp(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv, send}));

    Binary(a, call, b2.opts().WithName("G").WithControlInputs({s, e}));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one function to transform and two outside_compilation clusters.
TEST(EncapsulateSubgraphsTest, OneFunctionTwoOutside) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Binary(c, d,
                     b1.opts()
                         .WithName("E")
                         .WithControlInputs({b, d})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* f = Binary(c, e,
                     b1.opts().WithName("F").WithControlInput(e).WithAttr(
                         "_encapsulate", "F1"));
    Node* g = Binary(e, f,
                     b1.opts()
                         .WithName("G")
                         .WithControlInputs({e, f})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O2"));
    Node* h = Binary(d, e,
                     b1.opts()
                         .WithName("H")
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O2"));
    Node* i = Unary(h, b1.opts().WithName("I").WithAttr("_encapsulate", "F1"));
    Binary(g, i, b1.opts().WithName("J"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"i_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}, {}},
          {{"I"}, "UnaryTest", {"outside_compilation_O2_recv:output:0"}},
          {{"F"},
           "BinaryTest",
           {"C:o:0", "outside_compilation_O1_recv:output:0"},
           {},
           {"outside_compilation_O1_recv"}},
          {{"outside_compilation_O2_send"},
           "_XlaSendToHost",
           {"D:o:0", "F:o:0"},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT, DT_FLOAT})}},
           {"F"}},
          {{"outside_compilation_O1_send"},
           "_XlaSendToHost",
           {"C:o:0", "D:o:0"},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT, DT_FLOAT})}},
           {"D"}},
          {{"outside_compilation_O2_recv"},
           "_XlaRecvFromHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}},
           {"outside_compilation_O2_send"}},
          {{"outside_compilation_O1_recv"},
           "_XlaRecvFromHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}},
           {"outside_compilation_O1_send"}},
      },
      {{"i_0_retval", "I:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    NodeBuilder node_builder("F1", "F1", lib_def.get());
    node_builder.Input(a).Input(b);
    Node* call = b2.opts().FinalizeBuilder(&node_builder);

    Node* recv1 =
        RecvAtHost({DT_FLOAT, DT_FLOAT},
                   b2.opts().WithName("outside_compilation_F1_O1_recv"));
    Node* e = Binary(ops::NodeOut(recv1, 0), ops::NodeOut(recv1, 1),
                     b2.opts().WithName("E").WithControlInputs({recv1, b}));
    Node* send1 = SendFromHost({e}, {DT_FLOAT},
                               b2.opts()
                                   .WithName("outside_compilation_F1_O1_send")
                                   .WithControlInput(e));

    Node* recv2 =
        RecvAtHost({DT_FLOAT, DT_FLOAT},
                   b2.opts().WithName("outside_compilation_F1_O2_recv"));
    Node* g = Binary(e, ops::NodeOut(recv2, 1),
                     b2.opts().WithName("G").WithControlInputs({recv2, e}));
    Node* h = Binary(ops::NodeOut(recv2, 0), e, b2.opts().WithName("H"));
    Node* send2 = SendFromHost(
        {h}, {DT_FLOAT}, b2.opts().WithName("outside_compilation_F1_O2_send"));

    Node* s = NoOp(b2.opts()
                       .WithName("F1_sequencer")
                       .WithControlInputs({recv1, send1, recv2, send2}));

    Binary(g, call, b2.opts().WithName("J").WithControlInput(s));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with two functions to transform, each with one outside_compilation
// cluster.
TEST(EncapsulateSubgraphsTest, TwoFunctionsTwoOutside) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Binary(c, d,
                     b1.opts()
                         .WithName("E")
                         .WithControlInputs({b, d})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* f = Binary(c, e,
                     b1.opts().WithName("F").WithControlInput(e).WithAttr(
                         "_encapsulate", "F1"));
    Node* g = Binary(e, f,
                     b1.opts().WithName("G").WithControlInputs({e, f}).WithAttr(
                         "_encapsulate", "F2"));
    Node* h = Binary(d, g,
                     b1.opts()
                         .WithName("H")
                         .WithAttr("_encapsulate", "F2")
                         .WithAttr("_outside", "O1"));
    Node* i =
        Binary(f, h, b1.opts().WithName("I").WithAttr("_encapsulate", "F2"));
    Binary(g, i, b1.opts().WithName("J"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"},
      {"f_0_retval:float", "d_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"},
           "BinaryTest",
           {"C:o:0", "outside_compilation_O1_recv:output:0"},
           {},
           {"outside_compilation_O1_recv"}},
          {{"outside_compilation_O1_send"},
           "_XlaSendToHost",
           {"C:o:0", "D:o:0"},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT, DT_FLOAT})}},
           {"D"}},
          {{"outside_compilation_O1_recv"},
           "_XlaRecvFromHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}},
           {"outside_compilation_O1_send"}},
      },
      {{"d_0_retval", "D:o:0"}, {"f_0_retval", "F:o:0"}});

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F2", {"e_0_arg:float", "f_0_arg:float"},
      {"g_0_retval:float", "i_0_retval:float"}, {},
      {
          {{"G"}, "BinaryTest", {"e_0_arg", "f_0_arg"}},
          {{"I"},
           "BinaryTest",
           {"f_0_arg", "outside_compilation_O1_recv:output:0"}},
          {{"outside_compilation_O1_send"},
           "_XlaSendToHost",
           {"G:o:0"},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}}},
          {{"outside_compilation_O1_recv"},
           "_XlaRecvFromHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}},
           {"outside_compilation_O1_send"}},
      },
      {{"g_0_retval", "G:o:0"}, {"i_0_retval", "I:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* recv1 =
        RecvAtHost({DT_FLOAT, DT_FLOAT},
                   b2.opts().WithName("outside_compilation_F1_O1_recv"));
    Node* e = Binary(ops::NodeOut(recv1, 0), ops::NodeOut(recv1, 1),
                     b2.opts().WithName("E").WithControlInputs({recv1, b}));
    Node* send1 = SendFromHost({e}, {DT_FLOAT},
                               b2.opts()
                                   .WithName("outside_compilation_F1_O1_send")
                                   .WithControlInput(e));
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 = b2.opts().FinalizeBuilder(&node_builder1);
    Node* s1 = NoOp(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv1, send1}));

    Node* recv2 = RecvAtHost(
        {DT_FLOAT}, b2.opts().WithName("outside_compilation_F2_O1_recv"));
    Node* h = Binary(ops::NodeOut(call1, 1), recv2,
                     b2.opts().WithName("H").WithControlInput(s1));
    Node* send2 = SendFromHost(
        {h}, {DT_FLOAT}, b2.opts().WithName("outside_compilation_F2_O1_send"));

    NodeBuilder node_builder2("F2", "F2", lib_def.get());
    node_builder2.Input(e).Input(call1);
    Node* call2 = b2.opts()
                      .WithControlInputs({s1, e, call1})
                      .FinalizeBuilder(&node_builder2);
    Node* s2 = NoOp(
        b2.opts().WithName("F2_sequencer").WithControlInputs({recv2, send2}));
    Binary(call2, ops::NodeOut(call2, 1),
           b2.opts().WithName("J").WithControlInput(s2));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no inputs from the
// compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationNoInputs) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(a, b1.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f =
        Binary(d, e, b1.opts().WithName("F").WithAttr("_encapsulate", "F1"));
    Unary(f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"},
           "BinaryTest",
           {"D:o:0", "outside_compilation_O1_recv:output:0"}},
          {{"outside_compilation_O1_recv"},
           "_XlaRecvFromHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* e = Unary(a, b2.opts().WithName("E"));
    Node* send1 = SendFromHost(
        {e}, {DT_FLOAT}, b2.opts().WithName("outside_compilation_F1_O1_send"));
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 = b2.opts().FinalizeBuilder(&node_builder1);
    Node* s1 = NoOp(b2.opts().WithName("F1_sequencer").WithControlInput(send1));

    Unary(call1, b2.opts().WithName("G").WithControlInput(s1));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no data inputs but has a
// control input from the compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationControlInput) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(a, b1.opts()
                           .WithName("E")
                           .WithControlInput(d)
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f =
        Binary(d, e, b1.opts().WithName("F").WithAttr("_encapsulate", "F1"));
    Unary(f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"},
           "BinaryTest",
           {"D:o:0", "outside_compilation_O1_recv:output:0"}},
          {{"outside_compilation_O1_send"},
           "_XlaSendToHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({})}},
           {"D"}},
          {{"outside_compilation_O1_recv"},
           "_XlaRecvFromHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}},
           {"outside_compilation_O1_send"}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* recv1 =
        RecvAtHost({}, b2.opts().WithName("outside_compilation_F1_O1_recv"));
    Node* e = Unary(a, b2.opts().WithName("E").WithControlInput(recv1));
    Node* send1 = SendFromHost(
        {e}, {DT_FLOAT}, b2.opts().WithName("outside_compilation_F1_O1_send"));
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 = b2.opts().FinalizeBuilder(&node_builder1);
    Node* s1 = NoOp(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv1, send1}));

    Unary(call1, b2.opts().WithName("G").WithControlInput(s1));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no outputs from the
// compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationNoOutputs) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(d, b1.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f = Unary(d, b1.opts().WithName("F").WithAttr("_encapsulate", "F1"));
    Binary(e, f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"}, "UnaryTest", {"D:o:0"}},
          {{"outside_compilation_O1_send"},
           "_XlaSendToHost",
           {"D:o:0"},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* recv1 = RecvAtHost(
        {DT_FLOAT}, b2.opts().WithName("outside_compilation_F1_O1_recv"));
    Node* e = Unary(recv1, b2.opts().WithName("E"));
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 = b2.opts().FinalizeBuilder(&node_builder1);
    Node* s1 = NoOp(b2.opts().WithName("F1_sequencer").WithControlInput(recv1));

    Binary(e, call1, b2.opts().WithName("G").WithControlInput(s1));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no data outputs but has a
// control output to the compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationControlOutput) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(d, b1.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f = Unary(d, b1.opts().WithName("F").WithControlInput(e).WithAttr(
                           "_encapsulate", "F1"));
    Binary(e, f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"}, "UnaryTest", {"D:o:0"}, {}, {"outside_compilation_O1_recv"}},
          {{"outside_compilation_O1_send"},
           "_XlaSendToHost",
           {"D:o:0"},
           {{"dtypes", gtl::ArraySlice<DataType>({DT_FLOAT})}}},
          {{"outside_compilation_O1_recv"},
           "_XlaRecvFromHost",
           {},
           {{"dtypes", gtl::ArraySlice<DataType>({})}},
           {"outside_compilation_O1_send"}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* recv1 = RecvAtHost(
        {DT_FLOAT}, b2.opts().WithName("outside_compilation_F1_O1_recv"));
    Node* e = Unary(recv1, b2.opts().WithName("E"));
    Node* send1 = SendFromHost({}, {},
                               b2.opts()
                                   .WithName("outside_compilation_F1_O1_send")
                                   .WithControlInput(e));
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 = b2.opts().FinalizeBuilder(&node_builder1);
    Node* s1 = NoOp(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv1, send1}));

    Binary(e, call1, b2.opts().WithName("G").WithControlInput(s1));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no outputs from the
// compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationNoInputsOrOutputs) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(a, b1.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f = Unary(d, b1.opts().WithName("F").WithAttr("_encapsulate", "F1"));
    Binary(e, f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"}, "UnaryTest", {"D:o:0"}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* e = Unary(a, b2.opts().WithName("E"));
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 = b2.opts().FinalizeBuilder(&node_builder1);

    Binary(e, call1, b2.opts().WithName("G"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

}  // namespace
}  // namespace tensorflow
