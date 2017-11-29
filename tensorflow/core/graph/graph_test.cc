/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/graph.h"

#include <set>
#include <vector>
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

REGISTER_OP("OneInput").Input("x: float");

REGISTER_OP("OneOutput").Output("y: float");

REGISTER_OP("OneInputTwoOutputs")
    .Input("x: float")
    .Output("y: float")
    .Output("z: float");

REGISTER_OP("TwoInputsOneOutput")
    .Input("x: float")
    .Input("y: float")
    .Output("z: float");

class GraphTest : public ::testing::Test {
 protected:
  GraphTest() : graph_(OpRegistry::Global()) {}
  ~GraphTest() override {}

  static void VerifyNodes(Node* node, const std::vector<Node*>& expected_in,
                          const std::vector<Node*>& expected_out) {
    std::vector<Node*> in;
    for (const Edge* e : node->in_edges()) {
      in.push_back(e->src());
    }
    EXPECT_EQ(Stringify(expected_in), Stringify(in));

    std::vector<Node*> out;
    for (const Edge* e : node->out_edges()) {
      out.push_back(e->dst());
    }
    EXPECT_EQ(Stringify(expected_out), Stringify(out));
  }

  void VerifyGraphStats() {
    int nodes = 0;
    for (const Node* n : graph_.nodes()) {
      VLOG(1) << n->id();
      ++nodes;
    }
    EXPECT_EQ(nodes, graph_.num_nodes());
    int edges = 0;
    for (const Edge* e : graph_.edges()) {
      VLOG(1) << e->id();
      ++edges;
    }
    EXPECT_EQ(edges, graph_.num_edges());
  }

  Node* AddNodeWithName(const string& name) {
    Node* node;
    TF_CHECK_OK(NodeBuilder(name, "NoOp").Finalize(&graph_, &node));
    return node;
  }

  Node* FromNodeDef(const string& name, const string& node_type,
                    int num_inputs) {
    auto builder = NodeDefBuilder(name, node_type);
    for (int i = 0; i < num_inputs; ++i) {
      builder = builder.Input(strings::StrCat("node_", i), i, DT_FLOAT);
    }

    NodeDef node_def;
    TF_CHECK_OK(builder.Finalize(&node_def));

    Status s;
    Node* node = graph_.AddNode(node_def, &s);
    TF_CHECK_OK(s);
    return node;
  }

  Graph graph_;

 private:
  // Convert a list of nodes to a sorted list of strings so failure messages
  // are readable.
  static std::vector<string> Stringify(const std::vector<Node*>& nodes) {
    std::vector<string> result;
    result.reserve(nodes.size());
    for (Node* n : nodes) {
      result.push_back(n->DebugString());
    }
    std::sort(result.begin(), result.end());
    return result;
  }
};

TEST_F(GraphTest, Constructor) {
  Node* source = graph_.source_node();
  EXPECT_NE(source, nullptr);
  Node* sink = graph_.sink_node();
  EXPECT_NE(sink, nullptr);
  VerifyNodes(source, {}, {sink});
  VerifyNodes(sink, {source}, {});
  EXPECT_EQ(2, graph_.num_node_ids());
  VerifyGraphStats();
}

TEST_F(GraphTest, RemoveThenAdd) {
  AddNodeWithName("A");
  Node* b = AddNodeWithName("B");
  const int b_id = b->id();
  AddNodeWithName("C");
  EXPECT_EQ(5, graph_.num_node_ids());
  graph_.RemoveNode(b);
  EXPECT_EQ(5, graph_.num_node_ids());
  Node* d = AddNodeWithName("D");
  EXPECT_NE(b_id, d->id());  // Ids should not be reused.
  EXPECT_EQ(6, graph_.num_node_ids());
  VerifyGraphStats();
}

TEST_F(GraphTest, InNodesAndOutNodes) {
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* b = AddNodeWithName("B");
  Node* c = FromNodeDef("C", "OneInput", 1);
  graph_.RemoveNode(b);
  Node* d = AddNodeWithName("D");

  const Edge* source_to_a = graph_.AddControlEdge(graph_.source_node(), a);
  graph_.AddControlEdge(a, graph_.sink_node());
  graph_.AddEdge(a, 0, c, 0);
  graph_.AddControlEdge(c, graph_.sink_node());

  EXPECT_EQ("A", a->name());
  VerifyNodes(a, {graph_.source_node()}, {c, graph_.sink_node()});

  EXPECT_EQ("C", c->name());
  VerifyNodes(c, {a}, {graph_.sink_node()});

  EXPECT_EQ("D", d->name());
  VerifyNodes(d, {}, {});

  VerifyNodes(graph_.source_node(), {}, {a, graph_.sink_node()});
  VerifyNodes(graph_.sink_node(), {a, c, graph_.source_node()}, {});

  graph_.RemoveEdge(source_to_a);
  VerifyNodes(a, {}, {c, graph_.sink_node()});
  VerifyNodes(graph_.source_node(), {}, {graph_.sink_node()});  // no more a

  graph_.RemoveNode(c);
  VerifyNodes(a, {}, {graph_.sink_node()});                        // no more c
  VerifyNodes(graph_.sink_node(), {a, graph_.source_node()}, {});  // no more c
  EXPECT_EQ(6, graph_.num_node_ids());
  EXPECT_EQ(5, graph_.num_edge_ids());
  VerifyGraphStats();
}

TEST_F(GraphTest, NodeByIndex) {
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* c = FromNodeDef("C", "OneInput", 1);
  graph_.AddEdge(a, 0, c, 0);

  // Ask for 'a' from 'c' by index.
  const Node* a_copy;
  TF_ASSERT_OK(c->input_node(0, &a_copy));
  EXPECT_EQ(a, a_copy);

  const Edge* e;
  TF_ASSERT_OK(c->input_edge(0, &e));
  EXPECT_EQ(0, e->dst_input());
  EXPECT_EQ(a, e->src());
  EXPECT_EQ(c, e->dst());
  EXPECT_EQ(0, e->src_output());

  Node* t = FromNodeDef("T", "TwoInputsOneOutput", 2);
  graph_.AddEdge(a, 0, t, 0);
  // Weird self edge
  graph_.AddEdge(t, 0, t, 1);

  const Node* t_0;
  const Node* t_1;
  TF_ASSERT_OK(t->input_node(0, &t_0));
  EXPECT_EQ(a, t_0);
  TF_ASSERT_OK(t->input_node(1, &t_1));
  EXPECT_EQ(t, t_1);

  TF_ASSERT_OK(t->input_edge(1, &e));
  EXPECT_EQ(1, e->dst_input());
  EXPECT_EQ(t, e->src());

  std::vector<const Edge*> t_input_edges;
  TF_ASSERT_OK(t->input_edges(&t_input_edges));
  ASSERT_EQ(2, t_input_edges.size());
  EXPECT_EQ(a, t_input_edges[0]->src());
  EXPECT_EQ(e, t_input_edges[1]);

  // Check out of bounds access
  EXPECT_FALSE(c->input_node(1, &a_copy).ok());
  EXPECT_FALSE(c->input_node(-1, &a_copy).ok());

  graph_.RemoveNode(a);

  // 'c's input_node entry should be invalidated.
  Status s = c->input_node(0, &a_copy);
  EXPECT_FALSE(s.ok());

  // Add two new nodes.
  Node* a_new = FromNodeDef("A_new", "OneOutput", 0);
  Node* b_new = FromNodeDef("B_new", "OneOutput", 0);

  // Connect one up to c.
  graph_.AddEdge(a_new, 0, c, 0);
  const Edge* a_new_c_edge;
  TF_ASSERT_OK(c->input_edge(0, &a_new_c_edge));

  // Connect up the second edge
  graph_.AddEdge(b_new, 0, c, 0);
  const Edge* b_new_c_edge;
  TF_ASSERT_OK(c->input_edge(0, &b_new_c_edge));

  // Now remove the old one
  graph_.RemoveEdge(a_new_c_edge);

  // Check that the second edge can still be retrieved
  TF_ASSERT_OK(c->input_edge(0, &b_new_c_edge));

  std::vector<const Edge*> c_input_edges;
  TF_ASSERT_OK(c->input_edges(&c_input_edges));
  ASSERT_EQ(1, c_input_edges.size());
  EXPECT_EQ(b_new_c_edge, c_input_edges[0]);
}

TEST_F(GraphTest, NodeIteration) {
  // Set up the graph with some holes due to removals.
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* b = AddNodeWithName("B");
  Node* c = FromNodeDef("C", "OneInput", 1);
  graph_.RemoveNode(b);
  Node* d = AddNodeWithName("D");
  const Edge* source_to_a = graph_.AddControlEdge(graph_.source_node(), a);
  graph_.AddControlEdge(a, graph_.sink_node());
  graph_.AddEdge(a, 0, c, 0);
  graph_.AddControlEdge(c, graph_.sink_node());
  graph_.RemoveEdge(source_to_a);
  graph_.RemoveNode(c);

  // expected = set of all node DebugStrings we expect in the graph
  std::set<string> expected;
  expected.insert(graph_.source_node()->DebugString());
  expected.insert(a->DebugString());
  expected.insert(d->DebugString());
  expected.insert(graph_.sink_node()->DebugString());

  // Verify that iterating through ids gets the same set of nodes.
  std::set<string> actual;
  for (int id = 0; id < graph_.num_node_ids(); ++id) {
    Node* node = graph_.FindNodeId(id);
    if (node != nullptr) {
      actual.insert(node->DebugString());
    }
  }
  EXPECT_EQ(expected, actual);

  // Verify that range-based for loop gets the same set of nodes.
  actual.clear();
  for (Node* node : graph_.nodes()) {
    actual.insert(node->DebugString());
  }
  EXPECT_EQ(expected, actual);
  VerifyGraphStats();
}

static void CheckType(Node* node, bool b) {
  EXPECT_TRUE(b) << node->DebugString();
  // Make sure none of the other IsFoo() methods return true.
  int count = 0;
  if (node->IsSource()) count++;
  if (node->IsSink()) count++;
  if (node->IsOp()) count++;
  EXPECT_EQ(1, count) << node->DebugString();
}

TEST_F(GraphTest, Type) {
  Node* op = AddNodeWithName("A");
  CheckType(graph_.source_node(), graph_.source_node()->IsSource());
  CheckType(graph_.sink_node(), graph_.sink_node()->IsSink());
  CheckType(op, op->IsOp());
  VerifyGraphStats();
}

TEST_F(GraphTest, AddAttr) {
  Node* n1 = AddNodeWithName("A");

  n1->AddAttr("_a", "new_attr");

  string attr;
  EXPECT_EQ(Status::OK(), GetNodeAttr(n1->attrs(), "_a", &attr));
  EXPECT_EQ("new_attr", attr);

  Node* n2 = graph_.CopyNode(n1);

  n1->AddAttr("_b", "new_attr_2");

  EXPECT_EQ(Status::OK(), GetNodeAttr(n1->attrs(), "_a", &attr));
  EXPECT_EQ("new_attr", attr);
  EXPECT_EQ(Status::OK(), GetNodeAttr(n1->attrs(), "_b", &attr));
  EXPECT_EQ("new_attr_2", attr);

  EXPECT_EQ(Status::OK(), GetNodeAttr(n2->attrs(), "_a", &attr));
  EXPECT_EQ("new_attr", attr);
  EXPECT_NE(Status::OK(), GetNodeAttr(n2->attrs(), "_b", &attr));
}

// Convert edge iteration results into a sorted string.
static string EdgeIter(const Graph& g) {
  std::vector<std::pair<int, int> > edges;
  for (const Edge* e : g.edges()) {
    edges.push_back(std::make_pair(e->src()->id(), e->dst()->id()));
  }
  std::sort(edges.begin(), edges.end());
  string result;
  for (auto& p : edges) {
    strings::StrAppend(&result, p.first, "->", p.second, ";");
  }
  return result;
}

TEST_F(GraphTest, EdgeIteration) {
  EXPECT_EQ("0->1;", EdgeIter(graph_));

  Node* a = FromNodeDef("A", "OneInputTwoOutputs", 1);
  Node* b = FromNodeDef("B", "OneInput", 1);
  EXPECT_EQ("0->1;", EdgeIter(graph_));  // Since a,b are currently disconnected

  graph_.AddEdge(a, 0, b, 0);
  EXPECT_EQ("0->1;2->3;", EdgeIter(graph_));

  graph_.AddControlEdge(graph_.source_node(), a);
  graph_.AddControlEdge(b, graph_.sink_node());
  EXPECT_EQ("0->1;0->2;2->3;3->1;", EdgeIter(graph_));

  graph_.AddEdge(a, 1, a, 0);
  EXPECT_EQ("0->1;0->2;2->2;2->3;3->1;", EdgeIter(graph_));
  VerifyGraphStats();
}

TEST_F(GraphTest, NewName) {
  string a1 = graph_.NewName("A");
  string a2 = graph_.NewName("A");
  string b1 = graph_.NewName("B");
  EXPECT_NE(a1, a2);
  EXPECT_NE(a1, b1);
  EXPECT_NE(a2, b1);
  EXPECT_TRUE(StringPiece(a1).starts_with("A")) << a1;
}

TEST_F(GraphTest, InputEdges) {
  Node* a = FromNodeDef("A", "OneOutput", 0);
  Node* b = FromNodeDef("B", "TwoInputsOneOutput", 2);
  graph_.AddEdge(a, 0, b, 0);
  std::vector<const Edge*> edges;
  EXPECT_EQ(error::INVALID_ARGUMENT, b->input_edges(&edges).code());
  graph_.AddEdge(a, 0, b, 1);
  TF_EXPECT_OK(b->input_edges(&edges));
}

TEST_F(GraphTest, AddFunctionLibrary) {
  // Basic functionality
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  *proto.add_function() = test::function::XTimesFour();
  TF_EXPECT_OK(graph_.AddFunctionLibrary(proto));
  EXPECT_TRUE(graph_.flib_def().Find("XTimesTwo") != nullptr);
  EXPECT_TRUE(graph_.flib_def().Find("XTimesFour") != nullptr);

  // Duplicate functions are ignored
  TF_EXPECT_OK(graph_.AddFunctionLibrary(proto));
  EXPECT_TRUE(graph_.flib_def().Find("XTimesTwo") != nullptr);
  EXPECT_TRUE(graph_.flib_def().Find("XTimesFour") != nullptr);

  // Duplicate names corresponding to different functions trigger an error
  FunctionDefLibrary error_proto = proto;
  *error_proto.mutable_function(0)->add_node_def() =
      error_proto.function(0).node_def(0);
  Status s = graph_.AddFunctionLibrary(error_proto);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Cannot add function 'XTimesTwo' because a different function with "
            "the same name already exists.");

  // Function with same name as an existing op triggers an error
  error_proto = proto;
  error_proto.mutable_function(0)->mutable_signature()->set_name("Add");
  s = graph_.AddFunctionLibrary(error_proto);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Cannot add function 'Add' because an op with the same name "
            "already exists.");

  // Adding a gradient function to an existing function is ok
  GradientDef* grad = proto.add_gradient();
  grad->set_function_name("XTimesTwo");
  grad->set_gradient_func("Undefined");  // undefined funcs in grads are ok
  TF_EXPECT_OK(graph_.AddFunctionLibrary(proto));
  EXPECT_EQ(graph_.flib_def().FindGradient("XTimesTwo"), "Undefined");

  // Duplicate gradients are ignored
  TF_EXPECT_OK(graph_.AddFunctionLibrary(proto));
  EXPECT_EQ(graph_.flib_def().FindGradient("XTimesTwo"), "Undefined");

  // Conflicting gradient triggers an error
  error_proto = proto;
  error_proto.mutable_gradient(0)->set_gradient_func("Undefined2");
  s = graph_.AddFunctionLibrary(error_proto);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Cannot assign gradient function 'Undefined2' to 'XTimesTwo' "
            "because it already has gradient function 'Undefined'");
}

REGISTER_OP("Input").Output("o: float");
REGISTER_OP("In2Out1").Input("a: float").Input("b: float").Output("o: float");

static void BM_InEdgeIteration(int iters, int num_nodes) {
  testing::StopTiming();
  string s;
  for (int in = 0; in < 10; in++) {
    s += strings::Printf("node { name: 'in%04d' op: 'Input' }", in);
  }
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int op = 0; op < num_nodes; op++) {
    s += strings::Printf(
        "node { name: 'op%04d' op: 'In2Out1' input: ['in%04d', 'in%04d' ] }",
        op, rnd.Uniform(10), rnd.Uniform(10));
  }

  Graph graph(OpRegistry::Global());
  GraphDef graph_def;
  CHECK(protobuf::TextFormat::ParseFromString(s, &graph_def));
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));

  int64 sum = 0;
  testing::StartTiming();
  for (int i = 0; i < iters; i += graph.num_node_ids()) {
    for (const Node* node : graph.nodes()) {
      for (auto e : node->in_edges()) {
        sum += e->id();
      }
    }
  }
  VLOG(1) << sum;
}
BENCHMARK(BM_InEdgeIteration)->Range(10, 100000);

}  // namespace
}  // namespace tensorflow
