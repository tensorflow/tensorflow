#include "tensorflow/core/graph/graph.h"

#include <set>
#include <gtest/gtest.h>
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class GraphTest : public ::testing::Test {
 protected:
  GraphTest() : graph_(OpRegistry::Global()) { RequireDefaultOps(); }
  ~GraphTest() override {}

  static void VerifyNodes(Node* node, std::vector<Node*> expected_in,
                          std::vector<Node*> expected_out) {
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

  Node* AddNodeWithName(const string& name) {
    Node* node;
    TF_CHECK_OK(NodeBuilder(name, "NoOp").Finalize(&graph_, &node));
    return node;
  }

  Graph graph_;

 private:
  // Convert a list of nodes to a sorted list of strings so failure messages
  // are readable.
  static std::vector<string> Stringify(const std::vector<Node*>& nodes) {
    std::vector<string> result;
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
}

TEST_F(GraphTest, InNodesAndOutNodes) {
  Node* a = AddNodeWithName("A");
  Node* b = AddNodeWithName("B");
  Node* c = AddNodeWithName("C");
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
}

TEST_F(GraphTest, NodeIteration) {
  // Set up the graph with some holes due to removals.
  Node* a = AddNodeWithName("A");
  Node* b = AddNodeWithName("B");
  Node* c = AddNodeWithName("C");
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

  Node* a = AddNodeWithName("A");
  Node* b = AddNodeWithName("B");
  EXPECT_EQ("0->1;", EdgeIter(graph_));  // Since a,b are currently disconnected

  graph_.AddEdge(a, 0, b, 0);
  EXPECT_EQ("0->1;2->3;", EdgeIter(graph_));

  graph_.AddControlEdge(graph_.source_node(), a);
  graph_.AddControlEdge(b, graph_.sink_node());
  EXPECT_EQ("0->1;0->2;2->3;3->1;", EdgeIter(graph_));

  graph_.AddEdge(a, 1, a, 0);
  EXPECT_EQ("0->1;0->2;2->2;2->3;3->1;", EdgeIter(graph_));
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
