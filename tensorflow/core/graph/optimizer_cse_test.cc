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

#include "tensorflow/core/graph/optimizer_cse.h"

#include <utility>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

static void InitGraph(const string& s, Graph* graph) {
  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  //  parser.AllowRelaxedWhitespace(true);
  CHECK(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
}

class OptimizerCSETest : public ::testing::Test {
 public:
  OptimizerCSETest() : graph_(OpRegistry::Global()) {}

  void InitGraph(const string& s) {
    ::tensorflow::InitGraph(s, &graph_);
    original_ = CanonicalGraphString(&graph_);
  }

  static bool IncludeNode(const Node* n) { return n->IsOp(); }

  static string EdgeId(const Node* n, int index) {
    if (index == 0) {
      return n->name();
    } else if (index == Graph::kControlSlot) {
      return strings::StrCat(n->name(), ":control");
    } else {
      return strings::StrCat(n->name(), ":", index);
    }
  }

  string CanonicalGraphString(Graph* g) {
    std::vector<string> nodes;
    std::vector<string> edges;
    for (const Node* n : g->nodes()) {
      if (IncludeNode(n)) {
        nodes.push_back(strings::StrCat(n->name(), "(", n->type_string(), ")"));
      }
    }
    for (const Edge* e : g->edges()) {
      if (IncludeNode(e->src()) && IncludeNode(e->dst())) {
        edges.push_back(strings::StrCat(EdgeId(e->src(), e->src_output()), "->",
                                        EdgeId(e->dst(), e->dst_input())));
      }
    }
    // Canonicalize
    std::sort(nodes.begin(), nodes.end());
    std::sort(edges.begin(), edges.end());
    return strings::StrCat(str_util::Join(nodes, ";"), "|",
                           str_util::Join(edges, ";"));
  }

  string DoCSE(const std::function<bool(const Node*)>& consider_fn = nullptr) {
    string before = CanonicalGraphString(&graph_);
    LOG(ERROR) << "Before rewrites: " << before;

    OptimizeCSE(&graph_, consider_fn);

    string result = CanonicalGraphString(&graph_);
    LOG(ERROR) << "After rewrites:  " << result;
    return result;
  }

  const string& OriginalGraph() const { return original_; }

  Graph graph_;
  string original_;
};

REGISTER_OP("Input").Output("o: float").SetIsStateful();

// Note that the "rules" in these tests are not meant to be logically correct
TEST_F(OptimizerCSETest, Simple) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);D(Mul)|"
            "A->D;B->D:1");
}

TEST_F(OptimizerCSETest, Simple_ThreeEquivalent) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);E(Mul)|"
            "A->E;B->E:1");
}

TEST_F(OptimizerCSETest, Simple_WithFixups) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'D'] }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);D(Mul);E(Mul)|"
            "A->D;B->D:1;D->E;D->E:1");
}

TEST_F(OptimizerCSETest, Simple_Commutative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'A'] }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);D(Mul)|"
            "A->D:1;B->D");
}

static bool IsNotMultiply(const Node* n) { return n->type_string() != "Mul"; }

// Like Simple_Commutative,
TEST_F(OptimizerCSETest, Simple_Filtered) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'A'] }");
  EXPECT_EQ(DoCSE(IsNotMultiply), OriginalGraph());
}

TEST_F(OptimizerCSETest, Simple_NotCommutative) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Sub' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Sub' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'A'] }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, NotEquivalent_Ops) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Sub' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, Simple_SameOps_SameAttrs1) {
  // Should still do CSE for ops with attrs if they match.
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] attr { key: 'shape'"
      "    value { shape: { dim: { size: 37 name: 'SAME_NAME' } } } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] attr { key: 'shape'"
      "    value { shape: { dim: { size: 37 name: 'SAME_NAME' } } } } }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);D(Mul)|"
            "A->D;B->D:1");
}

TEST_F(OptimizerCSETest, Simple_SameOps_SameAttrs2) {
  // Should still do CSE for ops with attrs if they match, even if they
  // are not in the same order.
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 'a' value { i: 3 } }"
      "    attr { key: 't' value { type: DT_INT32 } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 't' value { type: DT_INT32 } }"
      "    attr { key: 'a' value { i: 3 } } }");
  EXPECT_EQ(DoCSE(),
            "A(Input);B(Input);D(Mul)|"
            "A->D;B->D:1");
}

TEST_F(OptimizerCSETest, SameConstants) {
  // Should still do CSE for ops with constants if the values are identical
  InitGraph(
      "node { name: 'A' op: 'Const' "
      "  attr { key: 'dtype' value { type: DT_INT32 } }"
      "  attr { key: 'value' value {"
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'B' op: 'Const' "
      "  attr { key: 'dtype' value { type: DT_INT32 } }"
      "  attr { key: 'value' value {"
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_INT32 } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(),
            "B(Const);D(Mul)|"
            "B->D;B->D:1");
}

TEST_F(OptimizerCSETest, DifferentConstants) {
  // Should still do CSE for ops with extensions if the extensions are identical
  InitGraph(
      "node { name: 'A' op: 'Const' "
      "  attr { key: 'dtype' value { type: DT_INT32 } }"
      "  attr { key: 'value' value {"
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 0 } } } }"
      "node { name: 'B' op: 'Const' "
      "  attr { key: 'dtype' value { type: DT_INT32 } }"
      "  attr { key: 'value' value {"
      "    tensor { dtype: DT_INT32 tensor_shape { dim { size: 1 } } "
      "    int_val: 100000 } } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_INT32 } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoCSE(),
            "A(Const);B(Const);D(Mul)|"
            "A->D;B->D:1");
}

TEST_F(OptimizerCSETest, SameOps_DifferentAttrs1) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 'a' value { i: 3 } }"
      "    attr { key: 't' value { type: DT_INT32 } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 't' value { type: DT_INT32 } }"
      "    attr { key: 'a' value { i: 4 } } }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, SameOps_DifferentAttrs2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 'a' value { i: 3 } }"
      "    attr { key: 't' value { type: DT_FLOAT } } }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B']"
      "    attr { key: 't' value { type: DT_INT32 } }"
      "    attr { key: 'a' value { i: 3 } } }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, NotEquivalent_Inputs) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'E' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }");
  EXPECT_EQ(DoCSE(), OriginalGraph());
}

TEST_F(OptimizerCSETest, Constant_Dedup) {
  Tensor a(DT_FLOAT, TensorShape({1}));
  a.flat<float>()(0) = 1.0;
  Tensor b(DT_DOUBLE, TensorShape({1}));  // Different type
  b.flat<double>()(0) = 1.0;
  Tensor c(DT_FLOAT, TensorShape({1, 1}));  // Different shape
  c.flat<float>()(0) = 1.0;
  Tensor d(DT_FLOAT, TensorShape({1}));  // Different value
  d.flat<float>()(0) = 2.0;

  // A graph contains a bunch of constants.
  Graph g(OpRegistry::Global());
  for (const auto& val : {a, b, c, d, d, c, b, a}) {
    test::graph::Constant(&g, val);  // Node name is n/_0, n/_1, ...
  }
  GraphDef gdef;
  test::graph::ToGraphDef(&g, &gdef);
  InitGraph(gdef.DebugString());

  EXPECT_EQ(OriginalGraph(),
            "n/_0(Const);n/_1(Const);n/_2(Const);n/_3(Const);"
            "n/_4(Const);n/_5(Const);n/_6(Const);n/_7(Const)|");
  // In theory, there are 2^4 possible correct output of CSE.  In this
  // test, it happens to eliminate the first 4 nodes.
  EXPECT_EQ(DoCSE(), "n/_4(Const);n/_5(Const);n/_6(Const);n/_7(Const)|");
}

static void BM_CSE(int iters, int op_nodes) {
  testing::StopTiming();
  string s;
  for (int in = 0; in < 10; in++) {
    s += strings::Printf("node { name: 'in%04d' op: 'Input'}", in);
  }
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int op = 0; op < op_nodes; op++) {
    s += strings::Printf(
        "node { name: 'op%04d' op: 'Mul' attr { key: 'T' value { "
        "type: DT_FLOAT } } input: ['in%04d', 'in%04d' ] }",
        op, rnd.Uniform(10), rnd.Uniform(10));
  }

  bool first = true;
  while (iters > 0) {
    Graph* graph = new Graph(OpRegistry::Global());
    InitGraph(s, graph);
    int N = graph->num_node_ids();
    if (first) {
      testing::SetLabel(strings::StrCat("Per graph node.  Nodes: ", N));
      first = false;
    }
    {
      testing::StartTiming();
      OptimizeCSE(graph, nullptr);
      testing::StopTiming();
    }
    iters -= N;  // Our benchmark units are individual graph nodes,
                 // not whole graphs
    delete graph;
  }
}
BENCHMARK(BM_CSE)->Arg(1000)->Arg(10000);

}  // namespace
}  // namespace tensorflow
