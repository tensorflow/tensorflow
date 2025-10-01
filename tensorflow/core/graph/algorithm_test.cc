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

#include "tensorflow/core/graph/algorithm.h"

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/graph/benchmark_testlib.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

// TODO(josh11b): Test setting the "device" field of a NodeDef.
// TODO(josh11b): Test that feeding won't prune targets.

namespace tensorflow {
namespace {

REGISTER_OP("TestParams").Output("o: float");
REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestMul").Input("a: float").Input("b: float").Output("o: float");
REGISTER_OP("TestUnary").Input("a: float").Output("o: float");
REGISTER_OP("TestBinary")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float");

// Compares that the order of nodes in 'inputs' respects the
// pair orders described in 'ordered_pairs'.
bool ExpectBefore(const std::vector<std::pair<string, string>>& ordered_pairs,
                  const std::vector<Node*>& inputs, string* error) {
  for (const std::pair<string, string>& pair : ordered_pairs) {
    const string& before_node = pair.first;
    const string& after_node = pair.second;
    bool seen_before = false;
    bool seen_both = false;
    for (const Node* node : inputs) {
      if (!seen_before && after_node == node->name()) {
        *error = absl::StrCat("Saw ", after_node, " before ", before_node);
        return false;
      }

      if (before_node == node->name()) {
        seen_before = true;
      } else if (after_node == node->name()) {
        seen_both = seen_before;
        break;
      }
    }
    if (!seen_both) {
      *error =
          absl::StrCat("didn't see either ", before_node, " or ", after_node);
      return false;
    }
  }

  return true;
}

TEST(AlgorithmTest, ReversePostOrder) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  Node* w1 = SourceOp("TestParams", b.opts().WithName("W1"));
  Node* w2 = SourceOp("TestParams", b.opts().WithName("W2"));
  Node* input =
      SourceOp("TestInput", b.opts().WithName("input").WithControlInput(w1));
  Node* t1 = BinaryOp("TestMul", w1, {input, 1}, b.opts().WithName("t1"));
  BinaryOp("TestMul", w1, {input, 1},
           b.opts().WithName("t2").WithControlInput(t1));
  BinaryOp("TestMul", w2, {input, 1}, b.opts().WithName("t3"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));
  std::vector<Node*> order;

  // Test reverse post order:
  GetReversePostOrder(g, &order);

  // Check that the order respects the dependencies correctly.
  std::vector<std::pair<string, string>> reverse_orders = {
      {"W1", "input"}, {"W1", "t1"},    {"W1", "t2"}, {"W1", "t3"},
      {"input", "t1"}, {"input", "t3"}, {"t1", "t2"}, {"W2", "t3"}};
  string error;
  EXPECT_TRUE(ExpectBefore(reverse_orders, order, &error)) << error;

  // A false ordering should fail the check.
  reverse_orders = {{"input", "W1"}};
  EXPECT_FALSE(ExpectBefore(reverse_orders, order, &error));

  // Test post order:
  GetPostOrder(g, &order);

  // Check that the order respects the dependencies correctly.
  std::vector<std::pair<string, string>> orders = {
      {"input", "W1"}, {"t1", "W1"},    {"t2", "W1"}, {"t3", "W1"},
      {"t1", "input"}, {"t3", "input"}, {"t2", "t1"}, {"t3", "W2"}};
  EXPECT_TRUE(ExpectBefore(orders, order, &error)) << error;

  // A false ordering should fail the check.
  orders = {{"W1", "t3"}};
  EXPECT_FALSE(ExpectBefore(orders, order, &error));
}

TEST(AlgorithmTest, ReversePostOrderStable) {
  int64_t run_count = 100;
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  for (int64_t i = 0; i < run_count; ++i) {
    // One source of nondeterminism comes from unordered set with key of a
    // pointer type, for example the order of FlatSet<Node*> depends on the
    // raw pointer value of Node. Stable post order suppose to remove this
    // nondeterminism by enforcing an ordering based on node ids.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    string error;
    Node* w1 = SourceOp("TestParams", b.opts().WithName("W1"));
    Node* input =
        SourceOp("TestInput", b.opts().WithName("input").WithControlInput(w1));
    BinaryOp("TestMul", w1, {input, 1}, b.opts().WithName("t2"));
    // Insert different number of nodes between the allocation of t2 and t3,
    // this creates enough entropy in the memory distance between t2 and t3 thus
    // forces them to have randomized ordering had stable DFS was not
    // implemented correctly.
    for (int64_t j = 0; j < i; ++j) {
      BinaryOp("TestMul", w1, {input, 1},
               b.opts().WithName(absl::StrCat("internal", j)));
    }

    BinaryOp("TestMul", w1, {input, 1}, b.opts().WithName("t3"));

    Graph g(OpRegistry::Global());
    TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));
    std::vector<Node*> order;

    // Test reverse post order generates expected ordering.
    GetReversePostOrder(g, &order, /*stable_comparator=*/NodeComparatorName());
    EXPECT_TRUE(ExpectBefore({{"t2", "t3"}}, order, &error));
  }
}

TEST(AlgorithmTest, PostOrderWithEdgeFilter) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Node* n0 = ops::SourceOp("TestParams", b.opts().WithName("n0"));
  Node* n1 = ops::UnaryOp("TestUnary", n0, b.opts().WithName("n1"));
  Node* n2 = ops::UnaryOp("TestUnary", n1, b.opts().WithName("n2"));
  Node* n3 = ops::BinaryOp("TestBinary", n2, n0, b.opts().WithName("n3"));

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(GraphDefBuilderToGraph(b, &g));

  g.AddEdge(g.FindNodeId(n3->id()), 0, g.FindNodeId(n1->id()), 1);

  std::vector<Node*> post_order;
  auto edge_filter = [&](const Edge& e) {
    return !(e.src()->id() == n3->id() && e.dst()->id() == n1->id());
  };

  std::vector<Node*> expected_post_order = {
      g.sink_node(),          g.FindNodeId(n3->id()), g.FindNodeId(n2->id()),
      g.FindNodeId(n1->id()), g.FindNodeId(n0->id()), g.source_node()};

  std::vector<Node*> expected_reverse_post_order = expected_post_order;
  std::reverse(expected_reverse_post_order.begin(),
               expected_reverse_post_order.end());

  GetPostOrder(g, &post_order, /*stable_comparator=*/{},
               /*edge_filter=*/edge_filter);

  ASSERT_EQ(expected_post_order.size(), post_order.size());
  for (int i = 0; i < post_order.size(); i++) {
    CHECK_EQ(post_order[i], expected_post_order[i])
        << post_order[i]->name() << " vs. " << expected_post_order[i]->name();
  }

  std::vector<Node*> reverse_post_order;
  GetReversePostOrder(g, &reverse_post_order, /*stable_comparator=*/{},
                      /*edge_filter=*/edge_filter);

  ASSERT_EQ(expected_reverse_post_order.size(), reverse_post_order.size());
  for (int i = 0; i < reverse_post_order.size(); i++) {
    CHECK_EQ(reverse_post_order[i], expected_reverse_post_order[i])
        << reverse_post_order[i]->name() << " vs. "
        << expected_reverse_post_order[i]->name();
  }
}

void BM_PruneForReverseReachability(::testing::benchmark::State& state) {
  const int num_nodes = state.range(0);
  const int num_edges_per_node = state.range(1);
  const GraphDef graph_def =
      test::CreateGraphDef(num_nodes, num_edges_per_node);
  const auto registry = OpRegistry::Global();
  GraphConstructorOptions opts;
  for (auto s : state) {
    state.PauseTiming();
    Graph graph(registry);
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
    std::unordered_set<const Node*> visited;
    visited.insert(graph.FindNodeId(graph.num_nodes() - 1));
    state.ResumeTiming();
    PruneForReverseReachability(&graph, std::move(visited));
  }
}
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(10, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 6, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 9, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 12, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 15, 2);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(10, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 6, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 9, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 12, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 15, 4);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(10, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 6, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 9, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 12, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 15, 8);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(10, 16);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 6, 16);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 9, 16);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 12, 16);
BENCHMARK(BM_PruneForReverseReachability)->ArgPair(1 << 15, 16);

}  // namespace
}  // namespace tensorflow
