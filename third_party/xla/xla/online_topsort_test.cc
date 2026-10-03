/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/online_topsort.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/strings/str_join.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

struct TestNode {
  explicit TestNode(int id) : id(id) {}

  int id;
  std::vector<TestNode*> in;
  std::vector<TestNode*> out;

  std::vector<TestNode*>::const_iterator incoming_begin() const {
    return in.begin();
  }
  std::vector<TestNode*>::const_iterator incoming_end() const {
    return in.end();
  }
  std::vector<TestNode*>::const_iterator outgoing_begin() const {
    return out.begin();
  }
  std::vector<TestNode*>::const_iterator outgoing_end() const {
    return out.end();
  }
};

using Topsort =
    TopologicalSort<TestNode, int, &TestNode::id,
                    std::vector<TestNode*>::const_iterator,
                    &TestNode::incoming_begin, &TestNode::incoming_end,
                    std::vector<TestNode*>::const_iterator,
                    &TestNode::outgoing_begin, &TestNode::outgoing_end>;

struct TestGraph {
  void AddNode(int id) {
    if (id >= node_index.size()) {
      node_index.resize(id + 1, nullptr);
    }
    auto node = std::make_unique<TestNode>(id);
    CHECK(node_index[id] == nullptr) << id;
    node_index[id] = node.get();
    topsort.AddNode(node.get());
    nodes.push_back(std::move(node));
  }

  void RemoveNode(int id) {
    TestNode* node = node_index[id];
    for (TestNode* x : node->in) {
      RemoveEdge(x->id, node->id);
    }
    for (TestNode* x : node->out) {
      RemoveEdge(id, x->id);
    }
    node_index[id] = nullptr;
    topsort.RemoveNode(node);
    auto it = std::find_if(nodes.begin(), nodes.end(),
                           [node](const auto& x) { return x.get() == node; });
    CHECK(it != nodes.end());
    nodes.erase(it);
  }

  void AddEdge(int from, int to) {
    CHECK_GE(from, 0);
    CHECK_LT(from, node_index.size());
    CHECK_GE(to, 0);
    CHECK_LT(to, node_index.size());
    TestNode* from_node = node_index[from];
    TestNode* to_node = node_index[to];
    topsort.AddEdge(from_node, to_node);
    from_node->out.push_back(to_node);
    to_node->in.push_back(from_node);
  }

  bool HasEdge(int from, int to) const {
    TestNode* from_node = node_index[from];
    TestNode* to_node = node_index[to];
    return std::find(from_node->out.begin(), from_node->out.end(), to_node) !=
           from_node->out.end();
  }

  void RemoveEdge(int from, int to) {
    TestNode* from_node = node_index[from];
    TestNode* to_node = node_index[to];
    auto it = std::find(from_node->out.begin(), from_node->out.end(), to_node);
    CHECK(it != from_node->out.end());
    from_node->out.erase(it);
    it = std::find(to_node->in.begin(), to_node->in.end(), from_node);
    CHECK(it != to_node->in.end());
    to_node->in.erase(it);
  }

  // Returns std::nullopt if the topological order is valid. Otherwise, returns
  // an edge that is inconsistent with the topological order.
  std::optional<std::pair<int, int>> TopologicalOrderIsValid() const {
    std::vector<int> order(node_index.size(), -1);
    int i = 0;
    std::vector<int> forward;
    for (int id : topsort) {
      forward.push_back(id);
      order[id] = i++;
    }

    // Verifies that the reverse iterator gives the same order.
    std::vector<int> reverse;
    for (auto it = topsort.rbegin(); it != topsort.rend(); ++it) {
      reverse.push_back(*it);
    }
    absl::c_reverse(reverse);
    CHECK(forward == reverse);

    for (const auto& x : nodes) {
      for (TestNode* y : x->out) {
        if (order[x->id] >= order[y->id]) {
          return std::make_pair(x->id, y->id);
        }
      }
    }
    return std::nullopt;
  }

  std::vector<std::unique_ptr<TestNode>> nodes;
  std::vector<TestNode*> node_index;
  Topsort topsort;
};

std::string OrderString(const Topsort& top) {
  std::vector<int> order;
  for (int id : top) {
    order.push_back(id);
  }
  return absl::StrJoin(order, ",");
}

MATCHER(HasValidTopologicalOrder, "") {
  std::optional<std::pair<int, int>> result = arg.TopologicalOrderIsValid();
  if (!result) {
    return true;
  }
  *result_listener << "Topological order: " << OrderString(arg.topsort)
                   << " is inconsistent with edge " << result->first << "->"
                   << result->second;
  return false;
}

TEST(TopologicalSortTest, Basic) {
  TestGraph g;
  for (int i = 0; i < 10; ++i) {
    g.AddNode(i);
    ASSERT_THAT(g, HasValidTopologicalOrder());
  }
  g.AddEdge(0, 1);
  ASSERT_THAT(g, HasValidTopologicalOrder());
  g.AddEdge(1, 2);
  ASSERT_THAT(g, HasValidTopologicalOrder());
  g.RemoveNode(0);
  ASSERT_THAT(g, HasValidTopologicalOrder());
  g.RemoveNode(1);
  ASSERT_THAT(g, HasValidTopologicalOrder());
}

TEST(TopologicalSortTest, Stick) {
  TestGraph g;
  int n = 20;
  for (int i = 0; i < n; ++i) {
    g.AddNode(i);
    ASSERT_THAT(g, HasValidTopologicalOrder());
  }
  for (int i = 0; i < n - 1; ++i) {
    g.AddEdge(i, i + 1);
    ASSERT_THAT(g, HasValidTopologicalOrder());
  }
  for (int i = 0; i < n; ++i) {
    g.RemoveNode(i);
    ASSERT_THAT(g, HasValidTopologicalOrder());
  }
}

TEST(TopologicalSortTest, ChangeOrder) {
  TestGraph g;
  int n = 20;
  for (int i = 0; i < n; ++i) {
    g.AddNode(i);
    ASSERT_THAT(g, HasValidTopologicalOrder());
  }
  for (int i = 0; i < n - 1; ++i) {
    g.AddEdge(i, i + 1);
    ASSERT_THAT(g, HasValidTopologicalOrder());
  }
  g.RemoveEdge(13, 14);
  ASSERT_THAT(g, HasValidTopologicalOrder());
  g.AddEdge(n - 1, 0);
  ASSERT_THAT(g, HasValidTopologicalOrder());
}

TEST(TopologicalSortTest, Diamonds) {
  TestGraph g;
  g.AddNode(0);
  for (int i = 0; i < 500; ++i) {
    int j = 3 * i;
    for (int k = 1; k <= 3; ++k) {
      g.AddNode(j + k);
      ASSERT_THAT(g, HasValidTopologicalOrder());
    }
    g.AddEdge(j, j + 1);
    ASSERT_THAT(g, HasValidTopologicalOrder());
    g.AddEdge(j, j + 2);
    ASSERT_THAT(g, HasValidTopologicalOrder());
    g.AddEdge(j + 1, j + 3);
    ASSERT_THAT(g, HasValidTopologicalOrder());
    g.AddEdge(j + 2, j + 3);
    ASSERT_THAT(g, HasValidTopologicalOrder());
  }
  ASSERT_THAT(g, HasValidTopologicalOrder());
}

TEST(TopologicalSortTest, Random) {
  absl::BitGen gen;
  for (int trial = 0; trial < 10; ++trial) {
    int n = absl::Uniform(gen, 10, 1000);
    int m = absl::Uniform(gen, 0, std::min(n * 5, (n * (n - 1)) / 2));
    LOG(INFO) << "trial: " << trial << " n: " << n << " m: " << m;
    std::vector<int> order(n);
    TestGraph g;
    for (int i = 0; i < n; ++i) {
      g.AddNode(i);
    }
    absl::c_iota(order, 0);
    absl::c_shuffle(order, gen);
    for (int i = 0; i < m; ++i) {
      int a, b;
      do {
        a = absl::Uniform(gen, 0, n);
        b = absl::Uniform(gen, 0, n);
        if (a > b) {
          std::swap(a, b);
        }
      } while (a == b || g.HasEdge(order[a], order[b]));
      g.AddEdge(order[a], order[b]);
      // Note: this check makes the test O(m^2), but it's valuable to verify
      // the invariant is maintained.
      ASSERT_THAT(g, HasValidTopologicalOrder());
    }
  }
}

// Node sized to 320 bytes to match the memory footprint of HloInstruction.
struct LargeNode {
  explicit LargeNode(int id = 0) : id(id) {}

  int id;
  std::vector<LargeNode*> in;
  std::vector<LargeNode*> out;
  char padding[256];

  std::vector<LargeNode*>::const_iterator incoming_begin() const {
    return in.begin();
  }
  std::vector<LargeNode*>::const_iterator incoming_end() const {
    return in.end();
  }
  std::vector<LargeNode*>::const_iterator outgoing_begin() const {
    return out.begin();
  }
  std::vector<LargeNode*>::const_iterator outgoing_end() const {
    return out.end();
  }
};

using LargeTopsort =
    TopologicalSort<LargeNode, int, &LargeNode::id,
                    std::vector<LargeNode*>::const_iterator,
                    &LargeNode::incoming_begin, &LargeNode::incoming_end,
                    std::vector<LargeNode*>::const_iterator,
                    &LargeNode::outgoing_begin, &LargeNode::outgoing_end>;

std::vector<std::pair<int, int>> MakeRandomDagEdges(int n) {
  std::seed_seq seq{42};
  absl::BitGen gen(seq);
  // Locally permute node ranks with 10% probability so that ~10% of edges go
  // against insertion order (u < v) and trigger incremental topological search,
  // while guaranteeing an acyclic DAG.
  std::vector<int> perm(n);
  absl::c_iota(perm, 0);
  for (int i = 0; i + 1 < n; ++i) {
    if (absl::Bernoulli(gen, 0.1)) {
      int offset = absl::Uniform<int>(gen, 1, std::min(n - i, 64));
      std::swap(perm[i], perm[i + offset]);
    }
  }
  std::vector<std::pair<int, int>> edges;
  edges.reserve(n * 2);
  for (int i = 1; i < n; ++i) {
    int window = std::min(i, 64);
    int j1 = i - absl::Uniform<int>(gen, 1, window + 1);
    edges.emplace_back(perm[i], perm[j1]);
    if (i > 1) {
      int j2 = i - absl::Uniform<int>(gen, 1, window + 1);
      if (j2 != j1) {
        edges.emplace_back(perm[i], perm[j2]);
      }
    }
  }
  return edges;
}

void BM_RandomDagConstruction(::testing::benchmark::State& state) {
  int n = state.range(0);
  auto edges = MakeRandomDagEdges(n);

  for (auto _ : state) {
    state.PauseTiming();
    std::vector<std::unique_ptr<LargeNode>> nodes(n);
    for (int i = 0; i < n; ++i) {
      nodes[i] = std::make_unique<LargeNode>(i);
    }
    state.ResumeTiming();

    LargeTopsort topsort;
    for (int i = 0; i < n; ++i) {
      topsort.AddNode(nodes[i].get());
    }
    for (auto [u, v] : edges) {
      nodes[u]->out.push_back(nodes[v].get());
      nodes[v]->in.push_back(nodes[u].get());
      topsort.AddEdge(nodes[u].get(), nodes[v].get());
    }
  }
}
BENCHMARK(BM_RandomDagConstruction)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000);

void BM_PostOrderTraversal(::testing::benchmark::State& state) {
  int n = state.range(0);
  auto edges = MakeRandomDagEdges(n);

  std::vector<std::unique_ptr<LargeNode>> nodes(n);
  LargeTopsort topsort;
  for (int i = 0; i < n; ++i) {
    nodes[i] = std::make_unique<LargeNode>(i);
    topsort.AddNode(nodes[i].get());
  }
  for (auto [u, v] : edges) {
    nodes[u]->out.push_back(nodes[v].get());
    nodes[v]->in.push_back(nodes[u].get());
    topsort.AddEdge(nodes[u].get(), nodes[v].get());
  }

  for (auto _ : state) {
    int64_t sum = 0;
    for (auto it = topsort.rbegin(); it != topsort.rend(); ++it) {
      sum += nodes[*it]->id;
    }
    ::benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PostOrderTraversal)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000);

}  // namespace
}  // namespace xla
