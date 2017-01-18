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

// A test for the GraphCycles interface.

#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"

#include <random>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::int32;
using tensorflow::string;

// We emulate a GraphCycles object with a node vector and an edge vector.
// We then compare the two implementations.

typedef std::vector<int> Nodes;
struct Edge {
  int from;
  int to;
};
typedef std::vector<Edge> Edges;

// Return whether "to" is reachable from "from".
static bool IsReachable(Edges *edges, int from, int to,
                        std::unordered_set<int> *seen) {
  seen->insert(from);  // we are investigating "from"; don't do it again
  if (from == to) return true;
  for (int i = 0; i != edges->size(); i++) {
    Edge *edge = &(*edges)[i];
    if (edge->from == from) {
      if (edge->to == to) {  // success via edge directly
        return true;
      } else if (seen->find(edge->to) == seen->end() &&  // success via edge
                 IsReachable(edges, edge->to, to, seen)) {
        return true;
      }
    }
  }
  return false;
}

static void PrintNodes(Nodes *nodes) {
  LOG(INFO) << "NODES (" << nodes->size() << ")";
  for (int i = 0; i != nodes->size(); i++) {
    LOG(INFO) << (*nodes)[i];
  }
}

static void PrintEdges(Edges *edges) {
  LOG(INFO) << "EDGES (" << edges->size() << ")";
  for (int i = 0; i != edges->size(); i++) {
    int a = (*edges)[i].from;
    int b = (*edges)[i].to;
    LOG(INFO) << a << " " << b;
  }
  LOG(INFO) << "---";
}

static void PrintGCEdges(Nodes *nodes, tensorflow::GraphCycles *gc) {
  LOG(INFO) << "GC EDGES";
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      if (gc->HasEdge(a, b)) {
        LOG(INFO) << a << " " << b;
      }
    }
  }
  LOG(INFO) << "---";
}

static void PrintTransitiveClosure(Nodes *nodes, Edges *edges,
                                   tensorflow::GraphCycles *gc) {
  LOG(INFO) << "Transitive closure";
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      std::unordered_set<int> seen;
      if (IsReachable(edges, a, b, &seen)) {
        LOG(INFO) << a << " " << b;
      }
    }
  }
  LOG(INFO) << "---";
}

static void PrintGCTransitiveClosure(Nodes *nodes,
                                     tensorflow::GraphCycles *gc) {
  LOG(INFO) << "GC Transitive closure";
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      if (gc->IsReachable(a, b)) {
        LOG(INFO) << a << " " << b;
      }
    }
  }
  LOG(INFO) << "---";
}

static void CheckTransitiveClosure(Nodes *nodes, Edges *edges,
                                   tensorflow::GraphCycles *gc) {
  std::unordered_set<int> seen;
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      seen.clear();
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      bool gc_reachable = gc->IsReachable(a, b);
      CHECK_EQ(gc_reachable, gc->IsReachableNonConst(a, b));
      bool reachable = IsReachable(edges, a, b, &seen);
      if (gc_reachable != reachable) {
        PrintEdges(edges);
        PrintGCEdges(nodes, gc);
        PrintTransitiveClosure(nodes, edges, gc);
        PrintGCTransitiveClosure(nodes, gc);
        LOG(FATAL) << "gc_reachable " << gc_reachable << " reachable "
                   << reachable << " a " << a << " b " << b;
      }
    }
  }
}

static void CheckEdges(Nodes *nodes, Edges *edges,
                       tensorflow::GraphCycles *gc) {
  int count = 0;
  for (int i = 0; i != edges->size(); i++) {
    int a = (*edges)[i].from;
    int b = (*edges)[i].to;
    if (!gc->HasEdge(a, b)) {
      PrintEdges(edges);
      PrintGCEdges(nodes, gc);
      LOG(FATAL) << "!gc->HasEdge(" << a << ", " << b << ")";
    }
  }
  for (int i = 0; i != nodes->size(); i++) {
    for (int j = 0; j != nodes->size(); j++) {
      int a = (*nodes)[i];
      int b = (*nodes)[j];
      if (gc->HasEdge(a, b)) {
        count++;
      }
    }
  }
  if (count != edges->size()) {
    PrintEdges(edges);
    PrintGCEdges(nodes, gc);
    LOG(FATAL) << "edges->size() " << edges->size() << "  count " << count;
  }
}

// Returns the index of a randomly chosen node in *nodes.
// Requires *nodes be non-empty.
static int RandomNode(std::mt19937 *rnd, Nodes *nodes) {
  std::uniform_int_distribution<int> distribution(0, nodes->size() - 1);
  return distribution(*rnd);
}

// Returns the index of a randomly chosen edge in *edges.
// Requires *edges be non-empty.
static int RandomEdge(std::mt19937 *rnd, Edges *edges) {
  std::uniform_int_distribution<int> distribution(0, edges->size() - 1);
  return distribution(*rnd);
}

// Returns the index of edge (from, to) in *edges or -1 if it is not in *edges.
static int EdgeIndex(Edges *edges, int from, int to) {
  int i = 0;
  while (i != edges->size() &&
         ((*edges)[i].from != from || (*edges)[i].to != to)) {
    i++;
  }
  return i == edges->size() ? -1 : i;
}

TEST(GraphCycles, RandomizedTest) {
  Nodes nodes;
  Edges edges;  // from, to
  tensorflow::GraphCycles graph_cycles;
  static const int kMaxNodes = 7;     // use <= 7 nodes to keep test short
  static const int kDataOffset = 17;  // an offset to the node-specific data
  int n = 100000;
  int op = 0;
  std::mt19937 rnd(tensorflow::testing::RandomSeed() + 1);

  for (int iter = 0; iter != n; iter++) {
    if ((iter % 10000) == 0) VLOG(0) << "Iter " << iter << " of " << n;

    if (VLOG_IS_ON(3)) {
      LOG(INFO) << "===============";
      LOG(INFO) << "last op " << op;
      PrintNodes(&nodes);
      PrintEdges(&edges);
      PrintGCEdges(&nodes, &graph_cycles);
    }
    for (int i = 0; i != nodes.size(); i++) {
      ASSERT_EQ(reinterpret_cast<intptr_t>(graph_cycles.GetNodeData(i)),
                i + kDataOffset)
          << " node " << i;
    }
    CheckEdges(&nodes, &edges, &graph_cycles);
    CheckTransitiveClosure(&nodes, &edges, &graph_cycles);
    std::uniform_int_distribution<int> distribution(0, 5);
    op = distribution(rnd);
    switch (op) {
      case 0:  // Add a node
        if (nodes.size() < kMaxNodes) {
          int new_node = graph_cycles.NewNode();
          ASSERT_NE(-1, new_node);
          VLOG(1) << "adding node " << new_node;
          ASSERT_EQ(0, graph_cycles.GetNodeData(new_node));
          graph_cycles.SetNodeData(
              new_node, reinterpret_cast<void *>(
                            static_cast<intptr_t>(new_node + kDataOffset)));
          ASSERT_GE(new_node, 0);
          for (int i = 0; i != nodes.size(); i++) {
            ASSERT_NE(nodes[i], new_node);
          }
          nodes.push_back(new_node);
        }
        break;

      case 1:  // Remove a node
        if (nodes.size() > 0) {
          int node_index = RandomNode(&rnd, &nodes);
          int node = nodes[node_index];
          nodes[node_index] = nodes.back();
          nodes.pop_back();
          VLOG(1) << "removing node " << node;
          graph_cycles.RemoveNode(node);
          int i = 0;
          while (i != edges.size()) {
            if (edges[i].from == node || edges[i].to == node) {
              edges[i] = edges.back();
              edges.pop_back();
            } else {
              i++;
            }
          }
        }
        break;

      case 2:  // Add an edge
        if (nodes.size() > 0) {
          int from = RandomNode(&rnd, &nodes);
          int to = RandomNode(&rnd, &nodes);
          if (EdgeIndex(&edges, nodes[from], nodes[to]) == -1) {
            if (graph_cycles.InsertEdge(nodes[from], nodes[to])) {
              Edge new_edge;
              new_edge.from = nodes[from];
              new_edge.to = nodes[to];
              edges.push_back(new_edge);
            } else {
              std::unordered_set<int> seen;
              ASSERT_TRUE(IsReachable(&edges, nodes[to], nodes[from], &seen))
                  << "Edge " << nodes[to] << "->" << nodes[from];
            }
          }
        }
        break;

      case 3:  // Remove an edge
        if (edges.size() > 0) {
          int i = RandomEdge(&rnd, &edges);
          int from = edges[i].from;
          int to = edges[i].to;
          ASSERT_EQ(i, EdgeIndex(&edges, from, to));
          edges[i] = edges.back();
          edges.pop_back();
          ASSERT_EQ(-1, EdgeIndex(&edges, from, to));
          VLOG(1) << "removing edge " << from << " " << to;
          graph_cycles.RemoveEdge(from, to);
        }
        break;

      case 4:  // Check a path
        if (nodes.size() > 0) {
          int from = RandomNode(&rnd, &nodes);
          int to = RandomNode(&rnd, &nodes);
          int32 path[2 * kMaxNodes];
          int path_len = graph_cycles.FindPath(nodes[from], nodes[to],
                                               2 * kMaxNodes, path);
          std::unordered_set<int> seen;
          bool reachable = IsReachable(&edges, nodes[from], nodes[to], &seen);
          bool gc_reachable = graph_cycles.IsReachable(nodes[from], nodes[to]);
          ASSERT_EQ(gc_reachable,
                    graph_cycles.IsReachableNonConst(nodes[from], nodes[to]));
          ASSERT_EQ(path_len != 0, reachable);
          ASSERT_EQ(path_len != 0, gc_reachable);
          // In the following line, we add one because a node can appear
          // twice, if the path is from that node to itself, perhaps via
          // every other node.
          ASSERT_LE(path_len, kMaxNodes + 1);
          if (path_len != 0) {
            ASSERT_EQ(nodes[from], path[0]);
            ASSERT_EQ(nodes[to], path[path_len - 1]);
            for (int i = 1; i < path_len; i++) {
              ASSERT_NE(-1, EdgeIndex(&edges, path[i - 1], path[i]));
              ASSERT_TRUE(graph_cycles.HasEdge(path[i - 1], path[i]));
            }
          }
        }
        break;

      case 5:  // Check invariants
        CHECK(graph_cycles.CheckInvariants());
        break;

      default:
        LOG(FATAL);
    }

    // Very rarely, test graph expansion by adding then removing many nodes.
    std::bernoulli_distribution rarely(1.0 / 1024.0);
    if (rarely(rnd)) {
      VLOG(3) << "Graph expansion";
      CheckEdges(&nodes, &edges, &graph_cycles);
      CheckTransitiveClosure(&nodes, &edges, &graph_cycles);
      for (int i = 0; i != 256; i++) {
        int new_node = graph_cycles.NewNode();
        ASSERT_NE(-1, new_node);
        VLOG(1) << "adding node " << new_node;
        ASSERT_GE(new_node, 0);
        ASSERT_EQ(0, graph_cycles.GetNodeData(new_node));
        graph_cycles.SetNodeData(
            new_node, reinterpret_cast<void *>(
                          static_cast<intptr_t>(new_node + kDataOffset)));
        for (int j = 0; j != nodes.size(); j++) {
          ASSERT_NE(nodes[j], new_node);
        }
        nodes.push_back(new_node);
      }
      for (int i = 0; i != 256; i++) {
        ASSERT_GT(nodes.size(), 0);
        int node_index = RandomNode(&rnd, &nodes);
        int node = nodes[node_index];
        nodes[node_index] = nodes.back();
        nodes.pop_back();
        VLOG(1) << "removing node " << node;
        graph_cycles.RemoveNode(node);
        int j = 0;
        while (j != edges.size()) {
          if (edges[j].from == node || edges[j].to == node) {
            edges[j] = edges.back();
            edges.pop_back();
          } else {
            j++;
          }
        }
      }
      CHECK(graph_cycles.CheckInvariants());
    }
  }
}

class GraphCyclesTest : public ::testing::Test {
 public:
  tensorflow::GraphCycles g_;

  // Test relies on ith NewNode() call returning Node numbered i
  GraphCyclesTest() {
    for (int i = 0; i < 100; i++) {
      CHECK_EQ(i, g_.NewNode());
    }
    CHECK(g_.CheckInvariants());
  }

  bool AddEdge(int x, int y) { return g_.InsertEdge(x, y); }

  void AddMultiples() {
    // For every node x > 0: add edge to 2*x, 3*x
    for (int x = 1; x < 25; x++) {
      EXPECT_TRUE(AddEdge(x, 2 * x)) << x;
      EXPECT_TRUE(AddEdge(x, 3 * x)) << x;
    }
    CHECK(g_.CheckInvariants());
  }

  string Path(int x, int y) {
    static const int kPathSize = 5;
    int32 path[kPathSize];
    int np = g_.FindPath(x, y, kPathSize, path);
    string result;
    for (int i = 0; i < np; i++) {
      if (i >= kPathSize) {
        result += " ...";
        break;
      }
      if (!result.empty()) result.push_back(' ');
      char buf[20];
      snprintf(buf, sizeof(buf), "%d", path[i]);
      result += buf;
    }
    return result;
  }
};

TEST_F(GraphCyclesTest, NoCycle) {
  AddMultiples();
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, SimpleCycle) {
  AddMultiples();
  EXPECT_FALSE(AddEdge(8, 4));
  EXPECT_EQ("4 8", Path(4, 8));
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, IndirectCycle) {
  AddMultiples();
  EXPECT_TRUE(AddEdge(16, 9));
  CHECK(g_.CheckInvariants());
  EXPECT_FALSE(AddEdge(9, 2));
  EXPECT_EQ("2 4 8 16 9", Path(2, 9));
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, LongPath) {
  ASSERT_TRUE(AddEdge(2, 4));
  ASSERT_TRUE(AddEdge(4, 6));
  ASSERT_TRUE(AddEdge(6, 8));
  ASSERT_TRUE(AddEdge(8, 10));
  ASSERT_TRUE(AddEdge(10, 12));
  ASSERT_FALSE(AddEdge(12, 2));
  EXPECT_EQ("2 4 6 8 10 ...", Path(2, 12));
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, RemoveNode) {
  ASSERT_TRUE(AddEdge(1, 2));
  ASSERT_TRUE(AddEdge(2, 3));
  ASSERT_TRUE(AddEdge(3, 4));
  ASSERT_TRUE(AddEdge(4, 5));
  g_.RemoveNode(3);
  ASSERT_TRUE(AddEdge(5, 1));
}

TEST_F(GraphCyclesTest, ManyEdges) {
  const int N = 50;
  for (int i = 0; i < N; i++) {
    for (int j = 1; j < N; j++) {
      ASSERT_TRUE(AddEdge(i, i + j));
    }
  }
  CHECK(g_.CheckInvariants());
  ASSERT_TRUE(AddEdge(2 * N - 1, 0));
  CHECK(g_.CheckInvariants());
  ASSERT_FALSE(AddEdge(10, 9));
  CHECK(g_.CheckInvariants());
}

TEST_F(GraphCyclesTest, ContractEdge) {
  ASSERT_TRUE(AddEdge(1, 2));
  ASSERT_TRUE(AddEdge(1, 3));
  ASSERT_TRUE(AddEdge(2, 3));
  ASSERT_TRUE(AddEdge(2, 4));
  ASSERT_TRUE(AddEdge(3, 4));

  EXPECT_FALSE(g_.ContractEdge(1, 3));
  CHECK(g_.CheckInvariants());
  EXPECT_TRUE(g_.HasEdge(1, 3));

  EXPECT_TRUE(g_.ContractEdge(1, 2));
  CHECK(g_.CheckInvariants());
  EXPECT_TRUE(g_.HasEdge(1, 3));
  EXPECT_TRUE(g_.HasEdge(1, 4));
  EXPECT_TRUE(g_.HasEdge(3, 4));

  EXPECT_TRUE(g_.ContractEdge(1, 3));
  CHECK(g_.CheckInvariants());
  EXPECT_TRUE(g_.HasEdge(1, 4));
}

static void BM_StressTest(int iters, int num_nodes) {
  while (iters > 0) {
    tensorflow::GraphCycles g;
    int32 *nodes = new int32[num_nodes];
    for (int i = 0; i < num_nodes; i++) {
      nodes[i] = g.NewNode();
    }
    for (int i = 0; i < num_nodes && iters > 0; i++, iters--) {
      int end = std::min(num_nodes, i + 5);
      for (int j = i + 1; j < end; j++) {
        if (nodes[i] >= 0 && nodes[j] >= 0) {
          CHECK(g.InsertEdge(nodes[i], nodes[j]));
        }
      }
    }
    delete[] nodes;
  }
}
BENCHMARK(BM_StressTest)->Range(2048, 1048576);
