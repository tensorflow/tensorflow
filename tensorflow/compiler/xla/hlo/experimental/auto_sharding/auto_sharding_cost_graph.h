/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_COST_GRAPH_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_COST_GRAPH_H_

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/matrix.h"
namespace xla {
namespace spmd {

// A graph data structrue to simplify the edge cost graph.
// It merges nodes and does path compression.
class CostGraph {
 public:
  CostGraph(const LeafStrategies& leaf_strategies,
            const AssociativeDotPairs& associative_dot_pairs) {
    node_lens_.reserve(leaf_strategies.size());
    extra_node_costs_.reserve(leaf_strategies.size());
    adjacency_.assign(leaf_strategies.size(), StableHashSet<int>());

    // Build the cost graph
    for (const auto& strategies : leaf_strategies) {
      node_lens_.push_back(strategies->leaf_vector.size());
      extra_node_costs_.push_back(
          std::vector<double>(strategies->leaf_vector.size(), 0.0));

      for (size_t i = 0; i < strategies->in_nodes.size(); ++i) {
        if (!strategies->in_nodes[i]->is_tuple) {
          size_t src_idx = strategies->in_nodes[i]->id;
          size_t dst_idx = strategies->id;
          Matrix edge_cost = CreateEdgeCost(src_idx, dst_idx, i, strategies);
          AddEdgeCost(src_idx, dst_idx, edge_cost);
        } else {
          CHECK_EQ(strategies->in_nodes.size(), 1)
              << "Do not support instructions with more than one tuple "
                 "operand. If this CHECK fails, we will need to fix "
                 "b/233412625.";
          for (size_t l = 0; l < strategies->in_nodes[i]->childs.size(); l++) {
            size_t src_idx = strategies->in_nodes[i]->childs.at(l)->id;
            size_t dst_idx = strategies->id;
            // TODO(b/233412625) Support more general case, e.g., multiple tuple
            // operands. If there is only one operand and it's a tuple, the
            // first index of resharding_costs is for the tuple element.
            Matrix edge_cost =
                CreateEdgeCost(src_idx, dst_idx, /*in_node_idx=*/l, strategies);
            AddEdgeCost(src_idx, dst_idx, edge_cost);
          }
        }
      }

      if (strategies->following) {
        to_merge_pairs_.push_back({strategies->id, strategies->following->id});
      }
    }

    // Adjust the edge costs for dot pairs that can be optimized by
    // AllReduceReassociate
    for (const auto& pair : associative_dot_pairs) {
      size_t src_idx = pair.first->id;
      size_t dst_idx = pair.second->id;

      if (node_lens_[src_idx] != node_lens_[dst_idx]) {
        continue;
      }

      Matrix edge_cost(node_lens_[src_idx], node_lens_[dst_idx]);
      for (size_t i = 0; i < node_lens_[src_idx]; ++i) {
        if (leaf_strategies[src_idx]->leaf_vector[i].communication_cost > 0) {
          CHECK_LE(
              std::abs(
                  leaf_strategies[src_idx]->leaf_vector[i].communication_cost -
                  leaf_strategies[dst_idx]->leaf_vector[i].communication_cost),
              1e-6);
          edge_cost(i, i) =
              -leaf_strategies[src_idx]->leaf_vector[i].communication_cost;
        }
      }
      AddEdgeCost(src_idx, dst_idx, edge_cost);
    }
  }

  Matrix CreateEdgeCost(size_t src_idx, size_t dst_idx, size_t in_node_idx,
                        StrategyVector* strategies) {
    CHECK_GE(node_lens_.size(), src_idx);
    CHECK_GE(node_lens_.size(), dst_idx);
    Matrix edge_cost(node_lens_[src_idx], node_lens_[dst_idx]);
    for (size_t k = 0; k < strategies->leaf_vector.size(); ++k) {
      const ShardingStrategy& strategy = strategies->leaf_vector[k];
      for (size_t j = 0; j < strategy.resharding_costs[in_node_idx].size();
           ++j) {
        edge_cost(j, k) = strategy.resharding_costs[in_node_idx][j];
      }
    }
    return edge_cost;
  }

  Matrix GetEdgeCost(int i, int j) {
    if (i <= j) {
      return edge_costs_[{i, j}];
    }
    return edge_costs_[{j, i}].Transpose();
  }

  void AddEdgeCost(int i, int j, Matrix& cost) {
    if (i > j) {
      std::swap(i, j);
      cost = cost.Transpose();
    }

    if (edge_costs_.contains({i, j})) {
      CHECK(adjacency_[i].contains(j));
      CHECK(adjacency_[j].contains(i));
      edge_costs_[{i, j}] = edge_costs_[{i, j}] + cost;
    } else {
      adjacency_[i].insert(j);
      adjacency_[j].insert(i);
      edge_costs_[{i, j}] = cost;
    }
  }

  void RemoveEdge(int i, int j) {
    if (i > j) {
      std::swap(i, j);
    }

    CHECK(adjacency_[i].contains(j));
    CHECK(adjacency_[j].contains(i));
    CHECK(edge_costs_.contains({i, j}));

    adjacency_[i].erase(j);
    adjacency_[j].erase(i);
    edge_costs_.erase({i, j});
  }

  void MergeNode(int src, int dst) {
    // Merge node src into node dst. This is used when we set one operator to
    // follow another operator's sharding spec. For the following computation
    // graph:
    //   dst -- src -- adj1
    //           |
    //          adj2
    //
    // It will be transformed into the following graph:
    //   (src)
    //    dst -- adj1
    //     |
    //    adj2
    // Where all the edges costs between src and adjs will be added into
    // the edge costs between dst and adjs. The edge cost between src and
    // dst will be added to the extra node cost of dst. Other node costs of
    // src will be added into dst's node cost in the ILP.

    CHECK(adjacency_[src].contains(dst));
    CHECK(adjacency_[dst].contains(src));
    CHECK(!merged_to_.contains(src));
    CHECK(!merged_to_.contains(dst));
    CHECK_NE(src, dst);

    Matrix edge_cost = GetEdgeCost(dst, src);

    std::vector<int> reindexing(node_lens_[dst]);
    if (node_lens_[dst] == node_lens_[src]) {
      // Assume the orders of strategies in src and dst match
      // (i.e. i-th strategy in src follows i-th strategy in dst).
      // This is true in most cases because of how we create the
      // following strategies.
      std::iota(reindexing.begin(), reindexing.end(), 0);
    } else {
      // Otherwise, find the strategy to follow greedily.
      // For every straetgy in dst, find the strategy in src with
      // the lowest resharding cost.
      std::vector<int> arange(node_lens_[src]);
      std::iota(arange.begin(), arange.end(), 0);
      for (int i = 0; i < node_lens_[dst]; ++i) {
        std::vector<std::pair<double, int>> keys;

        // If there are multiple strategies with the same lowest costs,
        // prefer to follow "replicated", which has the largest index.
        // Node: We assume the strategy "Repilcated" is always appended
        // as the last strategy in BuildStrategyAndCost.
        keys.reserve(node_lens_[src]);
        for (int j = 0; j < node_lens_[src]; ++j) {
          keys.push_back({edge_cost(i, j), -j});
        }

        std::sort(arange.begin(), arange.end(), [&keys](int l, int r) {
          return (keys[l].first < keys[r].first) ||
                 (keys[l].first == keys[r].first &&
                  keys[l].second < keys[r].second);
        });

        reindexing[i] = arange.front();
      }
    }
    merged_to_[src] = dst;
    reindexing_vector_[src] = reindexing;

    // Merge edge cost matrix
    std::vector<int> adj_list(adjacency_[src].begin(), adjacency_[src].end());
    for (int adj : adj_list) {
      if (adj == dst) {
        for (int i = 0; i < node_lens_[dst]; ++i) {
          extra_node_costs_[dst][i] += edge_cost(i, reindexing[i]);
        }
      } else {
        Matrix added_edge_cost(node_lens_[dst], node_lens_[adj]);
        Matrix edge_cost_src_adj = GetEdgeCost(src, adj);

        for (int i = 0; i < node_lens_[dst]; ++i) {
          for (int k = 0; k < node_lens_[adj]; ++k) {
            added_edge_cost(i, k) = edge_cost_src_adj(reindexing[i], k);
          }
        }

        AddEdgeCost(dst, adj, added_edge_cost);
      }
    }

    // Remove edges
    for (int adj : adj_list) {
      RemoveEdge(src, adj);
    }
  }

  int QueryDestination(int node) {
    if (merged_to_.contains(node)) {
      int old_dst = merged_to_[node];
      int new_dst = QueryDestination(old_dst);
      if (old_dst != new_dst) {
        // Compresss path
        absl::Span<const int> old_reindexing_vector = reindexing_vector_[node];
        std::vector<int> new_reindexing_vector;
        new_reindexing_vector.reserve(node_lens_.size());
        for (int i = 0; i < node_lens_[new_dst]; ++i) {
          new_reindexing_vector.push_back(
              old_reindexing_vector[reindexing_vector_[old_dst][i]]);
        }
        reindexing_vector_[node] = new_reindexing_vector;
        merged_to_[node] = new_dst;
      }
      return new_dst;
    }
    return node;
  }

  void Simplify(bool enable) {
    // Merge nodes
    for (const auto& pair : to_merge_pairs_) {
      int src = pair.first;
      int dst = pair.second;
      dst = QueryDestination(dst);
      if (enable) {
        MergeNode(src, dst);
      }
    }

    // Build follow map
    follow_idx_.reserve(node_lens_.size());
    for (int i = 0; i < node_lens_.size(); ++i) {
      if (merged_to_.contains(i)) {
        follow_idx_.push_back(QueryDestination(i));
      } else {
        follow_idx_.push_back(-1);
      }
    }
  }

  int RemapIndex(int node_id, int value) const {
    if (follow_idx_[node_id] < 0) {
      return value;
    }
    return reindexing_vector_.at(node_id)[value];
  }

  std::string ToString() {
    std::string str;
    absl::StrAppend(&str, "Cost Graph:\n");

    for (int i = 0; i < node_lens_.size(); ++i) {
      absl::StrAppend(&str, "Node", i, ": ", node_lens_[i], "\n");
    }
    absl::StrAppend(&str, "\n");

    for (const auto& iter : edge_costs_) {
      absl::StrAppend(&str, "Edge (", iter.first.first, ", ", iter.first.second,
                      "):\n");
      absl::StrAppend(&str, iter.second.ToString(), "\n");
    }

    return str;
  }

  // The number of strategies of each node.
  std::vector<int> node_lens_;
  // The adjacency list of each node.
  std::vector<StableHashSet<int>> adjacency_;
  // The cost matrix between two nodes.

  StableHashMap<std::pair<int, int>, Matrix> edge_costs_;
  // The extra node costs introduced by merging nodes.
  std::vector<std::vector<double>> extra_node_costs_;
  // The reindexing vector of the node.
  // A reindexing vector maps a strategy index from the node being followed
  // to a strategy index of the curret node.
  StableHashMap<int, std::vector<int>> reindexing_vector_;
  // Maps a node id to the node id that is being followed by this node.
  // The value is -1 if the current node does not follow any node.
  std::vector<int> follow_idx_;

  // Save the destination of merged nodes.
  StableHashMap<int, int> merged_to_;
  // Save pairs that need to be merged.
  std::vector<std::pair<int, int>> to_merge_pairs_;
};

// Get the final sharding strategy according to the ilp solution.
inline const ShardingStrategy& GetShardingStrategy(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const int64_t> s_val) {
  const StrategyVector* strategies = strategy_map.at(inst).get();
  CHECK(!strategies->is_tuple);
  int node_idx = strategies->id;
  int stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
  return strategies->leaf_vector[stra_idx];
}

}  // namespace spmd
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_COST_GRAPH_H_
