/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_cost_graph.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/matrix.h"

namespace xla {
namespace spmd {

EdgeReshardingCostMatrix Normalize(const EdgeReshardingCostMatrix& edge_cost) {
  double min_communication_cost = std::numeric_limits<double>::max();
  for (int i = 0; i < edge_cost.n_; ++i) {
    for (int j = 0; j < edge_cost.m_; ++j) {
      min_communication_cost =
          std::min(min_communication_cost, edge_cost(i, j).communication_cost);
    }
  }
  if (min_communication_cost >= 0) return edge_cost;
  EdgeReshardingCostMatrix normalized_edge_cost = edge_cost;
  for (int i = 0; i < edge_cost.n_; ++i) {
    for (int j = 0; j < edge_cost.m_; ++j) {
      normalized_edge_cost(i, j).communication_cost -= min_communication_cost;
    }
  }
  return normalized_edge_cost;
}

CostGraph::CostGraph(const StrategyGroups& strategy_groups,
                     const AssociativeDotPairs& associative_dot_pairs) {
  node_lens_.reserve(strategy_groups.size());
  extra_node_costs_.reserve(strategy_groups.size());
  adjacency_.assign(strategy_groups.size(), StableSet<int>());

  // Build the cost graph.
  for (StrategyGroup* strategy_group : strategy_groups) {
    node_lens_.push_back(strategy_group->GetStrategies().size());
    extra_node_costs_.push_back(
        std::vector<double>(strategy_group->GetStrategies().size(), 0.0));

    const auto& in_nodes = strategy_group->in_nodes;
    for (size_t i = 0; i < in_nodes.size(); ++i) {
      if (!in_nodes[i]->is_tuple) {
        NodeIdx src_idx = in_nodes[i]->node_idx;
        NodeIdx dst_idx = strategy_group->node_idx;
        EdgeReshardingCostMatrix edge_cost =
            CreateEdgeCost(src_idx, dst_idx, i, strategy_group);
        AddEdgeCost(src_idx, dst_idx, edge_cost);
      } else if (in_nodes[i]->is_tuple && in_nodes.size() > 1) {
        for (const auto& child : in_nodes[i]->GetChildren()) {
          NodeIdx src_idx = child->node_idx;
          NodeIdx dst_idx = strategy_group->node_idx;
          EdgeReshardingCostMatrix edge_cost =
              CreateEdgeCost(src_idx, dst_idx, i, strategy_group, true);
          AddEdgeCost(src_idx, dst_idx, edge_cost);
        }
      } else {
        CHECK_EQ(in_nodes.size(), 1)
            << "Do not support instructions with more than one tuple "
               "operand. If this CHECK fails, we will need to fix "
               "b/233412625.";
        for (size_t l = 0; l < in_nodes[i]->GetChildren().size(); ++l) {
          NodeIdx src_idx = in_nodes[i]->GetChildren()[l]->node_idx;
          NodeIdx dst_idx = strategy_group->node_idx;
          // TODO(b/233412625) Support more general case, e.g., multiple tuple
          // operands. If there is only one operand and it's a tuple, the
          // first index of communication_resharding_costs is for the tuple
          // element.
          EdgeReshardingCostMatrix edge_cost = CreateEdgeCost(
              src_idx, dst_idx, /*in_node_idx=*/l, strategy_group);
          AddEdgeCost(src_idx, dst_idx, edge_cost);
        }
      }
    }

    if (strategy_group->following) {
      CHECK_EQ(strategy_group->GetStrategies().size(),
               strategy_group->following->GetStrategies().size())
          << "Different strategy counts for instruction ID "
          << strategy_group->instruction_id << " and following instruction ID "
          << strategy_group->following->instruction_id;
      to_merge_pairs_.push_back(
          {strategy_group->node_idx, strategy_group->following->node_idx});
    }
  }

  // Adjust the edge costs for dot pairs that can be optimized by
  // AllReduceReassociate.
  for (const auto& pair : associative_dot_pairs) {
    NodeIdx src_idx = pair.first->node_idx;
    NodeIdx dst_idx = pair.second->node_idx;
    StrategyGroup& src_strategy_group = *strategy_groups[src_idx];
    StrategyGroup& dst_strategy_group = *strategy_groups[dst_idx];

    EdgeReshardingCostMatrix edge_cost(node_lens_[src_idx],
                                       node_lens_[dst_idx]);
    absl::flat_hash_map<std::string, NodeStrategyIdx>
        src_strategy_name_to_idx_map;
    const auto& src_strategy_input_shardings =
        src_strategy_group.GetStrategyInputShardings();
    for (size_t iid = 0; iid < src_strategy_input_shardings.size(); ++iid) {
      const InputShardings& input_shardings = src_strategy_input_shardings[iid];
      NodeStrategyIdx i =
          src_strategy_group.GetStrategyIdxForInputShardings(iid);
      const ShardingStrategy& strategy = src_strategy_group.GetStrategy(i);
      if (strategy.communication_cost > 0) {
        src_strategy_name_to_idx_map[input_shardings.name] = i;
      }
    }
    const auto& dst_strategy_input_shardings =
        dst_strategy_group.GetStrategyInputShardings();
    for (size_t iid = 0; iid < dst_strategy_input_shardings.size(); ++iid) {
      const InputShardings& input_shardings = dst_strategy_input_shardings[iid];
      NodeStrategyIdx i =
          dst_strategy_group.GetStrategyIdxForInputShardings(iid);
      const ShardingStrategy& dst_strategy = dst_strategy_group.GetStrategy(i);
      if (dst_strategy.communication_cost > 0) {
        auto it = src_strategy_name_to_idx_map.find(input_shardings.name);
        if (it != src_strategy_name_to_idx_map.end()) {
          const auto& src_strategy = src_strategy_group.GetStrategy(it->second);
          CHECK_LE(std::abs(src_strategy.communication_cost -
                            dst_strategy.communication_cost),
                   1e-6);
          edge_cost(it->second, i).communication_cost =
              -src_strategy.communication_cost;
        }
      }
    }
    AddEdgeCost(src_idx, dst_idx, edge_cost);
  }
}

EdgeReshardingCostMatrix CostGraph::CreateEdgeCost(
    const NodeIdx src_idx, const NodeIdx dst_idx, const size_t in_node_idx,
    StrategyGroup* strategy_group, const bool zero_cost) {
  CHECK_LT(src_idx, node_lens_.size());
  CHECK_LT(dst_idx, node_lens_.size());
  EdgeReshardingCostMatrix edge_cost(node_lens_[src_idx], node_lens_[dst_idx]);
  const auto& strategies = strategy_group->GetStrategies();
  for (NodeStrategyIdx k = 0; k < strategies.size(); ++k) {
    const ShardingStrategy& strategy = strategies[k];
    size_t start_idx = 0;
    CHECK_LT(in_node_idx, strategy.memory_resharding_costs.size())
        << strategy_group->node_idx;
    if (strategy.memory_resharding_costs[in_node_idx].size() >
        node_lens_[src_idx]) {
      start_idx = strategy.memory_resharding_costs[in_node_idx].size() -
                  node_lens_[src_idx];
    }
    for (size_t j = start_idx;
         j < strategy.memory_resharding_costs[in_node_idx].size(); ++j) {
      double communication_cost = 0;
      double memory_cost = 0;
      if (!zero_cost) {
        communication_cost =
            strategy.communication_resharding_costs[in_node_idx][j];
        memory_cost = strategy.memory_resharding_costs[in_node_idx][j];
      }
      edge_cost(j - start_idx, k) =
          EdgeReshardingCost(communication_cost, memory_cost);
    }
  }
  return edge_cost;
}

EdgeReshardingCostMatrix CostGraph::GetEdgeCost(const NodeIdx i,
                                                const NodeIdx j) {
  if (i <= j) {
    return edge_costs_[{i, j}];
  }
  return edge_costs_[{j, i}].Transpose();
}

void CostGraph::AddEdgeCost(NodeIdx i, NodeIdx j,
                            EdgeReshardingCostMatrix& cost) {
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

void CostGraph::RemoveEdge(NodeIdx i, NodeIdx j) {
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

void CostGraph::MergeNode(const NodeIdx src, const NodeIdx dst) {
  CHECK(adjacency_[src].contains(dst));
  CHECK(adjacency_[dst].contains(src));
  CHECK(!merged_to_.contains(src));
  CHECK(!merged_to_.contains(dst));
  CHECK_NE(src, dst);

  EdgeReshardingCostMatrix edge_cost = GetEdgeCost(dst, src);

  std::vector<NodeStrategyIdx> reindexing(node_lens_[dst]);
  if (node_lens_[dst] == node_lens_[src]) {
    // Assume the orders of strategies in src and dst match
    // (i.e., i-th strategy in src follows i-th strategy in dst).
    // This is true in most cases because of how we create the
    // following strategies.
    std::iota(reindexing.begin(), reindexing.end(), 0);
  } else {
    // Otherwise, find the strategy to follow greedily.
    // For every strategy in dst, find the strategy in src with
    // the lowest resharding cost.
    std::vector<int> arange(node_lens_[src]);
    std::iota(arange.begin(), arange.end(), 0);
    for (NodeStrategyIdx i = 0; i < node_lens_[dst]; ++i) {
      std::vector<std::pair<double, int>> keys;

      // If there are multiple strategies with the same lowest costs,
      // prefer to follow "replicated", which has the largest index.
      // Node: We assume the strategy "Repilcated" is always appended
      // as the last strategy in BuildStrategyAndCost.
      keys.reserve(node_lens_[src]);
      for (NodeStrategyIdx j = 0; j < node_lens_[src]; ++j) {
        keys.push_back({edge_cost(i, j).communication_cost, -j});
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

  // Merge edge-cost matrix.
  std::vector<NodeIdx> adj_list(adjacency_[src].begin(), adjacency_[src].end());
  for (const NodeIdx adj : adj_list) {
    if (adj == dst) {
      for (NodeStrategyIdx i = 0; i < node_lens_[dst]; ++i) {
        extra_node_costs_[dst][i] +=
            edge_cost(i, reindexing[i]).communication_cost;
      }
    } else {
      EdgeReshardingCostMatrix added_edge_cost(node_lens_[dst],
                                               node_lens_[adj]);
      EdgeReshardingCostMatrix edge_cost_src_adj = GetEdgeCost(src, adj);
      for (NodeStrategyIdx i = 0; i < node_lens_[dst]; ++i) {
        for (NodeStrategyIdx k = 0; k < node_lens_[adj]; ++k) {
          added_edge_cost(i, k) = edge_cost_src_adj(reindexing[i], k);
        }
      }
      AddEdgeCost(dst, adj, added_edge_cost);
    }
  }
  // Remove edges
  for (const NodeIdx adj : adj_list) {
    RemoveEdge(src, adj);
  }
}

NodeIdx CostGraph::QueryDestination(const NodeIdx node_idx) {
  if (merged_to_.contains(node_idx)) {
    NodeIdx old_dst = merged_to_[node_idx];
    NodeIdx new_dst = QueryDestination(old_dst);
    if (old_dst != new_dst) {
      // Compress path.
      absl::Span<const NodeStrategyIdx> old_reindexing_vector =
          reindexing_vector_[node_idx];
      std::vector<NodeStrategyIdx> new_reindexing_vector;
      new_reindexing_vector.reserve(node_lens_.size());
      for (NodeStrategyIdx i = 0; i < node_lens_[new_dst]; ++i) {
        new_reindexing_vector.push_back(
            old_reindexing_vector[reindexing_vector_[old_dst][i]]);
      }
      reindexing_vector_[node_idx] = new_reindexing_vector;
      merged_to_[node_idx] = new_dst;
    }
    return new_dst;
  }
  return node_idx;
}

void CostGraph::Simplify(const bool enable) {
  // Merge nodes.
  if (enable) {
    for (const auto& [src, dst] : to_merge_pairs_) {
      MergeNode(src, QueryDestination(dst));
    }
  }
  // Build follow map.
  follow_idx_.reserve(node_lens_.size());
  for (NodeIdx i = 0; i < node_lens_.size(); ++i) {
    if (merged_to_.contains(i)) {
      follow_idx_.push_back(QueryDestination(i));
    } else {
      follow_idx_.push_back(-1);
    }
  }
}

NodeStrategyIdx CostGraph::RemapIndex(const NodeIdx node_id,
                                      const NodeStrategyIdx value) const {
  if (follow_idx_[node_id] < 0) {
    return value;
  }
  return reindexing_vector_.at(node_id)[value];
}

std::string CostGraph::ToString() const {
  std::string str;
  absl::StrAppend(&str, "Cost Graph:\n");

  for (NodeIdx i = 0; i < node_lens_.size(); ++i) {
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

}  // namespace spmd
}  // namespace xla
