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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_COST_GRAPH_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_COST_GRAPH_H_

#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/matrix.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"

namespace xla {
namespace spmd {

struct EdgeReshardingCost {
  double communication_cost = 0;
  double memory_cost = 0;

  EdgeReshardingCost() : communication_cost(0), memory_cost(0) {}

  EdgeReshardingCost(double communication_cost_, double memory_cost_)
      : communication_cost(communication_cost_), memory_cost(memory_cost_) {}

  EdgeReshardingCost operator+(const EdgeReshardingCost& other) const {
    return EdgeReshardingCost(other.communication_cost + communication_cost,
                              other.memory_cost + memory_cost);
  }

  std::string ToString() const {
    return absl::StrCat("{communication_cost=", communication_cost,
                        ", memory_cost=", memory_cost, "}");
  }
};

using EdgeReshardingCostMatrix = Matrix<EdgeReshardingCost>;

// Normalizes the edge cost matrix by a fixed constant to ensure there are no
// negative communication costs.
EdgeReshardingCostMatrix Normalize(const EdgeReshardingCostMatrix& edge_cost);

// A graph data structure to simplify the edge cost graph. It merges nodes and
// performs path compression.
class CostGraph {
 public:
  CostGraph(const StrategyGroups& strategy_groups,
            const AssociativeDotPairs& associative_dot_pairs);

  EdgeReshardingCostMatrix CreateEdgeCost(NodeIdx src_idx, NodeIdx dst_idx,
                                          size_t in_node_idx,
                                          StrategyGroup* strategy_group,
                                          bool zero_cost = false);

  EdgeReshardingCostMatrix GetEdgeCost(NodeIdx i, NodeIdx j);

  void AddEdgeCost(NodeIdx i, NodeIdx j, EdgeReshardingCostMatrix& cost);

  void RemoveEdge(NodeIdx i, NodeIdx j);

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
  // Where all the edges costs between src and adjs will be added into the edge
  // costs between dst and adjs. The edge cost between src and dst will be added
  // to the extra node cost of dst. Other node costs of src will be added into
  // dst's node cost in the ILP.
  void MergeNode(NodeIdx src, NodeIdx dst);

  NodeIdx QueryDestination(NodeIdx node_idx);

  void Simplify(bool enable);

  NodeStrategyIdx RemapIndex(NodeIdx node_id, NodeStrategyIdx value) const;

  std::string ToString() const;

  // TODO: Make class member variables private.

  // The number of strategies of each node.
  std::vector<int> node_lens_;
  // The adjacency list of each node.
  std::vector<StableSet<int>> adjacency_;
  // The cost matrix between two nodes.

  StableMap<std::pair<NodeIdx, NodeIdx>, EdgeReshardingCostMatrix> edge_costs_;
  // The extra node costs introduced by merging nodes.
  std::vector<std::vector<double>> extra_node_costs_;
  // The reindexing vector of the node.
  // A reindexing vector maps a strategy index from the node being followed
  // to a strategy index of the current node.
  StableMap<int, std::vector<NodeStrategyIdx>> reindexing_vector_;
  // Maps a node id to the node id that is being followed by this node.
  // The value is -1 if the current node does not follow any node.
  std::vector<NodeIdx> follow_idx_;

  // Save the destination of merged nodes.
  StableMap<NodeIdx, NodeIdx> merged_to_;
  // Save pairs that need to be merged.
  std::vector<std::pair<NodeIdx, NodeIdx>> to_merge_pairs_;
};

// Get the final sharding strategy according to the ILP solution.
inline const ShardingStrategy& GetShardingStrategy(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const NodeStrategyIdx> s_val) {
  const StrategyGroup* strategy_group = strategy_map.at(inst).get();
  CHECK(!strategy_group->is_tuple);
  NodeIdx node_idx = strategy_group->node_idx;
  NodeStrategyIdx stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
  return strategy_group->GetStrategies()[stra_idx];
}

// Get the input shardings according to the ILP solution.
inline const InputShardings& GetInputShardings(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const NodeStrategyIdx> s_val) {
  const StrategyGroup* strategy_group = strategy_map.at(inst).get();
  CHECK(!strategy_group->is_tuple);
  NodeIdx node_idx = strategy_group->node_idx;
  NodeStrategyIdx stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
  return strategy_group->GetInputShardingsForStrategy(stra_idx);
}

// Get the final sharding strategy according to the ILP solution.
inline const ShardingStrategy& GetShardingStrategyForTuple(
    const HloInstruction* inst, const ShapeIndex& index,
    const StrategyMap& strategy_map, const CostGraph& cost_graph,
    absl::Span<const NodeStrategyIdx> s_val) {
  const StrategyGroup* strategy_group = strategy_map.at(inst).get();
  CHECK(strategy_group->is_tuple);
  for (auto index_element : index) {
    CHECK_LT(index_element, strategy_group->GetChildren().size());
    const auto& strategies = strategy_group->GetChildren()[index_element];
    strategy_group = strategies.get();
  }
  NodeIdx node_idx = strategy_group->node_idx;
  NodeStrategyIdx stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
  return strategy_group->GetStrategies()[stra_idx];
}

// Get the input shardings according to the ILP solution.
inline const InputShardings& GetInputShardingsForTuple(
    const HloInstruction* inst, const ShapeIndex& index,
    const StrategyMap& strategy_map, const CostGraph& cost_graph,
    absl::Span<const NodeStrategyIdx> s_val) {
  const StrategyGroup* strategy_group = strategy_map.at(inst).get();
  CHECK(strategy_group->is_tuple);
  for (auto index_element : index) {
    CHECK_LT(index_element, strategy_group->GetChildren().size());
    const auto& strategies = strategy_group->GetChildren()[index_element];
    strategy_group = strategies.get();
  }
  NodeIdx node_idx = strategy_group->node_idx;
  NodeStrategyIdx stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
  return strategy_group->GetInputShardingsForStrategy(stra_idx);
}

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_COST_GRAPH_H_
