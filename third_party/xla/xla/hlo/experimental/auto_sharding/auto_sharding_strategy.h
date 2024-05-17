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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STRATEGY_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STRATEGY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace xla {
namespace spmd {

// A constant to represent infinity cost.
constexpr double kInfinityCost = 1e20;

// Type alias
template <typename Key, typename Value>
using StableHashMap = ::absl::flat_hash_map<Key, Value>;
template <typename Key>
using StableHashSet = ::absl::flat_hash_set<Key>;

// Map an instruction to its depth.
using InstructionDepthMap = StableHashMap<const HloInstruction*, int64_t>;
// Map an instruction to its batch dimension.
using InstructionBatchDimMap = StableHashMap<std::string, int>;
// Map an instruction to its alias source parameter.
using AliasMap = StableHashMap<const HloInstruction*, HloInstruction*>;
// Map an instruction to its resharding cache.
using ReshardingCache =
    StableHashMap<const HloInstruction*,
                  std::vector<std::pair<HloSharding, HloInstruction*>>>;
// Resharding costs for each operand
using ReshardingCosts = std::vector<std::vector<double>>;

// One sharding strategy
struct ShardingStrategy {
  std::string name;
  HloSharding output_sharding;
  double compute_cost;
  double communication_cost;
  double memory_cost;
  // resharding_costs[i][j] is the resharding cost from the output of
  // i-th operand's j-th strategy to this strategy.
  // If there is only one tuple operand,resharding_costs[i][j] is the resharding
  // cost from i-th tuple element's j-th strategy.
  ReshardingCosts communication_resharding_costs;
  ReshardingCosts memory_resharding_costs;
  // Optional: the required shardings of operands.
  // This is used to guide the SPMD partitioner.
  std::vector<std::optional<HloSharding>> input_shardings;

  std::string ToString() const {
    return absl::StrCat(name, ", ", output_sharding.ToString());
  }

  std::string ToStringLong() const {
    std::vector<std::string> communication_resharding_vector_strings;
    communication_resharding_vector_strings.reserve(
        communication_resharding_costs.size());
    for (const auto& v : communication_resharding_costs) {
      communication_resharding_vector_strings.push_back(
          absl::StrCat("[", absl::StrJoin(v, ", "), "]"));
    }
    std::string communication_resharding_cost_str = absl::StrCat(
        "{", absl::StrJoin(communication_resharding_vector_strings, ", "), "}");

    std::vector<std::string> memory_resharding_vector_strings;
    memory_resharding_vector_strings.reserve(memory_resharding_costs.size());
    for (const auto& v : memory_resharding_costs) {
      memory_resharding_vector_strings.push_back(
          absl::StrCat("[", absl::StrJoin(v, ", "), "]"));
    }
    std::string memory_resharding_cost_str = absl::StrCat(
        "{", absl::StrJoin(memory_resharding_vector_strings, ", "), "}");

    std::string input_sharding_str = "{";
    for (const auto& s : input_shardings) {
      if (!s.has_value()) {
        input_sharding_str += "[*],";
      } else if (s->IsReplicated()) {
        input_sharding_str += "[R],";
      } else {
        if (s->ReplicateOnLastTileDim()) {
          input_sharding_str +=
              "[" + absl::StrJoin(s->tile_assignment().dimensions(), ", ") +
              "]last_tile_dim_replicate,";
        } else {
          input_sharding_str +=
              "[" + absl::StrJoin(s->tile_assignment().dimensions(), ", ") +
              "],";
        }
      }
    }
    input_sharding_str += "}\n";
    return absl::StrCat(
        name, ", ", output_sharding.ToString(), ", compute_cost=", compute_cost,
        ", communication_cost=", communication_cost,
        ", memory_cost=", memory_cost,
        ", communication_resharding_costs=", communication_resharding_cost_str,
        ", memory_resharding_costs=", memory_resharding_cost_str,
        ", input_shardings=", input_sharding_str);
  }

  bool operator==(const ShardingStrategy& other) const {
    return name == other.name && output_sharding == other.output_sharding &&
           compute_cost == other.compute_cost &&
           communication_cost == other.communication_cost &&
           memory_cost == other.memory_cost &&
           communication_resharding_costs ==
               other.communication_resharding_costs &&
           memory_resharding_costs == other.memory_resharding_costs &&
           input_shardings == other.input_shardings;
  }
};

using NodeIdx = int64_t;          // An index into the solver's node list.
using EdgeIdx = int64_t;          // An index into the solver's edge list.
using NodeStrategyIdx = int64_t;  // An index into a node's strategy vector.
using EdgeStrategyIdx = int64_t;  // An index into an edge's strategy vector.
using LivenessIdx = int64_t;      // An index into the liveness vector.
using AliasIdx = int64_t;         // An index into the alias vector.

// Various classes needed to support strategy shaving.
using NodeStrategy = std::pair<NodeIdx, NodeStrategyIdx>;
using NodeStrategies = StableHashSet<NodeStrategy>;

// A group of strategy choices (along with details like index values)
// for each instruction.
struct StrategyGroup {
  bool is_tuple;
  // The index used in the solver. For non-leaf nodes, this is set to -1.
  NodeIdx node_idx;
  // The index of the HLO instruction that this strategy group belongs to.
  size_t instruction_id;
  // The connected nodes used for resharding costs;
  // The size must be the same as the size of resharding cost
  // each element in strategies's resharding_costs.size() needs to be the same
  // as strategies->in_nodes.size()
  std::vector<const StrategyGroup*> in_nodes;
  // The followed strategy. Used for merging nodes.
  const StrategyGroup* following = nullptr;
  // Used when is_tuple == False. Leaf strategy vector.
  // A vector of strategy choices for the non-tuple output.
  std::vector<ShardingStrategy> strategies;
  // Used when is_tuple == True. A vector of pointers, each pointer is one
  // StrategyGroup for one value in the output Tuple
  std::vector<std::unique_ptr<StrategyGroup>> childs;
  // The index of this instruction in the HLO operand (or tuple shape) list.
  std::optional<int64_t> tuple_element_idx;

  std::string ToString(size_t indention = 0) const {
    std::string str;
    const std::string indent(indention, ' ');
    absl::StrAppend(&str, indent, "node_idx: ", node_idx, "\n");
    absl::StrAppend(&str, indent, "instruction id: ", instruction_id, "\n");
    absl::StrAppend(&str, indent, "is_tuple: ", is_tuple, "\n");
    if (tuple_element_idx.has_value()) {
      absl::StrAppend(&str, indent,
                      "index in producer inst.: ", *tuple_element_idx, "\n");
    }
    if (following != nullptr) {
      absl::StrAppend(&str, indent,
                      "following instruction: ", following->instruction_id,
                      "\n");
    } else {
      absl::StrAppend(&str, indent, "source instruction\n");
    }
    for (auto i : in_nodes) {
      absl::StrAppend(&str, indent, "in nodes: node_idx=", i->node_idx,
                      " instruction_id=", i->instruction_id, "\n");
    }
    if (is_tuple) {
      for (size_t i = 0; i < childs.size(); ++i) {
        absl::StrAppend(&str, indent, "Tuple element #", i, ":\n");
        absl::StrAppend(&str, childs[i]->ToString(indention + 2));
      }
    } else {
      for (const auto& strategy : strategies) {
        absl::StrAppend(&str, indent, "Strategy ", strategy.ToStringLong());
      }
    }
    return str;
  }

  const StrategyGroup* GetSubStrategyGroup(const ShapeIndex& index) const {
    const StrategyGroup* result = this;
    for (auto index_element : index) {
      CHECK_LE(index_element, result->childs.size());
      result = result->childs.at(index_element).get();
    }
    return result;
  }
};

// Type aliases.
using LivenessSet = std::vector<std::vector<const HloValue*>>;
// A liveness set using node indices instead of HLO values.
using LivenessNodeSet = std::vector<std::vector<NodeIdx>>;
// A liveness set using edge indices instead of HLO values.
using LivenessEdgeSet = std::vector<std::vector<EdgeIdx>>;
// Map an instruction to its strategy group.
using StrategyMap =
    StableHashMap<const HloInstruction*, std::unique_ptr<StrategyGroup>>;
// The list of all strategy groups.
using StrategyGroups = std::vector<StrategyGroup*>;
// The list of all dot instruction pairs that can be optimized by
// AllReduceReassociate pass.
using AssociativeDotPairs =
    std::vector<std::pair<const StrategyGroup*, const StrategyGroup*>>;
// The set of all alias pairs
using AliasSet = StableHashSet<std::pair<NodeIdx, NodeIdx>>;

}  // namespace spmd
}  // namespace xla
#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STRATEGY_H_
