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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"

namespace xla {
namespace spmd {

// A constant to represent infinity cost.
constexpr double kInfinityCost = 1e20;

// Type alias
template <typename Key, typename Value>
using StableMap = absl::btree_map<Key, Value>;
template <typename Key>
using StableSet = absl::btree_set<Key>;

struct CompareHloInstruction {
  bool operator()(const HloInstruction* a, const HloInstruction* b) const {
    return a->name() < b->name();
  }
};

template <typename Value>
using ConstInstructionMap =
    absl::btree_map<const HloInstruction*, Value, CompareHloInstruction>;
template <typename Value>
using InstructionMap =
    absl::btree_map<HloInstruction*, Value, CompareHloInstruction>;

using ConstInstructionSet =
    absl::btree_set<const HloInstruction*, CompareHloInstruction>;
using InstructionSet = absl::btree_set<HloInstruction*, CompareHloInstruction>;

// Map an instruction to its depth.
using InstructionDepthMap = ConstInstructionMap<int64_t>;
// Map an instruction to its batch dimension.
using InstructionBatchDimMap = StableMap<std::string, int>;
// Map an instruction to its alias source parameter.
using AliasMap = ConstInstructionMap<HloInstruction*>;
// Map an instruction to its resharding cache.
using ReshardingCache =
    ConstInstructionMap<std::vector<std::pair<HloSharding, HloInstruction*>>>;
// Resharding costs for each operand
using ReshardingCosts = std::vector<std::vector<double>>;

// A named vector of optional shardings for each operand.
struct InputShardings {
  std::string name;
  std::vector<std::optional<HloSharding>> shardings;

  std::string ToString() const {
    std::string str = absl::StrCat(name, " ");
    for (const auto& s : shardings) {
      if (!s.has_value()) {
        absl::StrAppend(&str, "[*],");
      } else if (s->IsReplicated()) {
        absl::StrAppend(&str, "[R],");
      } else {
        if (s->ReplicateOnLastTileDim()) {
          absl::StrAppend(
              &str, "[", absl::StrJoin(s->tile_assignment().dimensions(), ", "),
              "]last_tile_dim_replicate,");
        } else {
          absl::StrAppend(
              &str, "[", absl::StrJoin(s->tile_assignment().dimensions(), ", "),
              "],");
        }
      }
    }
    return str;
  }
};

// One sharding strategy
struct ShardingStrategy {
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

  std::string ToString() const { return output_sharding.ToString(); }

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

    return absl::StrCat(
        output_sharding.ToString(), ", compute_cost=", compute_cost,
        ", communication_cost=", communication_cost,
        ", memory_cost=", memory_cost,
        ", communication_resharding_costs=", communication_resharding_cost_str,
        ", memory_resharding_costs=", memory_resharding_cost_str);
  }

  bool operator==(const ShardingStrategy& other) const {
    return output_sharding == other.output_sharding &&
           compute_cost == other.compute_cost &&
           communication_cost == other.communication_cost &&
           memory_cost == other.memory_cost &&
           communication_resharding_costs ==
               other.communication_resharding_costs &&
           memory_resharding_costs == other.memory_resharding_costs;
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
using NodeStrategies = StableSet<NodeStrategy>;

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
  // The index of this instruction in the HLO operand (or tuple shape) list.
  std::optional<int64_t> tuple_element_idx;

  StrategyGroup() = default;

  StrategyGroup(bool is_tuple, NodeIdx node_idx, size_t instruction_id)
      : is_tuple(is_tuple),
        node_idx(node_idx),
        instruction_id(instruction_id) {}

  StrategyGroup(bool is_tuple, NodeIdx node_idx, size_t instruction_id,
                const std::vector<const StrategyGroup*>& in_nodes,
                const StrategyGroup* following,
                const std::vector<ShardingStrategy>& strategies)
      : is_tuple(is_tuple),
        node_idx(node_idx),
        instruction_id(instruction_id),
        in_nodes(in_nodes),
        following(following) {
    for (const ShardingStrategy& strategy : strategies) {
      AddStrategy(strategy);
    }
  }

  std::string ToString(size_t indentation = 0) const {
    std::string str;
    const std::string indent(indentation, ' ');
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
      for (size_t i = 0; i < children.size(); ++i) {
        absl::StrAppend(&str, indent, "Tuple element #", i, ":\n");
        absl::StrAppend(&str, children[i]->ToString(indentation + 2));
      }
    } else {
      for (const auto& strategy : strategies) {
        absl::StrAppend(&str, indent, "Strategy ", strategy.ToStringLong(),
                        "\n");
      }
    }
    if (!is_tuple) {
      for (const auto& input_shardings : strategy_input_shardings) {
        const std::string input_sharding_str =
            absl::StrCat("{", input_shardings.ToString(), "}\n");
        absl::StrAppend(&str, indent, "Input Sharding ", input_sharding_str);
      }
    }
    return str;
  }

  const StrategyGroup* GetSubStrategyGroup(const ShapeIndex& index) const {
    const StrategyGroup* result = this;
    for (auto index_element : index) {
      CHECK_LE(index_element, result->children.size());
      result = result->children.at(index_element).get();
    }
    return result;
  }

  void ForEachLeafStrategyGroup(
      absl::FunctionRef<void(const StrategyGroup&)> fn) const {
    if (is_tuple) {
      for (const std::unique_ptr<StrategyGroup>& child : children) {
        fn(*child);
      }
    } else {
      fn(*this);
    }
  }

  void ForEachLeafStrategyGroup(absl::FunctionRef<void(StrategyGroup&)> fn) {
    if (is_tuple) {
      for (std::unique_ptr<StrategyGroup>& child : children) {
        fn(*child);
      }
    } else {
      fn(*this);
    }
  }

  //////// Accessor methods for strategies ////////

  void AddStrategy(const ShardingStrategy& strategy,
                   const InputShardings& input_shardings = {}) {
    // Create a new strategy if needed (otherwise, reuse an existing one).
    size_t strategy_idx = strategies.size();
    const size_t input_sharding_idx = strategy_input_shardings.size();
    const auto it = std::find(strategies.begin(), strategies.end(), strategy);
    if (it == strategies.end()) {
      strategies.push_back(strategy);
      strategy_idx_to_input_sharding_idx.push_back(input_sharding_idx);
    } else {
      strategy_idx = std::distance(strategies.begin(), it);
    }
    input_sharding_idx_to_strategy_idx.push_back(strategy_idx);
    strategy_input_shardings.push_back(input_shardings);
  }

  void ClearStrategies() {
    strategies.clear();
    strategy_input_shardings.clear();
    input_sharding_idx_to_strategy_idx.clear();
    strategy_idx_to_input_sharding_idx.clear();
  }

  ShardingStrategy& GetStrategy(size_t strategy_idx) {
    return strategies[strategy_idx];
  }

  const ShardingStrategy& GetStrategyForInputShardings(
      size_t input_sharding_idx) const {
    const size_t strategy_idx =
        input_sharding_idx_to_strategy_idx[input_sharding_idx];
    CHECK_LT(strategy_idx, strategies.size());
    return strategies[strategy_idx];
  }

  size_t GetStrategyIdxForInputShardings(size_t input_sharding_idx) const {
    return input_sharding_idx_to_strategy_idx[input_sharding_idx];
  }

  const InputShardings& GetInputShardings(size_t input_sharding_idx) const {
    return strategy_input_shardings[input_sharding_idx];
  }

  const InputShardings& GetInputShardingsForStrategy(
      size_t strategy_idx) const {
    const size_t input_sharding_idx =
        strategy_idx_to_input_sharding_idx[strategy_idx];
    CHECK_LT(input_sharding_idx, strategy_input_shardings.size());
    return strategy_input_shardings[input_sharding_idx];
  }

  const std::vector<ShardingStrategy>& GetStrategies() const {
    return strategies;
  }

  const std::vector<InputShardings>& GetStrategyInputShardings() const {
    return strategy_input_shardings;
  }

  //////// Accessor methods for children ////////

  void AddChild(std::unique_ptr<StrategyGroup> child) {
    children.push_back(std::move(child));
  }

  void ClearChildren() { children.clear(); }

  StrategyGroup& GetChild(size_t child_idx) { return *children[child_idx]; }

  const std::vector<std::unique_ptr<StrategyGroup>>& GetChildren() const {
    return children;
  }

 private:
  // Used when is_tuple == False. Leaf strategy vector.
  // A vector of strategy choices for the non-tuple output.
  std::vector<ShardingStrategy> strategies;
  std::vector<InputShardings> strategy_input_shardings;
  std::vector<size_t> input_sharding_idx_to_strategy_idx;
  std::vector<size_t> strategy_idx_to_input_sharding_idx;

  // Used when is_tuple == True. A vector of pointers, each pointer is one
  // StrategyGroup for one value in the output Tuple
  std::vector<std::unique_ptr<StrategyGroup>> children;
};

// Type aliases.
using LivenessSet = std::vector<std::vector<const HloValue*>>;
// A liveness set using node indices instead of HLO values.
using LivenessNodeSet = std::vector<std::vector<NodeIdx>>;
// A liveness set using edge indices instead of HLO values.
using LivenessEdgeSet = std::vector<std::vector<EdgeIdx>>;
// Map an instruction to its strategy group.
using StrategyMap = ConstInstructionMap<std::unique_ptr<StrategyGroup>>;
// The list of all strategy groups.
using StrategyGroups = std::vector<StrategyGroup*>;
// The list of all dot instruction pairs that can be optimized by
// AllReduceReassociate pass.
using AssociativeDotPairs =
    std::vector<std::pair<const StrategyGroup*, const StrategyGroup*>>;
// The set of all alias pairs
using AliasSet = StableSet<std::pair<NodeIdx, NodeIdx>>;

// Utilities for creating sharding objects
using MeshDimSet = StableSet<int>;
using DimMap = StableMap</*tensor dim*/ int, /*mesh dims*/ MeshDimSet>;

// Map tensor dims from [0, tensor_shape.dimensions_size() - 1] to (atmost one
// or more, depending on the value of allow_mixed_mesh_shape) mesh dims.
void Enumerate(std::function<void(const DimMap&)> split_func,
               int64_t tensor_rank,
               const std::vector<int>& unassigned_mesh_dims,
               bool allow_mixed_mesh_shape);
}  // namespace spmd
}  // namespace xla
#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STRATEGY_H_
