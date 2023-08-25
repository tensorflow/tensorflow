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

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_cost_graph.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver_option.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/matrix.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/metrics.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/profiling_result.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_live_range.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace spmd {
// Compute the resharding cost vector from multiple possible strategies
// to a desired sharding spec.
std::vector<double> ReshardingCostVector(
    const StrategyVector* strategies, const Shape& operand_shape,
    const HloSharding& required_sharding,
    const ClusterEnvironment& cluster_env) {
  CHECK(!strategies->is_tuple) << "Only works with strategy vector.";
  std::vector<double> ret;
  ret.reserve(strategies->leaf_vector.size());
  auto required_sharding_for_resharding = required_sharding.IsTileMaximal()
                                              ? HloSharding::Replicate()
                                              : required_sharding;
  for (const auto& x : strategies->leaf_vector) {
    ret.push_back(cluster_env.ReshardingCost(operand_shape, x.output_sharding,
                                             required_sharding_for_resharding));
  }
  return ret;
}

// Factory functions for StrategyVector.
std::unique_ptr<StrategyVector> CreateLeafStrategyVectorWithoutInNodes(
    size_t instruction_id, LeafStrategies& leaf_strategies) {
  auto strategies = std::make_unique<StrategyVector>();
  strategies->is_tuple = false;
  strategies->node_idx = leaf_strategies.size();
  leaf_strategies.push_back(strategies.get());
  strategies->instruction_id = instruction_id;
  return strategies;
}

// Factory functions for StrategyVector.
std::unique_ptr<StrategyVector> CreateLeafStrategyVector(
    size_t instruction_id, const HloInstruction* ins,
    const StrategyMap& strategy_map, LeafStrategies& leaf_strategies) {
  auto strategies =
      CreateLeafStrategyVectorWithoutInNodes(instruction_id, leaf_strategies);
  for (int64_t i = 0; i < ins->operand_count(); ++i) {
    strategies->in_nodes.push_back(strategy_map.at(ins->operand(i)).get());
  }
  return strategies;
}

std::unique_ptr<StrategyVector> CreateTupleStrategyVector(
    size_t instruction_id) {
  auto strategies = std::make_unique<StrategyVector>();
  strategies->is_tuple = true;
  strategies->node_idx = -1;
  strategies->instruction_id = instruction_id;
  return strategies;
}

std::optional<HloSharding> GetInputSharding(const HloInstruction* ins,
                                            const HloInstruction* operand,
                                            int64_t op_index,
                                            const HloSharding& output_sharding,
                                            const CallGraph& call_graph) {
  auto ins_clone = ins->Clone();
  ins_clone->set_sharding(output_sharding);
  auto operand_clone = operand->Clone();
  auto s = ins_clone->ReplaceOperandWith(op_index, operand_clone.get());
  CHECK_OK(s);
  return ShardingPropagation::GetShardingFromUser(*operand_clone, *ins_clone,
                                                  10, true, call_graph);
}

// ShardingPropagation::GetShardingFromUser does not handle TopK custom
// calls. Mirroring that function's handling of kSort, we handle TopK below.
HloSharding InferInputShardingForTopK(const HloInstruction* ins,
                                      const HloSharding& output_sharding) {
  return output_sharding;
}

// Compute the resharding costs as well as input shardings (when missing) for
// all operands of a given instruction, and an output sharding for that
// instruction.
std::vector<std::vector<double>>
GenerateReshardingCostsAndMissingShardingsForAllOperands(
    const HloInstruction* ins, const HloSharding& output_sharding,
    const StrategyMap& strategy_map, const ClusterEnvironment& cluster_env,
    const CallGraph& call_graph,
    std::vector<std::optional<HloSharding>>& input_shardings) {
  std::vector<std::vector<double>> resharding_costs;
  if (input_shardings.empty() && ins->operand_count() > 0) {
    input_shardings.resize(ins->operand_count());
  }
  for (int64_t k = 0; k < ins->operand_count(); ++k) {
    auto operand = ins->operand(k);
    if (operand->shape().IsToken() || operand->shape().rank() == 0) {
      resharding_costs.push_back(std::vector<double>(
          strategy_map.at(operand)->leaf_vector.size(), 0.0));
      if (!input_shardings[k].has_value()) {
        input_shardings[k] = HloSharding::Replicate();
      }
    } else {
      std::optional<HloSharding> cur_input_sharding;
      if (input_shardings[k].has_value()) {
        CHECK_EQ(input_shardings.size(), ins->operand_count());
        cur_input_sharding = input_shardings[k];
      } else {
        cur_input_sharding =
            GetInputSharding(ins, operand, k, output_sharding, call_graph);
      }
      bool is_sharding_default_replicated = false;
      if (!cur_input_sharding.has_value()) {
        if ((ins->opcode() == HloOpcode::kGather && k == 0) ||
            (ins->opcode() == HloOpcode::kScatter && k != 0)) {
          is_sharding_default_replicated = true;
          cur_input_sharding = HloSharding::Replicate();
        } else if (IsTopKCustomCall(ins)) {
          cur_input_sharding = InferInputShardingForTopK(ins, output_sharding);
        } else if (ins->opcode() == HloOpcode::kCustomCall) {
          is_sharding_default_replicated = true;
          cur_input_sharding = HloSharding::Replicate();
        }
      }
      CHECK(cur_input_sharding.has_value());
      if (!input_shardings[k].has_value()) {
        input_shardings[k] = cur_input_sharding;
      }
      auto operand_strategies = strategy_map.at(operand).get();
      auto operand_shape = operand->shape();
      if (ins->opcode() == HloOpcode::kGather && k == 0 &&
          is_sharding_default_replicated) {
        LOG(INFO)
            << "Zeroing out operand 0 resharding costs for gather sharding "
            << output_sharding.ToString();
        resharding_costs.push_back(
            std::vector<double>(operand_strategies->leaf_vector.size(), 0));
        input_shardings[k] = std::nullopt;
      } else {
        resharding_costs.push_back(
            ReshardingCostVector(operand_strategies, ins->operand(k)->shape(),
                                 *cur_input_sharding, cluster_env));
      }
    }
  }
  return resharding_costs;
}

std::pair<std::vector<std::vector<double>>,
          std::vector<std::optional<HloSharding>>>
GenerateReshardingCostsAndShardingsForAllOperands(
    const HloInstruction* ins, const HloSharding& output_sharding,
    const StrategyMap& strategy_map, const ClusterEnvironment& cluster_env,
    const CallGraph& call_graph) {
  std::vector<std::optional<HloSharding>> input_shardings_optional;
  auto resharding_costs =
      GenerateReshardingCostsAndMissingShardingsForAllOperands(
          ins, output_sharding, strategy_map, cluster_env, call_graph,
          input_shardings_optional);
  for (const auto& sharding_optional : input_shardings_optional) {
    CHECK(sharding_optional.has_value());
  }

  return std::make_pair(resharding_costs, input_shardings_optional);
}

std::unique_ptr<StrategyVector> MaybeFollowInsStrategyVector(
    const StrategyVector* src_strategies, const Shape& shape,
    size_t instruction_id, bool have_memory_cost,
    LeafStrategies& leaf_strategies, const ClusterEnvironment& cluster_env,
    StableHashMap<NodeIdx, std::vector<ShardingStrategy>>&
        pretrimmed_strategy_map) {
  std::unique_ptr<StrategyVector> strategies;
  if (src_strategies->is_tuple) {
    CHECK(shape.IsTuple());
    CHECK_EQ(shape.tuple_shapes_size(), src_strategies->childs.size());
    strategies = CreateTupleStrategyVector(instruction_id);
    strategies->childs.reserve(src_strategies->childs.size());
    for (size_t i = 0; i < src_strategies->childs.size(); ++i) {
      strategies->childs.push_back(MaybeFollowInsStrategyVector(
          src_strategies->childs[i].get(), shape.tuple_shapes(i),
          instruction_id, have_memory_cost, leaf_strategies, cluster_env,
          pretrimmed_strategy_map));
    }
  } else {
    CHECK(shape.IsArray() || shape.IsToken());
    strategies =
        CreateLeafStrategyVectorWithoutInNodes(instruction_id, leaf_strategies);
    strategies->in_nodes.push_back(src_strategies);
    // Only follows the given strategy when there is no other strategy to be
    // restored.
    if (!pretrimmed_strategy_map.contains(src_strategies->node_idx)) {
      strategies->following = src_strategies;
    }
    strategies->leaf_vector.reserve(src_strategies->leaf_vector.size());
    // Creates the sharding strategies and restores the trimmed strategies if
    // there is any.
    for (int64_t sid = 0;
         sid < src_strategies->leaf_vector.size() +
                   pretrimmed_strategy_map[src_strategies->node_idx].size();
         ++sid) {
      const HloSharding* output_spec;
      if (sid < src_strategies->leaf_vector.size()) {
        output_spec = &src_strategies->leaf_vector[sid].output_sharding;
      } else {
        output_spec =
            &pretrimmed_strategy_map[src_strategies->node_idx]
                                    [sid - src_strategies->leaf_vector.size()]
                                        .output_sharding;
        VLOG(1) << "Adding outspec from the trimmed strategy map: "
                << output_spec->ToString();
      }
      std::string name = ToStringSimple(*output_spec);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost =
          have_memory_cost ? GetBytes(shape) / output_spec->NumTiles() : 0;
      auto resharding_costs = ReshardingCostVector(src_strategies, shape,
                                                   *output_spec, cluster_env);
      strategies->leaf_vector.push_back(
          ShardingStrategy({name,
                            *output_spec,
                            compute_cost,
                            communication_cost,
                            memory_cost,
                            {std::move(resharding_costs)},
                            {*output_spec}}));
    }
  }
  return strategies;
}

std::unique_ptr<StrategyVector> FollowReduceStrategy(
    const HloInstruction* ins, const Shape& output_shape,
    const HloInstruction* operand, const HloInstruction* unit,
    size_t instruction_id, StrategyMap& strategy_map,
    LeafStrategies& leaf_strategies, const ClusterEnvironment& cluster_env,
    bool allow_mixed_mesh_shape) {
  std::unique_ptr<StrategyVector> strategies;
  if (output_shape.IsTuple()) {
    strategies = CreateTupleStrategyVector(instruction_id);
    strategies->childs.reserve(ins->shape().tuple_shapes_size());
    for (size_t i = 0; i < ins->shape().tuple_shapes_size(); ++i) {
      strategies->childs.push_back(FollowReduceStrategy(
          ins, ins->shape().tuple_shapes().at(i), ins->operand(i),
          ins->operand(i + ins->shape().tuple_shapes_size()), instruction_id,
          strategy_map, leaf_strategies, cluster_env, allow_mixed_mesh_shape));
    }
  } else if (output_shape.IsArray()) {
    strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                          leaf_strategies);
    const StrategyVector* src_strategies = strategy_map.at(operand).get();
    strategies->following = src_strategies;
    strategies->leaf_vector.reserve(src_strategies->leaf_vector.size());
    // Map operand dims to inst dim
    // Example: f32[1,16]{1,0} reduce(f32[1,16,4096]{2,1,0} %param0, f32[]
    // %param1), dimensions={2}
    // op_dim_to_output_dim = [0, 1, -1]
    std::vector<int64_t> op_dim_to_output_dim =
        GetDimensionMapping(/*reduced_dimensions=*/ins->dimensions(),
                            /*op_count*/ operand->shape().rank());
    CHECK_EQ(ins->dimensions().size() + output_shape.rank(),
             operand->shape().rank())
        << "Invalid kReduce: output size + reduced dimensions size != op count";
    // Follows the strategy of the operand.
    strategies->following = src_strategies;

    for (size_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
      HloSharding input_sharding =
          src_strategies->leaf_vector[sid].output_sharding;
      const auto& tensor_dim_to_mesh = cluster_env.GetTensorDimToMeshDimWrapper(
          operand->shape(), input_sharding);
      std::vector<int64_t> all_reduce_dims;
      for (int64_t op_dim = 0; op_dim < operand->shape().rank(); ++op_dim) {
        int64_t mesh_dim = tensor_dim_to_mesh[op_dim];
        // Replicates on this mesh dim.
        if (mesh_dim == -1) {
          continue;
        }
        if (op_dim_to_output_dim[op_dim] == -1) {
          // Reduce on a split dim. Require an allreduce
          all_reduce_dims.push_back(mesh_dim);
        }
      }
      std::unique_ptr<HloInstruction> operand_clone = operand->Clone();
      std::unique_ptr<HloInstruction> unit_clone = unit->Clone();
      // Creates a new reduce op with one output, which is easier to use
      // GetShardingFromUser() to get the input sharding.
      auto new_reduce = HloInstruction::CreateReduce(
          output_shape, operand_clone.get(), unit_clone.get(),
          ins->dimensions(), ins->to_apply());
      operand_clone->set_sharding(
          src_strategies->leaf_vector[sid].output_sharding);
      auto s = new_reduce->ReplaceOperandWith(0, operand_clone.get());
      if (!s.ok()) {
        continue;
      }
      ShardingPropagation::ComputationMap computation_map;
      bool changed =
          InferReduceShardingFromOperand(new_reduce.get(), false, true);
      CHECK(changed);
      HloSharding output_spec = new_reduce->sharding();
      new_reduce.reset();
      operand_clone.reset();
      unit_clone.reset();

      std::string name = ToStringSimple(output_spec);

      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(output_shape) / output_spec.NumTiles();
      for (auto mesh_dim : all_reduce_dims) {
        communication_cost += cluster_env.AllReduceCost(memory_cost, mesh_dim);
      }
      std::vector<std::vector<double>> resharding_costs;
      for (int64_t k = 0; k < ins->operand_count(); ++k) {
        auto cur_operand = ins->operand(k);
        if (ToString(cur_operand->shape().dimensions()) ==
            ToString(operand->shape().dimensions())) {
          auto operand_strategies = strategy_map.at(cur_operand).get();
          resharding_costs.push_back(ReshardingCostVector(
              operand_strategies, output_shape, input_sharding, cluster_env));
        } else {
          resharding_costs.push_back(std::vector<double>(
              strategy_map.at(cur_operand)->leaf_vector.size(), 0.0));
        }
      }
      const ShardingStrategy strategy = ShardingStrategy({name,
                                                          output_spec,
                                                          compute_cost,
                                                          communication_cost,
                                                          memory_cost,
                                                          resharding_costs,
                                                          {input_sharding}});
      strategies->leaf_vector.push_back(strategy);
    }
  } else {
    LOG(FATAL) << "Unhandled kReduce shape: " << ins->shape().ToString();
  }
  return strategies;
}

std::vector<size_t> FindReplicateStrategyIndices(
    const std::vector<ShardingStrategy>& strategies) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < strategies.size(); i++) {
    if (strategies.at(i).output_sharding.IsReplicated()) {
      indices.push_back(i);
    }
  }
  return indices;
}

std::pair<std::vector<std::vector<double>>,
          std::vector<std::optional<HloSharding>>>
ReshardingCostsForTupleOperand(const HloInstruction* operand,
                               StrategyVector* operand_strategy_vector) {
  // TODO(yuemmawang) Support instructions with more than one tuple operand.
  // Creates resharding costs such that favors when operand strategies are
  // replicated.
  std::vector<std::vector<double>> resharding_costs;
  std::vector<HloSharding> tuple_element_shardings;
  for (size_t tuple_element_idx = 0;
       tuple_element_idx < operand->shape().tuple_shapes_size();
       tuple_element_idx++) {
    auto tuple_element_strategies =
        operand_strategy_vector->childs.at(tuple_element_idx).get();
    std::vector<size_t> indices =
        FindReplicateStrategyIndices(tuple_element_strategies->leaf_vector);
    CHECK_GT(indices.size(), 0)
        << "There is no replicated strategy in instruction "
        << operand->ToString() << ".\nStrategies:\n"
        << tuple_element_strategies->ToString();
    resharding_costs.push_back(std::vector<double>(
        tuple_element_strategies->leaf_vector.size(), kInfinityCost));
    tuple_element_shardings.push_back(HloSharding::Replicate());
    for (const size_t i : indices) {
      resharding_costs.back().at(i) = 0.0;
    }
  }
  return std::make_pair(
      resharding_costs,
      std::vector<std::optional<HloSharding>>(
          {HloSharding::Tuple(operand->shape(), tuple_element_shardings)}));
}

std::vector<std::vector<double>> CreateZeroReshardingCostsForAllOperands(
    const HloInstruction* ins, const StrategyMap& strategy_map) {
  std::vector<std::vector<double>> resharding_costs;
  for (size_t i = 0; i < ins->operand_count(); ++i) {
    auto operand = ins->operand(i);
    const auto& operand_strategies = strategy_map.at(operand);
    if (operand->shape().IsTuple()) {
      if (ins->opcode() == HloOpcode::kConditional ||
          ins->opcode() == HloOpcode::kOutfeed) {
        resharding_costs.push_back(std::vector<double>(1, 0));
      } else {
        CHECK_EQ(ins->operand_count(), 0)
            << "Do not support instructions with more than one tuple "
               "operand.";
        for (size_t tuple_element_idx = 0;
             tuple_element_idx < operand->shape().tuple_shapes_size();
             tuple_element_idx++) {
          auto tuple_element_strategies =
              operand_strategies->childs.at(tuple_element_idx).get();
          resharding_costs.push_back(std::vector<double>(
              tuple_element_strategies->leaf_vector.size(), 0));
        }
      }
    } else {
      resharding_costs.push_back(
          std::vector<double>(operand_strategies->leaf_vector.size(), 0));
    }
  }
  return resharding_costs;
}

void GenerateOutfeedStrategy(const HloInstruction* ins, const Shape& shape,
                             const ClusterEnvironment& cluster_env,
                             const StrategyMap& strategy_map,
                             std::unique_ptr<StrategyVector>& strategies,
                             double replicated_penalty) {
  HloSharding output_spec = HloSharding::Replicate();
  std::vector<std::vector<double>> resharding_costs;
  std::vector<std::optional<HloSharding>> input_shardings;

  int tuple_size = ins->operand(0)->shape().tuple_shapes_size();
  if (ins->has_sharding()) {
    std::vector<Shape> operand_shapes(ins->operand_count());
    for (int i = 0; i < ins->operand_count(); ++i) {
      operand_shapes[i] = ins->operand(i)->shape();
    }
    auto all_operands_tuple_shape = ShapeUtil::MakeTupleShape(operand_shapes);
    auto get_input_sharding = [&](int index) {
      auto sharding = ins->sharding();
      if (sharding.IsTuple()) {
        return (index >= 0)
                   ? sharding.GetSubSharding(all_operands_tuple_shape,
                                             {0, static_cast<int64_t>(index)})
                   : sharding.GetSubSharding(all_operands_tuple_shape, {1});
      } else {
        return sharding;
      }
    };

    for (size_t i = 0; i < tuple_size; ++i) {
      auto input_sharding = get_input_sharding(i);
      input_shardings.push_back(input_sharding);
      resharding_costs.push_back(ReshardingCostVector(
          strategy_map.at(ins->operand(0))->childs[i].get(),
          ins->operand(0)->shape().tuple_shapes(i), input_sharding,
          cluster_env));
    }
    auto input_sharding = get_input_sharding(-1);
    input_shardings.push_back(input_sharding);
  } else {
    for (size_t i = 0; i < tuple_size; ++i) {
      resharding_costs.push_back(std::vector<double>(
          strategy_map.at(ins->operand(0))->childs[i].get()->leaf_vector.size(),
          0));
    }
  }
  resharding_costs.push_back({});
  double memory_cost = GetBytes(shape) / output_spec.NumTiles();
  strategies->leaf_vector.push_back(ShardingStrategy(
      {"R", HloSharding::Replicate(), replicated_penalty, 0, memory_cost,
       std::move(resharding_costs), input_shardings}));
}

double ComputeCommunicationCost(
    const HloInstruction* ins,
    const std::vector<std::optional<HloSharding>>& operand_shardings,
    const ClusterEnvironment& cluster_env) {
  switch (ins->opcode()) {
    case HloOpcode::kGather: {
      if (operand_shardings[0].has_value() &&
          !operand_shardings[0]->IsReplicated()) {
        auto mesh_shape = cluster_env.device_mesh_.dimensions();
        auto mesh_dim = std::distance(
            mesh_shape.begin(),
            std::max_element(mesh_shape.begin(), mesh_shape.end()));
        // As seen in the test
        // SpmdPartitioningTest.GatherPartitionedOnTrivialSliceDims (in file
        // third_party/tensorflow/compiler/xla/service/spmd/spmd_partitioner_test.cc),
        // when the gather op is replicated, and the first operand sharded, we
        // need an AllReduce to implement the gather op. We capture that cost
        // here.
        // TODO(pratikf) Model gather communication costs in a more principled
        // and exhaustive manner.
        return cluster_env.AllReduceCost(GetBytes(ins->shape()), mesh_dim);
      }
      return 0;
    }
    default:
      LOG(FATAL) << "Unhandled instruction " << ins->ToString();
  }
}

// Add "Replicate()" strategy
// By default, when adding a replicated strategy for an op, we specify that all
// its operands need to be replicated as well (via the input_shardings field on
// a ShardingStrategy). When operands_to_consider_all_strategies_for is
// non-empty however, instead of merely allowing the operands to be replicated,
// we allos greater freedom for the shardings of the operands included in the
// set. More specifically, for these operands, we consider all generated
// strategies for those operands (instead of just replication) as potentially
// allowable shardings.
void AddReplicatedStrategy(
    const HloInstruction* ins, const Shape& shape,
    const ClusterEnvironment& cluster_env, const StrategyMap& strategy_map,
    std::unique_ptr<StrategyVector>& strategies, double replicated_penalty,
    absl::flat_hash_set<int64_t> operands_to_consider_all_strategies_for = {}) {
  HloSharding replicated_strategy = HloSharding::Replicate();
  HloSharding output_spec = replicated_strategy;
  double memory_cost = GetBytes(shape) / output_spec.NumTiles();

  CHECK_LE(operands_to_consider_all_strategies_for.size(), 1);
  if (!operands_to_consider_all_strategies_for.empty()) {
    int64_t operand_to_consider_all_strategies_for =
        *operands_to_consider_all_strategies_for.begin();
    auto operand = ins->operand(operand_to_consider_all_strategies_for);
    CHECK(!operand->shape().IsTuple());
    auto operand_strategies_to_consider = strategy_map.at(operand).get();
    std::vector<std::vector<std::optional<HloSharding>>>
        possible_input_shardings(
            operand_strategies_to_consider->leaf_vector.size(),
            std::vector<std::optional<HloSharding>>(ins->operand_count()));
    std::vector<std::vector<std::vector<double>>> possible_resharding_costs(
        operand_strategies_to_consider->leaf_vector.size(),
        std::vector<std::vector<double>>(ins->operand_count()));

    for (int64_t k = 0; k < ins->operand_count(); ++k) {
      CHECK(!ins->operand(k)->shape().IsTuple());
      if (k == operand_to_consider_all_strategies_for) {
        CHECK_EQ(possible_input_shardings.size(),
                 operand_strategies_to_consider->leaf_vector.size());
        for (size_t j = 0; j < possible_input_shardings.size(); ++j) {
          possible_input_shardings[j][k] =
              operand_strategies_to_consider->leaf_vector[j].output_sharding;
          possible_resharding_costs[j][k] = ReshardingCostVector(
              strategy_map.at(ins->operand(k)).get(), ins->operand(k)->shape(),
              operand_strategies_to_consider->leaf_vector[j].output_sharding,
              cluster_env);
        }
      } else {
        for (size_t j = 0; j < possible_input_shardings.size(); ++j) {
          possible_input_shardings[j][k] = replicated_strategy;
          possible_resharding_costs[j][k] = ReshardingCostVector(
              strategy_map.at(ins->operand(k)).get(), ins->operand(k)->shape(),
              replicated_strategy, cluster_env);
        }
      }
    }

    for (size_t j = 0; j < possible_input_shardings.size(); ++j) {
      double communication_cost = ComputeCommunicationCost(
          ins, possible_input_shardings[j], cluster_env);
      strategies->leaf_vector.push_back(ShardingStrategy(
          {"R", replicated_strategy, replicated_penalty, communication_cost,
           memory_cost, std::move(possible_resharding_costs[j]),
           std::move(possible_input_shardings[j])}));
    }
  } else {
    std::vector<std::vector<double>> resharding_costs;
    std::vector<std::optional<HloSharding>> input_shardings;

    if (ins->operand_count() > 0 && ins->operand(0)->shape().IsTuple()) {
      CHECK_EQ(ins->operand_count(), 1)
          << "Do not support instructions with more than one tuple "
             "operand. If this CHECK fails, we will need to fix "
             "b/233412625.";
      std::tie(resharding_costs, input_shardings) =
          ReshardingCostsForTupleOperand(
              ins->operand(0), strategy_map.at(ins->operand(0)).get());
    } else {
      for (int64_t k = 0; k < ins->operand_count(); ++k) {
        auto operand = ins->operand(k);
        if (ins->opcode() == HloOpcode::kConditional) {
          resharding_costs.push_back(std::vector<double>(
              strategy_map.at(operand)->leaf_vector.size(), 0));
        } else {
          resharding_costs.push_back(ReshardingCostVector(
              strategy_map.at(operand).get(), ins->operand(k)->shape(),
              output_spec, cluster_env));
          input_shardings.push_back(output_spec);
        }
      }
    }
    strategies->leaf_vector.push_back(ShardingStrategy(
        {"R", HloSharding::Replicate(), replicated_penalty, 0, memory_cost,
         std::move(resharding_costs), input_shardings}));
  }
}

// TODO(pratikf) Communication costs for sort HLO ops. This is currently a
// placeholder approximation and should be improved.
double ComputeSortCommunicationCost(int64_t sort_dim,
                                    int64_t operand_sharded_dim,
                                    int64_t mesh_sharding_dim,
                                    const Shape& shape,
                                    const ClusterEnvironment& cluster_env) {
  if (sort_dim == operand_sharded_dim) {
    return cluster_env.AllToAllCost(GetBytes(shape), mesh_sharding_dim);
  }
  return 0;
}

// Enumerate all 1d partition strategies.
void EnumerateAll1DPartition(const HloInstruction* ins, const Shape& shape,
                             const Array<int64_t>& device_mesh,
                             const ClusterEnvironment& cluster_env,
                             const StrategyMap& strategy_map,
                             std::unique_ptr<StrategyVector>& strategies,
                             bool only_allow_divisible,
                             const std::string& suffix,
                             const CallGraph& call_graph) {
  for (int64_t i = 0; i < shape.rank(); ++i) {
    for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
      if (device_mesh.dim(j) == 1 || shape.dimensions(i) < device_mesh.dim(j) ||
          (only_allow_divisible &&
           !IsDivisible(shape.dimensions(i), device_mesh.dim(j)))) {
        continue;
      }

      std::string name = absl::StrFormat("S%d @ %d", i, j) + suffix;
      HloSharding output_spec = Tile(shape, {i}, {j}, device_mesh);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(shape) / output_spec.NumTiles();

      std::vector<std::vector<double>> resharding_costs;
      std::vector<std::optional<HloSharding>> input_shardings;
      if (ins->opcode() == HloOpcode::kConditional) {
        // TODO(pratikf): Compute input_shardings for kConditional ops
        resharding_costs =
            CreateZeroReshardingCostsForAllOperands(ins, strategy_map);
      } else if (ins->operand_count() > 0 &&
                 ins->operand(0)->shape().IsTuple()) {
        CHECK_EQ(ins->operand_count(), 1)
            << "Do not support instructions with more than one tuple "
               "operand.";
        std::tie(resharding_costs, input_shardings) =
            ReshardingCostsForTupleOperand(
                ins->operand(0), strategy_map.at(ins->operand(0)).get());
      } else if (ins->opcode() == HloOpcode::kRngBitGenerator &&
                 ins->operand(0)->shape().IsArray()) {
        auto replicated_sharding = HloSharding::Replicate();
        input_shardings.push_back(HloSharding::SingleTuple(
            ins->operand(0)->shape(), replicated_sharding));
        resharding_costs =
            GenerateReshardingCostsAndMissingShardingsForAllOperands(
                ins, output_spec, strategy_map, cluster_env, call_graph,
                input_shardings);
      } else {
        std::tie(resharding_costs, input_shardings) =
            GenerateReshardingCostsAndShardingsForAllOperands(
                ins, output_spec, strategy_map, cluster_env, call_graph);
      }
      if (ins->opcode() == HloOpcode::kSort) {
        auto sort_ins = xla::DynCast<HloSortInstruction>(ins);
        CHECK(sort_ins);
        communication_cost = ComputeSortCommunicationCost(
            sort_ins->sort_dimension(), i, j, shape, cluster_env);
      } else if (IsTopKCustomCall(ins)) {
        // TODO(pratikf) Better model topk communication costs. Currently using
        // the cost model for sort (which, as noted above in the comments for
        // the function) is also an approximation.
        communication_cost = ComputeSortCommunicationCost(
            ins->operand(0)->shape().rank() - 1, i, j, shape, cluster_env);
      }
      strategies->leaf_vector.push_back(ShardingStrategy(
          {name, output_spec, compute_cost, communication_cost, memory_cost,
           std::move(resharding_costs), input_shardings}));
    }
  }
}

// Enumerate 2D partition
void EnumerateAll2DPartition(const HloInstruction* ins, const Shape& shape,
                             const Array<int64_t>& device_mesh,
                             const ClusterEnvironment& cluster_env,
                             const StrategyMap& strategy_map,
                             std::unique_ptr<StrategyVector>& strategies,
                             const InstructionBatchDimMap& batch_dim_map,
                             bool only_allow_divisible,
                             const CallGraph& call_graph) {
  auto iter = batch_dim_map.find(GetBatchDimMapKey(ins));
  int64_t batch_dim = -1;
  if (iter != batch_dim_map.end()) {
    batch_dim = iter->second;
  }
  // Fully tile the buffer to 2-d mesh
  for (int64_t i = 0; i < shape.rank(); ++i) {
    for (int64_t j = 0; j < shape.rank(); ++j) {
      if ((batch_dim != -1 && !(batch_dim == i || batch_dim == j)) || i == j) {
        continue;
      }
      if (shape.dimensions(i) < device_mesh.dim(0) ||
          shape.dimensions(j) < device_mesh.dim(1)) {
        continue;
      }

      if (only_allow_divisible &&
          (!IsDivisible(shape.dimensions(i), device_mesh.dim(0)) ||
           !IsDivisible(shape.dimensions(j), device_mesh.dim(1)))) {
        continue;
      }

      std::string name = absl::StrFormat("S{%d,%d} @ {0,1}", i, j);
      HloSharding output_spec = Tile(shape, {i, j}, {0, 1}, device_mesh);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(shape) / output_spec.NumTiles();
      std::vector<std::optional<HloSharding>> input_shardings;
      std::vector<std::vector<double>> resharding_costs;
      if (ins->opcode() == HloOpcode::kConditional) {
        // TODO(pratikf): Compute input_shardings for kConditional ops
        resharding_costs =
            CreateZeroReshardingCostsForAllOperands(ins, strategy_map);
      } else if (ins->operand_count() > 0 &&
                 ins->operand(0)->shape().IsTuple()) {
        CHECK_EQ(ins->operand_count(), 1)
            << "Do not support instructions with more than one tuple "
               "operand. If this CHECK fails, we will need to fix "
               "b/233412625.";
        std::tie(resharding_costs, input_shardings) =
            ReshardingCostsForTupleOperand(
                ins->operand(0), strategy_map.at(ins->operand(0)).get());
        LOG(INFO) << absl::StrJoin(resharding_costs.back(), ",");
      } else {
        std::tie(resharding_costs, input_shardings) =
            GenerateReshardingCostsAndShardingsForAllOperands(
                ins, output_spec, strategy_map, cluster_env, call_graph);
      }
      // TODO(pratikf) Communication costs for sort HLO ops. This is currently a
      // placeholder approximation and should be improved.
      if (ins->opcode() == HloOpcode::kSort) {
        auto sort_ins = xla::DynCast<HloSortInstruction>(ins);
        CHECK(sort_ins);

        if (sort_ins->sort_dimension() == i) {
          communication_cost = ComputeSortCommunicationCost(
              sort_ins->sort_dimension(), i, 0, shape, cluster_env);
        } else if (sort_ins->sort_dimension() == j) {
          communication_cost = ComputeSortCommunicationCost(
              sort_ins->sort_dimension(), j, 1, shape, cluster_env);
        }
      } else if (IsTopKCustomCall(ins)) {
        auto topk_dim = ins->operand(0)->shape().rank() - 1;
        if (topk_dim == i) {
          communication_cost =
              ComputeSortCommunicationCost(topk_dim, i, 0, shape, cluster_env);
        } else if (topk_dim == j) {
          communication_cost =
              ComputeSortCommunicationCost(topk_dim, j, 1, shape, cluster_env);
        }
      }
      strategies->leaf_vector.push_back(ShardingStrategy(
          {name, output_spec, compute_cost, communication_cost, memory_cost,
           std::move(resharding_costs), input_shardings}));
    }
  }
}

// Enumerate all 1d partition strategies for reshape.
void EnumerateAll1DPartitionReshape(const HloInstruction* ins,
                                    const Array<int64_t>& device_mesh,
                                    const ClusterEnvironment& cluster_env,
                                    const StrategyMap& strategy_map,
                                    std::unique_ptr<StrategyVector>& strategies,
                                    bool only_allow_divisible,
                                    const std::string& suffix) {
  const HloInstruction* operand = ins->operand(0);

  for (int64_t i = 0; i < ins->shape().rank(); ++i) {
    for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
      if (device_mesh.dim(j) == 1 ||
          (only_allow_divisible &&
           !IsDivisible(ins->shape().dimensions(i), device_mesh.dim(j)))) {
        continue;
      }
      HloSharding output_spec = Tile(ins->shape(), {i}, {j}, device_mesh);

      std::optional<HloSharding> input_spec =
          hlo_sharding_util::ReshapeSharding(ins->shape(), operand->shape(),
                                             output_spec);
      if (!input_spec.has_value()) {  // invalid reshape
        continue;
      }

      if (cluster_env.IsDeviceMesh1D() &&
          VectorGreaterThanOneElementCount(
              input_spec->tile_assignment().dimensions()) > 1) {
        continue;
      }

      std::string name = absl::StrFormat("S%d @ %d", i, j) + suffix;
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

      std::vector<std::vector<double>> resharding_costs{
          ReshardingCostVector(strategy_map.at(operand).get(), operand->shape(),
                               *input_spec, cluster_env)};
      strategies->leaf_vector.push_back(
          ShardingStrategy({name,
                            output_spec,
                            compute_cost,
                            communication_cost,
                            memory_cost,
                            std::move(resharding_costs),
                            {*input_spec}}));
    }
  }
}

// Enumerate 2D partition for reshape. Batch dim is always partitioned.
void Enumerate2DPartitionReshape(const HloInstruction* ins,
                                 const Array<int64_t>& device_mesh,
                                 const ClusterEnvironment& cluster_env,
                                 const StrategyMap& strategy_map,
                                 const InstructionBatchDimMap& batch_dim_map,
                                 std::unique_ptr<StrategyVector>& strategies,
                                 bool only_allow_divisible) {
  auto iter = batch_dim_map.find(GetBatchDimMapKey(ins));
  int64_t batch_dim = -1;
  if (iter != batch_dim_map.end()) {
    batch_dim = iter->second;
  }

  const HloInstruction* operand = ins->operand(0);

  // Split batch dim + another dim
  for (int64_t i = 0; i < ins->shape().rank(); ++i) {
    for (int64_t j = 0; j < ins->shape().rank(); ++j) {
      if ((batch_dim != -1 && !(batch_dim == i || batch_dim == j)) || i == j) {
        continue;
      }
      if (ins->shape().dimensions(i) < device_mesh.dim(0) ||
          ins->shape().dimensions(j) < device_mesh.dim(1)) {
        continue;
      }
      if (only_allow_divisible &&
          (!IsDivisible(ins->shape().dimensions(i), device_mesh.dim(0)) ||
           !IsDivisible(ins->shape().dimensions(j), device_mesh.dim(1)))) {
        continue;
      }

      HloSharding output_spec = Tile(ins->shape(), {i, j}, {0, 1}, device_mesh);
      std::optional<HloSharding> input_spec =
          hlo_sharding_util::ReshapeSharding(ins->shape(), operand->shape(),
                                             output_spec);
      if (!input_spec.has_value()) {  // invalid reshape
        continue;
      }

      std::string name = absl::StrFormat("S%d%d @ {%d,%d}", i, j, 0, 1);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

      std::vector<std::vector<double>> resharding_costs{
          ReshardingCostVector(strategy_map.at(operand).get(), operand->shape(),
                               *input_spec, cluster_env)};
      strategies->leaf_vector.push_back(
          ShardingStrategy({name,
                            output_spec,
                            compute_cost,
                            communication_cost,
                            memory_cost,
                            std::move(resharding_costs),
                            {*input_spec}}));
    }
  }
}

// Return the maximum number of tiles among all strategies of an instruction.
int64_t MaxNumTiles(const StrategyMap& strategy_map,
                    const HloInstruction* ins) {
  const StrategyVector* strategies = strategy_map.at(ins).get();
  // TODO(zhuohan): optimize with path compression.
  while (strategies->following != nullptr) {
    strategies = strategies->following;
  }
  int64_t max_num_tiles = -1;
  for (size_t i = 0; i < strategies->leaf_vector.size(); ++i) {
    max_num_tiles = std::max(
        max_num_tiles, strategies->leaf_vector[i].output_sharding.NumTiles());
  }

  return max_num_tiles;
}

// Choose an operand to follow.  We choose to follow the operand with the
// highest priority.  The priority is defined as a function of two entities as
// below:
//
// priority(operand) =
//   max(x.output_spec.num_tiles for x in operand.strategies) +
//   depth(operand) * depth_normalizer
//
// For example, We therefore prefer one operand with strategies with a high
// number of tiles and operands that have a higher depth in the graph. When the
// priorities are similar (within range_delta), we set these operators to be
// "tied" and let the ILP solver to pick which one to follow.
//
// The function returns (follow_idx, tie), where the follow_idx is the id of
// the operand to follow and tie is a boolean variable that indicates whether
// there are multiple operands have similar priority. Return `tie == True` if
// there are two operands with very close priorities and we cannot decide which
// one to follow.
std::pair<int64_t, bool> ChooseOperandToFollow(
    const StrategyMap& strategy_map, const InstructionDepthMap& depth_map,
    const AliasMap& alias_map, int64_t max_depth, const HloInstruction* ins) {
  // If an alias constraint is set, always follow its alias source.
  auto it = alias_map.find(ins);
  if (it != alias_map.end()) {
    for (int64_t i = 0; i < ins->operand_count(); ++i) {
      const HloInstruction* operand = ins->operand(i);
      if (operand == it->second) {
        return std::make_pair(i, false);
      }
    }
  }

  std::optional<int64_t> follow_idx;
  bool tie = false;
  double max_priority = -1e20;
  double depth_normalizer = 0.1 / max_depth;
  double range_delta = 4 * depth_normalizer;

  for (int64_t i = 0; i < ins->operand_count(); ++i) {
    const HloInstruction* operand = ins->operand(i);
    double priority = MaxNumTiles(strategy_map, operand) +
                      depth_map.at(operand) * depth_normalizer;
    if (priority > max_priority + range_delta) {
      follow_idx = i;
      tie = false;
      max_priority = priority;
    } else if (priority >= max_priority - range_delta) {
      tie = true;
    }
  }
  CHECK(follow_idx.has_value());

  return std::make_pair(*follow_idx, tie);
}

// Return whether an instruciton can follow one of its operand when
// more than one operand have the same priority.
// Consider adding special cases here if the auto sharding following strategy
// behaves weird for your model.
bool AllowTieFollowing(const HloInstruction* ins) {
  if (ins->opcode() == HloOpcode::kCompare ||
      ins->opcode() == HloOpcode::kAnd) {
    // This is used to resolve tricky cases in our testing models where an iota
    // and a parameter has the same priority when comparing to each other. This
    // happens for embedding, onehot or make_attention_mask.
    return false;
  }
  if (ins->operand_count() == 3) {
    return false;
  }
  return true;
}

// 1. Disable mixed mesh shape if the batch dim is not divisible by the
// number of devices.
// 2. Disable force_batch_dim_to_mesh_dim if the batch dim is 1. In this case,
// the batch dim analysis can be wrong because the batch dim might be dropped.
void DisableIncompatibleMixedMeshShapeAndForceBatchDim(
    const InstructionBatchDimMap& batch_dim_map,
    const std::vector<HloInstruction*>& instructions, int num_devices,
    AutoShardingSolverOption& solver_option) {
  int64_t batch_size = INT_MAX;
  for (const auto& iter : batch_dim_map) {
    batch_size = std::min(batch_size, FindInstruction(instructions, iter.first)
                                          ->shape()
                                          .dimensions(iter.second));
  }

  if (IsDivisible(batch_size, num_devices)) {
    if (solver_option.allow_mixed_mesh_shape) {
      solver_option.allow_mixed_mesh_shape = false;
      LOG(WARNING)
          << "Mixed mesh shape is disabled due to indivisible batch size.";
    }
  }

  if (batch_size == 1) {
    solver_option.force_batch_dim_to_mesh_dim = -1;
  }
}

StatusOr<std::unique_ptr<StrategyVector>> CreateAllStrategiesVector(
    const HloInstruction* ins, const Shape& shape, size_t instruction_id,
    LeafStrategies& leaf_strategies, const ClusterEnvironment& cluster_env,
    const StrategyMap& strategy_map,
    const AutoShardingSolverOption& solver_option, double replicated_penalty,
    const InstructionBatchDimMap& batch_dim_map, const CallGraph& call_graph,
    bool only_allow_divisible, bool create_replicated_strategies) {
  std::unique_ptr<StrategyVector> strategies;
  if (shape.IsTuple()) {
    strategies = CreateTupleStrategyVector(instruction_id);
    strategies->childs.reserve(shape.tuple_shapes_size());
    for (size_t i = 0; i < shape.tuple_shapes_size(); ++i) {
      strategies->childs.push_back(
          CreateAllStrategiesVector(
              ins, shape.tuple_shapes().at(i), instruction_id, leaf_strategies,
              cluster_env, strategy_map, solver_option, replicated_penalty,
              batch_dim_map, call_graph, only_allow_divisible,
              create_replicated_strategies)
              .value());
    }
  } else if (shape.IsArray()) {
    strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                          leaf_strategies);
    EnumerateAll1DPartition(ins, shape, cluster_env.device_mesh_, cluster_env,
                            strategy_map, strategies, only_allow_divisible, "",
                            call_graph);
    // Split 2 dims
    if (cluster_env.IsDeviceMesh2D()) {
      // NOTE(zhuohan): In full alpa, we only include 2D partition strategy
      //                for operators with batch dimension. We didn't include
      //                this logic here since this pass might be used for
      //                more general cases.
      EnumerateAll2DPartition(ins, shape, cluster_env.device_mesh_, cluster_env,
                              strategy_map, strategies, batch_dim_map,
                              only_allow_divisible, call_graph);
    }

    if (solver_option.allow_mixed_mesh_shape && cluster_env.IsDeviceMesh2D()) {
      // Set penalty for 1d partial tiled layout
      for (size_t i = 0; i < strategies->leaf_vector.size(); ++i) {
        strategies->leaf_vector[i].compute_cost += replicated_penalty * 0.8;
      }

      // Split 1 dim, but for 1d mesh
      EnumerateAll1DPartition(ins, shape, cluster_env.device_mesh_1d_,
                              cluster_env, strategy_map, strategies,
                              only_allow_divisible, " 1d", call_graph);
    }
    if (create_replicated_strategies || strategies->leaf_vector.empty()) {
      AddReplicatedStrategy(ins, shape, cluster_env, strategy_map, strategies,
                            replicated_penalty);
    }

    // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
    // and only keep the data parallel strategies.
    if (solver_option.force_batch_dim_to_mesh_dim >= 0 &&
        batch_dim_map.contains(GetBatchDimMapKey(ins))) {
      TF_RETURN_IF_ERROR(FilterStrategy(ins, shape, strategies, cluster_env,
                                        batch_dim_map, solver_option));
    }
  } else if (shape.IsToken()) {
    strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                          leaf_strategies);
    AddReplicatedStrategy(ins, shape, cluster_env, strategy_map, strategies,
                          replicated_penalty);
  } else {
    LOG(FATAL) << "Unsupported instruction shape: " << shape.DebugString();
  }
  return strategies;
}

StatusOr<std::unique_ptr<StrategyVector>> CreateParameterStrategyVector(
    const HloInstruction* ins, const Shape& shape, size_t instruction_id,
    LeafStrategies& leaf_strategies, const ClusterEnvironment& cluster_env,
    const StrategyMap& strategy_map,
    const AutoShardingSolverOption& solver_option, double replicated_penalty,
    const InstructionBatchDimMap& batch_dim_map, const CallGraph& call_graph,
    bool only_allow_divisible) {
  return CreateAllStrategiesVector(
      ins, shape, instruction_id, leaf_strategies, cluster_env, strategy_map,
      solver_option, replicated_penalty, batch_dim_map, call_graph,
      only_allow_divisible, solver_option.allow_replicated_parameters);
}

// The sharding is replicated or the total number of tiles is over or equal to
// the total number of devices. If returns true, this sharding is likely
// provided by users.
bool ShardingIsComplete(const HloSharding& sharding, size_t total_num_devices) {
  return sharding.TotalNumTiles() >= total_num_devices ||
         sharding.IsReplicated();
}

// Two shardings shard the same dimension of a given tensor.
bool ShardingIsConsistent(const HloSharding& partial_sharding,
                          const HloSharding& complete_sharding, bool strict) {
  if (partial_sharding.tile_assignment().num_dimensions() >
      complete_sharding.tile_assignment().num_dimensions()) {
    return false;
  }
  for (size_t i = 0; i < partial_sharding.tile_assignment().num_dimensions();
       ++i) {
    if (strict && partial_sharding.tile_assignment().dim(i) > 1 &&
        partial_sharding.tile_assignment().dim(i) ==
            complete_sharding.tile_assignment().dim(i)) {
      return true;
    }
    if (!strict && partial_sharding.tile_assignment().dim(i) > 1 &&
        complete_sharding.tile_assignment().dim(i) > 1) {
      return true;
    }
  }
  return false;
}

// Existing shardings refer to the HloSharding field in the given
// HloInstruction. This function handles two cases:
// 1. Existing sharding is from outside of XLA, which we refer to as user
// sharding. We need to preserve user shardings when the HloModule exit from
// AutoSharding.
// 2. Existing sharding is from previous iteration when
// solve_nd_sharding_iteratively is true. We use such shardings as hints to
// reduce the current iteration's problem size, by keeping sharding strategies
// that shard the same tensor dimensions as specified in the existing
// HloSharding.
// These two are distinguished by ShardingIsComplete().
void TrimOrGenerateStrategiesBasedOnExistingSharding(
    const Shape& output_shape, StrategyVector* strategies,
    const StrategyMap& strategy_map,
    const std::vector<HloInstruction*> instructions,
    const HloSharding& existing_sharding, const ClusterEnvironment& cluster_env,
    StableHashMap<int64_t, std::vector<ShardingStrategy>>&
        pretrimmed_strategy_map,
    const CallGraph& call_graph, bool strict) {
  if (strategies->is_tuple) {
    for (size_t i = 0; i < strategies->childs.size(); ++i) {
      TrimOrGenerateStrategiesBasedOnExistingSharding(
          output_shape.tuple_shapes(i), strategies->childs.at(i).get(),
          strategy_map, instructions, existing_sharding.tuple_elements().at(i),
          cluster_env, pretrimmed_strategy_map, call_graph, strict);
    }
  } else {
    if (ShardingIsComplete(existing_sharding,
                           cluster_env.device_mesh_.num_elements())) {
      // Sharding provided by XLA users, we need to keep them.
      strategies->following = nullptr;
      int32_t strategy_index = -1;
      for (size_t i = 0; i < strategies->leaf_vector.size(); i++) {
        if (strategies->leaf_vector[i].output_sharding == existing_sharding) {
          strategy_index = i;
        }
      }
      if (strategy_index >= 0) {
        VLOG(1) << "Keeping strategy index: " << strategy_index;
        // Stores other strategies in the map, removes them in the vector and
        // only keeps the one we found.
        ShardingStrategy found_strategy =
            strategies->leaf_vector[strategy_index];
        pretrimmed_strategy_map[strategies->node_idx] = strategies->leaf_vector;
        strategies->leaf_vector.clear();
        strategies->leaf_vector.push_back(found_strategy);
      } else {
        VLOG(1) << "Generate a new strategy based on user sharding.";
        std::string name = ToStringSimple(existing_sharding);
        std::vector<std::vector<double>> resharding_costs;
        std::vector<std::optional<HloSharding>> input_shardings;
        if (strategies->in_nodes.empty()) {
          resharding_costs = {};
        } else {
          HloInstruction* ins = instructions.at(strategies->instruction_id);
          for (size_t i = 0; i < strategies->in_nodes.size(); i++) {
            HloInstruction* operand =
                instructions.at(strategies->in_nodes.at(i)->instruction_id);
            std::optional<HloSharding> input_sharding_or =
                ShardingPropagation::GetShardingFromUser(*operand, *ins, 10,
                                                         true, call_graph);
            if (input_sharding_or.has_value()) {
              input_shardings.push_back(input_sharding_or.value());
            }

            StrategyVector* operand_strategies;
            Shape operand_shape;
            if (ins->opcode() == HloOpcode::kGetTupleElement) {
              operand_strategies =
                  strategy_map.at(operand)->childs[ins->tuple_index()].get();
              operand_shape = operand->shape().tuple_shapes(ins->tuple_index());
            } else {
              operand_strategies = strategy_map.at(operand).get();
              operand_shape = operand->shape();
            }
            resharding_costs.push_back(
                ReshardingCostVector(operand_strategies, operand_shape,
                                     existing_sharding, cluster_env));
          }
        }
        double memory_cost =
            GetBytes(output_shape) / existing_sharding.NumTiles();
        if (!strategies->leaf_vector.empty()) {
          pretrimmed_strategy_map[strategies->node_idx] =
              strategies->leaf_vector;
        }
        strategies->leaf_vector.clear();
        strategies->leaf_vector.push_back(
            ShardingStrategy({name, existing_sharding, 0, 0, memory_cost,
                              resharding_costs, input_shardings}));
      }
      CHECK_EQ(strategies->leaf_vector.size(), 1);
      // If there is only one option for resharding, and the cost computed for
      // that option is kInfinityCost, set the cost to zero. This is okay
      // because there is only one option anyway, and having the costs set to
      // kInfinityCost is problematic for the solver.
      for (auto& operand_resharding_costs :
           strategies->leaf_vector[0].resharding_costs) {
        if (operand_resharding_costs.size() == 1 &&
            operand_resharding_costs[0] >= kInfinityCost) {
          operand_resharding_costs[0] = 0;
        }
      }
    } else if (!strategies->following) {
      // If existing sharding is a partial sharding from previous iteration,
      // find the strategies that are 1D&&complete or align with user
      // sharding.
      // It is IMPORTANT that we do this only for instructions that do no follow
      // others, to keep the number of ILP variable small.
      std::vector<ShardingStrategy> new_vector;
      for (const auto& strategy : strategies->leaf_vector) {
        if (strategy.output_sharding.IsReplicated() ||
            ShardingIsConsistent(existing_sharding, strategy.output_sharding,
                                 strict) ||
            (VectorGreaterThanOneElementCount(
                 strategy.output_sharding.tile_assignment().dimensions()) ==
                 1 &&
             ShardingIsComplete(
                 strategy.output_sharding,
                 cluster_env.original_device_mesh_.num_elements()))) {
          new_vector.push_back(std::move(strategy));
        }
      }
      // If no sharding strategy left, just keep the original set, because we do
      // not have to strictly keep those shardings and the only purpose is to
      // reduce problem size for the last iteration.
      if (!new_vector.empty() &&
          new_vector.size() != strategies->leaf_vector.size()) {
        strategies->following = nullptr;
        strategies->leaf_vector = std::move(new_vector);
      }
    }
  }
}

void CheckMemoryCosts(StrategyVector* strategies, const Shape& shape) {
  if (strategies->is_tuple) {
    for (size_t i = 0; i < strategies->childs.size(); i++) {
      CheckMemoryCosts(strategies->childs[i].get(), shape.tuple_shapes().at(i));
    }
  } else {
    double full_mem = 0.0;
    for (const auto& strategy : strategies->leaf_vector) {
      if (strategy.output_sharding.IsReplicated()) {
        full_mem = strategy.memory_cost;
        size_t size = GetInstructionSize(shape);
        CHECK_EQ(strategy.memory_cost, size);
      }
    }
    for (const auto& strategy : strategies->leaf_vector) {
      if (!strategy.output_sharding.IsReplicated() && full_mem > 0.0) {
        CHECK_EQ(strategy.memory_cost * strategy.output_sharding.NumTiles(),
                 full_mem);
      }
    }
  }
}

void RemoveInvalidShardingsWithShapes(const Shape& shape,
                                      StrategyVector* strategies) {
  if (strategies->is_tuple) {
    for (size_t i = 0; i < strategies->childs.size(); i++) {
      RemoveInvalidShardingsWithShapes(shape.tuple_shapes().at(i),
                                       strategies->childs[i].get());
    }
  } else {
    std::vector<ShardingStrategy> new_vector;
    for (const auto& strategy : strategies->leaf_vector) {
      if (strategy.output_sharding.IsReplicated()) {
        new_vector.push_back(strategy);
        continue;
      }
      const auto& tile_assignment = strategy.output_sharding.tile_assignment();
      bool is_strategy_valid = true;
      for (int64_t i = 0; i < shape.rank(); ++i) {
        if (tile_assignment.dim(i) > 1 &&
            tile_assignment.dim(i) > shape.dimensions(i)) {
          VLOG(1) << "Removing invalid strategy: " << strategy.ToString();
          is_strategy_valid = false;
          break;
        }
      }
      if (is_strategy_valid) {
        new_vector.push_back(strategy);
      }
    }
    strategies->leaf_vector = std::move(new_vector);
  }
}

void CheckReshardingCostsShape(StrategyVector* strategies) {
  if (strategies->is_tuple) {
    for (size_t i = 0; i < strategies->childs.size(); i++) {
      CheckReshardingCostsShape(strategies->childs[i].get());
    }
  } else {
    for (const auto& strategy : strategies->leaf_vector) {
      if (strategies->in_nodes.size() == 1 &&
          strategies->in_nodes.at(0)->is_tuple) {
        // This is when current instruction's only operand is tuple, and the
        // first dimension of resharding_costs should equal its number of
        // tuple elements.
        CHECK_EQ(strategy.resharding_costs.size(),
                 strategies->in_nodes.at(0)->childs.size())
            << "Instruction ID: " << strategies->instruction_id << "\n"
            << strategies->ToString();
      } else {
        // The rest of the time, the first dimension of resharding_costs
        // should equal its number of operands (in_nodes).
        CHECK_EQ(strategy.resharding_costs.size(), strategies->in_nodes.size())
            << "Instruction ID: " << strategies->instruction_id << "\n"
            << strategies->ToString();
      }
      for (size_t i = 0; i < strategy.resharding_costs.size(); i++) {
        size_t to_compare;
        if (strategies->in_nodes.size() == 1 &&
            strategies->in_nodes.at(0)->is_tuple) {
          to_compare =
              strategies->in_nodes.at(0)->childs.at(i)->leaf_vector.size();
        } else if (strategies->is_tuple) {
          to_compare = strategies->in_nodes.at(i)->childs.size();
        } else {
          to_compare = strategies->in_nodes.at(i)->leaf_vector.size();
        }
        CHECK_EQ(strategy.resharding_costs[i].size(), to_compare)
            << "\nIndex of resharding_costs: " << i
            << "\nInstruction ID: " << strategies->instruction_id
            << "\nCurrent strategies:\n"
            << strategies->ToString();
      }
    }
  }
}

bool LeafVectorsAreConsistent(const std::vector<ShardingStrategy>& one,
                              const std::vector<ShardingStrategy>& two,
                              bool is_reshape) {
  if (one.size() != two.size()) {
    return false;
  }
  return true;
}

void ScaleCostsWithExecutionCounts(StrategyVector* strategies,
                                   int64_t execution_count) {
  if (strategies->is_tuple) {
    for (size_t i = 0; i < strategies->childs.size(); ++i) {
      ScaleCostsWithExecutionCounts(strategies->childs[i].get(),
                                    execution_count);
    }
  } else {
    for (auto& strategy : strategies->leaf_vector) {
      strategy.compute_cost *= execution_count;
      strategy.communication_cost *= execution_count;
      for (auto i = 0; i < strategy.resharding_costs.size(); ++i) {
        for (auto j = 0; j < strategy.resharding_costs[i].size(); ++j) {
          strategy.resharding_costs[i][j] *= execution_count;
        }
      }
    }
  }
}

// NOLINTBEGIN(readability/fn_size)
// TODO(zhuohan): Decompose this function into smaller pieces
// Build possible sharding strategies and their costs for all instructions.
StatusOr<std::tuple<StrategyMap, LeafStrategies, AssociativeDotPairs>>
BuildStrategyAndCost(const HloInstructionSequence& sequence,
                     const HloModule* module,
                     const absl::flat_hash_map<const HloInstruction*, int64_t>&
                         instruction_execution_counts,
                     const InstructionDepthMap& depth_map,
                     const InstructionBatchDimMap& batch_dim_map,
                     const AliasMap& alias_map,
                     const ClusterEnvironment& cluster_env,
                     AutoShardingSolverOption& solver_option,
                     const CallGraph& call_graph,
                     bool trying_multiple_mesh_shapes) {
  const Array<int64_t>& device_mesh = cluster_env.device_mesh_;
  const Array<int64_t>& device_mesh_1d = cluster_env.device_mesh_1d_;
  StrategyMap strategy_map;
  // This map stores all of the trimmed strategies due to user specified
  // sharding. The key is the instruction id, the value is the strategies. This
  // is useful when the operand is forced to use a user sharding, and the op
  // doesn't need to strictly follow it. We restore the trimmed strategies in
  // this situation.
  StableHashMap<int64_t, std::vector<ShardingStrategy>> pretrimmed_strategy_map;
  LeafStrategies leaf_strategies;
  AssociativeDotPairs associative_dot_pairs;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Count the non-one mesh dimension.
  int mesh_nn_dims = VectorGreaterThanOneElementCount(device_mesh.dimensions());

  // Add penalty for replicated tensors
  double replicated_penalty = std::round(cluster_env.AllReduceCost(1, 0) +
                                         cluster_env.AllReduceCost(1, 1));

  int64_t max_depth = -1;
  for (auto iter : depth_map) {
    max_depth = std::max(max_depth, iter.second);
  }

  // Register strategies and their costs for each instruction.
  for (size_t instruction_id = 0; instruction_id < instructions.size();
       ++instruction_id) {
    const HloInstruction* ins = instructions[instruction_id];
    VLOG(2) << "instruction_id = " << instruction_id << ": " << ins->ToString();
    std::unique_ptr<StrategyVector> strategies;

    HloOpcode opcode = ins->opcode();

    bool only_allow_divisible;
    if (IsEntryComputationInputOrOutput(module, ins)) {
      // With IsEntryComputationInputOrOutput(module, ins) == true, entry
      // computation's root instruction may still be unevenly sharded because it
      // usually "follows" other instruction's sharding. If the instruction it
      // follows is an intermediate instruction, it may be able to choose
      // unevenly sharded strategiyes. Usually if we constraint input's sharding
      // strategies, outputs would be constrained as welll, but if outputs are
      // still unevely sharded in some cases, we need to fix the implementation
      // in auto sharding.
      only_allow_divisible = solver_option.only_allow_divisible_input_output;
    } else {
      only_allow_divisible = solver_option.only_allow_divisible_intermediate;
    }
    switch (opcode) {
      case HloOpcode::kParameter:
      case HloOpcode::kRngBitGenerator:
      case HloOpcode::kRng: {
        strategies =
            CreateParameterStrategyVector(
                ins, ins->shape(), instruction_id, leaf_strategies, cluster_env,
                strategy_map, solver_option, replicated_penalty, batch_dim_map,
                call_graph, only_allow_divisible)
                .value();
        break;
      }
      case HloOpcode::kConstant: {
        strategies = CreateLeafStrategyVectorWithoutInNodes(instruction_id,
                                                            leaf_strategies);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategies, 0);
        break;
      }
      case HloOpcode::kScatter: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        // We follow the first operand (the array we're scattering into)
        auto src_strategies = strategy_map.at(ins->operand(0)).get();
        CHECK(!src_strategies->is_tuple);
        for (int64_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          HloSharding output_spec =
              src_strategies->leaf_vector[sid].output_sharding;
          std::string name = ToStringSimple(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

          std::vector<std::optional<HloSharding>> input_shardings_optional(
              {output_spec, std::nullopt, std::nullopt});
          std::vector<std::vector<double>> resharding_cost =
              GenerateReshardingCostsAndMissingShardingsForAllOperands(
                  ins, output_spec, strategy_map, cluster_env, call_graph,
                  input_shardings_optional);

          for (const auto& sharding_optional : input_shardings_optional) {
            CHECK(sharding_optional.has_value());
          }

          strategies->leaf_vector.push_back(ShardingStrategy(
              {name, output_spec, compute_cost, communication_cost, memory_cost,
               std::move(resharding_cost), input_shardings_optional}));
        }
        break;
      }
      case HloOpcode::kGather: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        // Follows the strategy of start_indices (operend 1)
        const HloInstruction* indices = ins->operand(1);
        const Shape& shape = ins->shape();
        const StrategyVector* src_strategies = strategy_map.at(indices).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;
        for (int32_t index_dim = 0; index_dim < indices->shape().rank();
             index_dim++) {
          // Shard on indices dimensions that correspond to output dimensions
          // TODO(b/220935014) Shard the last dim of output (model dim) with
          // AllGather cost and no follow.
          if (index_dim == ins->gather_dimension_numbers().index_vector_dim()) {
            continue;
          }
          for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
            // Split only when the tensor shape is divisible by device
            // mesh.
            if (device_mesh.dim(j) == 1 ||
                (only_allow_divisible &&
                 !IsDivisible(shape.dimensions(index_dim),
                              device_mesh.dim(j)))) {
              continue;
            }
            std::string name = absl::StrCat("S", index_dim, " @ ", j);

            HloSharding output_spec =
                Tile(shape, {index_dim}, {j}, device_mesh);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost = GetBytes(shape) / output_spec.NumTiles();
            std::optional<HloSharding> input_spec =
                hlo_sharding_util::ReshapeSharding(shape, indices->shape(),
                                                   output_spec);
            if (!input_spec.has_value()) {  // invalid reshape
              continue;
            }
            std::vector<std::optional<HloSharding>> input_shardings_optional(
                {std::nullopt, input_spec});
            std::vector<std::vector<double>> resharding_cost =
                GenerateReshardingCostsAndMissingShardingsForAllOperands(
                    ins, output_spec, strategy_map, cluster_env, call_graph,
                    input_shardings_optional);

            strategies->leaf_vector.push_back(ShardingStrategy(
                {name, output_spec, compute_cost, communication_cost,
                 memory_cost, std::move(resharding_cost),
                 input_shardings_optional}));
          }
        }
        AddReplicatedStrategy(
            ins, ins->shape(), cluster_env, strategy_map, strategies, 0,
            /* operands_to_consider_all_strategies_for */ {0});
        break;
      }
      case HloOpcode::kBroadcast: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        const HloInstruction* operand = ins->operand(0);

        const StrategyVector* operand_strategies =
            strategy_map.at(operand).get();
        CHECK(!operand_strategies->is_tuple);
        if (ins->shape().rank() == 1 || cluster_env.IsDeviceMesh1D()) {
          EnumerateAll1DPartition(ins, ins->shape(), cluster_env.device_mesh_,
                                  cluster_env, strategy_map, strategies,
                                  only_allow_divisible, "", call_graph);
        } else {
          EnumerateAll2DPartition(ins, ins->shape(), cluster_env.device_mesh_,
                                  cluster_env, strategy_map, strategies,
                                  batch_dim_map, only_allow_divisible,
                                  call_graph);
          if (solver_option.allow_mixed_mesh_shape) {
            EnumerateAll1DPartition(ins, ins->shape(),
                                    cluster_env.device_mesh_1d_, cluster_env,
                                    strategy_map, strategies,
                                    only_allow_divisible, "1d", call_graph);
          }
        }
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategies, replicated_penalty);

        break;
      }
      case HloOpcode::kReshape: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        const HloInstruction* operand = ins->operand(0);
        if (!(mesh_nn_dims >= 2 && solver_option.allow_mixed_mesh_shape)) {
          // Create follow strategies
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          CHECK(!src_strategies->is_tuple);
          strategies->following = src_strategies;

          for (int64_t sid = 0; sid < src_strategies->leaf_vector.size();
               ++sid) {
            std::optional<HloSharding> output_spec =
                hlo_sharding_util::ReshapeSharding(
                    operand->shape(), ins->shape(),
                    src_strategies->leaf_vector[sid].output_sharding);

            if (!output_spec.has_value()) {
              continue;
            }

            if (!IsValidTileAssignment(*output_spec)) {
              continue;
            }

            if (!TileAssignmentMatchesMesh(*output_spec, device_mesh)) {
              continue;
            }
            std::string name = ToStringSimple(*output_spec);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost =
                GetBytes(ins->shape()) / output_spec->NumTiles();
            auto resharding_costs = ReshardingCostVector(
                src_strategies, operand->shape(),
                src_strategies->leaf_vector[sid].output_sharding, cluster_env);
            strategies->leaf_vector.push_back(ShardingStrategy(
                {name,
                 *output_spec,
                 compute_cost,
                 communication_cost,
                 memory_cost,
                 {resharding_costs},
                 {src_strategies->leaf_vector[sid].output_sharding}}));
          }
        }

        // Fail to create follow strategies, enumerate all possible cases
        if (strategies->leaf_vector.empty()) {
          strategies->leaf_vector.clear();
          strategies->following = nullptr;

          // Split 1 dim
          if (cluster_env.IsDeviceMesh1D()) {
            EnumerateAll1DPartitionReshape(ins, device_mesh, cluster_env,
                                           strategy_map, strategies,
                                           only_allow_divisible, "");
          }
          if (solver_option.allow_mixed_mesh_shape &&
              cluster_env.IsDeviceMesh2D()) {
            // Split 1 dim, but for 1d mesh
            EnumerateAll1DPartitionReshape(ins, device_mesh_1d, cluster_env,
                                           strategy_map, strategies,
                                           only_allow_divisible, " 1d");
          }
          if (cluster_env.IsDeviceMesh2D()) {
            // Split 2 dim, one is always the batch dim
            Enumerate2DPartitionReshape(ins, device_mesh, cluster_env,
                                        strategy_map, batch_dim_map, strategies,
                                        only_allow_divisible);
          }

          // Replicate
          AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                strategies, replicated_penalty);
        }
        break;
      }
      case HloOpcode::kTranspose:
      case HloOpcode::kReverse: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        const HloInstruction* operand = ins->operand(0);

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          HloSharding output_spec = Undefined();
          auto input_spec = src_strategies->leaf_vector[sid].output_sharding;
          if (opcode == HloOpcode::kTranspose) {
            output_spec = hlo_sharding_util::TransposeSharding(
                input_spec, ins->dimensions());
          } else {
            output_spec = hlo_sharding_util::ReverseSharding(input_spec,
                                                             ins->dimensions());
          }

          std::string name = ToStringSimple(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
          auto resharding_costs = ReshardingCostVector(
              src_strategies, operand->shape(), input_spec, cluster_env);
          strategies->leaf_vector.push_back(
              ShardingStrategy({name,
                                output_spec,
                                compute_cost,
                                communication_cost,
                                memory_cost,
                                {resharding_costs},
                                {input_spec}}));
        }
        break;
      }
      case HloOpcode::kPad:
      case HloOpcode::kSlice:
      case HloOpcode::kConcatenate:  // TODO(zhuohan): revisit concatenate
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kSelectAndScatter: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        int64_t follow_idx;
        switch (opcode) {
          // TODO(yuemmawang) Re-evaluate the follow_idx choices for the
          // following 3.
          case HloOpcode::kPad:
          case HloOpcode::kReduceWindow:
          case HloOpcode::kSelectAndScatter:
          case HloOpcode::kConcatenate:
            // Follow the operand according to the follow heuristics
            follow_idx = ChooseOperandToFollow(strategy_map, depth_map,
                                               alias_map, max_depth, ins)
                             .first;
            break;
          // The following types are better to follow the first operand.
          case HloOpcode::kSlice:
          case HloOpcode::kDynamicSlice:
          case HloOpcode::kDynamicUpdateSlice:
            follow_idx = 0;
            break;
          default:
            LOG(FATAL) << "Selecting follow index encounters an unhandled "
                          "instruction type: " +
                              ins->ToShortString();
        }
        // Create follow strategies
        const HloInstruction* operand = ins->operand(follow_idx);
        StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          std::optional<HloSharding> output_spec;
          HloSharding input_spec =
              src_strategies->leaf_vector[sid].output_sharding;

          // Find output shardings.
          switch (opcode) {
            case HloOpcode::kPad:
            case HloOpcode::kSlice:
            case HloOpcode::kConcatenate:
            case HloOpcode::kDynamicSlice:
            case HloOpcode::kDynamicUpdateSlice:
              output_spec = PropagateDimwiseSharding(
                  input_spec, operand->shape(), ins->shape());
              break;
            case HloOpcode::kReduceWindow:
            case HloOpcode::kSelectAndScatter:
              output_spec = PropagateReduceWindowSharding(
                  input_spec, operand->shape(), ins->window());
              break;
            default:
              LOG(FATAL) << "Unhandled instruction: " + ins->ToString();
          }

          // Get a list of input shardings, each corresponds to an operand.
          std::vector<std::optional<HloSharding>> input_shardings;
          for (int64_t k = 0; k < ins->operand_count(); ++k) {
            if (k == follow_idx ||
                ToString(ins->operand(k)->shape().dimensions()) ==
                    ToString(operand->shape().dimensions())) {
              input_shardings.push_back(input_spec);
            } else {
              input_shardings.push_back(std::nullopt);
            }
          }
          if (!output_spec.has_value()) {
            continue;
          }

          std::string name = ToStringSimple(*output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec->NumTiles();
          std::vector<std::vector<double>> resharding_costs =
              GenerateReshardingCostsAndMissingShardingsForAllOperands(
                  ins, *output_spec, strategy_map, cluster_env, call_graph,
                  input_shardings);

          strategies->leaf_vector.push_back(
              ShardingStrategy({name,
                                *output_spec,
                                compute_cost,
                                communication_cost,
                                memory_cost,
                                std::move(resharding_costs),
                                {input_spec}}));
        }

        if (strategies->leaf_vector.empty()) {
          strategies->following = nullptr;
          AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                strategies, 0);
        }

        break;
      }
      // Unary elementwise operations.
      case HloOpcode::kAbs:
      case HloOpcode::kRoundNearestAfz:
      case HloOpcode::kRoundNearestEven:
      case HloOpcode::kCeil:
      case HloOpcode::kClz:
      case HloOpcode::kConvert:
      case HloOpcode::kBitcast:
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kCopy:
      case HloOpcode::kCos:
      case HloOpcode::kExp:
      case HloOpcode::kExpm1:
      case HloOpcode::kFloor:
      case HloOpcode::kImag:
      case HloOpcode::kIsFinite:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
      case HloOpcode::kNot:
      case HloOpcode::kNegate:
      case HloOpcode::kPopulationCount:
      case HloOpcode::kReal:
      case HloOpcode::kReducePrecision:
      case HloOpcode::kRsqrt:
      case HloOpcode::kLogistic:
      case HloOpcode::kSign:
      case HloOpcode::kSin:
      case HloOpcode::kSqrt:
      case HloOpcode::kCbrt:
      case HloOpcode::kTan:
      case HloOpcode::kTanh:
      // Binary elementwise operations
      case HloOpcode::kAdd:
      case HloOpcode::kAtan2:
      case HloOpcode::kCompare:
      case HloOpcode::kComplex:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kMultiply:
      case HloOpcode::kPower:
      case HloOpcode::kRemainder:
      case HloOpcode::kSubtract:
      case HloOpcode::kAnd:
      case HloOpcode::kOr:
      case HloOpcode::kXor:
      case HloOpcode::kShiftLeft:
      case HloOpcode::kShiftRightArithmetic:
      case HloOpcode::kShiftRightLogical:
      case HloOpcode::kStochasticConvert:
      // Ternary elementwise operations.
      case HloOpcode::kSelect:
      case HloOpcode::kClamp: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        // Choose an operand to follow
        int64_t follow_idx;
        bool tie;
        std::tie(follow_idx, tie) = ChooseOperandToFollow(
            strategy_map, depth_map, alias_map, max_depth, ins);

        if (!tie || AllowTieFollowing(ins)) {
          strategies->following =
              strategy_map.at(ins->operand(follow_idx)).get();
        } else {
          strategies->following = nullptr;
        }

        // Get all possible sharding specs from operands
        for (int64_t i = 0; i < ins->operand_count(); ++i) {
          if (strategies->following != nullptr && i != follow_idx) {
            // If ins follows one operand, do not consider sharding specs from
            // other operands.
            continue;
          }

          auto process_src_strategies = [&](const std::vector<ShardingStrategy>
                                                src_strategies_leaf_vector) {
            for (int64_t sid = 0; sid < src_strategies_leaf_vector.size();
                 ++sid) {
              HloSharding output_spec =
                  src_strategies_leaf_vector[sid].output_sharding;
              std::string name = ToStringSimple(output_spec);
              double compute_cost = 0, communication_cost = 0;
              double memory_cost =
                  GetBytes(ins->shape()) / output_spec.NumTiles();
              std::vector<std::vector<double>> resharding_costs;
              std::vector<std::optional<HloSharding>> input_shardings;
              for (int64_t k = 0; k < ins->operand_count(); ++k) {
                resharding_costs.push_back(ReshardingCostVector(
                    strategy_map.at(ins->operand(k)).get(),
                    ins->operand(k)->shape(), output_spec, cluster_env));
                input_shardings.push_back(output_spec);
              }

              strategies->leaf_vector.push_back(ShardingStrategy(
                  {name, output_spec, compute_cost, communication_cost,
                   memory_cost, std::move(resharding_costs), input_shardings}));
            }
          };
          auto src_strategies = strategy_map.at(ins->operand(i)).get();
          CHECK(!src_strategies->is_tuple);

          process_src_strategies(src_strategies->leaf_vector);
          if (pretrimmed_strategy_map.contains(src_strategies->node_idx)) {
            process_src_strategies(
                pretrimmed_strategy_map.at(src_strategies->node_idx));
          }
        }
        if (ins->opcode() == HloOpcode::kAdd) {
          // Adjust the resharding costs for AllReduceReassociate pass.
          // The AllReduceReassociate pass can simplify
          // allreduce(x) + allreduce(y) to allreduce(x + y),
          // so we adjust the resharidng costs to reflect this optimization.

          // TODO(zhuohan): The current implementation only works for
          // x = a + b. We also need to cover cases where there are
          // more than two operands (i.e., x = a + b + c).
          if (ins->operand(0)->opcode() == HloOpcode::kDot &&
              ins->operand(1)->opcode() == HloOpcode::kDot) {
            associative_dot_pairs.push_back(
                {strategy_map.at(ins->operand(0)).get(),
                 strategy_map.at(ins->operand(1)).get()});
          }
        }
        break;
      }
      case HloOpcode::kReduce: {
        strategies = FollowReduceStrategy(
            ins, ins->shape(), ins->operand(0), ins->operand(1), instruction_id,
            strategy_map, leaf_strategies, cluster_env,
            solver_option.allow_mixed_mesh_shape);
        break;
      }
      case HloOpcode::kDot: {
        TF_RETURN_IF_ERROR(HandleDot(strategies, leaf_strategies, strategy_map,
                                     ins, instruction_id, cluster_env,
                                     batch_dim_map, solver_option));
        if (solver_option.allow_replicated_strategy_for_dot_and_conv) {
          AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                strategies, 0);
        }
        break;
      }
      case HloOpcode::kConvolution: {
        TF_RETURN_IF_ERROR(HandleConv(strategies, leaf_strategies, strategy_map,
                                      ins, instruction_id, cluster_env,
                                      batch_dim_map, solver_option));
        if (solver_option.allow_replicated_strategy_for_dot_and_conv) {
          AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                strategies, 0);
        }
        break;
      }
      case HloOpcode::kRngGetAndUpdateState: {
        strategies = CreateLeafStrategyVectorWithoutInNodes(instruction_id,
                                                            leaf_strategies);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategies, 0);
        break;
      }
      case HloOpcode::kIota: {
        strategies = CreateLeafStrategyVectorWithoutInNodes(instruction_id,
                                                            leaf_strategies);
        if (cluster_env.IsDeviceMesh1D()) {
          EnumerateAll1DPartition(ins, ins->shape(), device_mesh, cluster_env,
                                  strategy_map, strategies,
                                  only_allow_divisible, "", call_graph);
        }
        if (cluster_env.IsDeviceMesh2D()) {
          // Split 2 dims
          EnumerateAll2DPartition(ins, ins->shape(), device_mesh, cluster_env,
                                  strategy_map, strategies, batch_dim_map,
                                  only_allow_divisible, call_graph);
        }
        if (cluster_env.IsDeviceMesh2D() &&
            solver_option.allow_mixed_mesh_shape) {
          // Split 1 dim, but for 1d flattened version of the 2d mesh
          // For example, when the mesh shape is (2, 4), we add strategies for
          // mesh shape (1, 8) here in addition.
          EnumerateAll1DPartition(ins, ins->shape(), device_mesh_1d,
                                  cluster_env, strategy_map, strategies,
                                  only_allow_divisible, " 1d", call_graph);
        }

        if (strategies->leaf_vector.empty() || IsFollowedByBroadcast(ins)) {
          // Replicate
          AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                strategies, replicated_penalty * 5);
        }

        break;
      }
      case HloOpcode::kTuple: {
        strategies = CreateTupleStrategyVector(instruction_id);
        strategies->childs.reserve(ins->operand_count());
        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          strategies->childs.push_back(MaybeFollowInsStrategyVector(
              src_strategies, operand->shape(), instruction_id,
              /* have_memory_cost= */ true, leaf_strategies, cluster_env,
              pretrimmed_strategy_map));
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* operand = ins->operand(0);
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(src_strategies->is_tuple);
        strategies = MaybeFollowInsStrategyVector(
            src_strategies->childs[ins->tuple_index()].get(), ins->shape(),
            instruction_id,
            /* have_memory_cost= */ true, leaf_strategies, cluster_env,
            pretrimmed_strategy_map);
        break;
      }
      case HloOpcode::kCustomCall: {
        auto generate_non_following_strategies =
            [&](bool only_replicated,
                absl::flat_hash_set<int64_t>
                    operands_to_consider_all_strategies_for = {}) {
              if (ins->shape().IsTuple()) {
                if (only_replicated) {
                  strategies = CreateTupleStrategyVector(instruction_id);
                  strategies->childs.reserve(ins->shape().tuple_shapes_size());
                  for (size_t i = 0; i < ins->shape().tuple_shapes_size();
                       ++i) {
                    std::unique_ptr<StrategyVector> child_strategies =
                        CreateLeafStrategyVector(instruction_id, ins,
                                                 strategy_map, leaf_strategies);
                    AddReplicatedStrategy(ins, ins->shape().tuple_shapes(i),
                                          cluster_env, strategy_map,
                                          child_strategies, replicated_penalty);
                    strategies->childs.push_back(std::move(child_strategies));
                  }
                } else {
                  strategies =
                      CreateAllStrategiesVector(
                          ins, ins->shape(), instruction_id, leaf_strategies,
                          cluster_env, strategy_map, solver_option,
                          replicated_penalty, batch_dim_map, call_graph,
                          only_allow_divisible, true)
                          .value();
                }
              } else {
                if (only_replicated) {
                  strategies = CreateLeafStrategyVector(
                      instruction_id, ins, strategy_map, leaf_strategies);
                  AddReplicatedStrategy(ins, ins->shape(), cluster_env,
                                        strategy_map, strategies,
                                        replicated_penalty);
                } else {
                  strategies =
                      CreateAllStrategiesVector(
                          ins, ins->shape(), instruction_id, leaf_strategies,
                          cluster_env, strategy_map, solver_option,
                          replicated_penalty, batch_dim_map, call_graph,
                          only_allow_divisible, true)
                          .value();
                }
              }
            };

        if (IsCustomCallMarker(ins)) {
          const HloInstruction* operand = ins->operand(0);
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          CHECK(src_strategies->is_tuple);
          strategies = MaybeFollowInsStrategyVector(
              src_strategies, ins->shape(), instruction_id,
              /* have_memory_cost= */ true, leaf_strategies, cluster_env,
              pretrimmed_strategy_map);
        } else if (ins->has_sharding()) {
          generate_non_following_strategies(false);
        } else if (OutputInputSameShapes(ins)) {
          auto* partitioner =
              GetCustomCallPartitioner(ins->custom_call_target());
          if (partitioner && partitioner->IsCustomCallShardable(ins)) {
            // Follows operand 0's strategies if this custom-call op is
            // shardable and has the same input and output sizes.
            const HloInstruction* operand = ins->operand(0);
            const StrategyVector* src_strategies =
                strategy_map.at(operand).get();
            strategies = MaybeFollowInsStrategyVector(
                src_strategies, ins->shape(), instruction_id,
                /* have_memory_cost= */ true, leaf_strategies, cluster_env,
                pretrimmed_strategy_map);
          }
        } else if (IsTopKCustomCall(ins)) {
          generate_non_following_strategies(false, {0});
        } else {
          // TODO (b/258723035) Handle CustomCall ops for GPUs in a better way.
          generate_non_following_strategies(true);
        }
        break;
      }
      case HloOpcode::kWhile: {
        strategies = CreateTupleStrategyVector(instruction_id);
        strategies->childs.reserve(ins->shape().tuple_shapes_size());
        const StrategyVector* src_strategies =
            strategy_map.at(ins->operand(0)).get();
        for (size_t i = 0; i < ins->shape().tuple_shapes_size(); ++i) {
          strategies->childs.push_back(MaybeFollowInsStrategyVector(
              src_strategies->childs[i].get(),
              ins->shape().tuple_shapes().at(i), instruction_id,
              /* have_memory_cost= */ true, leaf_strategies, cluster_env,
              pretrimmed_strategy_map));
        }

        break;
      }
      case HloOpcode::kConditional:
      case HloOpcode::kInfeed:
      case HloOpcode::kSort: {
        strategies =
            CreateAllStrategiesVector(
                ins, ins->shape(), instruction_id, leaf_strategies, cluster_env,
                strategy_map, solver_option, replicated_penalty, batch_dim_map,
                call_graph, only_allow_divisible,
                /*create_replicated_strategies*/ true)
                .value();
        break;
      }
      case HloOpcode::kOutfeed: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        GenerateOutfeedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                strategies, replicated_penalty);
        break;
      }
      case HloOpcode::kAfterAll: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategies, replicated_penalty);
        break;
      }
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->ToString();
    }
    RemoveDuplicatedStrategy(strategies);
    if (ins->has_sharding() && ins->opcode() != HloOpcode::kOutfeed) {
      // Finds the sharding strategy that aligns with the given sharding spec
      // Do not merge nodes if this one instruction has annotations.
      TrimOrGenerateStrategiesBasedOnExistingSharding(
          ins->shape(), strategies.get(), strategy_map, instructions,
          ins->sharding(), cluster_env, pretrimmed_strategy_map, call_graph,
          solver_option.nd_sharding_iteratively_strict_search_space);
    }
    if (!strategies->is_tuple && strategies->following) {
      if (!LeafVectorsAreConsistent(
              strategies->leaf_vector, strategies->following->leaf_vector,
              /*is_reshape*/ ins->opcode() == HloOpcode::kReshape)) {
        // It confuses the solver if two instructions have different number of
        // sharding strategies but share the same ILP variable. The solver
        // would run much longer and/or return infeasible solutions.
        // So if two strategies' leaf_vectors are inconsistent, we unfollow
        // them.
        strategies->following = nullptr;
      }
    } else if (strategies->is_tuple) {
      for (size_t i = 0; i < strategies->childs.size(); i++) {
        if (strategies->childs.at(i)->following &&
            !LeafVectorsAreConsistent(
                strategies->childs.at(i)->leaf_vector,
                strategies->childs.at(i)->following->leaf_vector,
                /*is_reshape*/ ins->opcode() == HloOpcode::kReshape)) {
          strategies->childs.at(i)->following = nullptr;
        }
      }
    }
    RemoveInvalidShardingsWithShapes(ins->shape(), strategies.get());

    if (instruction_execution_counts.contains(ins)) {
      ScaleCostsWithExecutionCounts(strategies.get(),
                                    instruction_execution_counts.at(ins));
    } else {
      VLOG(5) << "No execution count available for " << ins->name();
    }
    XLA_VLOG_LINES(2, absl::StrCat("strategies:\n", strategies->ToString()));

    // Debug options: forcibly set the strategy of some instructions.
    if (solver_option.force_strategy) {
      std::vector<int64_t> inst_indices =
          solver_option.force_strategy_inst_indices;
      std::vector<std::string> stra_names =
          solver_option.force_strategy_stra_names;
      CHECK_EQ(inst_indices.size(), stra_names.size());
      auto it = absl::c_find(inst_indices, strategies->node_idx);
      if (it != inst_indices.end()) {
        CHECK(!strategies->is_tuple);
        std::vector<ShardingStrategy> new_leaf_vector;
        int64_t idx = it - inst_indices.begin();
        for (const auto& stra : strategies->leaf_vector) {
          if (stra.name == stra_names[idx]) {
            new_leaf_vector.push_back(stra);
          }
        }
        strategies->leaf_vector = std::move(new_leaf_vector);
      }
    }

    // When trying out multiple mesh shapes in the presence of user specified
    // sharding (as in
    // AutoShardingTest.AutoShardingKeepUserShardingInputOutput), there may be a
    // situation when we cannot generate any shardings for an instruction when
    // the mesh shape we're trying does not match with the mesh shape used in
    // user specified shardings. So we disable the check in that situation.
    if (!trying_multiple_mesh_shapes) {
      CHECK(strategies->is_tuple || !strategies->leaf_vector.empty())
          << ins->ToString() << " does not have any valid strategies.";
    } else if (!(strategies->is_tuple || !strategies->leaf_vector.empty())) {
      return Status(absl::StatusCode::kFailedPrecondition,
                    "Could not generate any shardings for an instruction due "
                    "to mismatched mesh shapes.");
    }
    // Checks the shape of resharding_costs is valid. It will check fail if the
    // shape is not as expected.
    // CheckReshardingCostsShape(strategies.get());
    CheckMemoryCosts(strategies.get(), ins->shape());
    strategy_map[ins] = std::move(strategies);
  }  // end of for loop

  // If gradient accumulation is used, adjust the cost of all-reduce for
  // gradient synchronization.
  if (solver_option.grad_acc_num_micro_batches > 1) {
    // find gradient-computation instructions
    std::vector<const HloInstruction*> grad_insts =
        GetGradientComputationInstructions(instructions);
    for (const HloInstruction* inst : grad_insts) {
      StrategyVector* stra_vector = strategy_map[inst].get();
      CHECK(!stra_vector->is_tuple);

      for (auto& stra : stra_vector->leaf_vector) {
        if (absl::StrContains(stra.name, "allreduce")) {
          stra.communication_cost /= solver_option.grad_acc_num_micro_batches;
        }
      }
    }
  }

  return std::make_tuple(std::move(strategy_map), std::move(leaf_strategies),
                         std::move(associative_dot_pairs));
}

// NOLINTEND

AutoShardingSolverResult CallSolver(
    const HloLiveRange& hlo_live_range, const LivenessSet& liveness_set,
    const StrategyMap& strategy_map, const LeafStrategies& leaf_strategies,
    const CostGraph& cost_graph, const AliasSet& alias_set,
    const std::vector<NodeStrategyIdx>& s_hint,
    int64_t memory_budget_per_device, bool crash_at_infinity_costs_check,
    bool compute_iis, int64_t solver_timeout_in_seconds,
    bool allow_alias_to_follower_conversion) {
  // Serialize edges and edge costs to 1d numpy arrays
  AutoShardingSolverRequest request;
  request.num_nodes = leaf_strategies.size();
  request.memory_budget = memory_budget_per_device;
  request.s_len = cost_graph.node_lens_;
  request.s_follow = cost_graph.follow_idx_;
  request.s_hint = s_hint;
  request.solver_timeout_in_seconds = solver_timeout_in_seconds;
  request.crash_at_infinity_costs_check = crash_at_infinity_costs_check;
  request.compute_iis = compute_iis;
  for (const auto& iter : cost_graph.edge_costs_) {
    request.e.push_back(iter.first);
    std::vector<double> rij;
    Matrix edge_cost = iter.second;
    for (NodeStrategyIdx i = 0; i < edge_cost.n_; i++) {
      for (NodeStrategyIdx j = 0; j < edge_cost.m_; j++) {
        rij.push_back(edge_cost(i, j));
      }
    }
    request.r.push_back(std::move(rij));
  }

  const HloInstructionSequence& sequence =
      hlo_live_range.flattened_instruction_sequence();
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Serialize node costs
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    const StrategyVector* strategies = leaf_strategies[i];
    request.instruction_names.push_back(absl::StrCat(
        instructions.at(strategies->instruction_id)->name(), " (id: ", i, ")"));
    std::vector<double> ci, di, mi;
    for (NodeStrategyIdx j = 0; j < strategies->leaf_vector.size(); ++j) {
      ci.push_back(strategies->leaf_vector[j].compute_cost);
      di.push_back(strategies->leaf_vector[j].communication_cost +
                   cost_graph.extra_node_costs_[i][j]);
      mi.push_back(strategies->leaf_vector[j].memory_cost);
    }
    request.c.push_back(ci);
    request.d.push_back(di);
    request.m.push_back(mi);
  }

  // Serialize special edges that forces a alias pair have the same sharding
  // spec
  std::vector<std::pair<NodeIdx, NodeIdx>> new_followers;
  for (const auto& pair : alias_set) {
    const StrategyVector* src_strategies = leaf_strategies[pair.first];
    const StrategyVector* dst_strategies = leaf_strategies[pair.second];
    Matrix raw_cost(src_strategies->leaf_vector.size(),
                    dst_strategies->leaf_vector.size());
    for (NodeStrategyIdx i = 0; i < src_strategies->leaf_vector.size(); ++i) {
      for (NodeStrategyIdx j = 0; j < dst_strategies->leaf_vector.size(); ++j) {
        if (src_strategies->leaf_vector[i].output_sharding ==
            dst_strategies->leaf_vector[j].output_sharding) {
          raw_cost(i, j) = 0.0;
        } else {
          raw_cost(i, j) = 1.0;
        }
      }
    }
    NodeIdx idx_a = pair.first;
    NodeIdx idx_b = pair.second;
    std::vector<NodeStrategyIdx> row_indices;
    std::vector<NodeStrategyIdx> col_indices;

    if (request.s_follow[idx_a] >= 0) {
      row_indices = cost_graph.reindexing_vector_.at(idx_a);
      idx_a = request.s_follow[idx_a];
    } else {
      row_indices.assign(request.s_len[idx_a], 0);
      std::iota(row_indices.begin(), row_indices.end(), 0);
    }

    if (request.s_follow[idx_b] >= 0) {
      col_indices = cost_graph.reindexing_vector_.at(idx_b);
      idx_b = request.s_follow[idx_b];
    } else {
      col_indices.assign(request.s_len[idx_b], 0);
      std::iota(col_indices.begin(), col_indices.end(), 0);
    }

    CHECK_EQ(request.s_len[idx_a], row_indices.size());
    CHECK_EQ(request.s_len[idx_b], col_indices.size());

    std::vector<double> vij;
    for (NodeStrategyIdx i : row_indices) {
      for (NodeStrategyIdx j : col_indices) {
        vij.push_back(raw_cost(i, j));
      }
    }
    bool convertable = (row_indices.size() == col_indices.size());
    for (NodeStrategyIdx i = 0; i < row_indices.size() && convertable; ++i) {
      if (vij[i * col_indices.size() + i] != 0.0) convertable = false;
    }
    if (convertable && allow_alias_to_follower_conversion) {
      new_followers.push_back(std::make_pair(idx_a, idx_b));
    } else {
      request.a.push_back(std::make_pair(idx_a, idx_b));
      request.v.push_back(vij);
    }
  }

  // Process any new followers that had originally been modeled as aliases.
  std::vector<NodeIdx>& s_follow = request.s_follow;
  for (auto [follower, followee] : new_followers) {
    // New followers may have introduced chains, so find the root nodes.
    while (s_follow[follower] >= 0) follower = s_follow[follower];
    while (s_follow[followee] >= 0) followee = s_follow[followee];
    if (follower != followee) s_follow[follower] = followee;
  }

  // Flatten the follower indices to remove any transitive arcs.
  for (NodeIdx i = 0; i < request.num_nodes; ++i) {
    if (s_follow[i] < 0) continue;
    while (s_follow[s_follow[i]] >= 0) s_follow[i] = s_follow[s_follow[i]];
  }

  // Serialize liveness_set
  request.live.resize(liveness_set.size());
  for (LivenessIdx t = 0; t < liveness_set.size(); ++t) {
    for (const HloValue* value : liveness_set[t]) {
      const HloInstruction* instruction = value->instruction();
      const ShapeIndex& index = value->index();
      if (instruction->shape().IsTuple() && index.empty()) continue;
      const StrategyVector* strategies = strategy_map.at(instruction).get();
      const NodeIdx node_idx =
          strategies->GetSubStrategyVector(index)->node_idx;
      if (node_idx >= 0) request.live[t].push_back(node_idx);
    }
  }
  const AutoShardingSolverResult result = CallORToolsSolver(request);
  if (result.status.ok()) {
    const AutoShardingEvaluation evaluation = Evaluate(request, result);
    LOG(INFO) << "Total Communication Cost: "
              << evaluation.total_communication_cost
              << " (lower bound: " << evaluation.lower_bound_communication_cost
              << ")";
    LOG(INFO) << "Total Computation Cost: " << evaluation.total_computation_cost
              << " (lower bound: " << evaluation.lower_bound_computation_cost
              << ")";
    LOG(INFO) << "Total Resharding Cost: " << evaluation.total_resharding_cost
              << " (lower bound: " << evaluation.lower_bound_resharding_cost
              << ")";
    LOG(INFO) << "Total Cost: " << evaluation.total_cost
              << " (lower bound: " << evaluation.lower_bound_cost << ")";
    LOG(INFO) << "Total Violations: " << evaluation.violation_codes.size();
  }
  return result;
}

void CheckHloSharding(const HloInstructionSequence& sequence,
                      size_t total_num_devices) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  std::vector<std::pair<size_t, std::string>> size_string;
  for (HloInstruction* ins : instructions) {
    if (!ins->has_sharding()) {
      continue;
    }
    if (!ins->shape().IsTuple() &&
        ins->opcode() != HloOpcode::kGetTupleElement) {
      // TODO(yuemmawang) Check other cases when it's helpful (it's not
      // needed so far).
      double size = GetInstructionSize(ins->shape()) / 1024 / 1024 / 1024;
      if ((!ShardingIsComplete(ins->sharding(), total_num_devices) ||
           ins->sharding().IsReplicated()) &&
          size > 1) {
        LOG(INFO) << "Instruction is not fully sharded: (" << size << " GB) "
                  << ins->ToString();
      } else if (!ins->has_sharding()) {
        LOG(INFO) << "Instruction does not have sharding: " << ins->name();
      }
      for (const auto& op : ins->operands()) {
        if (op->has_sharding()) {
          if (op->sharding().IsReplicated() || ins->sharding().IsReplicated()) {
            continue;
          }
          std::vector<int64_t> ins_sharded_dims =
              VectorGreaterThanOneElementIndices(
                  ins->sharding().tile_assignment().dimensions(),
                  ins->sharding().ReplicateOnLastTileDim());
          std::vector<int64_t> op_sharded_dims =
              VectorGreaterThanOneElementIndices(
                  op->sharding().tile_assignment().dimensions(),
                  op->sharding().ReplicateOnLastTileDim());
          bool not_consistent = false;
          if (ins_sharded_dims.size() != op_sharded_dims.size()) {
            not_consistent = true;
          } else {
            for (size_t i = 0; i < ins_sharded_dims.size(); i++) {
              if (op->shape().dimensions().at(op_sharded_dims.at(i)) !=
                  ins->shape().dimensions().at(ins_sharded_dims.at(i))) {
                not_consistent = true;
              }
            }
          }
          if (not_consistent) {
            // Prints the inconsistent shardings, which may indicate causes
            // of resharding overheads, and some inconsistent shardings are
            // unavoidable.
            size_t op_size =
                GetInstructionSize(op->shape()) / (1024.0 * 1024 * 1024);
            std::string str = absl::StrCat("Shardings not consistent (op size ",
                                           op_size, " GB):", ins->ToString(),
                                           "\n Operand: ", op->ToString());
            size_string.push_back(std::make_pair(op_size, std::move(str)));
          }
        } else {
          LOG(INFO) << "Instruction " << op->name()
                    << " does not have sharding.";
        }
      }
    }
  }
  struct {
    bool operator()(const std::pair<size_t, std::string>& a,
                    const std::pair<size_t, std::string>& b) const {
      return a.first > b.first;
    }
  } MemLarger;
  std::sort(size_string.begin(), size_string.end(), MemLarger);
  size_t k = 10;
  k = std::min(k, size_string.size());
  for (size_t t = 0; t < k; t++) {
    LOG(INFO) << size_string.at(t).second;
  }
}

// Set the HloSharding for all instructions according to the ILP solution.
void SetHloSharding(const HloInstructionSequence& sequence,
                    const StrategyMap& strategy_map,
                    const CostGraph& cost_graph,
                    absl::Span<const NodeStrategyIdx> s_val,
                    bool last_iteration) {
  // Set the HloSharding for every instruction
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  for (HloInstruction* inst : instructions) {
    if (inst->opcode() == HloOpcode::kOutfeed) {
      continue;
    }
    auto iter = strategy_map.find(inst);
    if (iter == strategy_map.end()) {
      continue;
    }

    const StrategyVector* strategies = iter->second.get();
    if (strategies->is_tuple) {
      const Shape& out_shape = inst->shape();
      ShapeTree<HloSharding> output_tuple_sharding(out_shape, Undefined());
      std::vector<HloSharding> output_flattened_shardings;

      std::function<void(const StrategyVector*)> extract_tuple_shardings;
      bool set_tuple_sharding = true;

      extract_tuple_shardings = [&](const StrategyVector* strategies) {
        if (strategies->is_tuple) {
          for (const auto& child_strategies : strategies->childs) {
            extract_tuple_shardings(child_strategies.get());
          }
        } else {
          NodeIdx node_idx = strategies->node_idx;
          NodeStrategyIdx stra_idx = s_val[node_idx];
          // Do not set completed sharding before the last iteration
          if (strategies->leaf_vector[stra_idx]
                  .output_sharding.IsReplicated() &&
              !last_iteration) {
            set_tuple_sharding = false;
          }
          output_flattened_shardings.push_back(
              strategies->leaf_vector[stra_idx].output_sharding);
        }
      };
      extract_tuple_shardings(strategies);

      // Create Tuple HloSharding.
      int i = 0;
      for (auto& leaf : output_tuple_sharding.leaves()) {
        leaf.second = output_flattened_shardings[i++];
      }
      if (set_tuple_sharding) {
        inst->set_sharding(HloSharding::Tuple(output_tuple_sharding));
      }
    } else {
      const HloSharding& sharding_spec =
          GetShardingStrategy(inst, strategy_map, cost_graph, s_val)
              .output_sharding;
      if (IsUndefined(sharding_spec)) {
        continue;
      }
      // Do not overwrite existing complete shardings.
      if (sharding_spec.IsReplicated() && !last_iteration) {
        LOG(INFO) << "skip setting shardings for inst " << inst->name();
      } else {
        inst->set_sharding(sharding_spec);
      }
    }
  }
}

void SetHloShardingPostProcessing(
    const HloInstructionSequence& sequence, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const NodeStrategyIdx> s_val,
    const ClusterEnvironment& cluster_env,
    absl::flat_hash_map<std::string, std::vector<HloSharding>>*
        preserve_shardings) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  const Array<int64_t>& device_mesh = cluster_env.device_mesh_;
  // Post process: fix some corner cases.
  ReshardingCache resharding_cache_entity;
  ReshardingCache* resharding_cache = &resharding_cache_entity;

  for (HloInstruction* inst : instructions) {
    // For some dot instructions and resharding cases, our formulation thinks
    // they are valid. But the spmd partitioner cannot infer the correct
    // dot algorithms or resharding algorithm from the input/output sharding.
    // It then generates bad fallback code.
    // Here we insert some extra annotated identity instructions to help the
    // spmd partitioner generate correct code.

    if (inst->opcode() == HloOpcode::kDot) {
      const ShardingStrategy& stra =
          GetShardingStrategy(inst, strategy_map, cost_graph, s_val);
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      const HloSharding& lhs_sharding = lhs->sharding();
      const HloSharding& rhs_sharding = rhs->sharding();
      const DotDimensionNumbers& dot_dnums = inst->dot_dimension_numbers();
      const auto& lhs_con_dims = dot_dnums.lhs_contracting_dimensions();
      const auto& rhs_con_dims = dot_dnums.rhs_contracting_dimensions();

      const auto& lhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDimWrapper(
              lhs->shape(), lhs_sharding,
              /* consider_reverse_device_meshes */ true);
      const auto& rhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDimWrapper(
              rhs->shape(), rhs_sharding,
              /* consider_reverse_device_meshes */ true);

      if (absl::StrContains(stra.name, "allreduce") &&
          lhs_tensor_dim_to_mesh_dim[lhs_con_dims[0]] == -1 &&
          rhs_tensor_dim_to_mesh_dim[rhs_con_dims[0]] == -1) {
        // Allow duplicatd dot computation in this case to reduce
        // communication
      } else {
        CHECK(stra.input_shardings.size() == 2)
            << "Dot op requires both operands to have input shardings, "
               "but get instruction: "
            << inst->ToString() << ", strategy : " << stra.ToString();
        if (stra.input_shardings[0].has_value()) {
          FixMixedMeshShapeResharding(inst, 0, stra.input_shardings[0].value(),
                                      device_mesh, resharding_cache);
        }
        if (stra.input_shardings[1].has_value()) {
          FixMixedMeshShapeResharding(inst, 1, stra.input_shardings[1].value(),
                                      device_mesh, resharding_cache);
        }
      }
    } else if (inst->opcode() == HloOpcode::kConvolution) {
      const ShardingStrategy& stra =
          GetShardingStrategy(inst, strategy_map, cost_graph, s_val);
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      const HloSharding& lhs_sharding = lhs->sharding();
      const HloSharding& rhs_sharding = rhs->sharding();
      const ConvolutionDimensionNumbers& conv_dnums =
          inst->convolution_dimension_numbers();
      const int lhs_in_channel_dim = conv_dnums.input_feature_dimension();
      const int rhs_in_channel_dim =
          conv_dnums.kernel_input_feature_dimension();

      const auto& lhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDimWrapper(
              lhs->shape(), lhs_sharding,
              /* consider_reverse_device_meshes */ true);
      const auto& rhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDimWrapper(
              rhs->shape(), rhs_sharding,
              /* consider_reverse_device_meshes */ true);

      if (absl::StrContains(stra.name, "allreduce") &&
          lhs_tensor_dim_to_mesh_dim[lhs_in_channel_dim] == -1 &&
          rhs_tensor_dim_to_mesh_dim[rhs_in_channel_dim] == -1) {
        // Allow duplicatd conv computation in this case to reduce
        // communication
      } else {
        if (stra.input_shardings[0].has_value()) {
          FixMixedMeshShapeResharding(inst, 0, stra.input_shardings[0].value(),
                                      device_mesh, resharding_cache);
        }
        if (stra.input_shardings[1].has_value()) {
          FixMixedMeshShapeResharding(inst, 1, stra.input_shardings[1].value(),
                                      device_mesh, resharding_cache);
        }
      }
    } else if (inst->opcode() == HloOpcode::kOutfeed) {
      // Outfeed operand shardings are handled in downstream passes and so we
      // ignore outfeed ops here. However, we need to ensure that outfeed ops
      // which have user shardings have their shardings restored at the end. If
      // not, this can lead to errors downstream in the spmd_partitioner pass.
      auto preserved_sharding_iter = preserve_shardings->find(inst->name());
      if (preserved_sharding_iter != preserve_shardings->end()) {
        const auto& preserved_sharding = preserved_sharding_iter->second;
        if (preserved_sharding.size() > 1) {
          std::vector<Shape> tuple_elements_shape(
              inst->operand(0)->shape().tuple_shapes().begin(),
              inst->operand(0)->shape().tuple_shapes().end());
          tuple_elements_shape.push_back(inst->operand(1)->shape());
          Shape output_tuple_sharding_shape =
              ShapeUtil::MakeTupleShape(tuple_elements_shape);
          ShapeTree<HloSharding> output_tuple_sharding(
              output_tuple_sharding_shape, Undefined());
          size_t i = 0;
          for (auto& leaf : output_tuple_sharding.leaves()) {
            leaf.second = preserved_sharding.at(i++);
          }
          inst->set_sharding(HloSharding::Tuple(output_tuple_sharding));
        } else {
          inst->set_sharding(preserved_sharding.at(0));
        }
      }

      continue;
    } else {
      if (inst->shape().IsTuple()) {
        switch (inst->opcode()) {
          case HloOpcode::kReduce:
          case HloOpcode::kCustomCall:
          case HloOpcode::kSort: {
            for (size_t i = 0; i < inst->shape().tuple_shapes_size(); ++i) {
              const ShardingStrategy& stra =
                  GetShardingStrategyForTuple(inst, {static_cast<int64_t>(i)},
                                              strategy_map, cost_graph, s_val);
              if (stra.input_shardings.size() > i &&
                  stra.input_shardings[i].has_value()) {
                FixMixedMeshShapeResharding(inst, i,
                                            stra.input_shardings[i].value(),
                                            device_mesh, resharding_cache);
              }
            }
            break;
          }
          case HloOpcode::kTuple: {
            for (size_t i = 0; i < inst->shape().tuple_shapes_size(); ++i) {
              const ShardingStrategy& stra =
                  GetShardingStrategyForTuple(inst, {static_cast<int64_t>(i)},
                                              strategy_map, cost_graph, s_val);
              CHECK_EQ(stra.input_shardings.size(), 1);
              CHECK(stra.input_shardings[0].has_value());
              FixMixedMeshShapeResharding(inst, i,
                                          stra.input_shardings[0].value(),
                                          device_mesh, resharding_cache);
            }
            break;
          }
          case HloOpcode::kGetTupleElement: {
            std::vector<std::optional<HloSharding>> dst_shardings(
                inst->shape().tuple_shapes_size(), std::nullopt);
            for (size_t i = 0; i < inst->shape().tuple_shapes_size(); ++i) {
              CHECK(!inst->shape().tuple_shapes(i).IsTuple())
                  << "We currently do not support ops with nested tuples as "
                     "output.";
              const ShardingStrategy& stra =
                  GetShardingStrategyForTuple(inst, {static_cast<int64_t>(i)},
                                              strategy_map, cost_graph, s_val);
              if (!stra.input_shardings.empty() &&
                  stra.input_shardings[0].has_value()) {
                dst_shardings[i] = stra.input_shardings[0].value();
              }
            }
            FixMixedMeshShapeReshardingGetTupleElementWithTupleOutput(
                inst, dst_shardings, device_mesh, preserve_shardings);
            break;
          }

          case HloOpcode::kWhile:
          case HloOpcode::kInfeed:
          case HloOpcode::kConditional:
          case HloOpcode::kParameter: {
            break;
          }
          default:
            LOG(FATAL) << "Unhandled instruction: " + inst->ToString();
        }
      } else {
        const ShardingStrategy& stra =
            GetShardingStrategy(inst, strategy_map, cost_graph, s_val);

        if (stra.input_shardings.empty()) {
          continue;
        }
        if (inst->opcode() == HloOpcode::kGetTupleElement) {
          FixMixedMeshShapeReshardingGetTupleElement(
              inst, inst->sharding(), device_mesh, preserve_shardings);
        } else {
          for (size_t i = 0; i < inst->operand_count(); ++i) {
            if (stra.input_shardings.size() > i &&
                stra.input_shardings[i].has_value()) {
              FixMixedMeshShapeResharding(inst, i,
                                          stra.input_shardings[i].value(),
                                          device_mesh, resharding_cache);
            }
          }
        }
      }
    }
  }
}

// Print liveness set for debugging.
std::string PrintLivenessSet(const LivenessSet& liveness_set) {
  std::string str("Liveness Set\n");
  for (LivenessIdx i = 0; i < liveness_set.size(); ++i) {
    std::vector<std::string> names;
    names.reserve(liveness_set[i].size());
    for (const HloValue* value : liveness_set[i]) {
      names.push_back(absl::StrCat(value->instruction()->name(),
                                   value->index().ToString()));
    }
    std::sort(names.begin(), names.end());
    absl::StrAppend(&str, "Time ", i, ": ", absl::StrJoin(names, ", "), "\n");
  }
  return str;
}

// Print sorted instructions.
std::string PrintInstructions(const HloInstructionSequence& sequence) {
  std::string str;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    absl::StrAppend(&str, "Instruction ", i, ": ", instructions[i]->ToString(),
                    "\n");
  }
  return str;
}

// Print strategy map for debugging.
std::string PrintStrategyMap(const StrategyMap& strategy_map,
                             const HloInstructionSequence& sequence) {
  std::string str("Strategy Map\n");
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    absl::StrAppend(&str, "Instruction ", i, ": ", instructions[i]->ToString(),
                    "\n", strategy_map.at(instructions[i])->ToString());
  }
  return str;
}

// Print the chosen auto sharding strategy for debugging.
// TODO (zhuohan): update the following function
std::string PrintAutoShardingSolution(const HloInstructionSequence& sequence,
                                      const LivenessSet& liveness_set,
                                      const StrategyMap& strategy_map,
                                      const LeafStrategies& leaf_strategies,
                                      const CostGraph& cost_graph,
                                      absl::Span<const NodeStrategyIdx> s_val,
                                      double objective) {
  std::string str("=== Auto sharding strategy ===\n");
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  size_t N = leaf_strategies.size();

  // Print the chosen strategy
  for (NodeIdx i = 0; i < N; ++i) {
    absl::StrAppend(&str, i, " ",
                    instructions[leaf_strategies[i]->instruction_id]->ToString(
                        HloPrintOptions::ShortParsable()),
                    " ");
    NodeStrategyIdx stra_idx = cost_graph.RemapIndex(i, s_val[i]);
    if (cost_graph.follow_idx_[i] < 0) {
      absl::StrAppend(
          &str, leaf_strategies[i]->leaf_vector[stra_idx].ToString(), "\n");
    } else {
      absl::StrAppend(&str,
                      leaf_strategies[i]->leaf_vector[stra_idx].ToString(),
                      " follow ", cost_graph.follow_idx_[i], "\n");
    }
  }

  return str;
}

std::string PrintSolutionMemoryUsage(const LivenessSet& liveness_set,
                                     const StrategyMap& strategy_map,
                                     const CostGraph& cost_graph,
                                     absl::Span<const NodeStrategyIdx> s_val) {
  // Print the memory usage
  std::string str("=== Memory ===\n");
  std::vector<std::pair<LivenessIdx, double>> time_memory_usage;
  // Function that gets the memory usage of a StrategyVector belongs to one
  // tensor.
  std::function<double(const StrategyVector*)> calculate_memory_usage;
  calculate_memory_usage = [&](const StrategyVector* strategies) {
    if (strategies->is_tuple) {
      double m = 0.0;
      for (const auto& child : strategies->childs) {
        m += calculate_memory_usage(child.get());
      }
      return m;
    }
    NodeIdx ins_idx = strategies->node_idx;
    NodeStrategyIdx stra_idx = cost_graph.RemapIndex(ins_idx, s_val[ins_idx]);
    const ShardingStrategy& strategy = strategies->leaf_vector[stra_idx];
    return strategy.memory_cost;
  };
  for (LivenessIdx t = 0; t < liveness_set.size(); ++t) {
    double mem = 0.0;
    for (const auto& val : liveness_set.at(t)) {
      const HloInstruction* ins = val->instruction();
      auto tmp = calculate_memory_usage(strategy_map.at(ins).get());
      mem += tmp;

      if (VLOG_IS_ON(6) && tmp / (1024 * 1024) > 1) {
        // Prints out the largest tensors.
        absl::StrAppend(&str, "  ", ins->name(),
                        ": mem += ", tmp / (1024 * 1024),
                        " MB; mem=", mem / (1024 * 1024), " MB\n");
      }
    }
    time_memory_usage.push_back(std::make_pair(t, mem));
    if (VLOG_IS_ON(6)) {
      absl::StrAppend(&str, "Time ", t, ": ", mem / (1024 * 1024), " MB\n");
    }
  }

  struct {
    bool operator()(std::pair<LivenessIdx, double> a,
                    std::pair<LivenessIdx, double> b) const {
      return a.second > b.second;
    }
  } TimeMemLarger;
  std::sort(time_memory_usage.begin(), time_memory_usage.end(), TimeMemLarger);

  absl::StrAppend(&str,
                  "Using memory costs from ShardingStrategy, the max memory "
                  "consumption is ",
                  time_memory_usage.front().second / (1024 * 1024 * 1024),
                  " GB at time ", time_memory_usage.front().first, "\n");

  // Gets largest tensors in top k time steps.
  size_t k = 3;
  k = std::min(k, time_memory_usage.size());
  std::vector<std::pair<std::string, double>> instruction_mem;
  for (size_t t = 0; t < k; t++) {
    for (const auto& val : liveness_set[time_memory_usage.at(t).first]) {
      const HloInstruction* ins = val->instruction();
      auto mem = calculate_memory_usage(strategy_map.at(ins).get());
      if (mem > 100 * 1024 * 1024) {
        instruction_mem.push_back(std::make_pair(
            absl::StrCat(ins->name(), val->index().ToString()), mem));
      }
    }
  }

  struct {
    bool operator()(std::pair<std::string, double> a,
                    std::pair<std::string, double> b) const {
      return a.second > b.second;
    }
  } NameMemLarger;
  std::sort(instruction_mem.begin(), instruction_mem.end(), NameMemLarger);

  size_t top_tensors = 10;
  top_tensors = std::min(top_tensors, instruction_mem.size());
  absl::StrAppend(&str, "Top ", top_tensors, " largest tensors:\n");
  for (size_t i = 0; i < top_tensors; i++) {
    absl::StrAppend(
        &str, "instruction name: ", instruction_mem.at(i).first,
        " memory usage: ", instruction_mem.at(i).second / (1024 * 1024 * 1024),
        "GB\n");
  }

  return str;
}

void SaveShardingForInstruction(
    absl::flat_hash_map<std::string, std::vector<HloSharding>>&
        preserve_shardings,
    HloInstruction* inst) {
  if (!inst->has_sharding()) {
    return;
  }
  if (!inst->sharding().IsTuple()) {
    preserve_shardings[inst->name()] = {inst->sharding()};
  } else {
    preserve_shardings[inst->name()] = inst->sharding().tuple_elements();
  }
}

// Saves the user shardings that need to be preserved, and check whether they
// are preserved after this pass.
absl::flat_hash_map<std::string, std::vector<HloSharding>> SaveUserShardings(
    HloModule* module,
    const absl::flat_hash_set<std::string>& replicated_small_tensors,
    AutoShardingOption::PreserveShardingsType type) {
  absl::flat_hash_map<std::string, std::vector<HloSharding>> preserve_shardings;
  if (type == AutoShardingOption::PreserveShardingsType::kKeepAllShardings) {
    // Saves shardings for all instructions.
    for (const auto computation : module->computations()) {
      for (const auto inst : computation->instructions()) {
        SaveShardingForInstruction(preserve_shardings, inst);
        for (const auto user : inst->users()) {
          // Also preserve the shardings of copy ops that are the users of those
          // instructions.
          if (user->opcode() == HloOpcode::kCopy) {
            SaveShardingForInstruction(preserve_shardings, user);
          }
        }
      }
    }
  } else if (type == AutoShardingOption::PreserveShardingsType::
                         kKeepInputOutputShardings) {
    // Saves parameter shardings
    for (const auto inst :
         module->entry_computation()->parameter_instructions()) {
      SaveShardingForInstruction(preserve_shardings, inst);
      for (const auto user : inst->users()) {
        // Also preserve the shardings of copy ops that are the users of those
        // instructions.
        if (user->opcode() == HloOpcode::kCopy) {
          SaveShardingForInstruction(preserve_shardings, user);
        }
      }
    }
    for (const auto computation : module->computations()) {
      for (const auto inst : computation->instructions()) {
        if (inst->opcode() == HloOpcode::kOutfeed ||
            replicated_small_tensors.count(inst->name())) {
          SaveShardingForInstruction(preserve_shardings, inst);
        }
      }
    }
    // Saves output shardings
    auto inst = module->entry_computation()->root_instruction();
    SaveShardingForInstruction(preserve_shardings, inst);
  }

  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "User shardings that need to be kept (printing only the 1st "
                 "elemenet of tuples): ";
    for (const auto& tmp : preserve_shardings) {
      std::string sharding;
      for (const auto& s : tmp.second) {
        sharding += s.ToString() + ",";
      }
      LOG(INFO) << tmp.first << ": " << sharding;
    }
  }
  return preserve_shardings;
}

// Check whether the shardings that need to be perserved are preserved.
void CheckUserShardingPreservation(
    HloModule* module,
    const absl::flat_hash_map<std::string, std::vector<HloSharding>>&
        preserve_shardings) {
  for (const auto computation : module->computations()) {
    for (const auto inst : computation->instructions()) {
      if (preserve_shardings.find(inst->name()) == preserve_shardings.end()) {
        continue;
      }
      if (!inst->has_sharding()) {
        LOG(FATAL) << "User sharding is not preserved! Instruction with name "
                   << inst->name() << " should be: "
                   << preserve_shardings.at(inst->name())[0].ToString()
                   << "\nbut it's empty.";
      } else if (!inst->sharding().IsTuple() &&
                 preserve_shardings.at(inst->name())[0] != inst->sharding()) {
        LOG(FATAL) << "User sharding is not preserved! Instruction with name "
                   << inst->name() << " should be: "
                   << preserve_shardings.at(inst->name())[0].ToString()
                   << "\nbut it's: " << inst->sharding().ToString();
      } else if (inst->sharding().IsTuple()) {
        const std::vector<HloSharding>* preserve_shardings_tuple =
            &preserve_shardings.at(inst->name());
        for (size_t i = 0; i < inst->shape().tuple_shapes_size(); i++) {
          if (preserve_shardings_tuple->at(i) !=
              inst->sharding().tuple_elements().at(i)) {
            LOG(FATAL) << "Tuple sharding is not preserved! Instruction "
                          "with name "
                       << inst->name() << " " << i << "th tuple element "
                       << " should be: "
                       << preserve_shardings_tuple->at(i).ToString()
                       << "\nbut it's: "
                       << inst->sharding().tuple_elements().at(i).ToString();
          }
        }
      }
    }
  }
}

int64_t MemoryBudgetLowerBound(const HloModule& module,
                               const LivenessSet& liveness_set,
                               const HloAliasAnalysis* alias_analysis,
                               int64_t num_devices) {
  auto get_value_sharding = [](const HloValue* value) {
    return !value->index().empty()
               ? value->instruction()->sharding().GetSubSharding(
                     value->instruction()->shape(), value->index())
               : value->instruction()->sharding();
  };

  // We below, that is HloValues A and B alias, and A has a sharding specified,
  // the same sharding is also used to compute the per-device memory
  // requirements of B. This can be done by associating shardings with buffers
  // as aliasing HloValues are mapped to the same buffer.
  absl::flat_hash_map<HloBuffer::Id, const HloValue*>
      buffer_to_sharded_value_mapping;
  for (LivenessIdx t = 0; t < liveness_set.size(); ++t) {
    for (const HloValue* value : liveness_set[t]) {
      auto buffer = alias_analysis->GetBufferContainingValue(*value);
      if (value->instruction()->has_sharding()) {
        auto this_value_sharding = get_value_sharding(value);
        auto iter = buffer_to_sharded_value_mapping.find(buffer.id());
        if (iter != buffer_to_sharded_value_mapping.end()) {
          auto buffer_value_sharding = get_value_sharding(iter->second);
          if (this_value_sharding != buffer_value_sharding) {
            // TODO(pratikf): This is an unavoidable situation, but possibly
            // there is a better design decision that can be made here.
            VLOG(1) << "We have a situation where two HloValues alias, but "
                       "they have different shardings. This can happen in the "
                       "presence of user-specified shardings, and is expected. "
                       "This, however, means that the memory budget estimate "
                       "is not very accurate. The aliasing HLOs are "
                    << value->ToShortString() << " and "
                    << iter->second->ToShortString();
          }
        }
        buffer_to_sharded_value_mapping[buffer.id()] = value;
      }
    }
  }

  int64_t max_memory_usage = 0;
  for (LivenessIdx t = 0; t < liveness_set.size(); ++t) {
    int64_t memory_usage = 0;
    for (const HloValue* value : liveness_set[t]) {
      if (value->instruction()->shape().IsTuple() && value->index().empty()) {
        continue;
      }
      Shape shape =
          ShapeUtil::GetSubshape(value->instruction()->shape(), value->index());
      auto buffer = alias_analysis->GetBufferContainingValue(*value);
      auto iter = buffer_to_sharded_value_mapping.find(buffer.id());
      std::optional<HloSharding> optional_sharding = std::nullopt;
      if (iter != buffer_to_sharded_value_mapping.end()) {
        optional_sharding = get_value_sharding(iter->second);
      }
      memory_usage +=
          GetShardedInstructionSize(shape, num_devices, optional_sharding);
    }
    max_memory_usage = std::max(max_memory_usage, memory_usage);
  }
  return max_memory_usage;
}

void RecoverShardingsFromPartialMesh(
    const HloInstructionSequence& sequence,
    const absl::flat_hash_map<std::string, std::vector<HloSharding>>&
        preserve_shardings) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  for (HloInstruction* ins : instructions) {
    auto preserved_sharding_iter = preserve_shardings.find(ins->name());
    if (preserved_sharding_iter != preserve_shardings.end()) {
      const auto& preserved_sharding = preserved_sharding_iter->second;

      if (ins->shape().IsTuple() || (ins->opcode() == HloOpcode::kOutfeed &&
                                     preserved_sharding.size() > 1)) {
        Shape output_tuple_sharding_shape = ins->shape();
        if (ins->opcode() == HloOpcode::kOutfeed) {
          std::vector<Shape> tuple_elements_shape(
              ins->operand(0)->shape().tuple_shapes().begin(),
              ins->operand(0)->shape().tuple_shapes().end());
          tuple_elements_shape.push_back(ins->operand(1)->shape());
          output_tuple_sharding_shape =
              ShapeUtil::MakeTupleShape(tuple_elements_shape);
        }
        ShapeTree<HloSharding> output_tuple_sharding(
            output_tuple_sharding_shape, Undefined());
        size_t i = 0;
        for (auto& leaf : output_tuple_sharding.leaves()) {
          leaf.second = preserved_sharding.at(i++);
        }
        ins->set_sharding(HloSharding::Tuple(output_tuple_sharding));
      } else {
        ins->set_sharding(preserved_sharding.at(0));
      }
    }
  }
}
// DFS to find the replicated set starting from cur instruction.
void FindReplicateSet(
    HloInstruction* cur, const AliasMap& alias_map, const CostGraph& cost_graph,
    absl::Span<const NodeStrategyIdx> s_val, const StrategyMap& strategy_map,
    const ShardingStrategy& strategy, const HloInstruction* output,
    bool do_all_gather_after_backward, HloInstruction*& transpose_inst,
    StableHashSet<HloInstruction*>& replicated_set,
    StableHashSet<HloInstruction*>& boundary_set,
    StableHashSet<HloInstruction*>& consumer_set,
    StableHashSet<const HloInstruction*>& visited) {
  visited.insert(cur);

  // Check whether the node is a boundary node.
  StableHashSet<HloInstruction*> users = UsersWithAlias(cur, alias_map, output);
  for (HloInstruction* consumer : users) {
    const HloInstruction* shape_inst = cur;

    // Allow at most one transpose
    if (consumer->opcode() == HloOpcode::kTranspose &&
        (transpose_inst == nullptr ||
         DimensionsEqual(transpose_inst->shape(), consumer->shape()))) {
      shape_inst = consumer;
      transpose_inst = consumer;
      // TODO(zhuohan): fix output_sharding comparison.
    }

    if (consumer->opcode() == HloOpcode::kTuple ||
        (do_all_gather_after_backward && IsParameterConvert(consumer)) ||
        GetShardingStrategy(consumer, strategy_map, cost_graph, s_val)
                .output_sharding != strategy.output_sharding ||
        !DimensionsEqual(consumer->shape(), shape_inst->shape())) {
      boundary_set.insert(cur);
      return;
    }
  }

  // If this node is not a boundary node, propagate from this node.
  replicated_set.insert(cur);
  for (HloInstruction* consumer : users) {
    if (!visited.contains(consumer)) {
      consumer_set.insert(consumer);
      FindReplicateSet(consumer, alias_map, cost_graph, s_val, strategy_map,
                       strategy, output, do_all_gather_after_backward,
                       transpose_inst, replicated_set, boundary_set,
                       consumer_set, visited);
    }
  }

  for (size_t i = 0; i < cur->operand_count(); ++i) {
    HloInstruction* operand = cur->mutable_operand(i);
    operand = PassThroughCustomCallMarkerOperand(operand, cur);

    if (!visited.contains(operand) && !IsAlwaysReplicated(operand) &&
        GetShardingStrategy(operand, strategy_map, cost_graph, s_val)
                .output_sharding == strategy.output_sharding &&
        DimensionsEqual(operand->shape(), cur->shape())) {
      FindReplicateSet(operand, alias_map, cost_graph, s_val, strategy_map,
                       strategy, output, do_all_gather_after_backward,
                       transpose_inst, replicated_set, boundary_set,
                       consumer_set, visited);
    }
  }
}

// Substitute all-reduce strategies with their reduce-scatter variants.
void GenerateReduceScatter(const HloInstructionSequence& sequence,
                           const AliasMap& alias_map,
                           const InstructionDepthMap& depth_map,
                           const StrategyMap& strategy_map,
                           const CostGraph& cost_graph,
                           absl::Span<const NodeStrategyIdx> s_val,
                           const ClusterEnvironment& cluster_env,
                           const AutoShardingSolverOption& solver_option) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Propagation ends at output
  const HloInstruction* output = instructions.back();
  if (IsCustomCallMarker(output)) {
    output = output->operand(0);
  }

  // A debug option: whether to do all-gather after backward pass.
  // This controls the location of all-gather.
  // If true, all-gather happens after backward pass, which is desired for
  // gradient accumulation. If false, all-gather happens before forward pass,
  // which can partitions more tensors.
  bool do_all_gather_after_backward = true;

  // If true, do not actually generate reduce-scatter + all-gather,
  // but generate all-reduce + all-gather instead.
  // This saves less memory but is more friendly to gradient accumulation.
  // This is a temporary workaround due to implementation difficulty.
  // Ideally, we should be able to generate a gradient-accumulation-friendly
  // reduce-scatter + all-gather, but for now it is not easy to implement this
  // in our current system. So we generate a gradient-accumulation-friendly
  // all-reduce + all-gather, which has the same memory consumption but with 50%
  // communication overhead.
  bool use_all_reduce_for_grad_acc =
      solver_option.reduce_scatter_grad_acc_friendly;

  std::vector<HloInstruction*> insert_all_gather;
  StableHashSet<const HloInstruction*> modified;

  for (HloInstruction* inst : instructions) {
    if (!HasReduceScatterOpportunity(inst, strategy_map, cost_graph, s_val,
                                     modified)) {
      continue;
    }
    const ShardingStrategy& strategy =
        GetShardingStrategy(inst, strategy_map, cost_graph, s_val);
    if (!absl::StrContains(strategy.name, "allreduce")) {
      continue;
    }

    StableHashSet<HloInstruction*> replicated_set;
    StableHashSet<HloInstruction*> boundary_set;
    StableHashSet<HloInstruction*> consumer_set;
    StableHashSet<const HloInstruction*> visited;

    // We allow at most one transpose in the path of replication analysis.
    HloInstruction* transpose_inst = nullptr;

    // Find the replicated set starting from the all-reduce instruction.
    visited.insert(output);
    FindReplicateSet(inst, alias_map, cost_graph, s_val, strategy_map, strategy,
                     output, do_all_gather_after_backward, transpose_inst,
                     replicated_set, boundary_set, consumer_set, visited);

    // Try to reduce the boundary set to its common ancestor
    TryReduceWithCommonAncestor(replicated_set, boundary_set, consumer_set,
                                alias_map);

    // Analyze the instructions after which all-gather should be inserted.
    std::vector<HloInstruction*> need_all_gather;
    for (HloInstruction* node : boundary_set) {
      if (consumer_set.contains(node)) {
        if (AllUsersAreReduce(node)) {
          // If users are reduce, the all-gather cost after this instruction
          // should be small, so we ignore all-gather cost of these
          // instructions.
          replicated_set.insert(node);
        } else {
          need_all_gather.push_back(node);
        }
      }
    }

    // If we do all-gather on some parameters, move this all-gather after
    // backward.
    if (do_all_gather_after_backward && need_all_gather.size() == 1) {
      HloInstruction* point = need_all_gather.front();
      std::vector<HloInstruction*> path;
      HloInstruction* root = point;
      while (true) {
        path.push_back(root);
        if (root->opcode() == HloOpcode::kGetTupleElement) {
          root = PassThroughCustomCallMarkerOperand(root->mutable_operand(0),
                                                    root);
        } else {
          break;
        }
      }

      if (root->opcode() == HloOpcode::kParameter) {
        for (auto x : path) {
          replicated_set.erase(x);
          boundary_set.erase(x);
        }
        need_all_gather.clear();
        for (auto x : replicated_set) {
          auto iter = alias_map.find(x);
          if (iter != alias_map.end() && iter->second == root) {
            boundary_set.insert(x);
            need_all_gather.push_back(x);
            break;
          }
        }
      }
    }

    // Analyze how many parameters can be partitioned if we do this
    // transformation.
    int num_replicated_parameters = 0;
    for (const HloInstruction* node : replicated_set) {
      if (node->opcode() == HloOpcode::kParameter) {
        num_replicated_parameters++;
      }
    }
    for (const HloInstruction* to_split : need_all_gather) {
      if (to_split->users().size() == 1 &&
          to_split->users().front() == output && alias_map.contains(to_split)) {
        // Move the all-gather to its alias parameter.
        num_replicated_parameters++;
      }
    }

    // Print replicated set and boundary set for debugging.
    VLOG(10) << inst->ToString(HloPrintOptions::ShortParsable()) << "\n";
    VLOG(10) << "replicated set (#parameter: " << num_replicated_parameters
             << "):\n";
    for (auto x : replicated_set) {
      VLOG(10) << "  " << x->ToString(HloPrintOptions::ShortParsable()) << "\n";
    }
    VLOG(10) << "boundary set (#incompatible: " << need_all_gather.size()
             << "):\n";
    for (auto x : boundary_set) {
      VLOG(10) << "  " << x->ToString(HloPrintOptions::ShortParsable()) << " "
               << absl::c_linear_search(need_all_gather, x) << "\n";
    }

    // If applicable, replace all-reduce with reduce-scatter by
    // setting instructions' sharding.
    if (num_replicated_parameters >= 1 && need_all_gather.size() <= 1 &&
        replicated_set.size() >= 5) {
      HloSharding output_spec =
          GetReduceScatterOutput(inst, strategy, cluster_env);
      if (IsUndefined(output_spec)) {
        continue;
      }

      VLOG(10) << "SET: " << output_spec.ToString();

      if (absl::StartsWith(strategy.name, "RR = RS x SR")) {
        // If set the sharding for this dot instruction, the SPMD
        // partitioner will generate bad fallback code.
        replicated_set.erase(inst);
      }

      if (use_all_reduce_for_grad_acc) {
        UseAllReduceForGradAcc(replicated_set, inst);
      }

      for (HloInstruction* to_split : replicated_set) {
        SetSharding(to_split, output_spec, inst, transpose_inst, modified);
      }

      if (!solver_option.reduce_scatter_aggressive_partition) {
        // The normal case
        for (HloInstruction* to_split : need_all_gather) {
          SetSharding(to_split, output_spec, inst, transpose_inst, modified);

          if (!do_all_gather_after_backward && to_split->users().size() == 1 &&
              to_split->users().front() == output &&
              alias_map.contains(to_split)) {
            // Move the all-gather to its alias parameter.
            // This partitions more tensors but introduces communication
            // in the forward pass, which is not desired in gradient
            // accumulation.
            SetSharding(alias_map.at(to_split), output_spec, inst,
                        transpose_inst, modified);
            insert_all_gather.push_back(alias_map.at(to_split));
          } else {
            insert_all_gather.push_back(to_split);

            if (to_split->opcode() == HloOpcode::kGetTupleElement &&
                IsCustomCallMarker(to_split->operand(0)) &&
                to_split->users().size() == 1 &&
                to_split->users().front() == output) {
              insert_all_gather.push_back(PassThroughCustomCallMarkerOperand(
                  to_split->mutable_operand(0), to_split));
            }
          }
        }
      } else {
        // Aggressively partition more parameter tensors.
        // This can result in a strategy similar to ZeRO stage 3.
        // NOTE: The combination of this branch with pipeline parallel is not
        // tested.
        for (HloInstruction* to_split : need_all_gather) {
          SetSharding(to_split, output_spec, inst, transpose_inst, modified);

          if (to_split->users().size() == 1 &&
              to_split->users().front() == output &&
              alias_map.contains(to_split)) {
            // Move the all-gather to its alias parameter.
            HloInstruction* param = alias_map.at(to_split);

            // Find the branching point (i.e., skip elementwise ops like
            // convert)
            HloInstruction* cur = param;
            while (cur->users().size() == 1) {
              // TODO(zhuohan): handle tuple.
              CHECK(cur->shape().IsArray());
              SetSharding(cur, output_spec, inst, transpose_inst, modified);
              cur = cur->users().front();
            }
            SetSharding(cur, output_spec, inst, transpose_inst, modified);

            CHECK(!cur->users().empty());

            // Find the first user
            HloInstruction* first_user = nullptr;
            int64_t min_depth = ((int64_t)1) << 50;
            for (const auto& x : cur->users()) {
              auto iter = depth_map.find(x);
              if (iter == depth_map.end()) {
                LOG(FATAL) << "ERROR: " << x->ToString();
              }
              if (x->opcode() != HloOpcode::kConvolution &&
                  x->opcode() != HloOpcode::kDot) {
                // Only apply this aggressive optimization for dot and conv
                continue;
              }
              if (iter->second < min_depth) {
                first_user = x;
                min_depth = iter->second;
              }
            }

            if (first_user != nullptr) {
              // Insert an identity to prevent CSE of all-gather
              HloInstruction* identity = inst->parent()->AddInstruction(
                  HloInstruction::CreateCustomCall(cur->shape(), {cur},
                                                   kIdentityMarker));
              SetSharding(identity, output_spec, inst, transpose_inst,
                          modified);
              ReplaceOperand(first_user, cur, identity);
            }
          }
        }
      }
    }

    VLOG(10) << "-----------------------done\n";
  }

  // Insert all-gather on the output of boundary nodes by setting
  // their shardings. This also works as CSE of all-gather.
  for (HloInstruction* inst : insert_all_gather) {
    HloInstruction* replace_with = inst->parent()->AddInstruction(
        HloInstruction::CreateReshape(inst->shape(), inst));
    replace_with->set_sharding(
        GetShardingStrategy(inst, strategy_map, cost_graph, s_val)
            .output_sharding);
    TF_CHECK_OK(inst->ReplaceAllUsesWith(replace_with));
  }
}

void AnnotateShardingWithSimpleHeuristic(
    HloModule* module, const std::string& heuristic, const AliasMap& alias_map,
    const ClusterEnvironment& cluster_env) {
  const Array<int64_t>& device_mesh = cluster_env.device_mesh_;
  const Array<int64_t>& device_mesh_1d = cluster_env.device_mesh_1d_;
  int64_t num_devices = device_mesh.num_elements();

  // Count the non-one mesh dimension.
  size_t mesh_nn_dims = 0;
  for (int dim : device_mesh.dimensions()) {
    if (dim > 1) {
      mesh_nn_dims++;
    }
  }

  // Shard instructions
  HloComputation* entry_computation = module->entry_computation();
  for (HloInstruction* inst : entry_computation->instructions()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      HloSharding output_spec = HloSharding::Replicate();
      inst->set_sharding(output_spec);

      if (heuristic == "shard-largest") {
        std::vector<int64_t> lengths;
        for (int64_t i = 0; i < inst->shape().rank(); ++i) {
          lengths.push_back(inst->shape().dimensions(i));
        }

        std::vector<int> indices = Argsort(lengths);
        int common_dims = std::min(mesh_nn_dims, indices.size());

        if (common_dims < 1) {
          continue;
        }

        if (common_dims == 1) {
          int dim = indices[0];
          int length = lengths[dim];
          if (length % num_devices == 0) {
            output_spec = Tile(inst->shape(), {dim}, {0}, device_mesh_1d);
          }
        } else {
          int dim1 = indices[0];
          int length1 = lengths[dim1];
          int dim0 = indices[1];
          int length0 = lengths[dim0];

          if (length0 % device_mesh.dim(0) == 0 &&
              length1 % device_mesh.dim(1) == 0) {
            output_spec =
                Tile(inst->shape(), {dim0, dim1}, {0, 1}, device_mesh);
          }
        }
      } else if (heuristic == "shard-first") {
        if (inst->shape().rank() > 0 &&
            inst->shape().dimensions(0) % num_devices == 0) {
          output_spec = Tile(inst->shape(), {0}, {0}, device_mesh_1d);
        }
      } else if (heuristic == "shard-last") {
        int64_t last_dim = inst->shape().rank() - 1;
        if (inst->shape().rank() > 0 &&
            inst->shape().dimensions(last_dim) % num_devices == 0) {
          output_spec = Tile(inst->shape(), {last_dim}, {0}, device_mesh_1d);
        }
      } else {
        LOG(FATAL) << "Invalid heuristic: " << heuristic;
      }

      inst->set_sharding(output_spec);
    } else if (inst->opcode() == HloOpcode::kDot) {
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      const DotDimensionNumbers& dot_dnums = inst->dot_dimension_numbers();
      // const auto& lhs_con_dims = dot_dnums.lhs_contracting_dimensions();
      // const auto& rhs_con_dims = dot_dnums.rhs_contracting_dimensions();
      std::vector<int64_t> lhs_space_dims, rhs_space_dims;
      std::tie(lhs_space_dims, rhs_space_dims) =
          GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);
    }
  }

  // Meet the alias requirement for the output tuple.
  HloInstruction* output = entry_computation->root_instruction();
  const Shape& out_shape = output->shape();
  ShapeTree<HloSharding> tuple_sharding(out_shape, HloSharding::Replicate());
  std::vector<HloSharding> flattened_shardings;

  std::function<void(HloInstruction*)> get_flattened_shardings;
  get_flattened_shardings = [&](HloInstruction* cur) {
    for (int64_t i = 0; i < cur->operand_count(); ++i) {
      HloInstruction* operand = cur->mutable_operand(i);

      if (operand->shape().IsTuple()) {
        get_flattened_shardings(operand);
      } else {
        if (alias_map.contains(operand)) {
          operand = alias_map.at(operand);
        }
        if (!operand->has_sharding()) {
          operand->set_sharding(HloSharding::Replicate());
        }
        CHECK(operand->has_sharding());
        flattened_shardings.push_back(operand->sharding());
      }
    }
  };
  get_flattened_shardings(output);
  int i = 0;
  for (auto& leaf : tuple_sharding.leaves()) {
    leaf.second = flattened_shardings[i++];
  }
  CHECK_EQ(i, flattened_shardings.size());
  output->set_sharding(HloSharding::Tuple(tuple_sharding));
}

// Filter strategies according to the solver_option.force_batch_dim_to_mesh_dim.
// This can be used to forcibly generate data-parallel strategies.
Status FilterStrategy(const HloInstruction* ins, const Shape& shape,
                      std::unique_ptr<StrategyVector>& strategies,
                      const ClusterEnvironment& cluster_env,
                      const InstructionBatchDimMap& batch_map,
                      const AutoShardingSolverOption& solver_option) {
  int mesh_dim = solver_option.force_batch_dim_to_mesh_dim;
  int batch_dim = batch_map.at(GetBatchDimMapKey(ins));
  const Array<int64_t>& device_mesh = cluster_env.device_mesh_;

  if (shape.dimensions(batch_dim) % device_mesh.dim(mesh_dim) != 0) {
    return absl::InvalidArgumentError(
        "The length of batch dimension is "
        "not divisible by the number of devices");
  }

  std::vector<ShardingStrategy> new_leaf_vector;
  for (auto& stra : strategies->leaf_vector) {
    std::vector<int64_t> tensor_dim_to_mesh_dim =
        cluster_env.GetTensorDimToMeshDimWrapper(shape, stra.output_sharding);

    if (device_mesh.dim(mesh_dim) > 1) {
      // If the mesh dim is not one, the output tensor must be
      // tiled along the mesh dim.
      if (tensor_dim_to_mesh_dim[batch_dim] == mesh_dim) {
        new_leaf_vector.push_back(std::move(stra));
      }
    } else {
      // If the mesh dim is one, the output tensor must be replicated
      // on the mesh dim.
      if (tensor_dim_to_mesh_dim[batch_dim] == -1) {
        new_leaf_vector.push_back(std::move(stra));
      }
    }
  }
  CHECK(!new_leaf_vector.empty())
      << ins->ToString() << " does not have any valid strategies";
  strategies->leaf_vector = std::move(new_leaf_vector);

  return OkStatus();
}

// Return the output sharding of the reduce-scatter variant of a given strategy.
HloSharding GetReduceScatterOutput(const HloInstruction* ins,
                                   const ShardingStrategy& strategy,
                                   const ClusterEnvironment& cluster_env) {
  const Array<int64_t>& device_mesh = cluster_env.device_mesh_;
  const Array<int64_t>& device_mesh_1d = cluster_env.device_mesh_1d_;

  if (ins->opcode() == HloOpcode::kDot) {
    const DotDimensionNumbers& dot_dnums = ins->dot_dimension_numbers();
    int64_t space_base_dim = dot_dnums.lhs_batch_dimensions_size();

    if (absl::StartsWith(strategy.name, "SR = SS x SR") ||
        absl::StartsWith(strategy.name, "RS = RS x SS")) {
      int mesh_dim0, mesh_dim1;
      std::tie(mesh_dim0, mesh_dim1) = ParseMeshDims(strategy.name);

      if (!IsDivisible(ins, device_mesh, {space_base_dim, space_base_dim + 1},
                       {mesh_dim0, mesh_dim1})) {
        // XLA supports uneven partitioning by adding padding.
        // However, the ShardingSpec in Jax does not support uneven
        // partitioning.
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                  {mesh_dim0, mesh_dim1}, device_mesh);
    }
    if (absl::StartsWith(strategy.name, "SbR = SbSk x SbSk")) {
      int mesh_dim0, mesh_dim1;
      std::tie(mesh_dim0, mesh_dim1) = ParseMeshDims(strategy.name);

      if (!IsDivisible(ins, device_mesh, {0, space_base_dim},
                       {mesh_dim0, mesh_dim1})) {
        // XLA supports uneven partitioning by adding padding.
        // However, the ShardingSpec in Jax does not support uneven
        // partitioning.
        return Undefined();
      }

      return Tile(ins->shape(), {0, space_base_dim}, {mesh_dim0, mesh_dim1},
                  device_mesh);
    }
    if (absl::StartsWith(strategy.name, "RR = RS x SR")) {
      int mesh_dim = absl::StrContains(strategy.name, "{0}") ? 0 : 1;

      if (!IsDivisible(ins, device_mesh, {space_base_dim}, {mesh_dim})) {
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim}, {mesh_dim}, device_mesh);
    }
    if (absl::StartsWith(strategy.name, "R = Sk x Sk")) {
      int mesh_dim = 0;

      if (!IsDivisible(ins, device_mesh_1d, {space_base_dim}, {mesh_dim})) {
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim}, {mesh_dim}, device_mesh_1d);
    }
  } else if (ins->opcode() == HloOpcode::kConvolution) {
    const ConvolutionDimensionNumbers& conv_dnums =
        ins->convolution_dimension_numbers();
    int out_batch_dim = conv_dnums.output_batch_dimension();
    int out_out_channel_dim = conv_dnums.output_feature_dimension();

    if (absl::StartsWith(strategy.name, "SR = SS x SR") ||
        absl::StartsWith(strategy.name, "RS = RS x SS")) {
      int mesh_dim0, mesh_dim1;
      std::tie(mesh_dim0, mesh_dim1) = ParseMeshDims(strategy.name);

      if (!IsDivisible(ins, device_mesh, {out_batch_dim, out_out_channel_dim},
                       {mesh_dim0, mesh_dim1})) {
        return Undefined();
      }

      return Tile(ins->shape(), {out_batch_dim, out_out_channel_dim},
                  {mesh_dim0, mesh_dim1}, device_mesh);
    }
    if (absl::StartsWith(strategy.name, "R = Sk x Sk")) {
      int mesh_dim = 0;

      if (!IsDivisible(ins, device_mesh_1d, {out_batch_dim}, {mesh_dim})) {
        return Undefined();
      }

      return Tile(ins->shape(), {out_batch_dim}, {mesh_dim}, device_mesh_1d);
    }
  } else if (ins->opcode() == HloOpcode::kReduce) {
    // TODO(zhuohan): support more cases.
    CHECK_EQ(ins->shape().rank(), 1);

    int mesh_dim;
    if (absl::StrContains(strategy.name, "allreduce @ [0]")) {
      mesh_dim = 0;
    } else {
      mesh_dim = 1;
    }

    if (strategy.output_sharding.IsReplicated()) {
      if (absl::StrContains(strategy.name, "1d")) {
        if (!IsDivisible(ins, device_mesh_1d, {0}, {mesh_dim})) {
          return Undefined();
        }

        return Tile(ins->shape(), {0}, {mesh_dim}, device_mesh_1d);
      }
      if (!IsDivisible(ins, device_mesh, {0}, {mesh_dim})) {
        return Undefined();
      }

      return Tile(ins->shape(), {0}, {mesh_dim}, device_mesh);
    }
    if (!IsDivisible(ins, device_mesh_1d, {0}, {0})) {
      return Undefined();
    }

    auto tile_assignment = strategy.output_sharding.tile_assignment().Reshape(
        {cluster_env.total_devices_});
    return HloSharding::Tile(std::move(tile_assignment));

  } else {
    LOG(FATAL) << "Invalid instruction: " << ins->ToString();
  }

  return Undefined();
}

// Return whether an instruction has the opportunity to generate reduce-scatter.
bool HasReduceScatterOpportunity(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const NodeStrategyIdx> s_val,
    const StableHashSet<const HloInstruction*>& modified) {
  // If the operand is already modified by other ops, skip this instruction to
  // avoid conflicts.
  for (const HloInstruction* operand : inst->operands()) {
    if (modified.contains(operand)) {
      return false;
    }
  }
  if (modified.contains(inst)) {
    return false;
  }

  if (inst->opcode() == HloOpcode::kReduce && inst->shape().rank() == 1) {
    return true;
  }
  if (inst->opcode() == HloOpcode::kDot) {
    if (GetShardingStrategy(inst->operand(0), strategy_map, cost_graph, s_val)
            .output_sharding.IsReplicated() &&
        GetShardingStrategy(inst->operand(1), strategy_map, cost_graph, s_val)
            .output_sharding.IsReplicated()) {
      // This dot is replicated on all devices. Do not split it.
      // TODO(zhuohan): improve this condition.
      return false;
    }

    return true;
  }
  if (inst->opcode() == HloOpcode::kConvolution) {
    return true;
  }

  return false;
}

}  // namespace spmd

StatusOr<bool> AutoShardingImplementation::RemoveShardingAnnotation(
    HloModule* module,
    const absl::flat_hash_set<std::string>& replicated_small_tensors,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (option_.preserve_shardings ==
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings) {
    return false;
  }
  VLOG(0) << "Removing user sharding annotations.";
  bool changed = false;
  absl::flat_hash_set<HloInstruction*> keep_inst;
  for (HloComputation* computation : module->computations(execution_threads)) {
    bool is_entry_computation = computation->IsEntryComputation();

    for (HloInstruction* ins : computation->instructions()) {
      // Do not remove sharding annotations from instructions replicated as they
      // are small tensors
      if (replicated_small_tensors.count(ins->name())) {
        keep_inst.insert(ins);
        continue;
      }

      // Do not remove entry computation's parameter and root instruction's
      // sharding if preserve_shardings is kKeepInputOutputShardings.
      if (option_.preserve_shardings ==
              AutoShardingOption::PreserveShardingsType::
                  kKeepInputOutputShardings &&
          (is_entry_computation &&
           (ins->opcode() == HloOpcode::kParameter || ins->IsRoot()))) {
        keep_inst.insert(ins);
        continue;
      }
      if (ins->opcode() == HloOpcode::kCopy &&
          keep_inst.find(ins->operand(0)) != keep_inst.end()) {
        continue;
      }
      if (ins->has_sharding()) {
        changed |= true;
        ins->clear_sharding();
      }
    }
  }
  return changed;
}

Status AutoShardingImplementation::CanonicalizeLayouts(HloModule* module) {
  if (!module->layout_canonicalization_callback()) {
    LOG(INFO) << "There is no registered layout_canonicalization_callback.";
    return OkStatus();
  }
  TF_ASSIGN_OR_RETURN(auto layouts,
                      module->layout_canonicalization_callback()(*module));
  std::vector<Shape>& argument_shapes = layouts.first;
  Shape& result_shape = layouts.second;
  ComputationLayout entry_computation_layout =
      module->config().entry_computation_layout();
  TF_RETURN_IF_ERROR(
      entry_computation_layout.mutable_result_layout()->CopyLayoutFromShape(
          result_shape));
  CHECK_NE(entry_computation_layout.parameter_count(), 0);
  CHECK_EQ(argument_shapes.size(), entry_computation_layout.parameter_count());
  for (int32_t i = 0; i < entry_computation_layout.parameter_count(); i++) {
    TF_RETURN_IF_ERROR(entry_computation_layout.mutable_parameter_layout(i)
                           ->CopyLayoutFromShape(argument_shapes.at(i)));
  }
  *module->config().mutable_entry_computation_layout() =
      entry_computation_layout;
  return OkStatus();
}

AutoShardingImplementation::AutoShardingImplementation(
    const AutoShardingOption& option)
    : option_(option) {}

StatusOr<AutoShardingResult> AutoShardingImplementation::RunAutoSharding(
    HloModule* module,
    const absl::flat_hash_set<std::string>& replicated_small_tensors,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!option_.enable) {
    return AutoShardingResult::kModuleUnchanged;
  }
  bool module_is_changed = false;

  bool set_to_memory_lower_bound = (option_.memory_budget_per_device == 0);
  // ----- Set options for this pass -----
  spmd::AutoShardingSolverOption solver_option;
  solver_option.override_all_gather_cost = false;
  solver_option.override_all_reduce_cost = false;
  solver_option.override_reduce_scatter_cost = false;
  solver_option.override_all_to_all_cost = false;

  if (option_.force_all_gather_cost) {
    solver_option.override_all_gather_cost = true;
    solver_option.all_gather_cost = option_.all_gather_cost;
  }
  if (option_.force_all_to_all_cost) {
    solver_option.override_all_to_all_cost = true;
    solver_option.all_to_all_cost = option_.all_to_all_cost;
  }
  solver_option.force_batch_dim_to_mesh_dim =
      option_.force_batch_dim_to_mesh_dim;
  solver_option.allow_replicated_parameters =
      option_.allow_replicated_parameters;
  solver_option.prefer_reduce_scatter = option_.prefer_reduce_scatter;
  solver_option.reduce_scatter_grad_acc_friendly =
      option_.reduce_scatter_grad_acc_friendly;
  solver_option.reduce_scatter_aggressive_partition =
      option_.reduce_scatter_aggressive_partition;
  solver_option.batch_matmul_always_split_batch =
      option_.batch_matmul_always_split_batch;
  solver_option.allow_recompute_heavy_op = option_.allow_recompute_heavy_op;
  solver_option.allow_mixed_mesh_shape = option_.allow_mixed_mesh_shape;
  solver_option.grad_acc_num_micro_batches = option_.grad_acc_num_micro_batches;
  solver_option.load_solution_vector = option_.load_solution_vector;
  solver_option.force_simple_heuristic = option_.force_simple_heuristic;
  solver_option.force_strategy = option_.force_strategy;
  solver_option.force_strategy_inst_indices =
      option_.force_strategy_inst_indices;
  solver_option.force_strategy_stra_names = option_.force_strategy_stra_names;
  solver_option.only_allow_divisible_input_output = true;
  solver_option.only_allow_divisible_intermediate = false;
  solver_option.nd_sharding_iteratively_strict_search_space = false;
  solver_option.allow_replicated_strategy_for_dot_and_conv =
      option_.allow_replicated_strategy_for_dot_and_conv;
  solver_option.allow_alias_to_follower_conversion =
      option_.allow_alias_to_follower_conversion;

  // Remove CustomCalls with custom_call_target="Sharding" and move their
  // shardings to their input ops.
  absl::flat_hash_map<const HloInstruction*, std::vector<int64_t>>
      unspecified_dims;
  auto status_or_changed = ProcessShardingInstruction(
      module, execution_threads, /*replace_sharding_with_copy=*/true,
      &unspecified_dims, /*saved_root_shardings=*/nullptr,
      /*saved_parameter_shardings=*/nullptr);
  if (!status_or_changed.ok()) {
    return status_or_changed.status();
  }
  if (status_or_changed.value()) {
    module_is_changed = true;
    VLOG(3) << "CustomCalls with custom_call_target=Sharding are removed and "
               "their shardings are moved to their input ops.";
  } else {
    VLOG(3) << "This workload does not have CustomCalls with "
               "custom_call_target=Sharding.";
  }

  absl::flat_hash_map<std::string, std::vector<HloSharding>>
      preserve_shardings = spmd::SaveUserShardings(
          module, replicated_small_tensors, option_.preserve_shardings);

  // Remove xla sharding annotations, if there is any.
  if (option_.preserve_shardings !=
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings) {
    StatusOr<bool> status_or_changed = RemoveShardingAnnotation(
        module, replicated_small_tensors, execution_threads);
    if (!status_or_changed.ok()) {
      return status_or_changed.status();
    }
    if (status_or_changed.value()) {
      module_is_changed = true;
      VLOG(3) << "XLA sharding annotations are removed.";
    } else {
      VLOG(3) << "This workload does not have XLA sharding annotations.";
    }
  }

  // ----- Get a sequential schedule and do liveness analysis -----
  auto size_fn = [](const BufferValue& buffer) {
    return spmd::GetBytes(buffer.shape());
  };
  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleModule(module, size_fn,
                     ComputationSchedulerToModuleScheduler(DFSMemoryScheduler),
                     execution_threads));
  const HloComputation* entry_computation = module->entry_computation();
  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module).value();
  spmd::AliasMap alias_map = spmd::BuildAliasMap(module);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloLiveRange> hlo_live_range,
      HloLiveRange::Run(schedule, *alias_analysis, entry_computation));
  absl::flat_hash_map<const HloValue*, HloLiveRange::TimeBound>&
      buffer_live_ranges = hlo_live_range->buffer_live_ranges();
  spmd::LivenessSet liveness_set(hlo_live_range->schedule_end_time() + 1);
  for (const auto& iter : buffer_live_ranges) {
    for (spmd::LivenessIdx i = iter.second.start; i <= iter.second.end; ++i) {
      liveness_set[i].push_back(iter.first);
    }
  }
  VLOG(10) << hlo_live_range->ToString();
  VLOG(10) << spmd::PrintLivenessSet(liveness_set);
  XLA_VLOG_LINES(10, spmd::PrintLivenessSet(liveness_set));
  const HloInstructionSequence& sequence =
      hlo_live_range->flattened_instruction_sequence();

  absl::flat_hash_map<const HloInstruction*, int64_t>
      instruction_execution_counts = spmd::ComputeInstructionExecutionCounts(
          module, option_.loop_iteration_count_estimate);

  // ----- Analyze the batch dim -----
  spmd::InstructionBatchDimMap batch_dim_map;
  // TODO(yuemmawang) Enable the batch_dim_map if it becomes helpful. This is
  // supposed to make the solver faster, but it makes it much much slower for
  // both 1D and 2D mesh shapes.
  // batch_dim_map = spmd::BuildInstructionBatchDimMap(sequence);
  // ----- Read parameters of device mesh ----
  Array<int64_t> original_device_mesh(option_.device_mesh_shape);
  original_device_mesh.SetValues(option_.device_mesh_ids);
  int64_t original_memory_budget = option_.memory_budget_per_device;

  std::vector<std::vector<int64_t>> partial_mesh_shapes;
  if (option_.solve_nd_sharding_iteratively) {
    // Generate partial mesh shapes to optimize iteratively.
    partial_mesh_shapes = spmd::DecomposeMeshShapes(option_.device_mesh_shape);
  } else {
    partial_mesh_shapes = {option_.device_mesh_shape};
  }

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  for (size_t mesh_idx = 0; mesh_idx < partial_mesh_shapes.size(); ++mesh_idx) {
    // Adjust existing shardings with current partial mesh shapes;
    std::vector<int64_t> mesh_shape = partial_mesh_shapes[mesh_idx];
    LOG(INFO) << "Processing partial mesh shape: "
              << spmd::ToString(mesh_shape);
    Array<int64_t> device_mesh(mesh_shape);

    int64_t total_devices = 1;
    for (auto i : mesh_shape) {
      total_devices *= i;
    }
    if (mesh_idx != partial_mesh_shapes.size() - 1) {
      auto changed_or = spmd::AdjustShardingsWithPartialMeshShape(
          sequence.instructions(), mesh_shape, total_devices,
          /* crash_on_error */ !option_.try_multiple_mesh_shapes);
      if (changed_or.ok()) {
        LOG(INFO)
            << "Shardings are adjusted based on current partial mesh shape: "
            << *changed_or;
      } else {
        return changed_or.status();
      }
    }
    std::vector<int64_t> device_mesh_ids = std::vector<int64_t>(total_devices);
    std::iota(device_mesh_ids.begin(), device_mesh_ids.end(), 0);
    device_mesh.SetValues(device_mesh_ids);

    // TODO (zhuohan): include the prof result as an option.
    spmd::ProfilingResult prof_result;
    spmd::ClusterEnvironment cluster_env(
        original_device_mesh, device_mesh, option_.device_mesh_alpha,
        option_.device_mesh_beta, prof_result, solver_option);

    XLA_VLOG_LINES(1, module->ToString());
    int64_t memory_lower_bound = spmd::MemoryBudgetLowerBound(
        *module, liveness_set, alias_analysis.get(),
        device_mesh.num_elements());
    // Rounds up to the next GB.
    int64_t memory_lower_bound_gb =
        1 + memory_lower_bound / (1024 * 1024 * 1024);
    LOG(INFO) << "Memory consumption lower bound is " << memory_lower_bound_gb
              << " GB.";
    if (set_to_memory_lower_bound) {
      LOG(INFO)
          << "--xla_tpu_auto_spmd_partitioning_memory_budget_gb is 0, and "
             "--xla_tpu_auto_spmd_partitioning_memory_budget_ratio is "
          << option_.memory_budget_ratio
          << ", so setting "
             "option.memory_budget_per_device to "
          << memory_lower_bound_gb << " x " << option_.memory_budget_ratio
          << " = " << memory_lower_bound_gb * option_.memory_budget_ratio
          << " GB";
      option_.memory_budget_per_device = memory_lower_bound_gb *
                                         (1024 * 1024 * 1024) *
                                         option_.memory_budget_ratio;
    } else if (option_.memory_budget_per_device > 0) {
      option_.memory_budget_per_device = original_memory_budget *
                                         original_device_mesh.num_elements() /
                                         device_mesh.num_elements();
      LOG(INFO) << "Setting option.memory_budget_per_device to "
                << option_.memory_budget_per_device;
    }

    if (!solver_option.force_simple_heuristic.empty()) {
      AnnotateShardingWithSimpleHeuristic(
          module, solver_option.force_simple_heuristic, alias_map, cluster_env);
      return AutoShardingResult::kModuleChangedShardingPerformed;
    }

    if (solver_option.force_batch_dim_to_mesh_dim >= 0) {
      DisableIncompatibleMixedMeshShapeAndForceBatchDim(
          batch_dim_map, sequence.instructions(), device_mesh.num_elements(),
          solver_option);
    }

    // ----- Analyze depth -----
    spmd::InstructionDepthMap ins_depth_map;
    ins_depth_map = spmd::BuildInstructionDepthMap(sequence, batch_dim_map);
    // ----- Build strategies and costs -----
    spmd::StrategyMap strategy_map;
    spmd::LeafStrategies leaf_strategies;
    spmd::AssociativeDotPairs associative_dot_pairs;

    TF_ASSIGN_OR_RETURN(
        std::tie(strategy_map, leaf_strategies, associative_dot_pairs),
        BuildStrategyAndCost(sequence, module, instruction_execution_counts,
                             ins_depth_map, batch_dim_map, alias_map,
                             cluster_env, solver_option, *call_graph,
                             option_.try_multiple_mesh_shapes));
    spmd::AliasSet alias_set = spmd::BuildAliasSet(module, strategy_map);
    CheckAliasSetCompatibility(alias_set, leaf_strategies, sequence);
    XLA_VLOG_LINES(8, PrintStrategyMap(strategy_map, sequence));

    // ----- Build cost graph and merge unimportant nodes -----
    spmd::CostGraph cost_graph(leaf_strategies, associative_dot_pairs);
    cost_graph.Simplify(option_.simplify_graph);

    // ----- Call the ILP Solver -----
    std::vector<spmd::NodeStrategyIdx> s_val;
    std::vector<spmd::EdgeStrategyIdx> e_val;
    double objective = -1.0;
    if (!solver_option.load_solution_vector) {
      auto solver_result =
          Solve(*hlo_live_range, liveness_set, strategy_map, leaf_strategies,
                cost_graph, alias_set, option_);
      if (solver_result.skip_auto_sharding) {
        return AutoShardingResult::kModuleUnchangedNoShardingPerfomed;
      } else if (!solver_result.status.ok()) {
        return AutoShardingResult::kModuleUnchanged;
      } else {
        TF_ASSIGN_OR_RETURN(auto solution, solver_result.status);
        std::tie(s_val, e_val, objective) = solution;
        if (mesh_idx == partial_mesh_shapes.size() - 1) {
          this->solver_optimal_objective_value_ = objective;
        }
      }
    } else {
      s_val = option_.strategy_vector;
    }

    XLA_VLOG_LINES(5, PrintAutoShardingSolution(sequence, liveness_set,
                                                strategy_map, leaf_strategies,
                                                cost_graph, s_val, objective));
    XLA_VLOG_LINES(1, PrintSolutionMemoryUsage(liveness_set, strategy_map,
                                               cost_graph, s_val));

    // ----- Substitute all-reduce with reduce-scatter -----
    if (solver_option.prefer_reduce_scatter) {
      GenerateReduceScatter(sequence, alias_map, ins_depth_map, strategy_map,
                            cost_graph, s_val, cluster_env, solver_option);
    }
    // ----- Set Sharding -----
    SetHloSharding(sequence, strategy_map, cost_graph, s_val,
                   (mesh_idx == partial_mesh_shapes.size() - 1));
    if (mesh_idx == partial_mesh_shapes.size() - 1) {
      SetHloShardingPostProcessing(sequence, strategy_map, cost_graph, s_val,
                                   cluster_env, &preserve_shardings);
    } else {
      spmd::RecoverShardingsFromPartialMesh(sequence, preserve_shardings);
    }
  }

  if (VLOG_IS_ON(1)) {
    spmd::CheckHloSharding(sequence, original_device_mesh.num_elements());
  }
  module_is_changed = true;

  if (VLOG_IS_ON(1)) {
    spmd::CheckUserShardingPreservation(module, preserve_shardings);
  }

  // ----- Canonicalize layouts based on LayoutCanonicalizationCallback. -----
  TF_RETURN_IF_ERROR(CanonicalizeLayouts(module));

  return module_is_changed ? AutoShardingResult::kModuleChangedShardingPerformed
                           : AutoShardingResult::kModuleUnchanged;
}

bool ModuleHasUserShardings(const HloModule* module) {
  bool has_shardings = false;
  for (auto computation : module->computations()) {
    for (auto instruction : computation->instructions()) {
      if (instruction->has_sharding()) {
        has_shardings = true;
        break;
      }
    }
    if (has_shardings) {
      break;
    }
  }
  return has_shardings;
}

AutoSharding::AutoSharding(const AutoShardingOption& option)
    : option_(option) {}

bool IsSmallTensor(const HloInstruction* ins,
                   const AutoShardingOption& option) {
  return spmd::GetInstructionSize(ins->shape()) <=
         option.small_tensor_byte_size;
}

StatusOr<bool> AutoSharding::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!option_.enable) {
    return false;
  }
  VLOG(1) << "Start auto sharding pass";

  XLA_VLOG_LINES(6,
                 absl::StrCat("Before auto sharding:\n", module->ToString()));
  DumpHloModuleIfEnabled(*module, "before_auto_spmd_sharding");

  absl::Time start_time = absl::Now();
#if !defined(__APPLE__)
  // Streamz metrics.
  metrics::RecordAutoShardingInvocations();
#endif

  TF_RETURN_IF_ERROR(option_.CheckAndSetup());
  VLOG(1) << "AutoShardingOptions:\n" << option_.ToString();

  absl::flat_hash_set<std::string> replicated_small_tensors;
  if (option_.small_tensor_byte_size > 0) {
    for (auto computation : module->computations()) {
      for (auto instruction : computation->instructions()) {
        if (!instruction->has_sharding() &&
            IsSmallTensor(instruction, option_)) {
          VLOG(1) << "Replicated small tensor: " << instruction->name();
          instruction->set_sharding(
              instruction->shape().IsTuple()
                  ? HloSharding::SingleTuple(instruction->shape(),
                                             HloSharding::Replicate())
                  : HloSharding::Replicate());
          replicated_small_tensors.insert(std::string(instruction->name()));
        }
      }
    }
  }

  bool asymmetrical_mesh_dims = false;
  for (size_t i = 0; i < option_.device_mesh_shape.size(); ++i) {
    if (option_.device_mesh_beta[0] != option_.device_mesh_beta[i] ||
        option_.device_mesh_alpha[0] != option_.device_mesh_alpha[i]) {
      asymmetrical_mesh_dims = true;
      break;
    }
  }

  std::vector<std::vector<int64_t>> mesh_shapes;
  if (option_.try_multiple_mesh_shapes) {
    mesh_shapes = spmd::CreateDifferentMeshShapesToTry(
        absl::c_accumulate(option_.device_mesh_shape, 1,
                           [](int64_t a, int64_t b) { return a * b; }),
        option_.device_mesh_shape.size(),
        /* symmetrical_mesh_dims */ !asymmetrical_mesh_dims);
  } else {
    mesh_shapes.push_back(option_.device_mesh_shape);
  }

  size_t num_meshes = mesh_shapes.size();
  std::vector<std::unique_ptr<HloModule>> modules(num_meshes);
  std::vector<StatusOr<AutoShardingResult>> changed(
      num_meshes, AutoShardingResult::kModuleUnchanged);
  std::vector<double> objective_values(num_meshes, -1);

  VLOG(1) << "Original mesh shape "
          << spmd::ToString(option_.device_mesh_shape);
  double min_objective_value = std::numeric_limits<double>::max();
  int min_mesh_shape_index = -1;
  bool skip_auto_sharding = true;
  for (size_t i = 0; i < mesh_shapes.size(); ++i) {
    VLOG(1) << "Trying mesh shape " << spmd::ToString(mesh_shapes[i]);
    AutoShardingOption this_option = option_;
    this_option.device_mesh_shape = mesh_shapes[i];
    auto pass = new AutoShardingImplementation(this_option);
    auto module_clone = module->Clone("");
    module_clone->set_layout_canonicalization_callback(
        module->layout_canonicalization_callback());
    auto pass_result = pass->RunAutoSharding(
        module_clone.get(), replicated_small_tensors, execution_threads);

    changed[i] = pass_result;
    objective_values[i] = pass->GetSolverOptimalObjectiveValue();
    modules[i] = std::move(module_clone);
    delete pass;
    if (!pass_result.ok()) {
      continue;
    }
    VLOG(1) << "Mesh shape " << spmd::ToString(mesh_shapes[i])
            << " has objective value " << objective_values[i];
    if (objective_values[i] >= 0 && min_objective_value > objective_values[i]) {
      min_mesh_shape_index = i;
      min_objective_value = objective_values[i];
    }
    if (pass_result.ok() &&
        pass_result.value() !=
            AutoShardingResult::kModuleUnchangedNoShardingPerfomed) {
      skip_auto_sharding = false;
    }
  }

  StatusOr<bool> module_is_changed;
  if (skip_auto_sharding) {
    VLOG(1) << "Solver timed out. Will now rely on sharding propagation to "
               "perform sharding.";
    if (!ModuleHasUserShardings(module)) {
      LOG(WARNING)
          << "The auto-sharding solver has timed out without a solution. "
             "Further, as the input module does not contain any sharding "
             "annotations, we cannot rely on sharding propagation to perform "
             "heuristic-guided sharding. The module therefore may not be "
             "sharded leading to low performance.";
    }
    module_is_changed = false;
  } else {
    CHECK_GE(min_mesh_shape_index, 0)
        << "The auto-sharding pass could not find a device mesh that works for "
           "this input. This could be the result of a low memory budget. If "
           "you think you have set a reasonably large memory budget, please "
           "report this as a bug.";

    if (!changed[min_mesh_shape_index].ok()) {
      module_is_changed = changed[min_mesh_shape_index].status();
    } else {
      solver_optimal_objective_value_ = min_objective_value;
      if (changed[min_mesh_shape_index].value() ==
          AutoShardingResult::kModuleChangedShardingPerformed) {
        VLOG(1) << "Choosing mesh shape "
                << spmd::ToString(mesh_shapes[min_mesh_shape_index])
                << " which had the minimal solver objective value of "
                << min_objective_value;

        absl::flat_hash_map<HloComputation*, HloComputation*>
            computation_replacements;
        for (size_t i = 0; i < module->computation_count(); ++i) {
          auto original_computation = module->mutable_computation(i);
          auto new_computation =
              modules[min_mesh_shape_index]->mutable_computation(i);
          computation_replacements[original_computation] = new_computation;
        }

        module->ReplaceComputations(computation_replacements);
        module->MoveComputationsFrom(modules[min_mesh_shape_index].get());

        *module->config().mutable_entry_computation_layout() =
            modules[min_mesh_shape_index]->entry_computation_layout();

        module_is_changed = true;
      } else if (changed[min_mesh_shape_index].value() ==
                 AutoShardingResult::kModuleUnchanged) {
        module_is_changed = false;
      } else {
        module_is_changed = false;
      }
    }
  }

  absl::Time end_time = absl::Now();
  auto duration = end_time - start_time;
  LOG(INFO) << "Auto Sharding took " << absl::ToInt64Seconds(duration)
            << " seconds";
#if !defined(__APPLE__)
  metrics::RecordAutoShardingCompilationTime(
      absl::ToInt64Microseconds(duration));
#endif

  XLA_VLOG_LINES(6, absl::StrCat("After auto sharding:\n", module->ToString()));
  DumpHloModuleIfEnabled(*module, "after_auto_spmd_sharding");

  return module_is_changed;
}

StatusOr<bool> DummyAutoSharding::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // ----- Set Dummy Replicated Sharding -----
  HloComputation* entry = module->entry_computation();

  for (HloInstruction* inst : entry->instructions()) {
    const Shape& out_shape = inst->shape();
    if (out_shape.IsTuple()) {
      ShapeTree<HloSharding> tuple_sharding(out_shape,
                                            HloSharding::Replicate());
      inst->set_sharding(HloSharding::Tuple(tuple_sharding));
    } else {
      inst->set_sharding(HloSharding::Replicate());
    }
  }

  return true;
}

}  // namespace xla
