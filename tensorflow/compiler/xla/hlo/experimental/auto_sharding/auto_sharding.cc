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
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_cost_graph.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/matrix.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/metrics.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver.pb.h"
#ifdef PLATFORM_GOOGLE
#include "file/base/helpers.h"
#include "util/task/status.pb.h"
#endif

using MPConstraint = operations_research::MPConstraint;
using MPSolver = operations_research::MPSolver;
using MPSolverParameters = operations_research::MPSolverParameters;
using MPVariable = operations_research::MPVariable;

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
  for (const auto& x : strategies->leaf_vector) {
    ret.push_back(cluster_env.ReshardingCost(operand_shape, x.output_sharding,
                                             required_sharding));
  }
  return ret;
}

// Factory functions for StrategyVector.
std::unique_ptr<StrategyVector> CreateLeafStrategyVectorWithoutInNodes(
    size_t instruction_id, LeafStrategies& leaf_strategies) {
  auto strategies = std::make_unique<StrategyVector>();
  strategies->is_tuple = false;
  strategies->id = leaf_strategies.size();
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
  strategies->id = -1;
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
  CHECK(s.ok());
  return ShardingPropagation::GetShardingFromUser(*operand_clone, *ins_clone,
                                                  10, true, call_graph);
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
    if (operand->shape().rank() == 0) {
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
      if (!cur_input_sharding.has_value() &&
          ins->opcode() == HloOpcode::kGather && k == 0) {
        cur_input_sharding = HloSharding::Replicate();
      }
      CHECK(cur_input_sharding.has_value());
      if (!input_shardings[k].has_value()) {
        input_shardings[k] = cur_input_sharding;
      }
      auto operand_strategies = strategy_map.at(operand).get();
      auto operand_shape = operand->shape();
      resharding_costs.push_back(
          ReshardingCostVector(operand_strategies, ins->operand(k)->shape(),
                               *cur_input_sharding, cluster_env));
    }
  }
  return resharding_costs;
}

std::pair<std::vector<std::vector<double>>, std::vector<HloSharding>>
GenerateReshardingCostsAndShardingsForAllOperands(
    const HloInstruction* ins, const HloSharding& output_sharding,
    const StrategyMap& strategy_map, const ClusterEnvironment& cluster_env,
    const CallGraph& call_graph) {
  std::vector<std::optional<HloSharding>> input_shardings_optional;
  auto resharding_costs =
      GenerateReshardingCostsAndMissingShardingsForAllOperands(
          ins, output_sharding, strategy_map, cluster_env, call_graph,
          input_shardings_optional);
  std::vector<HloSharding> input_shardings;
  for (auto sharding_optional : input_shardings_optional) {
    CHECK(sharding_optional.has_value());
    input_shardings.push_back(sharding_optional.value());
  }

  return std::make_pair(resharding_costs, input_shardings);
}

std::vector<std::vector<double>> GenerateReshardingCostsForAllOperands(
    const HloInstruction* ins, const HloSharding& output_sharding,
    const StrategyMap& strategy_map, const ClusterEnvironment& cluster_env,
    const CallGraph& call_graph,
    std::vector<std::optional<HloSharding>> input_shardings) {
  return GenerateReshardingCostsAndMissingShardingsForAllOperands(
      ins, output_sharding, strategy_map, cluster_env, call_graph,
      input_shardings);
}

std::unique_ptr<StrategyVector> MaybeFollowInsStrategyVector(
    const StrategyVector* src_strategies, const Shape& shape,
    size_t instruction_id, bool have_memory_cost,
    LeafStrategies& leaf_strategies, const ClusterEnvironment& cluster_env,
    StableHashMap<int64_t, std::vector<ShardingStrategy>>&
        trimmed_strategy_map) {
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
          trimmed_strategy_map));
    }
  } else {
    CHECK(shape.IsArray());
    strategies =
        CreateLeafStrategyVectorWithoutInNodes(instruction_id, leaf_strategies);
    strategies->in_nodes.push_back(src_strategies);
    // Only follows the given strategy when there is no other strategy to be
    // restored.
    if (!trimmed_strategy_map.contains(src_strategies->id)) {
      strategies->following = src_strategies;
    }
    strategies->leaf_vector.reserve(src_strategies->leaf_vector.size());
    // Creates the sharding strategies and restores the trimmed strategies if
    // there is any.
    for (int64_t sid = 0;
         sid < src_strategies->leaf_vector.size() +
                   trimmed_strategy_map[src_strategies->id].size();
         ++sid) {
      const HloSharding* output_spec;
      if (sid < src_strategies->leaf_vector.size()) {
        output_spec = &src_strategies->leaf_vector[sid].output_sharding;
      } else {
        output_spec =
            &trimmed_strategy_map[src_strategies->id]
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
                            // {}}));
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

std::pair<std::vector<std::vector<double>>, std::vector<HloSharding>>
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
  return std::make_pair(resharding_costs,
                        std::vector<HloSharding>({HloSharding::Tuple(
                            operand->shape(), tuple_element_shardings)}));
}

// Add "Replicate()" strategy
void AddReplicatedStrategy(const HloInstruction* ins, const Shape& shape,
                           const ClusterEnvironment& cluster_env,
                           const StrategyMap& strategy_map,
                           std::unique_ptr<StrategyVector>& strategies,
                           double replicated_penalty) {
  HloSharding output_spec = HloSharding::Replicate();

  std::vector<std::vector<double>> resharding_costs;
  std::vector<HloSharding> input_shardings;
  if (ins->operand_count() > 0 && ins->operand(0)->shape().IsTuple()) {
    CHECK_EQ(ins->operand_count(), 1)
        << "Do not support instructions with more than one tuple "
           "operand. If this CHECK fails, we will need to fix "
           "b/233412625.";
    std::tie(resharding_costs, input_shardings) =
        ReshardingCostsForTupleOperand(ins->operand(0),
                                       strategy_map.at(ins->operand(0)).get());
    LOG(INFO) << absl::StrJoin(resharding_costs.back(), ",");
  } else {
    for (int64_t k = 0; k < ins->operand_count(); ++k) {
      auto operand = ins->operand(k);
      resharding_costs.push_back(ReshardingCostVector(
          strategy_map.at(operand).get(), ins->operand(k)->shape(), output_spec,
          cluster_env));
      input_shardings.push_back(output_spec);
    }
  }
  double memory_cost = GetBytes(shape) / output_spec.NumTiles();
  strategies->leaf_vector.push_back(ShardingStrategy(
      {"R", HloSharding::Replicate(), replicated_penalty, 0, memory_cost,
       std::move(resharding_costs), input_shardings}));
}

std::vector<std::vector<double>> CreateZeroReshardingCostsForAllOperands(
    const HloInstruction* ins, const StrategyMap& strategy_map) {
  std::vector<std::vector<double>> resharding_costs;
  for (size_t i = 0; i < ins->operand_count(); ++i) {
    auto operand = ins->operand(i);
    const auto& operand_strategies = strategy_map.at(operand);
    if (operand->shape().IsTuple()) {
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
    } else {
      resharding_costs.push_back(
          std::vector<double>(operand_strategies->leaf_vector.size(), 0));
    }
  }
  return resharding_costs;
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
      std::vector<HloSharding> input_shardings;
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
        LOG(INFO) << absl::StrJoin(resharding_costs.back(), ",");
      } else if (ins->opcode() == HloOpcode::kRngBitGenerator &&
                 ins->operand(0)->shape().IsArray()) {
        auto replicated_sharding = HloSharding::Replicate();
        input_shardings.push_back(HloSharding::SingleTuple(
            ins->operand(0)->shape(), replicated_sharding));
        resharding_costs = GenerateReshardingCostsForAllOperands(
            ins, output_spec, strategy_map, cluster_env, call_graph,
            {replicated_sharding});
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
  std::vector<int64_t> shardable_mesh_dims =
      VectorGreaterThanOneElementIndices(device_mesh.dimensions());
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
      if (shape.dimensions(i) < device_mesh.dim(shardable_mesh_dims[0]) ||
          shape.dimensions(j) < device_mesh.dim(shardable_mesh_dims[1])) {
        continue;
      }

      if (only_allow_divisible &&
          (!IsDivisible(shape.dimensions(i),
                        device_mesh.dim(shardable_mesh_dims[0])) ||
           !IsDivisible(shape.dimensions(j),
                        device_mesh.dim(shardable_mesh_dims[1])))) {
        continue;
      }

      std::string name = absl::StrFormat("S{%d,%d} @ {0,1}", i, j);
      HloSharding output_spec =
          Tile(shape, {i, j}, {shardable_mesh_dims[0], shardable_mesh_dims[1]},
               device_mesh);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(shape) / output_spec.NumTiles();
      std::vector<HloSharding> input_shardings;
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
  std::vector<int64_t> shardable_mesh_dims =
      VectorGreaterThanOneElementIndices(device_mesh.dimensions());
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
      if (ins->shape().dimensions(i) <
              device_mesh.dim(shardable_mesh_dims[0]) ||
          ins->shape().dimensions(j) <
              device_mesh.dim(shardable_mesh_dims[1])) {
        continue;
      }
      if (only_allow_divisible &&
          (!IsDivisible(ins->shape().dimensions(i),
                        device_mesh.dim(shardable_mesh_dims[0])) ||
           !IsDivisible(ins->shape().dimensions(j),
                        device_mesh.dim(shardable_mesh_dims[1])))) {
        continue;
      }

      HloSharding output_spec =
          Tile(ins->shape(), {i, j},
               {shardable_mesh_dims[0], shardable_mesh_dims[1]}, device_mesh);
      std::optional<HloSharding> input_spec =
          hlo_sharding_util::ReshapeSharding(ins->shape(), operand->shape(),
                                             output_spec);
      if (!input_spec.has_value()) {  // invalid reshape
        continue;
      }

      std::string name =
          absl::StrFormat("S%d%d @ {%d,%d}", i, j, shardable_mesh_dims[0],
                          shardable_mesh_dims[1]);
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

// Choose an operand to follow.
// We choose to follow the operand with the highest priority.
// The priority is defined as a two element tuple as below, where we compare
// the first key first, and if the first key is the same, we compare the second
// key:
//
// priority(operand) = (
//   max(x.output_spec.num_tiles for x in operand.strategies),
//   depth(operand),
// )
//
// For example, When one operand has a sharding strategy that splits into
// N slices, while another operand only has replicated strategy, we will choose
// the first operand to follow since it can be split into more slices. When
// both operands can be sliced into the same number of slices, we follow the
// deeper one in the computational graph. When the depth is also similar,
// we set these operators to be "tied" and let the ILP solver to pick which one
// to follow.
//
// The function returns (follow_idx, tie), where the follow_idx is the id of
// the operand to follow and tie is a boolean variable that indicates whether
// there are multiple operands have similar priority. Return `tie == True` if
// there are two operands with very close priorities and we cannot decide which
// one to follow.
std::pair<int64_t, bool> ChooseOperandToFollow(
    const StrategyMap& strategy_map, const InstructionDepthMap& depth_map,
    const AliasMap& alias_map,
    const absl::flat_hash_set<const HloInstruction*>& undefined_set,
    int64_t max_depth, const HloInstruction* ins) {
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
    if (!undefined_set.count(operand)) {
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
  for (auto iter : batch_dim_map) {
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
    StableHashMap<int64_t, std::vector<ShardingStrategy>>& trimmed_strategy_map,
    const CallGraph& call_graph, bool strict) {
  if (strategies->is_tuple) {
    for (size_t i = 0; i < strategies->childs.size(); ++i) {
      TrimOrGenerateStrategiesBasedOnExistingSharding(
          output_shape.tuple_shapes(i), strategies->childs.at(i).get(),
          strategy_map, instructions, existing_sharding.tuple_elements().at(i),
          cluster_env, trimmed_strategy_map, call_graph, strict);
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
        trimmed_strategy_map[strategies->id] = strategies->leaf_vector;
        strategies->leaf_vector.clear();
        strategies->leaf_vector.push_back(found_strategy);
      } else {
        VLOG(1) << "Generate a new strategy based on user sharding.";
        std::string name = ToStringSimple(existing_sharding);
        std::vector<std::vector<double>> resharding_costs;
        std::vector<HloSharding> input_shardings;
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
            std::vector<double> in_resharding_costs =
                ReshardingCostVector(operand_strategies, operand_shape,
                                     existing_sharding, cluster_env);
            // If there is only one option for resharding, and the cost
            // computed for that option is kInfinityCost, set the cost to
            // zero. This is okay because there is only one option anyway, and
            // having the costs set to kInfinityCost is problematic for the
            // solver.
            if (in_resharding_costs.size() == 1 &&
                in_resharding_costs[0] == kInfinityCost) {
              in_resharding_costs[0] = 0;
            }
            resharding_costs.push_back(in_resharding_costs);
          }
        }
        double memory_cost =
            GetBytes(output_shape) / existing_sharding.NumTiles();
        if (!strategies->leaf_vector.empty()) {
          trimmed_strategy_map[strategies->id] = strategies->leaf_vector;
        }
        strategies->leaf_vector.clear();
        strategies->leaf_vector.push_back(
            ShardingStrategy({name, existing_sharding, 0, 0, memory_cost,
                              resharding_costs, input_shardings}));
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

// NOLINTBEGIN(readability/fn_size)
// TODO(zhuohan): Decompose this function into smaller pieces
// Build possible sharding strategies and their costs for all instructions.
StatusOr<std::tuple<StrategyMap, LeafStrategies, AssociativeDotPairs>>
BuildStrategyAndCost(const HloInstructionSequence& sequence,
                     const HloModule* module,
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
  StableHashMap<int64_t, std::vector<ShardingStrategy>> trimmed_strategy_map;
  LeafStrategies leaf_strategies;
  AssociativeDotPairs associative_dot_pairs;
  absl::flat_hash_set<const HloInstruction*> undefined_set;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Count the non-one mesh dimension.
  int mesh_nn_dims = 0;
  for (int dim : device_mesh.dimensions()) {
    if (dim > 1) {
      mesh_nn_dims++;
    }
  }

  // Gather all output values
  absl::flat_hash_set<const HloInstruction*> output_set;
  for (size_t i = 0; i < instructions.back()->operand_count(); ++i) {
    output_set.insert(instructions.back()->operand(i));
  }

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
      case HloOpcode::kGather: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        // Follows the strategy of start_indices (operend 1)
        const HloInstruction* indices = ins->operand(1);
        const Shape& shape = ins->shape();
        const StrategyVector* src_strategies = strategy_map.at(indices).get();
        CHECK(!src_strategies->is_tuple);
        if (undefined_set.contains(indices)) {
          break;
        }
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
            // Split only when the tensor shape is divisable by device
            // mesh.
            // TODO(b/220942808) Shard non-divisible dimensions.
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

            std::vector<HloSharding> input_shardings;
            for (auto sharding_optional : input_shardings_optional) {
              CHECK(sharding_optional.has_value());
              input_shardings.push_back(sharding_optional.value());
            }

            strategies->leaf_vector.push_back(ShardingStrategy(
                {name, output_spec, compute_cost, communication_cost,
                 memory_cost, std::move(resharding_cost), input_shardings}));
          }
        }
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategies, 0);
        break;
      }
      case HloOpcode::kBroadcast: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        const HloInstruction* operand = ins->operand(0);
        if (undefined_set.contains(operand)) {
          break;
        }

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
        if (!undefined_set.count(operand) &&
            !(mesh_nn_dims >= 2 && solver_option.allow_mixed_mesh_shape)) {
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
        if (undefined_set.contains(operand)) {
          break;
        }

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
            follow_idx =
                ChooseOperandToFollow(strategy_map, depth_map, alias_map,
                                      undefined_set, max_depth, ins)
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
              GenerateReshardingCostsForAllOperands(
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
            strategy_map, depth_map, alias_map, undefined_set, max_depth, ins);

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

          const StrategyVector* src_strategies =
              strategy_map.at(ins->operand(i)).get();
          CHECK(!src_strategies->is_tuple);

          for (int64_t sid = 0; sid < src_strategies->leaf_vector.size();
               ++sid) {
            HloSharding output_spec =

                src_strategies->leaf_vector[sid].output_sharding;
            std::string name = ToStringSimple(output_spec);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost =
                GetBytes(ins->shape()) / output_spec.NumTiles();
            std::vector<std::vector<double>> resharding_costs;
            std::vector<HloSharding> input_shardings;
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
        break;
      }
      case HloOpcode::kConvolution: {
        TF_RETURN_IF_ERROR(HandleConv(strategies, leaf_strategies, strategy_map,
                                      ins, instruction_id, cluster_env,
                                      batch_dim_map, solver_option));
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
              trimmed_strategy_map));
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
            trimmed_strategy_map);
        break;
      }
      case HloOpcode::kCustomCall: {
        if (IsCustomCallMarker(ins)) {
          const HloInstruction* operand = ins->operand(0);
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          CHECK(src_strategies->is_tuple);
          strategies = MaybeFollowInsStrategyVector(
              src_strategies, ins->shape(), instruction_id,
              /* have_memory_cost= */ true, leaf_strategies, cluster_env,
              trimmed_strategy_map);
        } else if (ins->has_sharding()) {
          if (ins->shape().IsTuple()) {
            strategies = CreateTupleStrategyVector(instruction_id);
          } else {
            strategies = CreateLeafStrategyVector(
                instruction_id, ins, strategy_map, leaf_strategies);
          }
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
                trimmed_strategy_map);
          }
        } else {
          // TODO (b/258723035) Handle CustomCall ops for GPUs in a better way.
          if (ins->shape().IsTuple()) {
            strategies = CreateTupleStrategyVector(instruction_id);
            strategies->childs.reserve(ins->shape().tuple_shapes_size());
            for (size_t i = 0; i < ins->shape().tuple_shapes_size(); ++i) {
              std::unique_ptr<StrategyVector> child_strategies =
                  CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                           leaf_strategies);
              AddReplicatedStrategy(ins, ins->shape().tuple_shapes(i),
                                    cluster_env, strategy_map, child_strategies,
                                    replicated_penalty);
              strategies->childs.push_back(std::move(child_strategies));
            }
          } else {
            strategies = CreateLeafStrategyVector(
                instruction_id, ins, strategy_map, leaf_strategies);
            AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                  strategies, replicated_penalty);
          }
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
              trimmed_strategy_map));
        }

        break;
      }
      case HloOpcode::kConditional:
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
    if (ins->has_sharding()) {
      // Finds the sharding strategy that aligns with the given sharding spec
      // Do not merge nodes if this one instruction has annotations.
      // TODO(b/208668853) If needed, we can make auto sharding faster by using
      // this sharding spec when merging node using strategies->following.
      TrimOrGenerateStrategiesBasedOnExistingSharding(
          ins->shape(), strategies.get(), strategy_map, instructions,
          ins->sharding(), cluster_env, trimmed_strategy_map, call_graph,
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
    XLA_VLOG_LINES(2, absl::StrCat("strategies:\n", strategies->ToString()));

    // Debug options: forcibly set the strategy of some instructions.
    if (solver_option.force_strategy) {
      std::vector<int64_t> inst_indices =
          solver_option.force_strategy_inst_indices;
      std::vector<std::string> stra_names =
          solver_option.force_strategy_stra_names;
      CHECK_EQ(inst_indices.size(), stra_names.size());
      auto it = absl::c_find(inst_indices, strategies->id);
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
    CheckReshardingCostsShape(strategies.get());
    CheckMemoryCosts(strategies.get(), ins->shape());
    strategy_map[ins] = std::move(strategies);
  }  // end of for loop

  // If gradient accumulation is used, adjust the cost of all-reduce for
  // gradient synchronization.
  if (solver_option.grad_acc_num_micro_batches > 1) {
    // find gradientt-computation instructions
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

void PrintLargestInstructions(
    const std::vector<int64_t>& chosen_strategy,
    const std::vector<std::vector<double>>& memory_cost,
    const std::vector<std::vector<int>>& liveness,
    const std::vector<std::string>& instruction_names) {
  // This memory consumption computation is different from
  // that in PrintAutoShardingSolution() because how L and m are created to be
  // different from liveness_set and strategy.memory_cost.

  std::vector<int64_t> instruction_ids;
  std::vector<std::pair<size_t, double>> time_memory_usage;
  for (size_t t = 0; t < liveness.size(); ++t) {
    double mem = 0.0;
    for (auto i : liveness[t]) {
      mem += memory_cost[i][chosen_strategy[i]];
    }
    time_memory_usage.push_back(std::make_pair(t, mem));
  }
  struct {
    bool operator()(std::pair<size_t, double> a,
                    std::pair<size_t, double> b) const {
      return a.second > b.second;
    }
  } MemLarger;
  std::sort(time_memory_usage.begin(), time_memory_usage.end(), MemLarger);

  LOG(INFO) << "using m[] and L[], max memory usage: "
            << time_memory_usage.front().second / (1024 * 1024 * 1024)
            << " GB at time " << time_memory_usage.front().first;
  // Gets largest tensors in top k time steps.
  size_t k = 3;
  k = std::min(k, time_memory_usage.size());
  std::vector<std::pair<size_t, double>> instruction_mem;
  absl::flat_hash_set<size_t> instruction_set;
  for (size_t t = 0; t < k; t++) {
    for (auto i : liveness[time_memory_usage.at(t).first]) {
      double mem = memory_cost[i][chosen_strategy[i]];
      if (mem > 100 * 1024 * 1024 &&
          instruction_set.find(i) == instruction_set.end()) {
        instruction_mem.push_back(std::make_pair(i, mem));
        instruction_set.insert(i);
      }
    }
  }
  std::sort(instruction_mem.begin(), instruction_mem.end(), MemLarger);

  size_t top_tensors = 10;
  top_tensors = std::min(top_tensors, instruction_mem.size());
  VLOG(1) << "Top " << top_tensors << " largest tensors:";
  for (size_t i = 0; i < top_tensors; i++) {
    VLOG(1) << "instruction name: "
            << instruction_names.at(instruction_mem.at(i).first)
            << " memory usage: "
            << instruction_mem.at(i).second / (1024 * 1024 * 1024) << "GB";
  }
}

// NOLINTEND

// We formulate the auto sharding process as the following ILP problem:
// Variables:
//   s[i]: Sharding strategy one-hot vector.
//         dim(s[i]) == # sharding strategies of the i-th XLA op
//         s_len[i] := dim(s[i]) in the arguments
//   e[i, j]: Strategy one-hot vector of edge i -> j.
//            dim(e[i, j]) == dim(s[i]) * dim(s[j])
// Constants:
//   N: Number of total XLA ops
//   M: Memory budget
//   E: Edge set {(i, j)}
//   L[t]: Index of live instructions at time t
//   c[i]: Computation cost vector of instruction i
//   d[i]: Communication cost vector of instruction i
//   m[i]: Memory cost vector of instruction i
//         dim(c[i]) == dim(d[i]) == dim(m[i]) == dim(s[i])
//   r[i, j]: The resharding cost vector of edge i -> j
//            dim(e[i, j]) == dim(r[i, j])
//   A: Alias set {(i, j)}
//   v[i, j]: v[i, j](p, q) == 1 if strategy p is different than q, otherwise
//            v[i, j](p, q) == 0
//            dim(e[i, j]) == dim(v[i, j])
// Problem:
//   Minimize sum_{0 <= i < N} s[i]^T * (c[i] + d[i])
//            + sum_{(i, j) in E} e[i, j]^T * r[i, j]
//   s.t.
//       Make sure s is one-hot:
//     0. Do not choose solutions with infinity cost (b/238210866).
//     a. For 0 <= i < N, s[i] in {0, 1} ^ dim(s[i])
//     b. For 0 <= i < N, s[i]^T * 1 == 1
//       Memory constraint:
//     c. For all t: sum_{i in L[t]} s[i]^T * m[i] <= M
//       Make sure e is one-hot:
//     d. For all (i, j) in E, e[i, j] in {0, 1} ^ dim(e[i, j])
//     e. For all (i, j) in E, e[i, j]^T * 1 == 1
//       Make sure s[i] and s[j] align with e[i, j]:
//     f. For all (i, j) in E, 0 <= p < dim(s[i]),
//        sum_{0 <= q < dim(s[j])} e[i, j](p * dim(s[j]) + q) <= s[i](p)
//     g. For all (i, j) in E, 0 <= q < dim(s[j]),
//        sum_{0 <= p < dim(s[i])} e[i, j](p * dim(s[j]) + q) <= s[j](q)
//     h. For all (i, j) in A and all (p, q),
//        s[i][p] + s[j][q] <= 1 if v[p, q] == 1.0
// Serialize parameters of the ILP problem as numpy arrays and call the python
// solver.
StatusOr<std::tuple<std::vector<int64_t>, std::vector<int64_t>, double>>
CallORToolsSolver(int64_t N, int64_t M, const std::vector<int>& s_len,
                  const std::vector<int>& s_follow,
                  const std::vector<std::pair<int, int>>& E,
                  const std::vector<std::vector<int>>& L,
                  const std::vector<std::vector<double>>& c,
                  const std::vector<std::vector<double>>& d,
                  const std::vector<std::vector<double>>& m,
                  const std::vector<std::vector<double>>& r,
                  const std::vector<std::pair<int, int>>& A,
                  const std::vector<std::vector<double>>& v,
                  const std::vector<std::string>& instruction_names,
                  bool crash_at_infinity_costs_check) {
  size_t num_edges = E.size();

  int32_t num_workers = 32;
  // SAT or SCIP
  std::unique_ptr<MPSolver> solver(std::make_unique<MPSolver>("", MPSolver::SAT_INTEGER_PROGRAMMING));
  CHECK(solver);
  solver->MutableObjective()->SetMinimization();
  std::string solver_parameter_str;
#ifdef PLATFORM_GOOGLE
  if (solver->ProblemType() ==
      operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
    // Set random_seed, interleave_search and share_binary_clauses for
    // determinism, and num_workers for parallelism.
    solver_parameter_str = absl::StrCat(
        "share_binary_clauses:false,random_seed:1,interleave_"
        "search:true,num_workers:",
        num_workers);
    solver->SetSolverSpecificParametersAsString(solver_parameter_str);
  }
#endif
  // Create variables
  std::vector<std::vector<MPVariable*>> s(N);
  std::vector<std::vector<MPVariable*>> e(num_edges);

  size_t var_vector_cnt = 0;
  for (size_t i = 0; i < N; ++i) {
    if (s_follow[i] < 0) {
      var_vector_cnt += 1;
      // Creates variables for instructions that do not follow others.
      solver->MakeBoolVarArray(s_len[i], absl::StrCat("s[", i, "]"), &s[i]);
    }
  }

  for (size_t i = 0; i < N; ++i) {
    if (s_follow[i] >= 0) {
      // Copies the variable of followed instruction to the following
      // instruction.
      s[i] = s[s_follow[i]];
    }
  }

  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = E[i];
    solver->MakeBoolVarArray(
        s_len[edge.first] * s_len[edge.second],
        absl::StrCat("e[", edge.first, ",", edge.second, "]"), &e[i]);
  }

  // Objective
  // Node costs
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < s[i].size(); ++j) {
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(s[i][j]);
      solver->MutableObjective()->SetCoefficient(
          s[i][j], accumulated_coefficient + c[i][j] + d[i][j]);
    }
  }
  // Edge costs
  for (size_t i = 0; i < num_edges; ++i) {
    for (size_t j = 0; j < e[i].size(); ++j) {
      double accumulated_coefficient =
          solver->MutableObjective()->GetCoefficient(e[i][j]);
      solver->MutableObjective()->SetCoefficient(
          e[i][j], accumulated_coefficient + r[i][j]);
    }
  }

  // Constraints
  // 0. Do not choose solutions with infinity costs, as it will make the
  // objective value so large that other solution choices do not matter anymore.
  // Remove these constraints once b/238210866 is done.
  for (size_t i = 0; i < N; ++i) {
    if (s[i].empty()) {
      continue;
    }
    bool all_infinity = true;
    for (size_t j = 0; j < s[i].size(); ++j) {
      if (solver->MutableObjective()->GetCoefficient(s[i][j]) >=
          kInfinityCost) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0, absl::StrCat("infinitycost: s[", i, "][", j, "] = 0"));
        constraint->SetCoefficient(s[i][j], 1.0);
      } else {
        all_infinity = false;
      }
    }
    if (all_infinity) {
      LOG(FATAL) << "All of s[" << i << "][*] have infinity costs";
    }
  }

  for (size_t i = 0; i < num_edges; ++i) {
    if (e[i].empty()) {
      continue;
    }
    bool all_infinity = true;
    for (size_t j = 0; j < e[i].size(); ++j) {
      std::pair<int, int> edge = E[i];
      solver->MutableObjective()->SetCoefficient(e[i][j], r[i][j]);
      if (r[i][j] >= kInfinityCost) {
        MPConstraint* constraint = solver->MakeRowConstraint(
            0.0, 0.0,
            absl::StrCat("infinitycost: e[", edge.first, "][", edge.second,
                         "][", j, "] = 0"));
        constraint->SetCoefficient(e[i][j], 1.0);
      } else {
        all_infinity = false;
      }
    }
    if (all_infinity) {
      auto err_msg = absl::StrCat("All of e[", E[i].first, "][", E[i].second,
                                  "][*] have infinity costs");
      if (crash_at_infinity_costs_check) {
        LOG(FATAL) << err_msg;
      } else {
        LOG(WARNING) << err_msg;
        return tsl::errors::Internal(err_msg);
      }
    }
  }

  // a. specified via "BoolVarArray"
  // b.
  for (size_t i = 0; i < N; ++i) {
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(s[", i, "][j] for j = [0 .. ", s[i].size(),
                     ")) = 1"));
    for (size_t j = 0; j < s[i].size(); ++j) {
      constraint->SetCoefficient(s[i][j], 1.0);
    }
  }
  // c.
  if (M > 0) {
    for (size_t t = 0; t < L.size(); ++t) {
      std::string str = "[";
      for (auto i : L[t]) {
        absl::StrAppend(&str, i, ", ");
      }
      str += "]";
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), M, absl::StrCat("mem[", t, "] = ", str));
      for (auto i : L[t]) {
        for (size_t j = 0; j < s[i].size(); ++j) {
          double accumulated_coefficient = constraint->GetCoefficient(s[i][j]);
          constraint->SetCoefficient(s[i][j],
                                     accumulated_coefficient + m[i][j]);
        }
      }
    }
  }

  // d. specified via "BoolVarArray"
  // e.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = E[i];
    MPConstraint* constraint = solver->MakeRowConstraint(
        1.0, 1.0,
        absl::StrCat("sum(e[", edge.first, "][", edge.second, "][*]) = 1"));
    for (size_t j = 0; j < e[i].size(); ++j) {
      constraint->SetCoefficient(e[i][j], 1.0);
    }
  }
  // f.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = E[i];
    for (size_t p = 0; p < s[edge.first].size(); ++p) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0, absl::StrCat("f for i = ", i, ", p = ", p));
      constraint->SetCoefficient(s[edge.first][p], -1.0);
      for (size_t q = 0; q < s[edge.second].size(); ++q) {
        constraint->SetCoefficient(e[i][p * s[edge.second].size() + q], 1.0);
      }
    }
  }
  // g.
  for (size_t i = 0; i < num_edges; ++i) {
    std::pair<int, int> edge = E[i];
    for (size_t q = 0; q < s[edge.second].size(); ++q) {
      MPConstraint* constraint = solver->MakeRowConstraint(
          -MPSolver::infinity(), 0, absl::StrCat("g for i = ", i, ", q = ", q));
      constraint->SetCoefficient(s[edge.second][q], -1.0);
      for (size_t p = 0; p < s[edge.first].size(); ++p) {
        constraint->SetCoefficient(e[i][p * s[edge.second].size() + q], 1.0);
      }
    }
  }
  // h.
  for (size_t i = 0; i < A.size(); ++i) {
    std::pair<int, int> alias = A[i];
    for (size_t p = 0; p < s[alias.first].size(); ++p) {
      for (size_t q = 0; q < s[alias.second].size(); ++q) {
        // if lhs == 1
        if (v[i][p * s[alias.second].size() + q] > 0.5) {
          MPConstraint* constraint = solver->MakeRowConstraint(
              -MPSolver::infinity(), 1,
              absl::StrCat("s[", alias.first, "][", p, "] + s[", alias.second,
                           "][", q, "] <= 1"));
          constraint->SetCoefficient(s[alias.first][p], 1.0);
          constraint->SetCoefficient(s[alias.second][q], 1.0);
        }
      }
    }
  }

#ifdef PLATFORM_GOOGLE
  // Exports the model for debugging.
  bool dump_model = false;
  if (dump_model) {
    operations_research::MPModelProto model_proto;
    solver->ExportModelToProto(&model_proto);
    auto write_status = file::SetTextProto(
        // Modify this file path if needed.
        absl::StrCat("/tmp/model_", solver->NumVariables(), ".proto"),
        model_proto, file::Defaults());
    if (!write_status.ok()) {
      LOG(ERROR) << write_status.message();
    }
  }
#endif
  solver->set_time_limit(3600 * 1000);  // in ms
  VLOG(0) << "Starting solver " << solver->ProblemType() << "\n"
          << "Solver parameter string: " << solver_parameter_str << "\n"
          << "Number of workers: " << num_workers << "\n"
          << "Number of threads: " << solver->GetNumThreads() << "\n"
          << "Time limit: " << solver->time_limit() << "\n"
          << "Number variables for ILP: " << solver->NumVariables() << "\n"
          << "Total vector of variables: " << var_vector_cnt << "\n"
          << "Total instructions: " << N << "\n"
          << "Memory budget: " << M / (1024 * 1024 * 1024) << "GB\n"
          << "Number of ILP constraints: " << solver->NumConstraints();
  auto status = solver->Solve();
  if (status == operations_research::MPSolver::INFEASIBLE) {
    LOG(ERROR) << "MPSolver could not find any feasible solution.";
#ifdef PLATFORM_GOOGLE
    operations_research::MPModelRequest model_request;
    solver->ExportModelToProto(model_request.mutable_model());
    if (solver->ProblemType() ==
        operations_research::MPSolver::SAT_INTEGER_PROGRAMMING) {
      model_request.set_solver_type(
          operations_research::MPModelRequest::SAT_INTEGER_PROGRAMMING);
    } else if (solver->ProblemType() ==
               operations_research::MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING) {
      model_request.set_solver_type(
          operations_research::MPModelRequest::SCIP_MIXED_INTEGER_PROGRAMMING);
    }
    model_request.set_solver_time_limit_seconds(100);
    auto iis = MPSolver::ComputeIrreducibleInfeasibleSubset(model_request);
    LOG(INFO) << iis.status().DebugString();
    LOG(INFO) << "Infeasible constraints: ";
    for (int index : iis.constraint_index()) {
      LOG(INFO) << " - " << model_request.model().constraint(index).name();
    }
    for (int index : iis.general_constraint_index()) {
      LOG(INFO)
          << " - "
          << model_request.model().general_constraint(index).DebugString();
    }
#endif

    return tsl::errors::Internal(
        "MPSolver could not find any feasible solution.");
  }
  if (status != operations_research::MPSolver::OPTIMAL) {
    return tsl::errors::Internal("Solver errors.");
  }

  LOG(INFO) << "Solver Status: " << status
            << " Objective value: " << solver->Objective().Value();
  if (solver->Objective().Value() >= kInfinityCost) {
    LOG(WARNING) << "Objective (" << solver->Objective().Value()
                 << ") is larger than kInfinityCost. It means the solver "
                    "chooses a solution with kInfinityCost and there may be "
                    "numerical issues when the solver considering other costs.";
  }
  if (VLOG_IS_ON(10)) {
    // Print solver information for debugging. This hasn't been useful so far,
    // so leave it at VLOG level 10.
    operations_research::MPModelProto model_proto;
    solver->ExportModelToProto(&model_proto);
    VLOG(10) << "MODEL:";
    XLA_VLOG_LINES(10, model_proto.DebugString());
    VLOG(10) << "RESPONSE:";
    operations_research::MPSolutionResponse response;
    solver->FillSolutionResponseProto(&response);
    XLA_VLOG_LINES(10, response.DebugString());
  }

  // Return value
  std::vector<int64_t> chosen_strategy(N, -1), e_val(num_edges, -1);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < s[i].size(); ++j) {
      // if lhs == 1
      if (s[i][j]->solution_value() > 0.5) {
        chosen_strategy[i] = j;
        break;
      }
    }
  }
  for (int i = 0; i < num_edges; ++i) {
    for (int j = 0; j < e[i].size(); ++j) {
      // if lhs == 1
      if (e[i][j]->solution_value() > 0.5) {
        e_val[i] = j;
        break;
      }
    }
  }

  LOG(INFO) << "N = " << N;
  if (M < 0) {
    LOG(INFO) << "memory budget: -1";
  } else {
    LOG(INFO) << "memory budget: " << M / (1024 * 1024 * 1024) << " GB";
  }
  PrintLargestInstructions(chosen_strategy, m, L, instruction_names);
  return std::make_tuple(std::move(chosen_strategy), std::move(e_val),
                         solver->Objective().Value());
}

StatusOr<std::tuple<std::vector<int64_t>, std::vector<int64_t>, double>>
CallSolver(const HloInstructionSequence& sequence,
           const LivenessSet& liveness_set, const StrategyMap& strategy_map,
           const LeafStrategies& leaf_strategies, const CostGraph& cost_graph,
           const AliasSet& alias_set, int64_t memory_budget_per_device,
           bool crash_at_infinity_costs_check) {
  // Serialize edges and edge costs to 1d numpy arrays
  int64_t N = leaf_strategies.size();
  int64_t M = memory_budget_per_device;
  std::vector<int> s_len = cost_graph.node_lens_;
  const std::vector<int>& s_follow = cost_graph.follow_idx_;
  std::vector<std::pair<int, int>> E;
  std::vector<std::vector<double>> r;
  for (const auto& iter : cost_graph.edge_costs_) {
    E.push_back(iter.first);
    std::vector<double> rij;
    Matrix edge_cost = iter.second;
    for (size_t i = 0; i < edge_cost.n_; i++) {
      for (size_t j = 0; j < edge_cost.m_; j++) {
        rij.push_back(edge_cost(i, j));
      }
    }
    r.push_back(std::move(rij));
  }

  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  std::vector<std::string> instruction_names;

  // Serialize node costs
  std::vector<std::vector<double>> c, d, m;
  for (size_t i = 0; i < N; ++i) {
    const StrategyVector* strategies = leaf_strategies[i];
    instruction_names.push_back(absl::StrCat(
        instructions.at(strategies->instruction_id)->name(), " (id: ", i, ")"));
    std::vector<double> ci, di, mi;
    for (size_t j = 0; j < strategies->leaf_vector.size(); ++j) {
      ci.push_back(strategies->leaf_vector[j].compute_cost);
      di.push_back(strategies->leaf_vector[j].communication_cost +
                   cost_graph.extra_node_costs_[i][j]);
      mi.push_back(strategies->leaf_vector[j].memory_cost);
    }
    c.push_back(ci);
    d.push_back(di);
    m.push_back(mi);
  }

  // Serialize special edges that forces a alias pair have the same sharding
  // spec
  std::vector<std::pair<int, int>> A;
  std::vector<std::vector<double>> v;
  for (const auto& pair : alias_set) {
    const StrategyVector* src_strategies = leaf_strategies[pair.first];
    const StrategyVector* dst_strategies = leaf_strategies[pair.second];
    Matrix raw_cost(src_strategies->leaf_vector.size(),
                    dst_strategies->leaf_vector.size());
    for (size_t i = 0; i < src_strategies->leaf_vector.size(); ++i) {
      for (size_t j = 0; j < dst_strategies->leaf_vector.size(); ++j) {
        if (src_strategies->leaf_vector[i].output_sharding ==
            dst_strategies->leaf_vector[j].output_sharding) {
          raw_cost(i, j) = 0.0;
        } else {
          raw_cost(i, j) = 1.0;
        }
      }
    }
    int idx_a = pair.first;
    int idx_b = pair.second;
    std::vector<int> row_indices;
    std::vector<int> col_indices;

    if (s_follow[idx_a] >= 0) {
      row_indices = cost_graph.reindexing_vector_.at(idx_a);
      idx_a = s_follow[idx_a];
    } else {
      row_indices.assign(s_len[idx_a], 0);
      std::iota(row_indices.begin(), row_indices.end(), 0);
    }

    if (s_follow[idx_b] >= 0) {
      col_indices = cost_graph.reindexing_vector_.at(idx_b);
      idx_b = s_follow[idx_b];
    } else {
      col_indices.assign(s_len[idx_b], 0);
      std::iota(col_indices.begin(), col_indices.end(), 0);
    }

    CHECK_EQ(s_len[idx_a], row_indices.size());
    CHECK_EQ(s_len[idx_b], col_indices.size());

    A.push_back(std::make_pair(idx_a, idx_b));
    std::vector<double> vij;
    for (int i : row_indices) {
      for (int j : col_indices) {
        vij.push_back(raw_cost(i, j));
      }
    }
    v.push_back(vij);
  }

  // Serialize liveness_set
  std::vector<std::vector<int>> L(liveness_set.size());
  for (size_t t = 0; t < liveness_set.size(); ++t) {
    std::vector<int>& current_liveness_set_indices = L[t];
    std::function<void(const StrategyVector*, const ShapeIndex&)>
        traverse_live_instructions;
    traverse_live_instructions = [&](const StrategyVector* strategies,
                                     const ShapeIndex& index) {
      if (!index.empty()) {
        current_liveness_set_indices.push_back(
            strategies->childs.at(index.front())->id);
      } else {
        current_liveness_set_indices.push_back(strategies->id);
      }
    };
    for (const HloValue* value : liveness_set[t]) {
      if (value->instruction()->shape().IsTuple() && value->index().empty()) {
        continue;
      }
      traverse_live_instructions(strategy_map.at(value->instruction()).get(),
                                 value->index());
    }
  }
  return CallORToolsSolver(N, M, s_len, s_follow, E, L, c, d, m, r, A, v,
                           instruction_names, crash_at_infinity_costs_check);
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
                    absl::Span<const int64_t> s_val, bool last_iteration) {
  // Set the HloSharding for every instruction
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  for (HloInstruction* inst : instructions) {
    auto iter = strategy_map.find(inst);
    if (iter == strategy_map.end()) {
      continue;
    }

    const StrategyVector* strategies = iter->second.get();
    if (strategies->is_tuple) {
      const Shape& out_shape = inst->shape();
      ShapeTree<HloSharding> output_tuple_sharding(out_shape, Undefined());
      std::vector<HloSharding> output_flattened_shardings;

      bool set_tuple_sharding = true;
      for (const auto& t : strategies->childs) {
        int node_idx = t->id;
        int stra_idx = s_val[node_idx];
        // Do not set completed sharding before the last iteration
        if (t->leaf_vector[stra_idx].output_sharding.IsReplicated() &&
            !last_iteration) {
          set_tuple_sharding = false;
        }
        output_flattened_shardings.push_back(
            t->leaf_vector[stra_idx].output_sharding);
      }
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

void SetHloShardingPostProcessing(const HloInstructionSequence& sequence,
                                  const StrategyMap& strategy_map,
                                  const CostGraph& cost_graph,
                                  absl::Span<const int64_t> s_val,
                                  const ClusterEnvironment& cluster_env) {
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
          cluster_env.GetTensorDimToMeshDimWrapper(lhs->shape(), lhs_sharding);
      const auto& rhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDimWrapper(rhs->shape(), rhs_sharding);

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
        FixMixedMeshShapeResharding(inst, 0, stra.input_shardings[0],
                                    device_mesh, resharding_cache);
        FixMixedMeshShapeResharding(inst, 1, stra.input_shardings[1],
                                    device_mesh, resharding_cache);
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
          cluster_env.GetTensorDimToMeshDimWrapper(lhs->shape(), lhs_sharding);
      const auto& rhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDimWrapper(rhs->shape(), rhs_sharding);

      if (absl::StrContains(stra.name, "allreduce") &&
          lhs_tensor_dim_to_mesh_dim[lhs_in_channel_dim] == -1 &&
          rhs_tensor_dim_to_mesh_dim[rhs_in_channel_dim] == -1) {
        // Allow duplicatd conv computation in this case to reduce
        // communication
      } else {
        FixMixedMeshShapeResharding(inst, 0, stra.input_shardings[0],
                                    device_mesh, resharding_cache);
        FixMixedMeshShapeResharding(inst, 1, stra.input_shardings[1],
                                    device_mesh, resharding_cache);
      }
    } else if (inst->opcode() == HloOpcode::kReshape) {
      const ShardingStrategy& stra =
          GetShardingStrategy(inst, strategy_map, cost_graph, s_val);
      if (!stra.input_shardings.empty()) {
        FixMixedMeshShapeResharding(inst, 0, stra.input_shardings[0],
                                    device_mesh, resharding_cache);
      }
    } else {
      // TODO(pratikf): We currently skip over tuple shaped instructions here as
      // GetShardingStrategy, which is invoked below does not currently support
      // such instructions. Implement this support.
      if (inst->shape().IsTuple()) {
        switch (inst->opcode()) {
          case HloOpcode::kReduce:
          case HloOpcode::kSort: {
            for (size_t i = 0; i < inst->shape().tuple_shapes_size(); ++i) {
              const ShardingStrategy& stra = GetShardingStrategyForTuple(
                  inst, i, strategy_map, cost_graph, s_val);
              if (stra.input_shardings.size() > i) {
                FixMixedMeshShapeResharding(inst, i, stra.input_shardings[i],
                                            device_mesh, resharding_cache);
              }
            }
            break;
          }
          case HloOpcode::kTuple: {
            for (size_t i = 0; i < inst->shape().tuple_shapes_size(); ++i) {
              const ShardingStrategy& stra = GetShardingStrategyForTuple(
                  inst, i, strategy_map, cost_graph, s_val);
              CHECK_EQ(stra.input_shardings.size(), 1);
              FixMixedMeshShapeResharding(inst, i, stra.input_shardings[0],
                                          device_mesh, resharding_cache);
            }
            break;
          }
          case HloOpcode::kWhile:
          case HloOpcode::kConditional: {
            break;
          }
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
          FixMixedMeshShapeReshardingGetTupleElement(inst, inst->sharding(),
                                                     device_mesh);
        } else {
          for (size_t i = 0; i < inst->operand_count(); ++i) {
            if (stra.input_shardings.size() > i) {
              FixMixedMeshShapeResharding(inst, i, stra.input_shardings[i],
                                          device_mesh, resharding_cache);
            }
          }
        }
      }
    }
  }

  for (HloInstruction* inst : sequence.instructions()) {
    if (inst->opcode() == HloOpcode::kIota) {
      if (inst->sharding().IsReplicated()) {
        // For fully replicated iota, leave its sharding annotation to the
        // ShardingPropagation pass, which can typically do a better job.
        inst->clear_sharding();
      }
    }
  }
}

// Print liveness set for debugging.
std::string PrintLivenessSet(const LivenessSet& liveness_set) {
  std::string str("Liveness Set\n");
  for (size_t i = 0; i < liveness_set.size(); ++i) {
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
                                      absl::Span<const int64_t> s_val,
                                      double objective) {
  std::string str("=== Auto sharding strategy ===\n");
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  size_t N = leaf_strategies.size();

  // Print the chosen strategy
  for (size_t i = 0; i < N; ++i) {
    absl::StrAppend(&str, i, " ",
                    instructions[leaf_strategies[i]->instruction_id]->ToString(
                        HloPrintOptions::ShortParsable()),
                    " ");
    int stra_idx = cost_graph.RemapIndex(i, s_val[i]);
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
                                     absl::Span<const int64_t> s_val) {
  // Print the memory usage
  std::string str("=== Memory ===\n");
  std::vector<std::pair<size_t, double>> time_memory_usage;
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
    int64_t ins_idx = strategies->id;
    int stra_idx = cost_graph.RemapIndex(ins_idx, s_val[ins_idx]);
    const ShardingStrategy& strategy = strategies->leaf_vector[stra_idx];
    return strategy.memory_cost;
  };
  for (size_t t = 0; t < liveness_set.size(); ++t) {
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
    bool operator()(std::pair<size_t, double> a,
                    std::pair<size_t, double> b) const {
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
    HloModule* module, AutoShardingOption::PreserveShardingsType type) {
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
                 preserve_shardings.at(inst->name())[0].ToString() !=
                     inst->sharding().ToString()) {
        LOG(FATAL) << "User sharding is not preserved! Instruction with name "
                   << inst->name() << " should be: "
                   << preserve_shardings.at(inst->name())[0].ToString()
                   << "\nbut it's: " << inst->sharding().ToString();
      } else if (inst->sharding().IsTuple()) {
        const std::vector<HloSharding>* preserve_shardings_tuple =
            &preserve_shardings.at(inst->name());
        for (size_t i = 0; i < inst->shape().tuple_shapes_size(); i++) {
          if (preserve_shardings_tuple->at(i).ToString() !=
              inst->sharding().tuple_elements().at(i).ToString()) {
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
                               int64_t num_devices) {
  int64_t max_memory_usage = 0;
  for (size_t t = 0; t < liveness_set.size(); ++t) {
    int64_t memory_usage = 0;
    for (const HloValue* value : liveness_set[t]) {
      size_t tmp;
      if (value->instruction()->shape().IsTuple() && value->index().empty()) {
        continue;
      }
      Shape shape =
          ShapeUtil::GetSubshape(value->instruction()->shape(), value->index());
      if (value->instruction()->has_sharding()) {
        tmp = GetShardedInstructionSize(
            shape, num_devices,
            !value->index().empty()
                ? value->instruction()->sharding().GetSubSharding(
                      value->instruction()->shape(), value->index())
                : value->instruction()->sharding());
      } else {
        tmp = GetShardedInstructionSize(shape, num_devices);
      }
      memory_usage += tmp;
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
    if (preserve_shardings.find(ins->name()) != preserve_shardings.end()) {
      if (ins->shape().IsTuple()) {
        ShapeTree<HloSharding> output_tuple_sharding(ins->shape(), Undefined());
        size_t i = 0;
        for (auto& leaf : output_tuple_sharding.leaves()) {
          leaf.second = preserve_shardings.at(ins->name()).at(i++);
        }
        ins->set_sharding(HloSharding::Tuple(output_tuple_sharding));
      } else {
        ins->set_sharding(preserve_shardings.at(ins->name()).at(0));
      }
    }
  }
}
// DFS to find the replicated set starting from cur instruction.
void FindReplicateSet(
    HloInstruction* cur, const AliasMap& alias_map, const CostGraph& cost_graph,
    absl::Span<const int64_t> s_val, const StrategyMap& strategy_map,
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
                           absl::Span<const int64_t> s_val,
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
      // std::cerr << "ins: " << inst->ToString() << ", spec: " <<
      // output_spec.ToString() << std::endl;
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
    return tsl::errors::InvalidArgument(
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

    Array<int64_t> tile_assignment = strategy.output_sharding.tile_assignment();
    tile_assignment.Reshape({cluster_env.total_devices_});
    return HloSharding::Tile(std::move(tile_assignment));

  } else {
    LOG(FATAL) << "Invalid instruction: " << ins->ToString();
  }

  return Undefined();
}

// Return whether an instruction has the opportunity to generate reduce-scatter.
bool HasReduceScatterOpportunity(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const int64_t> s_val,
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

StatusOr<bool> AutoShardingImplementation::RunAutoSharding(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!option_.enable) {
    return false;
  }
  bool module_is_changed = false;
  VLOG(1) << "Start auto sharding pass";

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

  // Remove CustomCalls with custom_call_target="Sharding" and move their
  // shardings to their input ops.
  absl::flat_hash_map<const HloInstruction*, std::vector<int64_t>>
      unspecified_dims;
  // TODO(b/208668853): Keep shardings in custom-calls. After auto
  // sharding pass, instead of fixing the shardings, mark all the replicated
  // dims as "unspecified" instead of replicated. (this would require using
  // custom-call annotations ops instead of in-place attributes). Then run the
  // sharding propagation pass after that before spmd partitioner.
  auto status_or_changed = ProcessShardingInstruction(
      module, execution_threads, /*replace_sharding_with_copy=*/true,
      &unspecified_dims, /*saved_root_shardings=*/nullptr,
      /*saved_parameter_shardings=*/nullptr);
  if (!status_or_changed.ok()) {
    return status_or_changed;
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
      preserve_shardings =
          spmd::SaveUserShardings(module, option_.preserve_shardings);

  // Remove xla sharding annotations, if there is any.
  if (option_.preserve_shardings !=
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings) {
    StatusOr<bool> status_or_changed =
        RemoveShardingAnnotation(module, execution_threads);
    if (!status_or_changed.ok()) {
      return status_or_changed;
    }
    if (status_or_changed.value()) {
      module_is_changed = true;
      VLOG(3) << "XLA sharding annotations are removed.";
    } else {
      VLOG(3) << "This workload does not have XLA sharding annotations.";
    }
  }

  XLA_VLOG_LINES(6,
                 absl::StrCat("Before auto sharding:\n", module->ToString()));
  DumpHloModuleIfEnabled(*module, "before_auto_spmd_sharding");
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
    for (int64_t i = iter.second.start; i <= iter.second.end; ++i) {
      liveness_set[i].push_back(iter.first);
    }
  }
  VLOG(10) << hlo_live_range->ToString();
  VLOG(10) << spmd::PrintLivenessSet(liveness_set);
  XLA_VLOG_LINES(10, spmd::PrintLivenessSet(liveness_set));
  const HloInstructionSequence& sequence =
      hlo_live_range->flattened_instruction_sequence();

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
        return changed_or;
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

    int64_t memory_lower_bound = spmd::MemoryBudgetLowerBound(
        *module, liveness_set, device_mesh.num_elements());
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
      return true;
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
        BuildStrategyAndCost(sequence, module, ins_depth_map, batch_dim_map,
                             alias_map, cluster_env, solver_option, *call_graph,
                             option_.try_multiple_mesh_shapes));
    spmd::AliasSet alias_set = spmd::BuildAliasSet(module, strategy_map);
    CheckAliasSetCompatibility(alias_set, leaf_strategies, sequence);
    XLA_VLOG_LINES(8, PrintStrategyMap(strategy_map, sequence));

    // ----- Build cost graph and merge unimporant nodes -----
    spmd::CostGraph cost_graph(leaf_strategies, associative_dot_pairs);
    cost_graph.Simplify(option_.simplify_graph);

    // ----- Call the ILP Solver -----
    std::vector<int64_t> s_val, e_val;
    double objective = -1.0;
    if (!solver_option.load_solution_vector) {
      TF_ASSIGN_OR_RETURN(
          auto solution,
          CallSolver(sequence, liveness_set, strategy_map, leaf_strategies,
                     cost_graph, alias_set, option_.memory_budget_per_device,
                     /*crash_at_infinity_costs_check*/
                     !option_.try_multiple_mesh_shapes));
      std::tie(s_val, e_val, objective) = solution;
      this->solver_optimal_objective_value_ = objective;
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
                                   cluster_env);
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
  XLA_VLOG_LINES(6, absl::StrCat("After auto sharding:\n", module->ToString()));
  DumpHloModuleIfEnabled(*module, "after_auto_spmd_sharding");

  return module_is_changed;
}

AutoSharding::AutoSharding(const AutoShardingOption& option)
    : option_(option) {}

StatusOr<bool> AutoSharding::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running auto-sharding pass";

#if !defined(__APPLE__)
  // Streamz metrics.
  absl::Time start_time = absl::Now();
  metrics::RecordAutoShardingInvocations();
#endif

  TF_RETURN_IF_ERROR(option_.CheckAndSetup());
  VLOG(1) << "AutoShardingOptions:\n" << option_.ToString();

  if (!option_.try_multiple_mesh_shapes) {
    AutoShardingImplementation pass(option_);
    auto result = pass.RunAutoSharding(module, execution_threads);
    this->solver_optimal_objective_value_ =
        pass.GetSolverOptimalObjectiveValue();
    this->chosen_mesh_shape_ = option_.device_mesh_shape;
    return result;
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
  std::vector<StatusOr<bool>> changed(num_meshes, false);
  std::vector<double> objective_values(num_meshes, -1);

  VLOG(1) << "Original mesh shape "
          << spmd::ToString(option_.device_mesh_shape);
  double min_objective_value = std::numeric_limits<double>::max();
  int min_mesh_shape_index = -1;
  for (size_t i = 0; i < mesh_shapes.size(); ++i) {
    VLOG(1) << "Trying mesh shape " << spmd::ToString(mesh_shapes[i]);
    AutoShardingOption this_option = option_;
    this_option.device_mesh_shape = mesh_shapes[i];
    auto pass = new AutoShardingImplementation(this_option);
    auto module_clone = module->Clone("");
    module_clone->set_layout_canonicalization_callback(
        module->layout_canonicalization_callback());
    auto pass_result =
        pass->RunAutoSharding(module_clone.get(), execution_threads);

    changed[i] = pass_result;
    objective_values[i] = pass->GetSolverOptimalObjectiveValue();
    modules[i] = std::move(module_clone);
    delete pass;
    VLOG(1) << "Mesh shape " << spmd::ToString(mesh_shapes[i])
            << " has objective value " << objective_values[i];
    if (objective_values[i] >= 0 && min_objective_value > objective_values[i]) {
      min_mesh_shape_index = i;
      min_objective_value = objective_values[i];
    }
  }

  CHECK_GE(min_mesh_shape_index, 0)
      << "The auto-sharding pass could not find a device mesh that works for "
         "this input. This could be the result of a low memory budget. If you "
         "think you have a reasonably large memory budget, please report this "
         "an a bug.";

  StatusOr<bool> module_is_changed;
  if (!changed[min_mesh_shape_index].ok()) {
    module_is_changed = changed[min_mesh_shape_index];
  } else {
    solver_optimal_objective_value_ = min_objective_value;
    chosen_mesh_shape_ = mesh_shapes[min_mesh_shape_index];
    if (*changed[min_mesh_shape_index]) {
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
    } else if (!*changed[min_mesh_shape_index]) {
      module_is_changed = false;
    } else {
      module_is_changed = false;
    }
  }

#if !defined(__APPLE__)
  absl::Time end_time = absl::Now();
  auto duration = end_time - start_time;
  metrics::RecordAutoShardingCompilationTime(
      absl::ToInt64Microseconds(duration));
#endif
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
