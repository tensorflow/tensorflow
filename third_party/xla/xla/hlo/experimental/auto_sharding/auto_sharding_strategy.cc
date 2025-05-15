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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_device_mesh.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_wrapper.h"
#include "xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/sharding_propagation.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace spmd {

void EnumerateHelper(std::function<void(const DimMap&)> split_func,
                     int tensor_rank, int current_mesh_dim_idx,
                     const std::vector<int>& unassigned_mesh_dims,
                     const DimMap& current_dim_map,
                     bool allow_mixed_mesh_shape) {
  if (current_mesh_dim_idx == unassigned_mesh_dims.size()) {
    split_func(current_dim_map);
    return;
  }
  // Current mesh dim is not assigned to any tensor dim
  EnumerateHelper(split_func, tensor_rank, current_mesh_dim_idx + 1,
                  unassigned_mesh_dims, current_dim_map,
                  allow_mixed_mesh_shape);

  for (int i = 0; i < tensor_rank; ++i) {
    DimMap updated_dim_map = current_dim_map;
    if (!updated_dim_map[i].empty() && !allow_mixed_mesh_shape) {
      continue;
    }
    updated_dim_map[i].insert(unassigned_mesh_dims[current_mesh_dim_idx]);
    EnumerateHelper(split_func, tensor_rank, current_mesh_dim_idx + 1,
                    unassigned_mesh_dims, updated_dim_map,
                    allow_mixed_mesh_shape);
  }
}

// Map tensor dims from [0, tensor_shape.dimensions_size() - 1] to (atmost one
// or more, depending on the value of allow_mixed_mesh_shape) mesh dims.
void Enumerate(std::function<void(const DimMap&)> split_func,
               int64_t tensor_rank,
               const std::vector<int>& unassigned_mesh_dims,
               bool allow_mixed_mesh_shape) {
  EnumerateHelper(split_func, tensor_rank, /*current_mesh_dim_idx=*/0,
                  unassigned_mesh_dims,
                  /*current_dim_map=*/{}, allow_mixed_mesh_shape);
}

bool LeafVectorsAreConsistent(const std::vector<ShardingStrategy>& one,
                              const std::vector<ShardingStrategy>& two) {
  if (one.size() != two.size()) return false;
  for (size_t sid = 0; sid < one.size(); ++sid) {
    const bool invalid_strategy_one = (one[sid].compute_cost >= kInfinityCost);
    const bool invalid_strategy_two = (two[sid].compute_cost >= kInfinityCost);
    if (invalid_strategy_one != invalid_strategy_two) return false;
  }
  return true;
}

std::optional<HloSharding> ConstructImprovedSharding(
    HloSharding from, const HloSharding& to_improved,
    const Shape& to_improved_shape, bool may_combine_partial_sharding,
    bool allow_aggressive_resharding) {
  return hlo_sharding_util::ReturnImprovedShardingImpl(
      from, &to_improved, to_improved_shape, may_combine_partial_sharding,
      allow_aggressive_resharding);
}

std::pair<HloSharding, double>
ComputeSliceShardingAndCommunicationCostFromOperand(
    const HloSharding& input_spec, const Shape& old_shape,
    const Shape& new_shape, const DeviceMesh& device_mesh,
    const ClusterEnvironment& cluster_env) {
  if (input_spec.IsReplicated()) {
    return std::make_pair(input_spec, 0);
  }

  CHECK(old_shape.IsArray());

  std::vector<int64_t> tensor_to_mesh_dim = GetTensorDimToMeshDim(
      new_shape.dimensions().size(), input_spec, device_mesh,
      /* consider_reverse_device_meshes */ true);

  std::vector<int64_t> mesh_dims_for_communication;
  std::vector<int64_t> tensor_dims;
  std::vector<int64_t> mesh_dims;
  for (size_t i = 0; i < new_shape.dimensions().size(); ++i) {
    if (tensor_to_mesh_dim[i] == -1) {
      continue;
    }
    tensor_dims.push_back(i);
    mesh_dims.push_back(tensor_to_mesh_dim[i]);
    if (new_shape.dimensions(i) != old_shape.dimensions(i)) {
      mesh_dims_for_communication.push_back(tensor_to_mesh_dim[i]);
    }
  }

  // When input_spec shards one or more or the sliced tensor dimensions, we
  // might be required to perform some collective communication. In the worst
  // case, the sliced output would be available on one machine, which we would
  // need to then re-shard across the devices per result. We approximate the
  // cost for this operation by adding up the ReduceScatter cost across the mesh
  // dimensions that shard sliced tensor dimensions.
  const HloSharding& result =
      Tile(new_shape, tensor_dims, mesh_dims, device_mesh);
  double num_bytes_to_transfer = ByteSizeOfShape(new_shape);
  double communication_cost = 0;
  for (size_t i = 0; i < mesh_dims_for_communication.size(); ++i) {
    int64_t mesh_dim = mesh_dims_for_communication[i];
    num_bytes_to_transfer /= device_mesh.dim(mesh_dim);
    communication_cost +=
        cluster_env.ReduceScatterCost(num_bytes_to_transfer, mesh_dim);
  }
  return std::make_pair(result, communication_cost);
}

// Generates strategies for scatter ops, given the shardings for its operands.
// This implementation is a simplified/modified version of the handling of
// scatter ops in ShardingPropagation::InferShardingFromOperands. This
// implementation currently does not support tuple-shaped scatter ops (nor did
// the original implementation), but it should be easy to generalize if needed.
void GenerateScatterShardingFromOperands(
    const HloScatterInstruction* scatter, const HloSharding& data_sharding,
    const HloSharding& update_sharding, const CallGraph& call_graph,
    absl::FunctionRef<void(const HloSharding& data_sharding,
                           const HloSharding& indices_sharding,
                           const HloSharding& update_sharding,
                           const HloSharding& scatter_sharding)>
        yield_sharding) {
  std::vector<HloSharding> scatter_shardings;
  auto scatter_shardings_insert = [&](const HloSharding& sharding) {
    const auto it =
        std::find(scatter_shardings.begin(), scatter_shardings.end(), sharding);
    if (it == scatter_shardings.end()) scatter_shardings.push_back(sharding);
  };
  CHECK_EQ(scatter->scatter_operand_count(), 1);

  const HloSharding& indices_sharding =
      hlo_sharding_util::ScatterIndexShardingFromUpdate(update_sharding,
                                                        scatter);

  scatter_shardings_insert(data_sharding);
  if (std::optional<HloSharding> maybe_from_update =
          hlo_sharding_util::ScatterOutputShardingFromUpdate(update_sharding,
                                                             *scatter)) {
    scatter_shardings_insert(*maybe_from_update);
  }

  std::optional<hlo_sharding_util::GatherScatterDims> scatter_parallel_dims =
      hlo_sharding_util::GetScatterParallelBatchDims(*scatter, call_graph);
  if (!scatter_parallel_dims) {
    for (const HloSharding& sharding : scatter_shardings) {
      yield_sharding(data_sharding, indices_sharding, update_sharding,
                     sharding);
    }
    return;
  }

  // Infer output sharding from scatter operand sharding.
  const Shape& shape = scatter->shape();
  scatter_shardings_insert(
      hlo_sharding_util::InferGatherScatterParallelShardingFromOperandSharding(
          data_sharding, shape,
          absl::MakeConstSpan(scatter_parallel_dims->operand_dims),
          absl::MakeConstSpan(scatter_parallel_dims->operand_dims)));

  // Infer output sharding from scatter indices sharding.
  scatter_shardings_insert(
      hlo_sharding_util::InferGatherScatterParallelShardingFromOperandSharding(
          indices_sharding, shape,
          absl::MakeConstSpan(scatter_parallel_dims->indices_dims),
          absl::MakeConstSpan(scatter_parallel_dims->operand_dims)));

  // Infer output sharding from scatter update sharding.
  scatter_shardings_insert(
      hlo_sharding_util::InferGatherScatterParallelShardingFromOperandSharding(
          update_sharding, shape,
          absl::MakeConstSpan(scatter_parallel_dims->output_dims),
          absl::MakeConstSpan(scatter_parallel_dims->operand_dims)));

  for (const HloSharding& scatter_sharding : scatter_shardings) {
    yield_sharding(data_sharding, indices_sharding, update_sharding,
                   scatter_sharding);
  }
}

// NOLINTBEGIN(readability/fn_size)
// TODO(zhuohan): Decompose this function into smaller pieces
absl::StatusOr<std::tuple<StrategyMap, StrategyGroups, AssociativeDotPairs>>
BuildStrategyAndCost(
    const HloInstructionSequence& sequence, const HloModule* module,
    const absl::flat_hash_set<const HloInstruction*>& instructions_to_shard,
    const absl::flat_hash_map<const HloInstruction*, int64_t>&
        instruction_execution_counts,
    const InstructionDepthMap& depth_map, const AliasMap& alias_map,
    const ClusterEnvironment& cluster_env, AutoShardingOption& option,
    const CallGraph& call_graph, const HloCostAnalysis& hlo_cost_analysis,
    bool trying_multiple_mesh_shapes) {
  // const DeviceMesh& device_mesh = cluster_env.device_mesh_;
  StrategyMap strategy_map;
  // This map stores all of the trimmed strategies due to user specified
  // sharding. The key is the instruction id, the value is the strategies. This
  // is useful when the operand is forced to use a user sharding, and the op
  // doesn't need to strictly follow it. We restore the trimmed strategies in
  // this situation.
  StableMap<int64_t, std::vector<ShardingStrategy>> pretrimmed_strategy_map;
  StrategyGroups strategy_groups;
  AssociativeDotPairs associative_dot_pairs;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Add penalty for replicated tensors
  double replicated_penalty = cluster_env.GetDefaultReplicatedPenalty();

  int64_t max_depth = -1;
  for (auto iter : depth_map) {
    max_depth = std::max(max_depth, iter.second);
  }

  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      while_body_args_to_input_tuple;
  // Register strategies and their costs for each instruction.
  for (size_t instruction_id = 0; instruction_id < instructions.size();
       ++instruction_id) {
    const HloInstruction* ins = instructions[instruction_id];
    VLOG(2) << "instruction_id = " << instruction_id << ": "
            << ToAdaptiveString(ins);
    std::unique_ptr<StrategyGroup> strategy_group;

    if (!instructions_to_shard.contains(ins)) {
      VLOG(2) << "  Manually sharded;";
      strategy_group = HandleManuallyShardedInstruction(
          ins, ins->shape(), instruction_id, strategy_groups, strategy_map);
      XLA_VLOG_LINES(2,
                     absl::StrCat("strategies:\n", strategy_group->ToString()));
      strategy_map[ins] = std::move(strategy_group);
      continue;
    }

    HloOpcode opcode = ins->opcode();

    bool only_allow_divisible;
    if (IsEntryComputationInputOrOutput(module, ins)) {
      // With IsEntryComputationInputOrOutput(module, ins) == true, entry
      // computation's root instruction may still be unevenly sharded because it
      // usually "follows" other instruction's sharding. If the instruction it
      // follows is an intermediate instruction, it may be able to choose
      // unevenly sharded strategiyes. Usually if we constraint input's sharding
      // strategies, outputs would be constrained as well, but if outputs are
      // still unevely sharded in some cases, we need to fix the implementation
      // in auto sharding.
      only_allow_divisible = option.only_allow_divisible_input_output;
    } else {
      only_allow_divisible = option.only_allow_divisible_intermediate;
    }

    bool is_follow_necessary_for_correctness = false;
    switch (opcode) {
      case HloOpcode::kParameter: {
        if (auto it = while_body_args_to_input_tuple.find(ins);
            it != while_body_args_to_input_tuple.end()) {
          const HloInstruction* while_input_tuple = it->second;
          const StrategyGroup* while_input_tuple_strategy_group =
              strategy_map.at(while_input_tuple).get();

          VLOG(5) << "Following while input " << while_input_tuple->name();
          strategy_group = CreateTupleStrategyGroup(instruction_id);
          // We use this following relationship to ensure that the input tuple
          // of the while loop, and the parameter of the body of that while
          // loop. Therefore, this followinf relationship is necessary for
          // correctness, and is not merely an optimization.
          is_follow_necessary_for_correctness = true;
          for (size_t i = 0; i < ins->shape().tuple_shapes().size(); ++i) {
            std::unique_ptr<StrategyGroup> child_strategies =
                MaybeFollowInsStrategyGroup(
                    *while_input_tuple_strategy_group->GetChildren()[i],
                    ins->shape().tuple_shapes().at(i), instruction_id,
                    strategy_groups, cluster_env, pretrimmed_strategy_map);
            child_strategies->tuple_element_idx = i;
            strategy_group->AddChild(std::move(child_strategies));
          }
          break;
        }
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, call_graph,
                only_allow_divisible, option.allow_replicated_parameters,
                /* create_partially_replicated_strategies */ true)
                .value();
        break;
      }
      case HloOpcode::kRngBitGenerator:
      case HloOpcode::kRng: {
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, call_graph,
                only_allow_divisible, option.allow_replicated_parameters,
                /* create_partially_replicated_strategies */ true)
                .value();
        break;
      }
      case HloOpcode::kConstant: {
        strategy_group = CreateLeafStrategyGroupWithoutInNodes(instruction_id,
                                                               strategy_groups);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map, 0,
                              {}, *strategy_group);
        break;
      }
      case HloOpcode::kScatter: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        auto add_scatter_sharding = [&](const HloSharding& data_sharding,
                                        const HloSharding& indices_sharding,
                                        const HloSharding& update_sharding,
                                        const HloSharding& scatter_sharding) {
          std::string name = ToStringSimple(scatter_sharding);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost =
              ByteSizeOfShapeWithSharding(ins->shape(), scatter_sharding);

          InputShardings input_shardings_optional(
              {name, {data_sharding, indices_sharding, update_sharding}});
          std::pair<ReshardingCosts, ReshardingCosts> resharding_costs =
              GenerateReshardingCostsAndMissingShardingsForAllOperands(
                  ins, scatter_sharding, strategy_map, cluster_env, call_graph,
                  input_shardings_optional);

          strategy_group->AddStrategy(
              ShardingStrategy({scatter_sharding, compute_cost,
                                communication_cost, memory_cost,
                                std::move(resharding_costs.first),
                                std::move(resharding_costs.second)}),
              input_shardings_optional);
        };

        const HloScatterInstruction* scatter = Cast<HloScatterInstruction>(ins);
        const HloInstruction* scatter_data = scatter->scatter_operands()[0];
        const HloInstruction* scatter_update = scatter->scatter_updates()[0];

        ForEachInCartesianProduct<ShardingStrategy>(
            {strategy_map.at(scatter_data)->GetStrategies(),
             strategy_map.at(scatter_update)->GetStrategies()},
            [&](const std::vector<ShardingStrategy>& operand_shardings) {
              GenerateScatterShardingFromOperands(
                  scatter, operand_shardings[0].output_sharding,
                  operand_shardings[1].output_sharding, call_graph,
                  add_scatter_sharding);
            });

        break;
      }
      case HloOpcode::kGather: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        const HloInstruction* data = ins->operand(0);
        const HloInstruction* indices = ins->operand(1);
        const Shape& gather_shape = ins->shape();

        const StrategyGroup* data_strategy_group = strategy_map.at(data).get();
        const StrategyGroup* indices_strategy_group =
            strategy_map.at(indices).get();

        auto add_sharding_strategy = [&](const HloSharding& data_sharding,
                                         const HloSharding& indices_sharding,
                                         const HloSharding& output_sharding) {
          if (output_sharding.IsReplicated()) {
            return;
          }
          double compute_cost = 0, communication_cost = 0;
          double memory_cost =
              ByteSizeOfShapeWithSharding(gather_shape, output_sharding);
          InputShardings input_shardings_optional(
              {output_sharding.ToString(), {data_sharding, indices_sharding}});
          std::pair<ReshardingCosts, ReshardingCosts> resharding_costs =
              GenerateReshardingCostsAndMissingShardingsForAllOperands(
                  ins, output_sharding, strategy_map, cluster_env, call_graph,
                  input_shardings_optional);

          strategy_group->AddStrategy(
              ShardingStrategy({output_sharding, compute_cost,
                                communication_cost, memory_cost,
                                std::move(resharding_costs.first),
                                std::move(resharding_costs.second)}),
              input_shardings_optional);
        };

        for (const ShardingStrategy& indices_strategy :
             indices_strategy_group->GetStrategies()) {
          const HloSharding& indices_spec = indices_strategy.output_sharding;
          const HloSharding& indices_to_combine_spec =
              hlo_sharding_util::GatherOutputShardingFromIndex(indices_spec,
                                                               ins);
          if (std::optional<HloSharding> data_spec =
                  hlo_sharding_util::GatherOperandShardingFromOutput(
                      indices_to_combine_spec, *ins, call_graph)) {
            add_sharding_strategy(*data_spec, indices_spec,
                                  indices_to_combine_spec);
          } else {
            add_sharding_strategy(HloSharding::Replicate(), indices_spec,
                                  indices_to_combine_spec);
          }

          for (const ShardingStrategy& data_strategy :
               data_strategy_group->GetStrategies()) {
            const HloSharding& data_spec = data_strategy.output_sharding;
            auto gather_parallel_dims =
                hlo_sharding_util::GetGatherParallelBatchDims(*ins, call_graph);
            HloSharding output_spec = indices_to_combine_spec;
            if (gather_parallel_dims) {
              // Infer output sharding from scatter operand sharding.
              if (hlo_sharding_util::IsSpatiallyPartitioned(data_spec)) {
                const HloSharding to_merge = hlo_sharding_util::
                    InferGatherScatterParallelShardingFromOperandSharding(
                        data_spec, gather_shape,
                        absl::MakeConstSpan(gather_parallel_dims->operand_dims),
                        absl::MakeConstSpan(gather_parallel_dims->output_dims));
                if (std::optional<HloSharding> improved_spec =
                        ConstructImprovedSharding(
                            to_merge, output_spec, gather_shape,
                            /*may_combine_partial_sharding=*/true,
                            /*allow_aggressive_resharding=*/false)) {
                  output_spec = *improved_spec;
                  add_sharding_strategy(data_spec, indices_spec, output_spec);
                }
              }
              // Infer output sharding from scatter indices sharding.
              if (hlo_sharding_util::IsSpatiallyPartitioned(indices_spec)) {
                const HloSharding to_merge = hlo_sharding_util::
                    InferGatherScatterParallelShardingFromOperandSharding(
                        indices_spec, gather_shape,
                        absl::MakeConstSpan(gather_parallel_dims->indices_dims),
                        absl::MakeConstSpan(gather_parallel_dims->output_dims));
                if (std::optional<HloSharding> improved_spec =
                        ConstructImprovedSharding(
                            to_merge, output_spec, gather_shape,
                            /*may_combine_partial_sharding=*/true,
                            /*allow_aggressive_resharding=*/false)) {
                  output_spec = *improved_spec;
                  add_sharding_strategy(data_spec, indices_spec, output_spec);
                }
              }
            }

            absl::Span<const int64_t> operand_parallel_dims;
            if (gather_parallel_dims) {
              operand_parallel_dims =
                  absl::MakeConstSpan(gather_parallel_dims->operand_dims);
            }
            HloSharding filtered_operand_sharding =
                hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
                    data_spec, operand_parallel_dims);
            std::optional<HloSharding> maybe_from_data = hlo_sharding_util::
                GatherOutputShardingFromOperandOperandPassthroughDimensions(
                    filtered_operand_sharding, *ins);

            if (!maybe_from_data) {
              continue;
            }

            if (std::optional<HloSharding> improved_spec =
                    ConstructImprovedSharding(
                        *maybe_from_data, output_spec, gather_shape,
                        /*may_combine_partial_sharding=*/true,
                        /*allow_aggressive_resharding=*/false)) {
              output_spec = *improved_spec;
              add_sharding_strategy(data_spec, indices_spec, output_spec);
            }
          }
        }
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map, 0,
                              /*operands_to_consider_all_strategies_for=*/{},
                              *strategy_group);
        break;
      }
      case HloOpcode::kBroadcast: {
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, call_graph,
                only_allow_divisible,
                /* create_replicated_strategies */ true,
                /* create_partially_replicated_strategies */ true)
                .value();
        break;
      }
      case HloOpcode::kReshape: {
        strategy_group = CreateReshapeStrategies(
            instruction_id, ins, strategy_map, cluster_env,
            only_allow_divisible, replicated_penalty, option, strategy_groups,
            call_graph);
        break;
      }
      case HloOpcode::kTranspose:
      case HloOpcode::kReverse: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);

        const HloInstruction* operand = ins->operand(0);

        // Create follow strategies
        const StrategyGroup& src_strategy_group = *strategy_map.at(operand);
        CHECK(!src_strategy_group.is_tuple);
        strategy_group->following = &src_strategy_group;

        for (const auto& strategy : src_strategy_group.GetStrategies()) {
          HloSharding output_spec = Undefined();
          const HloSharding& input_spec = strategy.output_sharding;
          if (opcode == HloOpcode::kTranspose) {
            output_spec = hlo_sharding_util::TransposeSharding(
                input_spec, ins->dimensions());
          } else {
            output_spec = hlo_sharding_util::ReverseSharding(input_spec,
                                                             ins->dimensions());
          }

          std::string name = ToStringSimple(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost =
              ByteSizeOfShapeWithSharding(ins->shape(), output_spec);
          std::vector<double> communication_resharding_costs =
              CommunicationReshardingCostVector(src_strategy_group,
                                                operand->shape(), input_spec,
                                                cluster_env);
          std::vector<double> memory_resharding_costs =
              MemoryReshardingCostVector(src_strategy_group, operand->shape(),
                                         input_spec, cluster_env);
          strategy_group->AddStrategy(
              ShardingStrategy({output_spec,
                                compute_cost,
                                communication_cost,
                                memory_cost,
                                {communication_resharding_costs},
                                {memory_resharding_costs}}),
              {name, {input_spec}});
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
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
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
        StrategyGroup* src_strategy_group = strategy_map.at(operand).get();
        CHECK(!src_strategy_group->is_tuple);
        strategy_group->following = src_strategy_group;

        for (const auto& strategy : src_strategy_group->GetStrategies()) {
          std::optional<HloSharding> output_spec;
          const HloSharding& input_spec = strategy.output_sharding;

          double compute_cost = 0, communication_cost = 0;
          // Find output shardings.
          switch (opcode) {
            case HloOpcode::kSlice: {
              // When solve_nd_sharding_iteratively is true, in some cases, we
              // can have 1D shardings where the total number of tiles is larger
              // than the number of elements in the partial mesh (and is
              // actually equal to the number of devices in the original
              // mesh). Below, we use the correct mesh depending on the number
              // of elements in the 1D sharding.
              bool is_1d_sharding =
                  VectorGreaterThanOneElementCount(
                      input_spec.tile_assignment().dimensions()) == 1;
              if (is_1d_sharding &&
                  input_spec.TotalNumTiles() ==
                      cluster_env.device_mesh_1d_.num_elements()) {
                std::pair<HloSharding, double>
                    output_spec_and_communication_cost =
                        ComputeSliceShardingAndCommunicationCostFromOperand(
                            input_spec, operand->shape(), ins->shape(),
                            cluster_env.device_mesh_1d_, cluster_env);
                output_spec = output_spec_and_communication_cost.first;
                communication_cost = output_spec_and_communication_cost.second;
              } else if (is_1d_sharding) {
                CHECK_EQ(input_spec.TotalNumTiles(),
                         cluster_env.original_device_mesh_1d_.num_elements());
                std::pair<HloSharding, double>
                    output_spec_and_communication_cost =
                        ComputeSliceShardingAndCommunicationCostFromOperand(
                            input_spec, operand->shape(), ins->shape(),
                            cluster_env.original_device_mesh_1d_, cluster_env);
                output_spec = output_spec_and_communication_cost.first;
                communication_cost = output_spec_and_communication_cost.second;
              } else {
                std::pair<HloSharding, double>
                    output_spec_and_communication_cost =
                        ComputeSliceShardingAndCommunicationCostFromOperand(
                            input_spec, operand->shape(), ins->shape(),
                            cluster_env.device_mesh_, cluster_env);
                output_spec = output_spec_and_communication_cost.first;
                communication_cost = output_spec_and_communication_cost.second;
              }
              break;
            }
            case HloOpcode::kPad:
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
          InputShardings input_shardings;
          for (int64_t k = 0; k < ins->operand_count(); ++k) {
            if (k == follow_idx ||
                ToString(ins->operand(k)->shape().dimensions()) ==
                    ToString(operand->shape().dimensions())) {
              input_shardings.shardings.push_back(input_spec);
            } else {
              input_shardings.shardings.push_back(std::nullopt);
            }
          }
          if (!output_spec.has_value()) {
            continue;
          }

          std::string name = ToStringSimple(*output_spec);
          double memory_cost =
              ByteSizeOfShapeWithSharding(ins->shape(), output_spec);
          std::pair<ReshardingCosts, ReshardingCosts> resharding_costs =
              GenerateReshardingCostsAndMissingShardingsForAllOperands(
                  ins, *output_spec, strategy_map, cluster_env, call_graph,
                  input_shardings);

          strategy_group->AddStrategy(
              ShardingStrategy({*output_spec, compute_cost, communication_cost,
                                memory_cost, std::move(resharding_costs.first),
                                std::move(resharding_costs.second)}),
              {name, {input_spec}});
        }

        if (strategy_group->GetStrategies().empty()) {
          strategy_group->following = nullptr;
          AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map, 0,
                                {}, *strategy_group);
        }
        break;
      }
      case HloOpcode::kOptimizationBarrier: {
        const auto& operand_strategy_group = *strategy_map.at(ins->operand(0));
        strategy_group = MaybeFollowInsStrategyGroup(
            operand_strategy_group, ins->shape(), instruction_id,
            strategy_groups, cluster_env, pretrimmed_strategy_map);
        break;
      }
      case HloOpcode::kBitcast: {
        if (ins->shape() == ins->operand(0)->shape()) {
          strategy_group = CreateElementwiseOperatorStrategies(
              instruction_id, ins, strategy_map, cluster_env, depth_map,
              alias_map, pretrimmed_strategy_map, max_depth, strategy_groups,
              associative_dot_pairs);
        } else {
          strategy_group = CreateReshapeStrategies(
              instruction_id, ins, strategy_map, cluster_env,
              only_allow_divisible, replicated_penalty, option, strategy_groups,
              call_graph);
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
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kCopy:
      case HloOpcode::kCos:
      case HloOpcode::kErf:
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
        strategy_group = CreateElementwiseOperatorStrategies(
            instruction_id, ins, strategy_map, cluster_env, depth_map,
            alias_map, pretrimmed_strategy_map, max_depth, strategy_groups,
            associative_dot_pairs);
        break;
      }
      case HloOpcode::kReduce: {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<StrategyGroup> new_strategy_group,
            FollowReduceStrategy(
                ins, ins->shape(), ins->operand(0), ins->operand(1),
                instruction_id, strategy_map, strategy_groups, cluster_env,
                option.allow_mixed_mesh_shape, !trying_multiple_mesh_shapes));
        strategy_group = std::move(new_strategy_group);
        break;
      }
      case HloOpcode::kDot: {
        TF_RETURN_IF_ERROR(HandleDot(
            strategy_group, strategy_groups, strategy_map, ins, instruction_id,
            sequence, hlo_cost_analysis, cluster_env, option, call_graph));

        if (option.allow_recompute_heavy_op) {
          AddReplicatedStrategy(
              ins, ins->shape(), cluster_env, strategy_map,
              GetDotConvReplicationPenalty(ins, instruction_id, /* window */ 10,
                                           sequence, hlo_cost_analysis),
              {}, *strategy_group);
        }
        break;
      }
      case HloOpcode::kConvolution: {
        TF_RETURN_IF_ERROR(HandleConv(
            strategy_group, strategy_groups, strategy_map, ins, instruction_id,
            sequence, hlo_cost_analysis, cluster_env, option, call_graph));
        if (option.allow_recompute_heavy_op) {
          AddReplicatedStrategy(
              ins, ins->shape(), cluster_env, strategy_map,
              GetDotConvReplicationPenalty(ins, instruction_id, /* window */ 10,
                                           sequence, hlo_cost_analysis),
              {}, *strategy_group);
        }
        break;
      }
      case HloOpcode::kRngGetAndUpdateState: {
        strategy_group = CreateLeafStrategyGroupWithoutInNodes(instruction_id,
                                                               strategy_groups);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map, 0,
                              {}, *strategy_group);
        break;
      }
      case HloOpcode::kIota: {
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, call_graph,
                only_allow_divisible,
                /* create_replicated_strategies */ true,
                /* create_partially_replicated_strategies */ true)
                .value();
        break;
      }
      case HloOpcode::kTuple: {
        strategy_group = CreateTupleStrategyGroup(instruction_id);
        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          const StrategyGroup& src_strategy_group = *strategy_map.at(operand);
          auto child_strategies = MaybeFollowInsStrategyGroup(
              src_strategy_group, operand->shape(), instruction_id,
              strategy_groups, cluster_env, pretrimmed_strategy_map);
          child_strategies->tuple_element_idx = i;
          strategy_group->AddChild(std::move(child_strategies));
        }

        if (ins->users().size() == 1 &&
            ins->users()[0]->opcode() == HloOpcode::kWhile) {
          const HloInstruction* while_op = ins->users()[0];
          while_body_args_to_input_tuple[while_op->while_body()
                                             ->parameter_instruction(0)] = ins;
          while_body_args_to_input_tuple[while_op->while_condition()
                                             ->parameter_instruction(0)] = ins;
        }

        break;
      }
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* operand = ins->operand(0);
        const StrategyGroup& src_strategy_group = *strategy_map.at(operand);
        CHECK(src_strategy_group.is_tuple);
        const auto& src_children = src_strategy_group.GetChildren();
        strategy_group = MaybeFollowInsStrategyGroup(
            *src_children[ins->tuple_index()], ins->shape(), instruction_id,
            strategy_groups, cluster_env, pretrimmed_strategy_map);
        break;
      }
      case HloOpcode::kCustomCall: {
        auto generate_non_following_strategies =
            [&](bool only_replicated,
                absl::flat_hash_set<int64_t>
                    operands_to_consider_all_strategies_for = {}) {
              if (only_replicated) {
                if (ins->shape().IsTuple()) {
                  strategy_group = CreateTupleStrategyGroup(instruction_id);
                  for (size_t i = 0; i < ins->shape().tuple_shapes().size();
                       ++i) {
                    std::unique_ptr<StrategyGroup> child_strategies =
                        CreateLeafStrategyGroup(instruction_id, ins,
                                                strategy_map, strategy_groups);
                    AddReplicatedStrategy(ins, ins->shape().tuple_shapes(i),
                                          cluster_env, strategy_map,
                                          replicated_penalty, {},
                                          *child_strategies);
                    strategy_group->AddChild(std::move(child_strategies));
                  }
                } else {
                  strategy_group = CreateLeafStrategyGroup(
                      instruction_id, ins, strategy_map, strategy_groups);
                  AddReplicatedStrategy(ins, ins->shape(), cluster_env,
                                        strategy_map, replicated_penalty, {},
                                        *strategy_group);
                }
                return;
              }
              strategy_group =
                  CreateAllStrategiesGroup(
                      ins, ins->shape(), instruction_id, strategy_groups,
                      cluster_env, strategy_map, option, replicated_penalty,
                      call_graph, only_allow_divisible,
                      /* create_replicated_strategies */ true,
                      /* create_partially_replicated_strategies */ true)
                      .value();
            };

        if (IsSPMDFullToShardShapeCustomCall(ins)) {
          return absl::InternalError(
              "An SPMDFullToShardShape call found outside a manually "
              "partitioned sub-graph.");
        } else if (IsSPMDShardToFullShapeCustomCall(ins)) {
          if (!ins->has_sharding()) {
            return absl::InternalError(
                "An SPMDShardToFullShape custom call found without a sharding "
                "annotation.");
          }
          generate_non_following_strategies(false);
        } else if (IsTopKCustomCall(ins)) {
          generate_non_following_strategies(false, {0});
        } else if (IsPartialReduceCustomCall(ins)) {
          strategy_group =
              HandlePartialReduce(ins, instruction_id, strategy_groups,
                                  cluster_env, strategy_map, call_graph);
        } else if (OutputInputSameShapes(ins)) {
          auto* partitioner =
              GetCustomCallPartitioner(ins->custom_call_target());
          if (partitioner && partitioner->IsCustomCallShardable(ins)) {
            // Follows operand 0's strategies if this custom-call op is
            // shardable and has the same input and output sizes.
            const HloInstruction* operand = ins->operand(0);
            const StrategyGroup& src_strategy_group = *strategy_map.at(operand);
            strategy_group = MaybeFollowInsStrategyGroup(
                src_strategy_group, ins->shape(), instruction_id,
                strategy_groups, cluster_env, pretrimmed_strategy_map);
          }
        } else if (ins->has_sharding()) {
          generate_non_following_strategies(false);
        } else {
          // TODO (b/258723035) Handle CustomCall ops for GPUs in a better way.
          generate_non_following_strategies(true);
        }
        break;
      }
      case HloOpcode::kWhile: {
        strategy_group = CreateTupleStrategyGroup(instruction_id);
        const auto& src_strategy_group = *strategy_map.at(ins->operand(0));
        const auto& src_children = src_strategy_group.GetChildren();
        for (size_t i = 0; i < ins->shape().tuple_shapes().size(); ++i) {
          auto child_strategies = MaybeFollowInsStrategyGroup(
              *src_children[i], ins->shape().tuple_shapes().at(i),
              instruction_id, strategy_groups, cluster_env,
              pretrimmed_strategy_map);
          child_strategies->tuple_element_idx = i;
          strategy_group->AddChild(std::move(child_strategies));
        }

        break;
      }
      case HloOpcode::kConditional:
      case HloOpcode::kInfeed:
      case HloOpcode::kSort: {
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, call_graph,
                only_allow_divisible,
                /* create_replicated_strategies */ true,
                /* create_partially_replicated_strategies */ true)
                .value();
        break;
      }
      case HloOpcode::kOutfeed: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        GenerateOutfeedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                replicated_penalty, *strategy_group);
        break;
      }
      case HloOpcode::kRecv:
      case HloOpcode::kRecvDone:
      case HloOpcode::kSend: {
        strategy_group = CreateTupleStrategyGroup(instruction_id);
        for (size_t i = 0; i < ins->shape().tuple_shapes().size(); ++i) {
          std::unique_ptr<StrategyGroup> child_strategies =
              CreateLeafStrategyGroup(instruction_id, ins, strategy_map,
                                      strategy_groups);
          AddReplicatedStrategy(ins, ins->shape().tuple_shapes(i), cluster_env,
                                strategy_map, 0, {}, *child_strategies);
          child_strategies->tuple_element_idx = i;
          strategy_group->AddChild(std::move(child_strategies));
        }
        break;
      }
      case HloOpcode::kSendDone: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map, 0,
                              {}, *strategy_group);
        break;
      }
      case HloOpcode::kAfterAll: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              replicated_penalty, {}, *strategy_group);
        break;
      }
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->ToString();
    }
    CHECK(strategy_group != nullptr);
    RemoveDuplicatedStrategy(*strategy_group);
    if (ins->has_sharding() && ins->opcode() != HloOpcode::kOutfeed) {
      // Finds the sharding strategy that aligns with the given sharding spec
      // Do not merge nodes if this one instruction has annotations.
      TrimOrGenerateStrategiesBasedOnExistingSharding(
          ins->shape(), strategy_map, instructions, ins->sharding(),
          cluster_env, pretrimmed_strategy_map, call_graph,
          option.nd_sharding_iteratively_strict_search_space, *strategy_group);
    }
    if (!option.allow_shardings_small_dims_across_many_devices) {
      RemoveShardingsWhereSmallDimsShardedAcrossManyDevices(
          ins->shape(), /* instruction_has_user_sharding */ ins->has_sharding(),
          *strategy_group);
    }
    if (!strategy_group->is_tuple && strategy_group->following) {
      if (!LeafVectorsAreConsistent(
              strategy_group->GetStrategies(),
              strategy_group->following->GetStrategies())) {
        // It confuses the solver if two instructions have different number of
        // sharding strategies but share the same ILP variable. The solver would
        // run much longer and/or return infeasible solutions. So if two
        // strategies are inconsistent, we unfollow them.
        CHECK(!is_follow_necessary_for_correctness)
            << "Reverting a following decision that is necessary for "
               "correctness. Please report this as a bug.";
        strategy_group->following = nullptr;
      }
    } else if (strategy_group->is_tuple) {
      for (size_t i = 0; i < strategy_group->GetChildren().size(); i++) {
        auto& child = strategy_group->GetChildren().at(i);
        if (child->following &&
            !LeafVectorsAreConsistent(child->GetStrategies(),
                                      child->following->GetStrategies())) {
          CHECK(!is_follow_necessary_for_correctness)
              << "Reverting a following decision that is necessary for "
                 "correctness. Please report this as a bug.";
          child->following = nullptr;
        }
      }
    }

    if (instruction_execution_counts.contains(ins)) {
      ScaleCostsWithExecutionCounts(instruction_execution_counts.at(ins),
                                    *strategy_group);
    } else {
      VLOG(5) << "No execution count available for " << ins->name();
    }
    XLA_VLOG_LINES(2,
                   absl::StrCat("strategies:\n", strategy_group->ToString()));

    // Debug options: forcibly set the strategy of some instructions.
    if (option.force_strategy) {
      std::vector<int64_t> inst_indices = option.force_strategy_inst_indices;
      std::vector<std::string> stra_names = option.force_strategy_stra_names;
      CHECK_EQ(inst_indices.size(), stra_names.size());
      auto it = absl::c_find(inst_indices, strategy_group->node_idx);
      if (it != inst_indices.end()) {
        CHECK(!strategy_group->is_tuple);
        std::vector<std::pair<ShardingStrategy, InputShardings>> new_strategies;
        int64_t idx = it - inst_indices.begin();
        const auto& strategy_input_shardings =
            strategy_group->GetStrategyInputShardings();
        for (size_t iid = 0; iid < strategy_input_shardings.size(); ++iid) {
          const InputShardings& input_shardings = strategy_input_shardings[iid];
          const ShardingStrategy& strategy =
              strategy_group->GetStrategyForInputShardings(iid);
          if (input_shardings.name == stra_names[idx]) {
            new_strategies.push_back({strategy, input_shardings});
          }
        }
        strategy_group->ClearStrategies();
        for (const auto& [strategy, input_shardings] : new_strategies) {
          strategy_group->AddStrategy(strategy, input_shardings);
        }
      }
    }

    // When trying out multiple mesh shapes in the presence of user specified
    // sharding (as in
    // AutoShardingTest.AutoShardingKeepUserShardingInputOutput), there may be a
    // situation when we cannot generate any shardings for an instruction when
    // the mesh shape we're trying does not match with the mesh shape used in
    // user specified shardings. So we disable the check in that situation.
    if (!trying_multiple_mesh_shapes) {
      CHECK(strategy_group->is_tuple ||
            !strategy_group->GetStrategies().empty())
          << ins->ToString() << " does not have any valid strategies.";
    } else if (!(strategy_group->is_tuple ||
                 !strategy_group->GetStrategies().empty())) {
      return absl::Status(
          absl::StatusCode::kFailedPrecondition,
          "Could not generate any shardings for an instruction due "
          "to mismatched mesh shapes.");
    }
    // Checks the shape of resharding_costs is valid. It will check fail if the
    // shape is not as expected.
    // CheckReshardingCostsShape(strategies.get());
    CheckMemoryCosts(*strategy_group, ins->shape());
    strategy_map[ins] = std::move(strategy_group);
  }  // end of for loop

  return std::make_tuple(std::move(strategy_map), std::move(strategy_groups),
                         std::move(associative_dot_pairs));
}

// NOLINTEND

}  // namespace spmd
}  // namespace xla
