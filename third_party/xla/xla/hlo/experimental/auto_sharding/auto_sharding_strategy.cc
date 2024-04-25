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
#include <memory>
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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_wrapper.h"
#include "xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/sharding_propagation.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace spmd {

bool LeafVectorsAreConsistent(const std::vector<ShardingStrategy>& one,
                              const std::vector<ShardingStrategy>& two) {
  return one.size() == two.size();
}

// NOLINTBEGIN(readability/fn_size)
// TODO(zhuohan): Decompose this function into smaller pieces
absl::StatusOr<std::tuple<StrategyMap, StrategyGroups, AssociativeDotPairs>>
BuildStrategyAndCost(const HloInstructionSequence& sequence,
                     const HloModule* module,
                     const absl::flat_hash_map<const HloInstruction*, int64_t>&
                         instruction_execution_counts,
                     const InstructionDepthMap& depth_map,
                     const InstructionBatchDimMap& batch_dim_map,
                     const AliasMap& alias_map,
                     const ClusterEnvironment& cluster_env,
                     AutoShardingOption& option, const CallGraph& call_graph,
                     const HloCostAnalysis& hlo_cost_analysis,
                     bool trying_multiple_mesh_shapes) {
  const Array<int64_t>& device_mesh = cluster_env.device_mesh_;
  StrategyMap strategy_map;
  // This map stores all of the trimmed strategies due to user specified
  // sharding. The key is the instruction id, the value is the strategies. This
  // is useful when the operand is forced to use a user sharding, and the op
  // doesn't need to strictly follow it. We restore the trimmed strategies in
  // this situation.
  StableHashMap<int64_t, std::vector<ShardingStrategy>> pretrimmed_strategy_map;
  StrategyGroups strategy_groups;
  AssociativeDotPairs associative_dot_pairs;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Add penalty for replicated tensors
  double replicated_penalty = std::round(cluster_env.AllReduceCost(1, 0) +
                                         cluster_env.AllReduceCost(1, 1));

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
      only_allow_divisible = option.only_allow_divisible_input_output;
    } else {
      only_allow_divisible = option.only_allow_divisible_intermediate;
    }

    bool is_follow_necessary_for_correctness = false;
    switch (opcode) {
      case HloOpcode::kParameter: {
        auto it = while_body_args_to_input_tuple.find(ins);
        if (it != while_body_args_to_input_tuple.end()) {
          const HloInstruction* while_input_tuple = it->second;
          const StrategyGroup* while_input_tuple_strategy_group =
              strategy_map.at(while_input_tuple).get();

          VLOG(5) << "Following while input " << while_input_tuple->name();
          strategy_group = CreateTupleStrategyGroup(instruction_id);
          strategy_group->childs.reserve(ins->shape().tuple_shapes_size());
          // We use this following relationship to ensure that the input tuple
          // of the while loop, and the parameter of the body of that while
          // loop. Therefore, this followinf relationship is necessary for
          // correctness, and is not merely an optmization.
          is_follow_necessary_for_correctness = true;
          for (size_t i = 0; i < ins->shape().tuple_shapes_size(); ++i) {
            std::unique_ptr<StrategyGroup> child_strategies =
                MaybeFollowInsStrategyGroup(
                    while_input_tuple_strategy_group->childs[i].get(),
                    ins->shape().tuple_shapes().at(i), instruction_id,
                    /* have_memory_cost= */ true, strategy_groups, cluster_env,
                    pretrimmed_strategy_map);
            child_strategies->tuple_element_idx = i;
            strategy_group->childs.push_back(std::move(child_strategies));
          }
        } else {
          strategy_group =
              CreateAllStrategiesGroup(
                  ins, ins->shape(), instruction_id, strategy_groups,
                  cluster_env, strategy_map, option, replicated_penalty,
                  batch_dim_map, call_graph, only_allow_divisible,
                  option.allow_replicated_parameters,
                  /* create_partially_replicated_strategies */ true)
                  .value();
        }
        break;
      }
      case HloOpcode::kRngBitGenerator:
      case HloOpcode::kRng: {
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, batch_dim_map,
                call_graph, only_allow_divisible,
                option.allow_replicated_parameters,
                /* create_partially_replicated_strategies */ true)
                .value();
        break;
      }
      case HloOpcode::kConstant: {
        strategy_group = CreateLeafStrategyGroupWithoutInNodes(instruction_id,
                                                               strategy_groups);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategy_group, 0);
        break;
      }
      case HloOpcode::kScatter: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        // We follow the first operand (the array we're scattering into)
        auto src_strategy_group = strategy_map.at(ins->operand(0)).get();
        CHECK(!src_strategy_group->is_tuple);
        for (int64_t sid = 0; sid < src_strategy_group->strategies.size();
             ++sid) {
          HloSharding output_spec =
              src_strategy_group->strategies[sid].output_sharding;
          std::string name = ToStringSimple(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

          std::vector<std::optional<HloSharding>> input_shardings_optional(
              {output_spec, std::nullopt, std::nullopt});
          std::pair<ReshardingCosts, ReshardingCosts> resharding_costs =
              GenerateReshardingCostsAndMissingShardingsForAllOperands(
                  ins, output_spec, strategy_map, cluster_env, call_graph,
                  input_shardings_optional);

          for (const auto& sharding_optional : input_shardings_optional) {
            CHECK(sharding_optional.has_value());
          }

          strategy_group->strategies.push_back(ShardingStrategy(
              {name, output_spec, compute_cost, communication_cost, memory_cost,
               std::move(resharding_costs.first),
               std::move(resharding_costs.second), input_shardings_optional}));
        }
        break;
      }
      case HloOpcode::kGather: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        // Follows the strategy of start_indices (operand 1)
        const HloInstruction* indices = ins->operand(1);
        const Shape& shape = ins->shape();
        const StrategyGroup* src_strategy_group =
            strategy_map.at(indices).get();
        CHECK(!src_strategy_group->is_tuple);
        strategy_group->following = src_strategy_group;
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
            std::pair<ReshardingCosts, ReshardingCosts> resharding_costs =
                GenerateReshardingCostsAndMissingShardingsForAllOperands(
                    ins, output_spec, strategy_map, cluster_env, call_graph,
                    input_shardings_optional);

            strategy_group->strategies.push_back(ShardingStrategy(
                {name, output_spec, compute_cost, communication_cost,
                 memory_cost, std::move(resharding_costs.first),
                 std::move(resharding_costs.second),
                 input_shardings_optional}));
          }
        }
        AddReplicatedStrategy(
            ins, ins->shape(), cluster_env, strategy_map, strategy_group, 0,
            /* operands_to_consider_all_strategies_for */ {0});
        break;
      }
      case HloOpcode::kBroadcast: {
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, batch_dim_map,
                call_graph, only_allow_divisible,
                /* create_replicated_strategies */ true,
                /* create_partially_replicated_strategies */ true)
                .value();
        break;
      }
      case HloOpcode::kReshape: {
        strategy_group = CreateReshapeStrategies(
            instruction_id, ins, strategy_map, cluster_env,
            only_allow_divisible, replicated_penalty, batch_dim_map, option,
            strategy_groups, call_graph);
        break;
      }
      case HloOpcode::kTranspose:
      case HloOpcode::kReverse: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);

        const HloInstruction* operand = ins->operand(0);

        // Create follow strategies
        const StrategyGroup* src_strategy_group =
            strategy_map.at(operand).get();
        CHECK(!src_strategy_group->is_tuple);
        strategy_group->following = src_strategy_group;

        for (int64_t sid = 0; sid < src_strategy_group->strategies.size();
             ++sid) {
          HloSharding output_spec = Undefined();
          auto input_spec = src_strategy_group->strategies[sid].output_sharding;
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
          std::vector<double> communication_resharding_costs =
              CommunicationReshardingCostVector(src_strategy_group,
                                                operand->shape(), input_spec,
                                                cluster_env);
          std::vector<double> memory_resharding_costs =
              MemoryReshardingCostVector(src_strategy_group, operand->shape(),
                                         input_spec, cluster_env);
          strategy_group->strategies.push_back(
              ShardingStrategy({name,
                                output_spec,
                                compute_cost,
                                communication_cost,
                                memory_cost,
                                {communication_resharding_costs},
                                {memory_resharding_costs},
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

        for (int64_t sid = 0; sid < src_strategy_group->strategies.size();
             ++sid) {
          std::optional<HloSharding> output_spec;
          HloSharding input_spec =
              src_strategy_group->strategies[sid].output_sharding;

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
                output_spec = PropagateDimwiseShardingSlice(
                    input_spec, operand->shape(), ins->shape(),
                    cluster_env.device_mesh_1d_);
              } else if (is_1d_sharding) {
                CHECK_EQ(input_spec.TotalNumTiles(),
                         cluster_env.original_device_mesh_1d_.num_elements());
                output_spec = PropagateDimwiseShardingSlice(
                    input_spec, operand->shape(), ins->shape(),
                    cluster_env.original_device_mesh_1d_);
              } else {
                output_spec = PropagateDimwiseShardingSlice(
                    input_spec, operand->shape(), ins->shape(),
                    cluster_env.device_mesh_);
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
          std::pair<ReshardingCosts, ReshardingCosts> resharding_costs =
              GenerateReshardingCostsAndMissingShardingsForAllOperands(
                  ins, *output_spec, strategy_map, cluster_env, call_graph,
                  input_shardings);

          strategy_group->strategies.push_back(
              ShardingStrategy({name,
                                *output_spec,
                                compute_cost,
                                communication_cost,
                                memory_cost,
                                std::move(resharding_costs.first),
                                std::move(resharding_costs.second),
                                {input_spec}}));
        }

        if (strategy_group->strategies.empty()) {
          strategy_group->following = nullptr;
          AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                strategy_group, 0);
        }
        break;
      }
      case HloOpcode::kOptimizationBarrier: {
        auto operand_strategies = strategy_map.at(ins->operand(0)).get();
        strategy_group = MaybeFollowInsStrategyGroup(
            operand_strategies, ins->shape(), instruction_id,
            /* have_memory_cost */ true, strategy_groups, cluster_env,
            pretrimmed_strategy_map);
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
              only_allow_divisible, replicated_penalty, batch_dim_map, option,
              strategy_groups, call_graph);
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
        auto strategies_status = FollowReduceStrategy(
            ins, ins->shape(), ins->operand(0), ins->operand(1), instruction_id,
            strategy_map, strategy_groups, cluster_env,
            option.allow_mixed_mesh_shape, !trying_multiple_mesh_shapes);
        if (strategies_status.ok()) {
          strategy_group = std::move(strategies_status.value());
        } else {
          return strategies_status.status();
        }
        break;
      }
      case HloOpcode::kDot: {
        TF_RETURN_IF_ERROR(HandleDot(strategy_group, strategy_groups,
                                     strategy_map, ins, instruction_id,
                                     sequence, hlo_cost_analysis, cluster_env,
                                     batch_dim_map, option, call_graph));

        if (option.allow_recompute_heavy_op) {
          AddReplicatedStrategy(
              ins, ins->shape(), cluster_env, strategy_map, strategy_group,
              GetDotConvReplicationPenalty(ins, instruction_id, /* window */ 10,
                                           sequence, hlo_cost_analysis));
        }
        break;
      }
      case HloOpcode::kConvolution: {
        TF_RETURN_IF_ERROR(HandleConv(strategy_group, strategy_groups,
                                      strategy_map, ins, instruction_id,
                                      sequence, hlo_cost_analysis, cluster_env,
                                      batch_dim_map, option, call_graph));
        if (option.allow_recompute_heavy_op) {
          AddReplicatedStrategy(
              ins, ins->shape(), cluster_env, strategy_map, strategy_group,
              GetDotConvReplicationPenalty(ins, instruction_id, /* window */ 10,
                                           sequence, hlo_cost_analysis));
        }
        break;
      }
      case HloOpcode::kRngGetAndUpdateState: {
        strategy_group = CreateLeafStrategyGroupWithoutInNodes(instruction_id,
                                                               strategy_groups);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategy_group, 0);
        break;
      }
      case HloOpcode::kIota: {
        // For an unknown reason, we do not generate partially replicated
        // strategies for iota ops. This can be changed if we find that our
        // search isn't exhaustive enough for certain ops.
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, batch_dim_map,
                call_graph, only_allow_divisible,
                /* create_replicated_strategies */ true,
                /* create_partially_replicated_strategies */ false)
                .value();
        break;
      }
      case HloOpcode::kTuple: {
        strategy_group = CreateTupleStrategyGroup(instruction_id);
        strategy_group->childs.reserve(ins->operand_count());
        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          const StrategyGroup* src_strategy_group =
              strategy_map.at(operand).get();
          auto child_strategies = MaybeFollowInsStrategyGroup(
              src_strategy_group, operand->shape(), instruction_id,
              /* have_memory_cost= */ true, strategy_groups, cluster_env,
              pretrimmed_strategy_map);
          child_strategies->tuple_element_idx = i;
          strategy_group->childs.push_back(std::move(child_strategies));
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
        const StrategyGroup* src_strategy_group =
            strategy_map.at(operand).get();
        CHECK(src_strategy_group->is_tuple);
        strategy_group = MaybeFollowInsStrategyGroup(
            src_strategy_group->childs[ins->tuple_index()].get(), ins->shape(),
            instruction_id,
            /* have_memory_cost= */ true, strategy_groups, cluster_env,
            pretrimmed_strategy_map);
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
                  strategy_group->childs.reserve(
                      ins->shape().tuple_shapes_size());
                  for (size_t i = 0; i < ins->shape().tuple_shapes_size();
                       ++i) {
                    std::unique_ptr<StrategyGroup> child_strategies =
                        CreateLeafStrategyGroup(instruction_id, ins,
                                                strategy_map, strategy_groups);
                    AddReplicatedStrategy(ins, ins->shape().tuple_shapes(i),
                                          cluster_env, strategy_map,
                                          child_strategies, replicated_penalty);
                    strategy_group->childs.push_back(
                        std::move(child_strategies));
                  }
                } else {
                  strategy_group = CreateLeafStrategyGroup(
                      instruction_id, ins, strategy_map, strategy_groups);
                  AddReplicatedStrategy(ins, ins->shape(), cluster_env,
                                        strategy_map, strategy_group,
                                        replicated_penalty);
                }
              } else {
                strategy_group =
                    CreateAllStrategiesGroup(
                        ins, ins->shape(), instruction_id, strategy_groups,
                        cluster_env, strategy_map, option, replicated_penalty,
                        batch_dim_map, call_graph, only_allow_divisible,
                        /* create_replicated_strategies */ true,
                        /* create_partially_replicated_strategies */ true)
                        .value();
              }
            };

        if (IsCustomCallMarker(ins)) {
          const HloInstruction* operand = ins->operand(0);
          const StrategyGroup* src_strategy_group =
              strategy_map.at(operand).get();
          CHECK(src_strategy_group->is_tuple);
          strategy_group = MaybeFollowInsStrategyGroup(
              src_strategy_group, ins->shape(), instruction_id,
              /* have_memory_cost= */ true, strategy_groups, cluster_env,
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
            const StrategyGroup* src_strategy_group =
                strategy_map.at(operand).get();
            strategy_group = MaybeFollowInsStrategyGroup(
                src_strategy_group, ins->shape(), instruction_id,
                /* have_memory_cost= */ true, strategy_groups, cluster_env,
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
        strategy_group = CreateTupleStrategyGroup(instruction_id);
        strategy_group->childs.reserve(ins->shape().tuple_shapes_size());
        const StrategyGroup* src_strategy_group =
            strategy_map.at(ins->operand(0)).get();
        for (size_t i = 0; i < ins->shape().tuple_shapes_size(); ++i) {
          auto child_strategies = MaybeFollowInsStrategyGroup(
              src_strategy_group->childs[i].get(),
              ins->shape().tuple_shapes().at(i), instruction_id,
              /* have_memory_cost= */ true, strategy_groups, cluster_env,
              pretrimmed_strategy_map);
          child_strategies->tuple_element_idx = i;
          strategy_group->childs.push_back(std::move(child_strategies));
        }

        break;
      }
      case HloOpcode::kConditional:
      case HloOpcode::kInfeed:
      case HloOpcode::kSort: {
        strategy_group =
            CreateAllStrategiesGroup(
                ins, ins->shape(), instruction_id, strategy_groups, cluster_env,
                strategy_map, option, replicated_penalty, batch_dim_map,
                call_graph, only_allow_divisible,
                /* create_replicated_strategies */ true,
                /* create_partially_replicated_strategies */ true)
                .value();
        break;
      }
      case HloOpcode::kOutfeed: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        GenerateOutfeedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                                strategy_group, replicated_penalty);
        break;
      }
      case HloOpcode::kRecv:
      case HloOpcode::kRecvDone:
      case HloOpcode::kSend: {
        strategy_group = CreateTupleStrategyGroup(instruction_id);
        strategy_group->childs.reserve(ins->shape().tuple_shapes_size());
        for (size_t i = 0; i < ins->shape().tuple_shapes_size(); ++i) {
          std::unique_ptr<StrategyGroup> child_strategies =
              CreateLeafStrategyGroup(instruction_id, ins, strategy_map,
                                      strategy_groups);
          AddReplicatedStrategy(ins, ins->shape().tuple_shapes(i), cluster_env,
                                strategy_map, child_strategies, 0);
          child_strategies->tuple_element_idx = i;
          strategy_group->childs.push_back(std::move(child_strategies));
        }
        break;
      }
      case HloOpcode::kSendDone: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategy_group, 0);
        break;
      }
      case HloOpcode::kAfterAll: {
        strategy_group = CreateLeafStrategyGroup(instruction_id, ins,
                                                 strategy_map, strategy_groups);
        AddReplicatedStrategy(ins, ins->shape(), cluster_env, strategy_map,
                              strategy_group, replicated_penalty);
        break;
      }
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->ToString();
    }
    RemoveDuplicatedStrategy(strategy_group);
    if (ins->has_sharding() && ins->opcode() != HloOpcode::kOutfeed) {
      // Finds the sharding strategy that aligns with the given sharding spec
      // Do not merge nodes if this one instruction has annotations.
      TrimOrGenerateStrategiesBasedOnExistingSharding(
          ins->shape(), strategy_group.get(), strategy_map, instructions,
          ins->sharding(), cluster_env, pretrimmed_strategy_map, call_graph,
          option.nd_sharding_iteratively_strict_search_space);
    }
    if (!strategy_group->is_tuple && strategy_group->following) {
      if (!LeafVectorsAreConsistent(strategy_group->strategies,
                                    strategy_group->following->strategies)) {
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
      for (size_t i = 0; i < strategy_group->childs.size(); i++) {
        if (strategy_group->childs.at(i)->following &&
            !LeafVectorsAreConsistent(
                strategy_group->childs.at(i)->strategies,
                strategy_group->childs.at(i)->following->strategies)) {
          CHECK(!is_follow_necessary_for_correctness)
              << "Reverting a following decision that is necessary for "
                 "correctness. Please report this as a bug.";
          strategy_group->childs.at(i)->following = nullptr;
        }
      }
    }
    RemoveInvalidShardingsWithShapes(
        ins->shape(), strategy_group.get(),
        /* instruction_has_user_sharding */ ins->has_sharding());

    if (instruction_execution_counts.contains(ins)) {
      ScaleCostsWithExecutionCounts(strategy_group.get(),
                                    instruction_execution_counts.at(ins));
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
        std::vector<ShardingStrategy> new_strategies;
        int64_t idx = it - inst_indices.begin();
        for (const auto& stra : strategy_group->strategies) {
          if (stra.name == stra_names[idx]) {
            new_strategies.push_back(stra);
          }
        }
        strategy_group->strategies = std::move(new_strategies);
      }
    }

    // When trying out multiple mesh shapes in the presence of user specified
    // sharding (as in
    // AutoShardingTest.AutoShardingKeepUserShardingInputOutput), there may be a
    // situation when we cannot generate any shardings for an instruction when
    // the mesh shape we're trying does not match with the mesh shape used in
    // user specified shardings. So we disable the check in that situation.
    if (!trying_multiple_mesh_shapes) {
      CHECK(strategy_group->is_tuple || !strategy_group->strategies.empty())
          << ins->ToString() << " does not have any valid strategies.";
    } else if (!(strategy_group->is_tuple ||
                 !strategy_group->strategies.empty())) {
      return Status(absl::StatusCode::kFailedPrecondition,
                    "Could not generate any shardings for an instruction due "
                    "to mismatched mesh shapes.");
    }
    // Checks the shape of resharding_costs is valid. It will check fail if the
    // shape is not as expected.
    // CheckReshardingCostsShape(strategies.get());
    CheckMemoryCosts(strategy_group.get(), ins->shape());
    strategy_map[ins] = std::move(strategy_group);
  }  // end of for loop

  // If gradient accumulation is used, adjust the cost of all-reduce for
  // gradient synchronization.
  if (option.grad_acc_num_micro_batches > 1) {
    // find gradient-computation instructions
    std::vector<const HloInstruction*> grad_insts =
        GetGradientComputationInstructions(instructions);
    for (const HloInstruction* inst : grad_insts) {
      StrategyGroup* stra_vector = strategy_map[inst].get();
      CHECK(!stra_vector->is_tuple);

      for (auto& stra : stra_vector->strategies) {
        if (absl::StrContains(stra.name, "allreduce")) {
          stra.communication_cost /= option.grad_acc_num_micro_batches;
        }
      }
    }
  }

  return std::make_tuple(std::move(strategy_map), std::move(strategy_groups),
                         std::move(associative_dot_pairs));
}

// NOLINTEND

}  // namespace spmd
}  // namespace xla
