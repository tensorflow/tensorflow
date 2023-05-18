/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/while_loop_all_reduce_code_motion.h"

#include <memory>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_replication_analysis.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

namespace {

struct AccumulationContext {
  HloInstruction* accumulation_instruction;
  HloInstruction* accumulation_buffer;
  int64_t param_tuple_index;
  std::optional<HloInstruction*> dynamic_slice;
  std::optional<HloInstruction*> dynamic_update_slice;
};

// Describes whether an all-reduce instruction can be sinked from a while body
// computation and all the accumulation uses of the all-reduce's result in the
// while body if movable.
struct MovableAllReduceContext {
  bool is_movable;
  // If movable, `accumulation_contexts` contains one accumulation
  // context for each accumulation in the while body that uses the all-reduce's
  // result. Otherwise, this field is undefined.
  std::vector<AccumulationContext> accumulation_contexts;
};

bool IsZero(const HloInstruction* hlo) {
  if (hlo->IsConstant() && hlo->shape().rank() == 0 &&
      hlo->literal().IsZero({})) {
    return true;
  }
  if (hlo->opcode() == HloOpcode::kBroadcast) {
    return IsZero(hlo->operand(0));
  }
  return false;
}

bool IsValueReplicatedWithinEachAllReduceGroup(
    const HloInstruction& instruction, const ShapeIndex& index,
    CollectiveOpGroupMode all_reduce_group_mode,
    absl::Span<const ReplicaGroup> replica_groups, int num_replicas,
    int num_partitions,
    const std::unique_ptr<HloReplicationAnalysis>&
        cross_replica_replication_analysis,
    const std::unique_ptr<HloReplicationAnalysis>&
        cross_partition_replication_analysis) {
  VLOG(5) << "IsValueReplicatedWithinEachAllReduceGroup,"
          << " all_reduce_group_mode: "
          << CollectiveOpGroupModeToString(all_reduce_group_mode);
  switch (all_reduce_group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      return cross_replica_replication_analysis == nullptr ||
             cross_replica_replication_analysis->HloInstructionIsReplicatedAt(
                 &instruction, index, replica_groups);
    }
    case CollectiveOpGroupMode::kCrossPartition: {
      return cross_partition_replication_analysis == nullptr ||
             cross_partition_replication_analysis->HloInstructionIsReplicatedAt(
                 &instruction, index, replica_groups);
    }
    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      return (cross_replica_replication_analysis == nullptr ||
              cross_replica_replication_analysis->HloInstructionIsReplicatedAt(
                  &instruction, index, replica_groups)) &&
             (cross_partition_replication_analysis == nullptr ||
              cross_partition_replication_analysis
                  ->HloInstructionIsReplicatedAt(&instruction, index));
    }
    case CollectiveOpGroupMode::kFlattenedID: {
      if (num_replicas == 1) {
        return cross_partition_replication_analysis == nullptr ||
               cross_partition_replication_analysis
                   ->HloInstructionIsReplicatedAt(&instruction, index,
                                                  replica_groups);
      }
      if (num_partitions == 1) {
        return cross_replica_replication_analysis == nullptr ||
               cross_replica_replication_analysis->HloInstructionIsReplicatedAt(
                   &instruction, index, replica_groups);
      }
      return (cross_replica_replication_analysis == nullptr ||
              cross_replica_replication_analysis->HloInstructionIsReplicatedAt(
                  &instruction, index)) &&
             (cross_partition_replication_analysis == nullptr ||
              cross_partition_replication_analysis
                  ->HloInstructionIsReplicatedAt(&instruction, index));
    }
  }
}

// If this instruction is effectively a scalar, returns the scalar value, else
// returns nullptr.
HloInstruction* GetEffectiveScalar(HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kBroadcast) {
    return nullptr;
  }
  HloInstruction* operand = instruction->mutable_operand(0);
  if (!ShapeUtil::IsScalar(operand->shape())) {
    return nullptr;
  }
  return operand;
}

// Checks if an all-reduce instruction is eligible for sinking and finds all of
// the all-reduce's accumulation uses inside the while body if eligible.
// An all-reduce is movable if all following conditions hold. This function
// checks each condition.
//   1) The all-reduce's reduction computation is summation.
//   2) All users of the all-reduce are additions, which we refer to as
//      accumulations. We refer the other operand and the output of the addition
//      as accumulation buffers.
//   3) Each accumulation buffer is a parameter to the loop body and also an
//      output of the loop at the same tuple index.
//   4) A limited set of HLOs can be applied to the all-reduce output before
//      accumulation, as well as the accumulation buffers before and after the
//      accumuation. These HLOs include
//      a. kConvert: the sinked all-reduce will have the same element type.
//      b. HLOs that change the shape of the all-reduce output and / or the
//         accumulation buffer. HLOs are supported as long as all all-reduce
//         participants have the same element-wise mapping between all-reduce
//         output and the accumulation buffer. Note that it is fine if at
//         different iterations, different all-reduce elements are mapped to
//         the same accumulation buffer element. These include kBitcast,
//         kReshape, and kTranspose.
//         We also support dynamic-slice and dynamic-update-slice pairs on the
//         accumulation buffer. We need to ensure the slice offset is the same
//         across all cores. It is possible but difficult to support the
//.        general case, so we use pattern matching to support the specific
//.        cases of interest.
//      c. Dynamically discarding the all-reduce result, i.e., kSelect between
//         all-reduce result and 0. The predicate to kSelect must have the same
//         value on all all-reduce cores.
//
// Extension for reduce-scatter: For reduce-scatter, we will allow a subset of
// the conditions above, given that we need to expand the size of the
// accumulation buffers in the loop and then execute the reduce-scatter outside
// the loop. This means we will only allow cases where we can do this
// expansion:
//  1. Do not allow dynamic-slice or dynamic-update-slice (not expected with
//     reduce-scatter)
//  2. Do not allow any forwarding instructions like convert (TBD, as their
//     shape can be changed).
//  3. Do not allow forwarding instructions like bitcast, transpose, and
//     reshape (as changing their shape may not be trivial)
//  3. Allow select only when the predicate is effectively scalar. This is
//     because we will be changing the shape of this select to match the
//     expanded accumulation buffer, and with scalar predicate its trivial to
//     change the predicate shape by broadcasting the scalar predicate to the
//     new shape. With non-scalar predicate, HLO replication analysis only has
//     HLO level info about whether the entire value is replicated or not, and
//     that may not be sufficient to change the predicate shape to a new shape.
MovableAllReduceContext IsAllReduceMovable(
    HloAllReduceInstructionBase* all_reduce, HloComputation* while_body,
    const std::unique_ptr<HloReplicationAnalysis>&
        cross_replica_replication_analysis,
    const std::unique_ptr<HloReplicationAnalysis>&
        cross_partition_replication_analysis) {
  VLOG(4) << "IsAllReduceMovable: " << all_reduce->ToString();
  std::optional<ReductionKind> reduction_type =
      MatchReductionComputation(all_reduce->to_apply());
  const bool all_reduce_is_summation =
      reduction_type.has_value() && *reduction_type == ReductionKind::SUM;

  // We only support numerical types.
  const absl::InlinedVector<PrimitiveType, 12> kSupportedTypes{
      BF16, F16, F32, F64, S8, S16, S32, S64, U8, U16, U32, U64};

  if (!absl::c_linear_search(kSupportedTypes,
                             all_reduce->shape().element_type()) ||
      !all_reduce_is_summation) {
    return MovableAllReduceContext{/*is_movable=*/false,
                                   /*accumulation_contexts=*/{}};
  }

  CollectiveOpGroupMode all_reduce_group_mode =
      GetCollectiveOpGroupMode(all_reduce->channel_id().has_value(),
                               all_reduce->use_global_device_ids())
          .value();

  auto is_value_replicated_within_replica_group =
      [&cross_replica_replication_analysis,
       &cross_partition_replication_analysis, &all_reduce_group_mode,
       all_reduce](const HloInstruction& instruction,
                   const ShapeIndex& index) -> bool {
    bool is_replicated = IsValueReplicatedWithinEachAllReduceGroup(
        instruction, index, all_reduce_group_mode, all_reduce->replica_groups(),
        all_reduce->GetModule()->config().replica_count(),
        all_reduce->GetModule()->config().num_partitions(),
        cross_replica_replication_analysis,
        cross_partition_replication_analysis);
    VLOG(5) << "instruction: " << instruction.name()
            << " is_replicate: " << is_replicated;
    return is_replicated;
  };
  struct BufferTupleIndex {
    bool unsupported_operation{false};
    std::optional<int64_t> tuple_index;
    bool returned_from_computation{false};
    std::optional<HloInstruction*> dynamic_slice;
    std::optional<HloInstruction*> dynamic_update_slice;
  };

  const bool is_reduce_scatter =
      all_reduce->opcode() == HloOpcode::kReduceScatter;

  // If the instruction is a buffer forwarded from a tuple element of the
  // computation's parameter, returns the indices of the buffer in the parameter
  // tuple. The returned_from_computation field in the result is unused.
  auto get_origin_tuple_index =
      [is_reduce_scatter](HloInstruction* instruction) -> BufferTupleIndex {
    VLOG(4) << "get_origin_tuple_index called on " << instruction->ToString();
    // The returned_from_computation is never touched in this function.
    BufferTupleIndex result;
    while (!result.unsupported_operation) {
      switch (instruction->opcode()) {
        default: {
          VLOG(4) << "get_origin_tuple_index, instruction: ("
                  << instruction->ToString()
                  << ") is an unsupported operation on accumulation buffer.";
          result.unsupported_operation = true;
          break;
        }
        case HloOpcode::kBitcast:
        case HloOpcode::kConvert:
        case HloOpcode::kReshape:
        case HloOpcode::kTranspose:
          if (is_reduce_scatter) {
            VLOG(4) << "get_origin_tuple_index, instruction: ("
                    << instruction->ToString()
                    << ") is an unsupported operation on accumulation buffer.";
            result.unsupported_operation = true;
          } else {
            instruction = instruction->mutable_operand(0);
          }
          break;
        case HloOpcode::kGetTupleElement: {
          if (result.tuple_index.has_value()) {
            // Note that we don't support nested tuples as of now.
            result.unsupported_operation = true;
          } else {
            result.tuple_index =
                Cast<HloGetTupleElementInstruction>(instruction)->tuple_index();
            instruction = instruction->mutable_operand(0);
          }
          break;
        }
        case HloOpcode::kDynamicSlice: {
          if (is_reduce_scatter) {
            VLOG(4) << "get_origin_tuple_index, instruction: ("
                    << instruction->ToString()
                    << ") is an unsupported operation on accumulation buffer.";
            result.unsupported_operation = true;
          } else if (result.dynamic_slice.has_value()) {
            VLOG(4) << "get_origin_tuple_index, instruction: ("
                    << instruction->ToString()
                    << "), we do not yet support more than 1 dynamic-slices on"
                    << " the accumulation buffer.";
            result.unsupported_operation = true;
          } else {
            result.dynamic_slice = instruction;
            instruction = instruction->mutable_operand(0);
          }
          break;
        }
        case HloOpcode::kParameter: {
          int parameter_number =
              Cast<HloParameterInstruction>(instruction)->parameter_number();
          CHECK_EQ(parameter_number, 0);
          break;
        }
      }
      if (instruction->opcode() == HloOpcode::kParameter) {
        break;
      }
    }
    return result;
  };

  // If the instruction's result is returned from its parent computation with
  // only forwarding operations, returns the index of the result buffer in the
  // output parameter tuple.
  auto get_output_tuple_index =
      [is_reduce_scatter](HloInstruction* instruction,
                          HloComputation* while_body) -> BufferTupleIndex {
    VLOG(4) << "get_output_tuple_index called on " << instruction->ToString();
    BufferTupleIndex result;
    std::stack<HloInstruction*> to_visit;
    to_visit.push(instruction);
    while (!to_visit.empty() && !result.unsupported_operation) {
      HloInstruction* instruction = to_visit.top();
      to_visit.pop();
      for (HloInstruction* user : instruction->users()) {
        switch (user->opcode()) {
          case HloOpcode::kBitcast:
          case HloOpcode::kConvert:
          case HloOpcode::kReshape:
          case HloOpcode::kGetTupleElement:
          case HloOpcode::kTranspose:
          case HloOpcode::kSlice: {
            if (is_reduce_scatter) {
              result.unsupported_operation = true;
            } else {
              to_visit.push(user);
            }
            break;
          }
          case HloOpcode::kDynamicUpdateSlice: {
            if (result.dynamic_update_slice.has_value() || is_reduce_scatter) {
              result.unsupported_operation = true;
            } else {
              result.dynamic_update_slice = user;
              to_visit.push(user);
            }
            break;
          }
          case HloOpcode::kTuple: {
            if (result.tuple_index.has_value()) {
              // Note that we don't support nested tuples as of now.
              result.unsupported_operation = true;
            } else {
              result.tuple_index = user->operand_index(instruction);
              if (while_body->root_instruction() == user) {
                if (result.returned_from_computation) {
                  result.unsupported_operation = true;
                }
                result.returned_from_computation = true;
              } else {
                to_visit.push(user);
              }
            }
            break;
          }
          default: {
            VLOG(4) << "get_output_tuple_index, instruction: ("
                    << instruction->ToString()
                    << ") is an unsupported operation on accumulation buffer.";
            result.unsupported_operation = true;
          }
        }
        if (result.unsupported_operation) {
          break;
        }
      }
    }
    return result;
  };

  // Checks whether any buffer in the list of accumulation contexts is used in
  // the parent computation except for forwarding uses.
  auto is_buffer_used =
      [&is_value_replicated_within_replica_group, is_reduce_scatter](
          absl::Span<const AccumulationContext> accumulation_contexts,
          HloComputation* while_body_computation) -> bool {
    // While loop body computation should have a single parameter.
    CHECK_EQ(while_body_computation->num_parameters(), 1);
    HloInstruction* parameter_instruction =
        while_body_computation->parameter_instruction(0);
    for (const auto& accumulation : accumulation_contexts) {
      HloInstruction* accumulation_instruction =
          accumulation.accumulation_instruction;
      int64_t tuple_index = accumulation.param_tuple_index;
      std::stack<HloInstruction*> to_visit;
      // TODO(b/176437845): simplify the logic below by using
      // TuplePointsToAnalysis.

      // Iterate over all users of the while body parameter and find all
      // instructions that use the accumulation buffer, as specified by
      // tuple_index.
      // This logic could be simplied by using TuplePointsToAnalysis, which
      // we leave to a future CL (see TODO above).
      for (HloInstruction* user : parameter_instruction->users()) {
        if (auto* gte = DynCast<HloGetTupleElementInstruction>(user)) {
          if (gte->tuple_index() == tuple_index) {
            to_visit.push(user);
          }
        } else {
          return true;
        }
      }

      while (!to_visit.empty()) {
        HloInstruction* instruction = to_visit.top();
        to_visit.pop();
        for (HloInstruction* user : instruction->users()) {
          VLOG(5) << "is_buffer_used, user: " << user->name();
          switch (user->opcode()) {
            case HloOpcode::kBitcast:
            case HloOpcode::kConvert:
            case HloOpcode::kReshape:
            case HloOpcode::kTranspose:
              if (is_reduce_scatter) {
                VLOG(4) << "buffer is used by " << user->ToString()
                        << ", preventing the motion of reduce-scatter.";
                return true;
              }
              to_visit.push(user);
              break;
            case HloOpcode::kSelect: {
              if (((user->operand_index(instruction) == 1 &&
                    IsZero(user->operand(2))) ||
                   (user->operand_index(instruction) == 2 &&
                    IsZero(user->operand(1)))) &&
                  is_value_replicated_within_replica_group(*(user->operand(0)),
                                                           {})) {
                to_visit.push(user);
              } else {
                return true;
              }
              break;
            }
            case HloOpcode::kAdd: {
              if (user != accumulation_instruction) {
                return true;
              }
              break;
            }
            case HloOpcode::kDynamicSlice: {
              if (!accumulation.dynamic_slice.has_value() ||
                  user != *accumulation.dynamic_slice) {
                return true;
              }
              break;
            }
            case HloOpcode::kDynamicUpdateSlice: {
              if (!accumulation.dynamic_update_slice.has_value() ||
                  user != *accumulation.dynamic_update_slice) {
                return true;
              }
              break;
            }
            default: {
              VLOG(4) << "buffer is used by " << user->ToString()
                      << ", preventing the motion of all-reduce.";
              return true;
            }
          }
        }
      }
    }
    return false;
  };

  auto dus_matches_ds_offsets =
      [](const HloInstruction& dynamic_slice,
         const HloInstruction& dynamic_update_slice) -> bool {
    if (dynamic_slice.operand_count() + 1 !=
        dynamic_update_slice.operand_count()) {
      return false;
    }
    for (int i = 1; i < dynamic_slice.operand_count(); ++i) {
      if (dynamic_slice.operand(i) != dynamic_update_slice.operand(i + 1)) {
        return false;
      }
    }
    return true;
  };

  auto dus_indices_are_replicated =
      [&is_value_replicated_within_replica_group](
          const HloInstruction& dynamic_update_slice) -> bool {
    for (int i = 2; i < dynamic_update_slice.operand_count(); ++i) {
      if (!is_value_replicated_within_replica_group(
              *dynamic_update_slice.operand(i), {})) {
        return false;
      }
    }
    return true;
  };

  // Finds all accumulation contexts of the given all-reduce instruction
  // if it is movable.
  std::vector<AccumulationContext> accumulation_contexts;
  // DFS starting from the all-reduce instruction and stops at the first
  // non-trival uses of the all-reduce result or finds all accmululations
  // of the all-reduce result.
  std::stack<HloInstruction*> to_visit;
  // By default movable unless we find that it's not.
  bool is_all_reduce_movable = true;
  to_visit.push(all_reduce);

  while (!to_visit.empty() && is_all_reduce_movable) {
    HloInstruction* instruction = to_visit.top();
    to_visit.pop();
    for (HloInstruction* user : instruction->users()) {
      switch (user->opcode()) {
        case HloOpcode::kConvert:
          to_visit.push(user);
          break;

        case HloOpcode::kBitcast:
        case HloOpcode::kReshape:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kTranspose:
        case HloOpcode::kSlice: {
          if (is_reduce_scatter) {
            is_all_reduce_movable = false;
          } else {
            to_visit.push(user);
          }
          break;
        }
        case HloOpcode::kSelect: {
          bool is_select_ok = [&]() {
            bool operand_1_match = user->operand_index(instruction) == 1 &&
                                   IsZero(user->operand(2));
            bool operand_2_match = user->operand_index(instruction) == 2 &&
                                   IsZero(user->operand(1));
            if (!operand_1_match && !operand_2_match) {
              return false;
            }

            if (!is_reduce_scatter) {
              return true;
            }
            // For reduce-scatter, check that the predicate is effectively
            // scalar.
            HloInstruction* predicate = user->mutable_operand(0);
            return GetEffectiveScalar(predicate) != nullptr;
          }();
          if (is_select_ok) {
            to_visit.push(user);
          } else {
            is_all_reduce_movable = false;
          }
          break;
        }
        case HloOpcode::kAdd: {
          int64_t buffer_index = 1 - user->operand_index(instruction);
          HloInstruction* accumulation_buffer =
              user->mutable_operand(buffer_index);

          auto origin_buffer_tuple_index =
              get_origin_tuple_index(accumulation_buffer);
          if (origin_buffer_tuple_index.unsupported_operation) {
            is_all_reduce_movable = false;
            break;
          }

          // TODO(b/276505779): We also need to check that the accumulators are
          // not referenced in the loop condition.
          auto output_buffer_tuple_index =
              get_output_tuple_index(user, while_body);
          if (!output_buffer_tuple_index.unsupported_operation &&
              output_buffer_tuple_index.returned_from_computation &&
              origin_buffer_tuple_index.tuple_index.has_value() &&
              output_buffer_tuple_index.tuple_index.has_value() &&
              origin_buffer_tuple_index.tuple_index ==
                  output_buffer_tuple_index.tuple_index &&
              (origin_buffer_tuple_index.dynamic_slice.has_value() ==
               output_buffer_tuple_index.dynamic_update_slice.has_value()) &&
              (!origin_buffer_tuple_index.dynamic_slice.has_value() ||
               (dus_matches_ds_offsets(
                    **origin_buffer_tuple_index.dynamic_slice,
                    **output_buffer_tuple_index.dynamic_update_slice) &&
                dus_indices_are_replicated(
                    **output_buffer_tuple_index.dynamic_update_slice)))) {
            accumulation_contexts.push_back(AccumulationContext{
                user, accumulation_buffer,
                *output_buffer_tuple_index.tuple_index,
                origin_buffer_tuple_index.dynamic_slice,
                output_buffer_tuple_index.dynamic_update_slice});
          } else {
            is_all_reduce_movable = false;
          }
          break;
        }
        default: {
          VLOG(4) << "get_accumulation_contexts, all-reduce result is used "
                  << " by " << user->ToString() << ", not movable.";
          is_all_reduce_movable = false;
        }
      }
    }
  }
  if (is_buffer_used(accumulation_contexts, while_body)) {
    is_all_reduce_movable = false;
  }
  return MovableAllReduceContext{is_all_reduce_movable, accumulation_contexts};
}

struct WhileInitContext {
  HloInstruction* while_init{nullptr};
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_old_buffer;
};

// Creates a new while init instruction, which replaces each accumulation buffer
// in the given accumulation contexts with a zero-initialized buffer. In other
// words, we are accumulating all the deltas in the while loop with a zero
// initial value.
WhileInitContext CreateNewWhileInit(
    HloInstruction* old_while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations) {
  HloInstruction* old_while_init = old_while_instruction->mutable_operand(0);
  HloComputation* while_parent = old_while_instruction->parent();
  std::vector<HloInstruction*> new_while_init_elements(
      old_while_init->operand_count(), nullptr);
  for (const auto& all_reduce_and_accumulations_pair :
       all_reduce_to_accumulations) {
    const std::vector<AccumulationContext>& accumulations =
        all_reduce_and_accumulations_pair.second;
    HloInstruction* loop_all_reduce = all_reduce_and_accumulations_pair.first;

    for (auto& accumulation_context : accumulations) {
      int64_t tuple_index = accumulation_context.param_tuple_index;
      HloInstruction* old_buffer = old_while_init->mutable_operand(tuple_index);
      // For reduce-scatter, the shape of the accumulator is the pre-scatter
      // shape.
      const Shape& accumulation_shape =
          loop_all_reduce->opcode() == HloOpcode::kAllReduce
              ? old_buffer->shape()
              : loop_all_reduce->operand(0)->shape();

      HloInstruction* new_buffer = while_parent->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateFromDimensions(
              accumulation_shape.element_type(),
              accumulation_shape.dimensions())));
      new_while_init_elements[tuple_index] = new_buffer;
    }
  }
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_old_buffer;
  for (int i = 0; i < old_while_init->operand_count(); i++) {
    if (!new_while_init_elements[i]) {
      new_while_init_elements[i] = old_while_init->mutable_operand(i);
    } else {
      tuple_index_to_old_buffer[i] = old_while_init->mutable_operand(i);
    }
  }
  HloInstruction* new_while_init = while_parent->AddInstruction(
      HloInstruction::CreateTuple(new_while_init_elements));
  return WhileInitContext{new_while_init, tuple_index_to_old_buffer};
}

// When moving reduce-scatter outside the while body, change the associated
// accumulation buffers to use the shape of the operand of the reduce-scatter
// (i.e., the pre-scatter shape).
Status ChangeAccumulatorShapesInLoopBodies(
    HloInstruction* old_while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations) {
  HloComputation* body = old_while_instruction->while_body();
  HloComputation* cond = old_while_instruction->while_condition();

  // Cache of zero shaped constants
  absl::flat_hash_map<Shape, HloInstruction*> zeros;

  auto create_zero_of_shape = [&zeros, body](const Shape& shape) {
    auto it = zeros.find(shape);
    if (it != zeros.end()) {
      return it->second;
    }
    HloInstruction* zero = body->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateFromShape(shape)));
    zeros[shape] = zero;
    return zero;
  };

  for (const auto& [loop_reduce_scatter, accumulations] :
       all_reduce_to_accumulations) {
    if (loop_reduce_scatter->opcode() != HloOpcode::kReduceScatter) {
      continue;
    }
    const Shape& accumulation_shape = loop_reduce_scatter->operand(0)->shape();
    for (auto& accumulation_context : accumulations) {
      const int64_t tuple_index = accumulation_context.param_tuple_index;
      // Change shape of parameter tuple element at index for while body.
      HloInstruction* param_body = body->parameter_instruction(0);

      std::vector<Shape> element_shapes = param_body->shape().tuple_shapes();
      element_shapes[tuple_index] = accumulation_shape;
      *param_body->mutable_shape() = ShapeUtil::MakeTupleShape(element_shapes);

      // Find the GTE for this index and change its type and its users.
      // For reduce-scatter, we do not allow any forwarding instructions, so
      // we expect that the user is the accumulation instruction.
      for (HloInstruction* user : param_body->users()) {
        if (user->opcode() != HloOpcode::kGetTupleElement) {
          continue;
        }
        HloGetTupleElementInstruction* gte =
            Cast<HloGetTupleElementInstruction>(user);
        if (gte->tuple_index() != tuple_index) {
          continue;
        }

        *gte->mutable_shape() = accumulation_shape;
        for (HloInstruction* gte_user : gte->users()) {
          CHECK_EQ(gte_user->opcode(), HloOpcode::kAdd);
          *gte_user->mutable_shape() = accumulation_shape;
        }
      }

      // Change the users of the reduce-scatter. We expect it to feed into an
      // add through an optional chain of selects.
      std::vector<HloInstruction*> reduce_scatter_users =
          loop_reduce_scatter->users();
      while (!reduce_scatter_users.empty()) {
        HloInstruction* user = reduce_scatter_users.back();
        reduce_scatter_users.pop_back();
        if (user->opcode() == HloOpcode::kSelect) {
          // Select should have a zero on one side and reduce-scatter on other
          // with an effectively scalar predicate.
          HloInstruction* zero = create_zero_of_shape(accumulation_shape);

          // We have verified that the predicate is effectively scalar.
          HloInstruction* scalar_predicate =
              GetEffectiveScalar(user->mutable_operand(0));
          Shape pred_shape =
              ShapeUtil::ChangeElementType(accumulation_shape, PRED);
          HloInstruction* pred =
              body->AddInstruction(HloInstruction::CreateBroadcast(
                  pred_shape, scalar_predicate, {}));
          TF_RETURN_IF_ERROR(user->ReplaceOperandWithDifferentShape(0, pred));
          HloInstruction *new_operand_1, *new_operand_2;
          if (user->operand_index(loop_reduce_scatter) == 1) {
            new_operand_1 = loop_reduce_scatter->mutable_operand(0);
            new_operand_2 = zero;
          } else {
            new_operand_1 = zero;
            new_operand_2 = loop_reduce_scatter->mutable_operand(0);
          }
          TF_RETURN_IF_ERROR(
              user->ReplaceOperandWithDifferentShape(1, new_operand_1));
          TF_RETURN_IF_ERROR(
              user->ReplaceOperandWithDifferentShape(2, new_operand_2));
          *user->mutable_shape() = accumulation_shape;
        } else {
          TF_RET_CHECK(user->opcode() == HloOpcode::kAdd);
          // We should have already changed the Add's shape when patching input
          // GTE above.
          TF_RET_CHECK(user->shape() == accumulation_shape);
        }
      }

      // Change result tuple of the while body.
      HloInstruction* root = body->root_instruction();
      *root->mutable_shape() = param_body->shape();

      // Change parameter type for condition.
      HloInstruction* param_cond = cond->parameter_instruction(0);
      *param_cond->mutable_shape() = param_body->shape();
    }
  }

  return OkStatus();
}

// Creates all the sinked all-reduce instructions in the while instruction's
// parent computation. Returns a map that maps a tuple index of an
// accumulation buffer to it's corresponding all-reduce.
absl::flat_hash_map<int, HloInstruction*> CreateSinkedAllReduces(
    HloInstruction* new_while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations,
    const absl::flat_hash_map<int, HloInstruction*>&
        tuple_index_to_old_buffer) {
  HloComputation* while_parent = new_while_instruction->parent();
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_new_buffer;
  for (const auto& all_reduce_and_accumulations_pair :
       all_reduce_to_accumulations) {
    HloInstruction* loop_all_reduce = all_reduce_and_accumulations_pair.first;
    const std::vector<AccumulationContext>& accumulations =
        all_reduce_and_accumulations_pair.second;
    for (const auto& accumulation_context : accumulations) {
      int64_t tuple_index = accumulation_context.param_tuple_index;
      const Shape& accumulation_buffer_shape =
          new_while_instruction->shape().tuple_shapes(tuple_index);
      HloInstruction* accumulation_buffer =
          while_parent->AddInstruction(HloInstruction::CreateGetTupleElement(
              accumulation_buffer_shape, new_while_instruction, tuple_index));
      HloInstruction* all_reduce_operand = accumulation_buffer;
      if (!ShapeUtil::SameElementType(loop_all_reduce->shape(),
                                      accumulation_buffer_shape)) {
        Shape all_reduce_shape =
            ShapeUtil::MakeShape(loop_all_reduce->shape().element_type(),
                                 accumulation_buffer_shape.dimensions());
        all_reduce_operand =
            while_parent->AddInstruction(HloInstruction::CreateConvert(
                all_reduce_shape, accumulation_buffer));
      }
      HloInstruction* all_reduced_delta;
      if (loop_all_reduce->opcode() == HloOpcode::kAllReduce) {
        auto* old_all_reduce = Cast<HloAllReduceInstruction>(loop_all_reduce);

        all_reduced_delta =
            while_parent->AddInstruction(HloInstruction::CreateAllReduce(
                all_reduce_operand->shape(), {all_reduce_operand},
                old_all_reduce->called_computations()[0],
                old_all_reduce->replica_groups(),
                old_all_reduce->constrain_layout(),
                hlo_query::NextChannelId(*(while_parent->parent())),
                old_all_reduce->use_global_device_ids()));
      } else {
        auto* old_reduce_scatter =
            Cast<HloReduceScatterInstruction>(loop_all_reduce);
        all_reduced_delta =
            while_parent->AddInstruction(HloInstruction::CreateReduceScatter(
                old_reduce_scatter->shape(), {all_reduce_operand},
                old_reduce_scatter->called_computations()[0],
                old_reduce_scatter->replica_groups(),
                old_reduce_scatter->constrain_layout(),
                hlo_query::NextChannelId(*(while_parent->parent())),
                old_reduce_scatter->use_global_device_ids(),
                old_reduce_scatter->scatter_dimension()));
      }

      if (!ShapeUtil::SameElementType(all_reduced_delta->shape(),
                                      accumulation_buffer_shape)) {
        all_reduced_delta =
            while_parent->AddInstruction(HloInstruction::CreateConvert(
                accumulation_buffer_shape, all_reduced_delta));
      }
      CHECK(ContainsKey(tuple_index_to_old_buffer, tuple_index));
      HloInstruction* old_buffer = tuple_index_to_old_buffer.at(tuple_index);
      CHECK(Shape::Equal().IgnoreLayout()(old_buffer->shape(),
                                          all_reduced_delta->shape()));
      HloInstruction* add_to_old_buffer =
          while_parent->AddInstruction(HloInstruction::CreateBinary(
              all_reduced_delta->shape(), HloOpcode::kAdd, old_buffer,
              all_reduced_delta));
      tuple_index_to_new_buffer[tuple_index] = add_to_old_buffer;
    }
  }
  return tuple_index_to_new_buffer;
}

// Creates a tuple which is equivalent to the original while instruction's
// output.
HloInstruction* CreateNewWhileResult(
    HloInstruction* new_while_instruction,
    const absl::flat_hash_map<int, HloInstruction*>&
        tuple_index_to_new_buffer) {
  HloComputation* while_parent = new_while_instruction->parent();
  CHECK(new_while_instruction->shape().IsTuple());
  std::vector<HloInstruction*> new_while_result_elements(
      new_while_instruction->shape().tuple_shapes_size(), nullptr);
  for (int i = 0; i < new_while_result_elements.size(); i++) {
    if (ContainsKey(tuple_index_to_new_buffer, i)) {
      new_while_result_elements[i] = tuple_index_to_new_buffer.at(i);
    } else {
      HloInstruction* gte =
          while_parent->AddInstruction(HloInstruction::CreateGetTupleElement(
              new_while_instruction->shape().tuple_shapes(i),
              new_while_instruction, i));
      new_while_result_elements[i] = gte;
    }
  }
  HloInstruction* new_while_result = while_parent->AddInstruction(
      HloInstruction::CreateTuple(new_while_result_elements));
  return new_while_result;
}

// Creates the sinked all-reduce instructions for all accumulation buffers.
// The all-reduce outputs are then added to the original accumulation buffers.
// Creates a tuple that groups the while loop output and the accumulated
// buffers and replaces all uses of the old while with this new tuple.
Status AddSinkedAllReducesAndReplaceWhile(
    HloInstruction* while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations) {
  // Note that we create all instructions before replacing and removing any
  // old instruction. This ensures that we do not accidentally access any
  // deleted instruction when creating new instructions.

  // Step 1) create the new while init instruction, which uses
  // zero-initialized tensors as the accumulation buffers for the all-reduce.
  auto new_while_init_context =
      CreateNewWhileInit(while_instruction, all_reduce_to_accumulations);

  // For reduce-scatter, we need to adjust all the accumulator shapes to use
  // the pre-scatter shape.
  TF_RETURN_IF_ERROR(ChangeAccumulatorShapesInLoopBodies(
      while_instruction, all_reduce_to_accumulations));

  // Step 2) create the new while instruction.
  HloInstruction* new_while_instruction =
      while_instruction->parent()->AddInstruction(HloInstruction::CreateWhile(
          new_while_init_context.while_init->shape(),
          while_instruction->while_condition(), while_instruction->while_body(),
          new_while_init_context.while_init));
  // Step 3) create the new all-reduce instructions after the while loop.
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_new_buffer =
      CreateSinkedAllReduces(new_while_instruction, all_reduce_to_accumulations,
                             new_while_init_context.tuple_index_to_old_buffer);
  // Step 4) create the tuple and replace the old while instruction for all of
  // its uses.
  HloInstruction* new_while_result =
      CreateNewWhileResult(new_while_instruction, tuple_index_to_new_buffer);
  TF_RETURN_IF_ERROR(while_instruction->parent()->ReplaceInstruction(
      while_instruction, new_while_result));
  return OkStatus();
}

}  // namespace

StatusOr<bool> WhileLoopAllReduceCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool is_changed = false;
  bool run_next_pass = true;
  // In case of MPMD, all-reduces might be cross-module and should preserve
  // their channel ID. Do not move all-reduces in this case since the channel
  // ID might be changed.
  if (module->config().num_partitions() > 1 &&
      !module->config().use_spmd_partitioning()) {
    return false;
  }
  std::unique_ptr<HloReplicationAnalysis> cross_replica_replication_analysis;
  if (module->config().replica_count() > 1) {
    VLOG(5) << "num_replicas: " << module->config().replica_count()
            << " run HloReplicationAnalysis across replicas";
    TF_ASSIGN_OR_RETURN(cross_replica_replication_analysis,
                        HloReplicationAnalysis::RunWithPartialReplication(
                            module, /*cross_partition_spmd=*/false));
  }
  std::unique_ptr<HloReplicationAnalysis> cross_partition_replication_analysis;
  if (module->config().use_spmd_partitioning() &&
      module->config().num_partitions() > 1) {
    VLOG(5) << "num_partitions: " << module->config().num_partitions()
            << " run HloReplicationAnalysis across partitions";
    TF_ASSIGN_OR_RETURN(cross_partition_replication_analysis,
                        HloReplicationAnalysis::RunWithPartialReplication(
                            module, /*cross_partition_spmd=*/true));
  }
  // The while instruction's parent could be a while body for another while
  // loop. We recursively sink the all-reduce through nested while loops if
  // applicable by repeating this process.
  uint32_t count_all_reduce = 0, count_reduce_scatter = 0;
  while (run_next_pass) {
    run_next_pass = false;
    std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
    // A computation could be the while body of multiple while instructions,
    // so we start from the computation and find all of its callers that is a
    // kWhile if there is any.
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      std::vector<HloInstruction*> computation_callers =
          call_graph->GetComputationCallers(computation);
      std::vector<HloInstruction*> while_caller_instructions;
      for (HloInstruction* caller_instruction : computation_callers) {
        // For simplicity, we only support while instructions whose shape is
        // tuple.
        if (caller_instruction->opcode() == HloOpcode::kWhile &&
            caller_instruction->shape().IsTuple() &&
            caller_instruction->while_body() == computation) {
          while_caller_instructions.push_back(caller_instruction);
        }
      }
      // Skip to next computation if this computation is not the while body of
      // any while instruction.
      if (while_caller_instructions.empty()) {
        continue;
      }
      std::vector<HloAllReduceInstructionBase*> while_body_all_reduces;
      for (HloInstruction* while_body_instruction :
           computation->MakeInstructionPostOrder()) {
        HloOpcode op = while_body_instruction->opcode();
        const bool is_candidate =
            (op == HloOpcode::kAllReduce) ||
            (enable_reduce_scatter_ && op == HloOpcode::kReduceScatter);
        if (!is_candidate) {
          continue;
        }
        auto* all_reduce_instruction =
            Cast<HloAllReduceInstructionBase>(while_body_instruction);
        if (all_reduce_instruction->constrain_layout()) {
          return false;
        } else {
          while_body_all_reduces.push_back(all_reduce_instruction);
        }
      }
      HloInstructionMap<std::vector<AccumulationContext>>
          all_reduce_to_accumulations;
      for (HloAllReduceInstructionBase* all_reduce : while_body_all_reduces) {
        auto movable_all_reduce_context = IsAllReduceMovable(
            all_reduce, computation, cross_replica_replication_analysis,
            cross_partition_replication_analysis);
        if (movable_all_reduce_context.is_movable) {
          all_reduce_to_accumulations[all_reduce] =
              std::move(movable_all_reduce_context.accumulation_contexts);
        }
        VLOG(3) << "WhileLoopAllReduceCodeMotion, all-reduce: "
                << all_reduce->ToString()
                << " is_movable: " << movable_all_reduce_context.is_movable
                << " while loop: " << while_caller_instructions.front()->name()
                << " num_accumulations: "
                << (movable_all_reduce_context.is_movable
                        ? all_reduce_to_accumulations[all_reduce].size()
                        : 0);
      }
      if (all_reduce_to_accumulations.empty()) {
        continue;
      }
      // For each while instruction calling this computation, create the
      // corresponding all-reduces after the while loop.
      for (HloInstruction* while_instruction : while_caller_instructions) {
        TF_RETURN_IF_ERROR(AddSinkedAllReducesAndReplaceWhile(
            while_instruction, all_reduce_to_accumulations));
        is_changed = true;
        run_next_pass = true;
      }
      // At last, remove the old all-reduce instructions in the while body.
      for (const auto& all_reduce_accumulations_pair :
           all_reduce_to_accumulations) {
        HloInstruction* all_reduce = all_reduce_accumulations_pair.first;
        if (all_reduce->opcode() == HloOpcode::kAllReduce) {
          count_all_reduce++;
        } else {
          count_reduce_scatter++;
        }
        TF_RETURN_IF_ERROR(computation->ReplaceInstructionWithDifferentShape(
            all_reduce, all_reduce->mutable_operand(0)));
      }
    }
  }
  VLOG(2) << "Hoisted " << count_all_reduce << " all-reduce and "
          << count_reduce_scatter << " reduce-scatter out of while loops";
  return is_changed;
}

}  // namespace xla
