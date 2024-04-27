/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/collective_pipeliner.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/map_util.h"
#include "xla/primitive_util.h"
#include "xla/service/constant_value.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/value_range.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

const char* const CollectivePipeliner::kInsertedByPreviousStep =
    "InsertedByPreviousStep";
const char* const CollectivePipeliner::kSunkByPreviousStep =
    "SunkByPreviousStep";

namespace {

using InstructionMap =
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>;
// Record the loop invariant parameters used in a chain as well as their
// parameter indices.
using LoopVariantParameterInfo =
    std::vector<std::pair<int64_t, HloInstruction*>>;

// Update all control dependencies for a cloned instruction to connect other
// cloned instructions rather than originals.
Status UpdateControlDependencies(HloInstruction* original,
                                 HloInstruction* new_instr,
                                 const InstructionMap& cloned_map) {
  for (auto* pred : original->control_predecessors()) {
    auto it = cloned_map.find(pred);
    if (it == cloned_map.end()) {
      continue;
    }
    TF_RETURN_IF_ERROR(it->second->AddControlDependencyTo(new_instr));
  }
  return OkStatus();
}

// Checks for the condition where all indices except the one passed as parameter
// of a dynamic slice are constants. Something like dynamic-slice(operand, i, c,
// c), where "c" are constants and "i" is a dynamic value.
bool AllIndicesConstantsExceptOne(
    const HloDynamicUpdateSliceInstruction* dyn_update, int64_t index) {
  if (dyn_update->operand(index)->IsConstant()) {
    return false;
  }
  for (int64_t i = dyn_update->first_index_operand_number();
       i < dyn_update->operand_count(); ++i) {
    if (i == index) {
      continue;
    }
    if (!dyn_update->operand(i)->IsConstant()) {
      return false;
    }
  }
  return true;
}

// Checks if a dynamic-update-slice() HLO has only the first dimension being
// actually inserted "sliced" and the other dimensions are the same size of the
// output on the tensor to be "inserted".
std::optional<int> GetSlicedDimension(
    const HloDynamicUpdateSliceInstruction* dyn_update) {
  std::optional<int> sliced_dim;
  for (int64_t i = dyn_update->first_index_operand_number();
       i < dyn_update->operand_count(); ++i) {
    const HloInstruction* idx = dyn_update->operand(i);
    if (!idx->IsConstant()) {
      if (sliced_dim.has_value()) {
        return std::nullopt;
      }
      sliced_dim = i - dyn_update->first_index_operand_number();
      continue;
    }
    if (Cast<HloConstantInstruction>(idx)->literal().GetFirstInteger() != 0) {
      return std::nullopt;
    }
  }
  return sliced_dim;
}

bool CheckIndexIsMonotonic(
    const HloInstruction* index,
    const absl::flat_hash_map<const HloInstruction*, Range>& induction_map) {
  // Because the only math operations supported by RecursivelyIdentifyRange()
  // are only sub/add then checking that we can compute the range here is enough
  // to guarantee that the index is monotonic if the base index is monotonic. If
  // we want to make the function more powerful we need to have a more
  // sophisticated check for monotonicity.
  Range range = RecursivelyIdentifyRange(index, induction_map);
  VLOG(5) << "Range for: " << index->ToString() << " " << range.ToString();
  return !range.IsEmpty() && range.IsLinear();
}

// Check that the parameter is only used in a pattern param -> gte ->
// dyn-slice(,i, ...) where the only users of the parameter are an extraction of
// a subslice of it driven by the loop iteration counter.
bool CheckParameterUsageIsCompatible(const HloInstruction* gte,
                                     const HloInstruction* dus,
                                     const HloInstruction* dus_idx,
                                     int64_t sliced_index) {
  for (auto* user : gte->users()) {
    // Expected all users are dynamic-slices
    if (dus != user) {
      VLOG(5) << "CheckParameterUsageIsCompatible(): User not a dynamic slice "
                 "or the dynamic-update-slice for the output."
              << user->ToString();
      return false;
    }
    // Expected same index as dynamic-update-slice().
    if (user->operand(static_cast<HloDynamicSliceInstruction*>(user)
                          ->first_index_operand_number() +
                      sliced_index) != dus_idx) {
      VLOG(5) << "CheckParameterUsageIsCompatible(): Idx is not the same as "
                 "dynamic-update-slice() "
              << user->ToString();
      return false;
    }
  }
  return true;
}

// Given a kInsertedByPreviousStep custom call return the level it represents.
std::optional<int64_t> GetLevelFromCustomCall(const HloInstruction* instr) {
  if (!instr->IsCustomCall(CollectivePipeliner::kInsertedByPreviousStep)) {
    return std::nullopt;
  }
  return Cast<HloConstantInstruction>(instr->operand(1))
      ->literal()
      .GetFirstInteger();
}

std::optional<std::vector<HloInstruction*>>
CollectDynamicSliceIndicesIfConstant(HloInstruction* instr) {
  CHECK_EQ(instr->opcode(), HloOpcode::kDynamicSlice);
  std::vector<HloInstruction*> indices;
  HloDynamicSliceInstruction* dyn_slice =
      Cast<HloDynamicSliceInstruction>(instr);
  for (int64_t i = dyn_slice->first_index_operand_number();
       i < instr->operand_count(); ++i) {
    HloInstruction* operand = dyn_slice->mutable_operand(i);
    CHECK_EQ(operand->shape().dimensions_size(), 0);
    std::vector<std::pair<HloInstruction*, int>> stack(
        1, std::make_pair(operand, 0));
    absl::flat_hash_set<HloInstruction*> visited;
    while (!stack.empty()) {
      auto& [curr_instr, operand_idx] = stack.back();
      if (operand_idx == curr_instr->operand_count()) {
        indices.push_back(curr_instr);
        stack.pop_back();
        continue;
      }
      HloInstruction* next_operand = curr_instr->mutable_operand(operand_idx++);
      if (next_operand->opcode() == HloOpcode::kParameter ||
          next_operand->HasSideEffect()) {
        return std::nullopt;
      }
      if (visited.insert(next_operand).second) {
        stack.push_back(std::make_pair(next_operand, 0));
      }
    }
  }
  return indices;
}

bool IsSupportedLoopIndexType(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::S32:
    case PrimitiveType::S64:
    case PrimitiveType::S16:
    case PrimitiveType::S8:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
    case PrimitiveType::U16:
    case PrimitiveType::U8:
      return true;
    default:
      return false;
  }
}

std::optional<Literal> CreateLiteralOfShape(const Shape& shape, int64_t value) {
  return primitive_util::PrimitiveTypeSwitch<std::optional<Literal>>(
      [&](auto kType) -> std::optional<Literal> {
        if constexpr (primitive_util::IsIntegralType(kType)) {
          using NativeT = typename primitive_util::NativeTypeOf<kType>;
          CHECK_LE(value, static_cast<absl::int128>(
                              std::numeric_limits<NativeT>::max()));
          CHECK_GE(value, static_cast<absl::int128>(
                              std::numeric_limits<NativeT>::min()));
          return LiteralUtil::CreateR0(static_cast<NativeT>(value));
        }
        return std::nullopt;
      },
      shape.element_type());
}

// Collect input data dependencies of instructions we want to pipeline that are
// simple to be cloned. Returns if an unexpected dependency has been found for
// pipelining.
bool CollectSimpleDependencies(HloInstruction* i,
                               std::vector<HloInstruction*>& deps_vector,
                               absl::flat_hash_set<HloInstruction*>& deps_set) {
  if (i->opcode() == HloOpcode::kDynamicSlice) {
    auto indices = CollectDynamicSliceIndicesIfConstant(i);
    if (!indices.has_value()) {
      return false;
    }
    deps_vector.insert(deps_vector.end(), indices->begin(), indices->end());
    deps_set.insert(indices->begin(), indices->end());
    return true;
  }
  for (HloInstruction* op : i->mutable_operands()) {
    absl::InlinedVector<HloInstruction*, 4> to_add;
    if (op->opcode() == HloOpcode::kBroadcast) {
      to_add.push_back(op);
      if (deps_set.insert(op).second) {
        op = op->mutable_operand(0);
        if (op->opcode() == HloOpcode::kConstant) {
          if (deps_set.insert(op).second) {
            to_add.push_back(op);
          }
        }
      }
    }
    deps_vector.insert(deps_vector.end(), to_add.rbegin(), to_add.rend());
  }
  return true;
}

// Check that the value we plan to push to the next iteration is stored
// in a way we support into an output to the loop.
// If this level 0 we require the unique dynamic update slice to feed directly
// into the root instruction. If this is level > 1 then we require that the
// unique dynamic_update slice is inserted using the index created in the
// previous level.
std::pair<HloDynamicUpdateSliceInstruction*, std::vector<HloInstruction*>>
CheckStoreIntoSliceIsCompatible(HloInstruction* instr,
                                const HloComputation* while_body,
                                int64_t level_to_operate_on,
                                bool multi_uses_pipelining,
                                HloPredicate acceptable_formatting) {
  if ((!multi_uses_pipelining && instr->user_count() != 1) ||
      instr->operand_count() != 1 || instr->HasControlDependencies()) {
    return std::make_pair(nullptr, std::vector<HloInstruction*>{});
  }
  // Set to collect instructions that have been already added.
  absl::flat_hash_set<HloInstruction*> added_instructions;
  HloInstruction* folded_instr = instr;
  std::vector<HloInstruction*> formatting_ops;
  // Returns if this is an acceptable user of a pipelined instruction.
  // Generic elementwise ops can have multiple operands that require the inputs
  // of being saved across the loop. So protect them through
  // "multi_uses_pipelining" flag.
  auto is_acceptable_user = [&](HloInstruction* i) {
    if (i->HasControlDependencies() || !acceptable_formatting(i)) {
      return false;
    }
    if (i->opcode() == HloOpcode::kReduce &&
        (ShapeUtil::ElementsIn(i->shape()) ==
             ShapeUtil::ElementsIn(instr->operand(0)->shape()) ||
         ShapeUtil::ElementsIn(instr->operand(0)->shape()) < 1024)) {
      return true;
    }
    return HloPredicateIsOp<HloOpcode::kSlice, HloOpcode::kDynamicSlice,
                            HloOpcode::kPad, HloOpcode::kCollectivePermute,
                            HloOpcode::kConvert, HloOpcode::kReshape,
                            HloOpcode::kAllReduce, HloOpcode::kTranspose>(i) ||
           (multi_uses_pipelining && i->IsElementwise()) ||
           i->IsCustomCall(CollectivePipeliner::kInsertedByPreviousStep);
  };
  // Returns if this instruction is a dynamic-update-slice inserting the value
  // into a bigger buffer that we are going to pipeline to the next iteration.
  auto is_final_slice_insertion = [&](HloInstruction* i) {
    HloDynamicUpdateSliceInstruction* dyn_update =
        DynCast<HloDynamicUpdateSliceInstruction>(i);
    if (dyn_update == nullptr || dyn_update->user_count() != 1) {
      return false;
    }
    if (level_to_operate_on == 0) {
      if (dyn_update->users()[0] == while_body->root_instruction()) {
        return true;
      }
      return false;
    }
    for (int64_t i = dyn_update->first_index_operand_number();
         i < dyn_update->operand_count(); ++i) {
      if (auto level = GetLevelFromCustomCall(dyn_update->operand(i))) {
        if (*level == level_to_operate_on) {
          return true;
        }
        return false;
      }
    }
    return false;
  };
  HloDynamicUpdateSliceInstruction* final_slice_insertion = nullptr;
  std::vector<std::pair<HloInstruction*, int>> stack;
  absl::flat_hash_map<HloInstruction*, int32_t> formatting_map;
  stack.push_back(std::make_pair(folded_instr, 0));
  // Post order traversal to discover formatting instructions.
  while (!stack.empty()) {
    auto& data = stack.back();
    HloInstruction* instr = data.first;
    if (data.second == 0 && instr != folded_instr) {
      formatting_map[instr] = 0;
    }
    if (data.second == instr->user_count()) {
      stack.pop_back();
      continue;
    }
    HloInstruction* next_user = instr->users()[data.second++];
    if (is_final_slice_insertion(next_user)) {
      if ((final_slice_insertion != nullptr &&
           final_slice_insertion != next_user) ||
          next_user->user_count() != 1 || next_user->operand(1) != instr) {
        return std::make_pair(nullptr, std::vector<HloInstruction*>{});
      }
      final_slice_insertion = Cast<HloDynamicUpdateSliceInstruction>(next_user);
      continue;
    }
    if (!is_acceptable_user(next_user)) {
      return std::make_pair(nullptr, std::vector<HloInstruction*>{});
    }
    if (added_instructions.insert(next_user).second) {
      stack.push_back(std::make_pair(next_user, 0));
    }
  }
  if (final_slice_insertion == nullptr) {
    return std::make_pair(nullptr, std::vector<HloInstruction*>{});
  }
  for (auto& op : formatting_map) {
    for (const HloInstruction* operand : final_slice_insertion->operands()) {
      if (formatting_map.count(operand)) {
        ++op.second;
      }
    }
  }
  stack.push_back(std::make_pair(folded_instr, 0));
  added_instructions.clear();
  // Post order traversal to determine the insert instruction order.
  while (!stack.empty()) {
    auto& data = stack.back();
    HloInstruction* instr = data.first;
    if (data.second == 0 && instr != folded_instr) {
      if (!CollectSimpleDependencies(instr, formatting_ops,
                                     added_instructions)) {
        return std::make_pair(nullptr, std::vector<HloInstruction*>{});
      }
      formatting_ops.push_back(instr);
    }
    if (data.second == instr->user_count()) {
      stack.pop_back();
      continue;
    }
    HloInstruction* next_user = instr->users()[data.second++];
    if (is_final_slice_insertion(next_user)) {
      if ((final_slice_insertion != nullptr &&
           final_slice_insertion != next_user) ||
          next_user->user_count() != 1 || next_user->operand(1) != instr) {
        return std::make_pair(nullptr, std::vector<HloInstruction*>{});
      }
      final_slice_insertion = Cast<HloDynamicUpdateSliceInstruction>(next_user);
      continue;
    }
    if (--formatting_map[next_user] > 0) {
      continue;
    }
    if (added_instructions.insert(next_user).second) {
      stack.push_back(std::make_pair(next_user, 0));
    }
  }
  return std::make_pair(final_slice_insertion, formatting_ops);
}

bool IsLoopIterator(const HloInstruction* instr,
                    int64_t loop_iteration_tuple_idx) {
  if (instr->opcode() != HloOpcode::kGetTupleElement ||
      instr->operand(0)->opcode() != HloOpcode::kParameter) {
    return false;
  }
  return instr->tuple_index() == loop_iteration_tuple_idx;
}

// Scavenge operands that are dependencies not included in the ops set and that
// aren't the source_op passed as input parameter and return them in a vector.
std::vector<HloInstruction*> CollectDependenciesToPipeline(
    HloInstruction* source_op, absl::Span<HloInstruction* const> ops) {
  absl::flat_hash_set<HloInstruction*> formatting_set(ops.begin(), ops.end());
  formatting_set.insert(source_op);
  std::vector<HloInstruction*> to_return;
  absl::flat_hash_set<HloInstruction*> already_inserted;
  for (const HloInstruction* op : ops) {
    for (HloInstruction* operand : op->operands()) {
      if (!formatting_set.count(operand)) {
        formatting_set.insert(operand);
        to_return.push_back(operand);
      }
    }
  }
  return to_return;
}

std::optional<std::vector<HloInstruction*>> CollectIndependentOperandChain(
    HloInstruction* instr, int64_t loop_iter,
    const absl::flat_hash_set<const HloInstruction*>& loop_invariant_params,
    HloPredicate should_allow_loop_variant_parameter_in_chain) {
  std::vector<HloInstruction*> chain;
  absl::flat_hash_set<const HloInstruction*> visited_set({instr});
  std::vector<std::pair<HloInstruction*, int>> stack(1, {instr, 0});
  auto is_loop_variant_parameter_input =
      [&loop_invariant_params, loop_iter](const HloInstruction* instr) {
        if (instr->opcode() != HloOpcode::kGetTupleElement ||
            instr->operand(0)->opcode() != HloOpcode::kParameter) {
          return false;
        }
        return !IsLoopIterator(instr, loop_iter) &&
               !loop_invariant_params.count(instr);
      };
  while (!stack.empty()) {
    auto& curr = stack.back();
    if (curr.second == curr.first->operand_count()) {
      if (curr.first != instr) {
        chain.push_back(curr.first);
      }
      stack.pop_back();
      continue;
    }
    HloInstruction* curr_operand = curr.first->mutable_operand(curr.second++);
    if (curr_operand->opcode() == HloOpcode::kParameter) {
      continue;
    }
    if (is_loop_variant_parameter_input(curr_operand) &&
        !should_allow_loop_variant_parameter_in_chain(curr_operand)) {
      return std::nullopt;
    }
    if (visited_set.insert(curr_operand).second) {
      stack.emplace_back(curr_operand, 0);
    }
  }
  for (auto* chain_instr : chain) {
    // Allow tokens in the chain.
    if (chain_instr->opcode() == HloOpcode::kAfterAll) {
      continue;
    }
    if (chain_instr->opcode() == HloOpcode::kRecvDone) {
      // Since we allow tokens in the chain, we need to exclude Recv-done in
      // the chain, to prevent pipelining Recv/Recv-done by accident.
      return std::nullopt;
    }
    const bool all_users_in_chain = absl::c_all_of(
        chain_instr->users(), [&visited_set](const HloInstruction* u) {
          return visited_set.contains(u);
        });
    const bool is_scalar_shaped =
        ShapeUtil::IsEffectiveScalar(chain_instr->shape());
    if (!all_users_in_chain) {
      if (!loop_invariant_params.contains(chain_instr) && !is_scalar_shaped &&
          (chain_instr->opcode() != HloOpcode::kGetTupleElement ||
           chain_instr->operand(0)->opcode() != HloOpcode::kParameter ||
           !should_allow_loop_variant_parameter_in_chain(chain_instr))) {
        return std::nullopt;
      }
    }
  }
  return std::move(chain);
}

// Collect chains of instructions that we can pipeline backwards.
// These are chains of instructions culminating in one of the instructions we
// are interested in pipelining (like all-gather for example), that have uses
// only inside the chain (except for scalar instructions that get duplicated)
// and use a parameter value from the loop that is invariant (doesn't get
// updated between loop iterations).
std::optional<std::vector<HloInstruction*>> CollectChainsToPushBackwards(
    HloInstruction* instr, int64_t loop_iter, const HloComputation* while_body,
    int64_t level_to_operate_on,
    const absl::flat_hash_set<const HloInstruction*>& loop_invariant_params,
    HloPredicate should_allow_loop_variant_parameter_in_chain) {
  if (instr->HasControlDependencies()) {
    return std::nullopt;
  }
  return CollectIndependentOperandChain(
      instr, loop_iter, loop_invariant_params,
      should_allow_loop_variant_parameter_in_chain);
}

// Given a dynamic-update-slice find the output index of the loop we feed into.
// We assume that the insertion instruction has been already validated.
std::optional<int64_t> FindOutputIndexForDynamicUpdateSlice(
    const HloInstruction* dus, const HloInstruction* root_instr) {
  std::optional<int64_t> output_idx;
  while (dus->opcode() == HloOpcode::kDynamicUpdateSlice) {
    if (dus->user_count() != 1) {
      output_idx = std::nullopt;
      break;
    }
    if (dus->users()[0] == root_instr) {
      auto indices = root_instr->OperandIndices(dus);
      if (indices.size() != 1) {
        output_idx = std::nullopt;
        break;
      }
      output_idx = indices[0];
      break;
    }
    dus = Cast<HloDynamicUpdateSliceInstruction>(dus->users()[0]);
  }
  return output_idx;
}

std::vector<HloInstruction*> MapNewOperands(
    absl::Span<HloInstruction* const> operands, const InstructionMap& clone_map,
    bool allow_unmapped = false) {
  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(operands.size());
  for (HloInstruction* operand : operands) {
    auto it = clone_map.find(operand);
    HloInstruction* mapped_operand = operand;
    CHECK(it != clone_map.end() || allow_unmapped)
        << operand->ToString() << " not present in map";
    if (it != clone_map.end()) {
      mapped_operand = it->second;
    }
    new_operands.push_back(mapped_operand);
  }
  return new_operands;
}

// Collect information regarding movement of data either backward or forward
// through loop iterations. Except collective_to_move every other information
// here can be empty/null/-1 to indicate absence.
struct WhileMoveInfo {
  HloInstruction* collective_to_move;
  HloDynamicUpdateSliceInstruction* dynamic_update_slice;
  std::vector<HloInstruction*> formatting_ops;
  int64_t sliced_idx;
  int64_t output_idx;
};

// Set channel_id of instruction to next available to avoid collisions.
void UpdateInstructionChannelId(HloInstruction* cloned_instr,
                                int64_t& next_channel_id) {
  // Avoid updating Send and Recv instructions because pipelined Send and Recv
  // instructions should keep the same channel-id to indicate that the group of
  // instructions need to cooperate.
  if (const auto* send_recv_instr =
          DynCast<HloSendRecvInstruction>(cloned_instr)) {
    if (!send_recv_instr->is_host_transfer()) {
      return;
    }
  }
  if (auto* channel_instr = DynCast<HloChannelInstruction>(cloned_instr)) {
    if (channel_instr->opcode() == HloOpcode::kSendDone ||
        channel_instr->opcode() == HloOpcode::kRecvDone) {
      auto* operand = channel_instr->operand(0);
      CHECK(operand->opcode() == HloOpcode::kSend ||
            operand->opcode() == HloOpcode::kRecv);
      channel_instr->set_channel_id(
          Cast<HloChannelInstruction>(operand)->channel_id());
      return;
    }
    if (channel_instr->channel_id()) {
      channel_instr->set_channel_id(next_channel_id++);
    }
  }
}

// Clones a chain of instructions from a move_info for backward movement, and
// returns the cloned of the last instruction in the chain. The last instruction
// in the chain is the collective instruction being pipelined and shouldn't be
// shared by multiple chains. As such, the last_cloned being returned shouldn't
// be nullptr.
template <typename Comp>
absl::StatusOr<HloInstruction*> CloneBackwardChain(
    Comp& target_computation, const WhileMoveInfo& move_info,
    InstructionMap& clone_map, int64_t loop_iter_idx, int64_t& next_channel_id,
    LoopVariantParameterInfo* loop_variant_parameter_info = nullptr) {
  std::vector<HloInstruction*> to_clone(move_info.formatting_ops.begin(),
                                        move_info.formatting_ops.end());
  to_clone.push_back(move_info.collective_to_move);
  HloInstruction* last_cloned = nullptr;
  for (auto* chain_op : to_clone) {
    // Do not clone a loop iterator or an op that is already cloned.
    if (IsLoopIterator(chain_op, loop_iter_idx) ||
        clone_map.contains(chain_op)) {
      continue;
    }
    auto new_operands = MapNewOperands(chain_op->operands(), clone_map);
    HloInstruction* cloned = target_computation.AddInstruction(
        chain_op->CloneWithNewOperands(chain_op->shape(), new_operands));
    TF_RETURN_IF_ERROR(UpdateControlDependencies(chain_op, cloned, clone_map));
    UpdateInstructionChannelId(cloned, next_channel_id);
    clone_map[chain_op] = cloned;
    last_cloned = cloned;
    if (loop_variant_parameter_info != nullptr &&
        chain_op->opcode() == HloOpcode::kGetTupleElement &&
        chain_op->operand(0)->opcode() == HloOpcode::kParameter &&
        chain_op->tuple_index() != loop_iter_idx) {
      loop_variant_parameter_info->push_back(
          std::make_pair(chain_op->tuple_index(), cloned));
    }
  }
  CHECK_NE(last_cloned, nullptr);
  return last_cloned;
}

// Analyzes a loop and collects information to understand if this transformation
// can be performed or if it should be performed (because there are collectives
// to optimize).
class WhileLoopAnalysis {
 public:
  explicit WhileLoopAnalysis(
      HloInstruction* while_instr, int64_t max_pipelining_per_loop,
      bool pipeline_use_tree, bool process_different_sized_options,
      std::optional<ConstantValue> known_start = std::nullopt)
      : while_(while_instr),
        loop_start_(known_start),
        max_pipelining_per_loop_(max_pipelining_per_loop),
        pipeline_use_tree_(pipeline_use_tree),
        process_different_sized_options_(process_different_sized_options) {}
  std::optional<ConstantValue> GetLoopIterationCount() const;
  std::optional<ConstantValue> GetLoopStart() const;
  std::optional<ConstantValue> GetLoopIncrement() const;
  const std::vector<WhileMoveInfo>& GetMoveInfos() const;
  std::optional<int64_t> GetLoopIterationIdx() const {
    return loop_iteration_idx_;
  }
  int64_t GetDUSIndex(const HloInstruction* dus) const;
  const absl::flat_hash_map<HloInstruction*, int64_t>& GetDUSIndices() const {
    return dus_index_map_;
  }
  int64_t GetUniqueDUSIndices() const { return dus_index_map_.size(); }
  int64_t GetMaxPipeliningPerLoop() const { return max_pipelining_per_loop_; }

  bool ComputeLoopStatistics();
  void CollectCollectivesToMove(
      int64_t level_to_operate_on,
      CollectivePipeliner::PipeliningDirection direction,
      HloPredicate should_process, HloPredicate acceptable_formatting,
      HloPredicate should_allow_loop_variant_parameter_in_chain =
          HloPredicateFalse);
  HloInstruction* while_loop_instruction() const { return while_; }

 private:
  HloInstruction* while_;
  std::optional<ConstantValue> loop_iteration_count_;
  std::optional<ConstantValue> loop_increment_;
  std::optional<ConstantValue> loop_start_;
  std::optional<ConstantValue> loop_bound_;
  std::optional<int64_t> loop_iteration_idx_;
  std::vector<WhileMoveInfo> move_infos_;
  absl::flat_hash_map<HloInstruction*, int64_t> dus_index_map_;
  absl::flat_hash_set<const HloInstruction*> invariant_loop_parameters_;
  int64_t max_pipelining_per_loop_;
  bool pipeline_use_tree_;
  bool process_different_sized_options_;
};

int64_t WhileLoopAnalysis::GetDUSIndex(const HloInstruction* dus) const {
  auto it = dus_index_map_.find(dus);
  CHECK(it != dus_index_map_.end());
  return it->second;
}

bool WhileLoopAnalysis::ComputeLoopStatistics() {
  // Loop iteration count already computed. This means a previous analysis as
  // been successful and we don't need to do anything.
  if (loop_iteration_count_) {
    return true;
  }
  std::optional<ParsedWhileLoop> parsed_loop =
      PatternMatchParseWhileLoop(while_);
  if (!parsed_loop || !parsed_loop->static_while_loop) {
    return false;
  }
  if (!IsSupportedLoopIndexType(
          while_->shape()
              .tuple_shapes(parsed_loop->static_while_loop->induction_var_index)
              .element_type())) {
    return false;
  }
  const HloInstruction* loop_root = while_->while_body()->root_instruction();
  const int64_t bitwidth = primitive_util::BitWidth(
      loop_root->operand(parsed_loop->static_while_loop->induction_var_index)
          ->shape()
          .element_type());
  const bool is_signed = primitive_util::IsSignedIntegralType(
      loop_root->operand(parsed_loop->static_while_loop->induction_var_index)
          ->shape()
          .element_type());
  const ConstantValue bound =
      is_signed ? ConstantValue::GetSigned(
                      parsed_loop->static_while_loop->loop_bound, bitwidth)
                : ConstantValue::GetUnsigned(
                      parsed_loop->static_while_loop->loop_bound, bitwidth);
  const ConstantValue increment =
      is_signed ? ConstantValue::GetSigned(
                      parsed_loop->static_while_loop->step_size, bitwidth)
                : ConstantValue::GetUnsigned(
                      parsed_loop->static_while_loop->step_size, bitwidth);
  loop_start_ =
      is_signed ? ConstantValue::GetSigned(
                      parsed_loop->static_while_loop->induction_var_init_value,
                      bitwidth)
                : ConstantValue::GetUnsigned(
                      parsed_loop->static_while_loop->induction_var_init_value,
                      bitwidth);

  auto iteration_range = bound.sub(*loop_start_);
  auto iter_count = iteration_range.div(increment);
  loop_iteration_count_ =
      iteration_range.mod(increment).gt(
          ConstantValue::GetZero(increment.GetBitwidth(), increment.IsSigned()))
          ? iter_count.add(ConstantValue::GetOne(increment.GetBitwidth(),
                                                 increment.IsSigned()))
          : iter_count;

  // Overflowing the iteration count.
  if (loop_iteration_count_->lt(iter_count)) {
    return false;
  }

  loop_bound_ = bound;
  loop_increment_ = increment;
  loop_iteration_idx_ = parsed_loop->static_while_loop->induction_var_index;

  VLOG(1) << "Bound: " << loop_bound_->ToString()
          << " Start: " << loop_start_->ToString()
          << " Increment: " << loop_increment_->ToString();
  // Simple invariant analysis. Just support arrays in the first nest of the
  // while() input.
  if (loop_root->opcode() == HloOpcode::kTuple) {
    for (int i = 0; i < loop_root->operand_count(); ++i) {
      if (loop_root->operand(i)->opcode() != HloOpcode::kGetTupleElement) {
        continue;
      }
      if (i != loop_root->operand(i)->tuple_index()) {
        continue;
      }
      invariant_loop_parameters_.insert(loop_root->operand(i));
    }
  }
  return true;
}

void WhileLoopAnalysis::CollectCollectivesToMove(
    int64_t level_to_operate_on,
    CollectivePipeliner::PipeliningDirection direction,
    HloPredicate should_process, HloPredicate acceptable_formatting,
    HloPredicate should_allow_loop_variant_parameter_in_chain) {
  move_infos_.clear();
  HloComputation* while_body = while_->while_body();
  const HloInstruction* loop_parameter =
      while_body->parameter_instructions()[0];

  // If the parameter tuple escapes then we can't guarantee that the replacement
  // for the next iteration is used by everybody unless we create a tuple with
  // the replacement that would probably limit overlap, so avoid this.
  if (absl::c_any_of(loop_parameter->users(), [](const HloInstruction* instr) {
        return instr->opcode() != HloOpcode::kGetTupleElement;
      })) {
    return;
  }

  if (absl::c_any_of(while_->users(), [](const HloInstruction* instr) {
        return instr->opcode() != HloOpcode::kGetTupleElement;
      })) {
    return;
  }
  absl::flat_hash_map<int64_t, int64_t> parameter_gtes_count;
  for (auto* user : loop_parameter->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    ++parameter_gtes_count[user->tuple_index()];
  }
  absl::flat_hash_map<const HloInstruction*, Range> index_ranges;
  absl::flat_hash_map<const HloInstruction*, int64_t>
      index_per_dyn_update_slice;
  std::optional<Range> index_range;
  if (loop_bound_) {
    // Compute the range of the index as "start + iteration_count * increment"
    index_range = Range{*loop_start_,
                        loop_start_->add(loop_iteration_count_
                                             ->sub(ConstantValue::GetOne(
                                                 loop_start_->GetBitwidth(),
                                                 loop_start_->IsSigned()))
                                             .mul(*loop_increment_)),
                        /*is_linear=*/true};
  }
  int64_t count = 0;
  absl::flat_hash_map<const HloInstruction*, int64_t> instruction_order;
  for (auto* instr : while_body->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kGetTupleElement) {
      if (index_range && instr->tuple_index() == 0) {
        index_ranges.insert({instr, *index_range});
      }
    }
    instruction_order[instr] = count++;
  }

  for (auto* instr : while_body->instructions()) {
    if (direction == CollectivePipeliner::PipeliningDirection::kForward &&
        (instr->operand_count() != 1 ||
         instr->shape().dimensions_size() !=
             instr->operand(0)->shape().dimensions_size())) {
      continue;
    }
    if (!should_process(instr)) {
      continue;
    }
    if (direction == CollectivePipeliner::PipeliningDirection::kForward ||
        direction == CollectivePipeliner::PipeliningDirection::kForwardSink) {
      auto [dyn_update, formatting_ops] = CheckStoreIntoSliceIsCompatible(
          instr, while_body, level_to_operate_on, pipeline_use_tree_,
          acceptable_formatting);
      if (dyn_update == nullptr) {
        VLOG(5)
            << "Skipping " << instr->ToString()
            << " because update users > 1 or single user is not the root of "
               "computation";
        continue;
      }
      std::optional<int64_t> sliced_dim = GetSlicedDimension(dyn_update);
      if (!sliced_dim.has_value()) {
        VLOG(5) << "Skipping " << instr->name()
                << " because couldn't find sliced dimension";
        continue;
      }
      if (direction == CollectivePipeliner::PipeliningDirection::kForwardSink &&
          (*sliced_dim != 0 || dyn_update->shape().dimensions(0) !=
                                   loop_iteration_count_->GetUnsignedValue())) {
        VLOG(5) << "Skipping " << instr->name()
                << " because number of iteration of the loop doesn't match "
                   "slices being inserted or slice dim is not 0. slice_dim = "
                << *sliced_dim << " loop count = "
                << loop_iteration_count_->GetUnsignedValue();
      }
      if (!process_different_sized_options_) {
        if (!formatting_ops.empty()) {
          if (instr->operand(0)->shape() != formatting_ops.back()->shape()) {
            continue;
          }
          auto dependencies_to_pipeline = CollectDependenciesToPipeline(
              instr, absl::MakeConstSpan(formatting_ops));
          bool skip_because_not_same_size = false;
          // If any instruction in the dependency chain is not of the same size
          // then we abort for this instruction.
          for (auto* dependency : dependencies_to_pipeline) {
            if (ShapeUtil::IsEffectiveScalar(dependency->shape())) {
              skip_because_not_same_size = true;
              break;
            }
          }
          if (skip_because_not_same_size) {
            continue;
          }
        } else if (instr->operand(0)->shape() != instr->shape()) {
          continue;
        }
      }
      const HloInstruction* to_insert_into = dyn_update->operand(0);
      if (level_to_operate_on == 0 &&
          (to_insert_into->opcode() != HloOpcode::kGetTupleElement ||
           to_insert_into->operand(0) != loop_parameter)) {
        VLOG(5) << "Skipping " << instr->name()
                << " because slice to insert into is not a GTE from input "
                   "parameter "
                << to_insert_into->ToString();
        continue;
      }
      if (dyn_update->user_count() != 1) {
        continue;
      }
      // If Level is > 0 then we already did our analysis in the previous
      // iteration for safeness of this index to transform.
      if (level_to_operate_on == 0) {
        if (to_insert_into->opcode() == HloOpcode::kGetTupleElement) {
          // GTE for this parameter is not CSEd. Abort because we don't analyze
          // every single use from other GTEs.
          if (parameter_gtes_count.at(to_insert_into->tuple_index()) != 1) {
            VLOG(5)
                << "Skipping " << instr->name()
                << " because there are multiple parameter GTEs for this slice";
            continue;
          }
        }
        HloInstruction* dyn_update_idx = dyn_update->mutable_operand(
            dyn_update->first_index_operand_number() + *sliced_dim);
        if (level_to_operate_on == 0 &&
            !CheckParameterUsageIsCompatible(to_insert_into, dyn_update,
                                             dyn_update_idx, *sliced_dim)) {
          VLOG(5)
              << "Skipping " << instr->name()
              << " because parameter usage doesn't follow the expected pattern";
          continue;
        }
        if (!AllIndicesConstantsExceptOne(
                dyn_update,
                dyn_update->first_index_operand_number() + *sliced_dim)) {
          VLOG(5) << "Skipping " << instr->name()
                  << " because update slicing doesn't match expectation";
          continue;
        }
        if (!CheckIndexIsMonotonic(dyn_update_idx, index_ranges)) {
          VLOG(5) << "Skipping " << instr->name()
                  << " because update index is not monotonic";
          continue;
        }
      }
      std::optional<int64_t> output_idx = FindOutputIndexForDynamicUpdateSlice(
          dyn_update, while_body->root_instruction());
      if (!output_idx.has_value()) {
        VLOG(5) << "Skipping " << instr->name()
                << " because couldn't find unique output index for insertion";
        continue;
      }
      auto merge_as_formatting =
          [this, &instruction_order](
              absl::flat_hash_map<const HloInstruction*, int64_t>::iterator it,
              HloInstruction* instr, HloInstruction* dyn_upd,
              absl::Span<HloInstruction* const> formatting_ops) {
            CHECK_EQ(move_infos_[it->second].dynamic_update_slice, dyn_upd)
                << "Not the same dynamic-update-slice for converging entry";
            absl::flat_hash_set<const HloInstruction*> existing_entry_instrs(
                move_infos_[it->second].formatting_ops.begin(),
                move_infos_[it->second].formatting_ops.end());
            existing_entry_instrs.insert(
                move_infos_[it->second].collective_to_move);
            // If instr is already in the set then this instruction is already
            // in formatting-ops of the other one, so its already pipelined.
            if (existing_entry_instrs.count(instr)) {
              return;
            }
            move_infos_[it->second].formatting_ops.push_back(instr);
            for (auto* op : formatting_ops) {
              if (!existing_entry_instrs.count(op)) {
                move_infos_[it->second].formatting_ops.push_back(op);
              }
            }
            absl::c_sort(move_infos_[it->second].formatting_ops,
                         [&](const HloInstruction* a, const HloInstruction* b) {
                           return instruction_order[a] < instruction_order[b];
                         });
          };
      auto it = index_per_dyn_update_slice.find(dyn_update);
      if (it != index_per_dyn_update_slice.end()) {
        // Merge stuff with existing entry.
        merge_as_formatting(it, instr, dyn_update, formatting_ops);
        continue;
      }
      index_per_dyn_update_slice[dyn_update] = move_infos_.size();
      move_infos_.push_back({instr, dyn_update, std::move(formatting_ops),
                             *sliced_dim, *output_idx});
    } else {
      CHECK_EQ(direction, CollectivePipeliner::PipeliningDirection::kBackward);
      auto chain_collected = CollectChainsToPushBackwards(
          instr, *loop_iteration_idx_, while_body, level_to_operate_on,
          invariant_loop_parameters_,
          should_allow_loop_variant_parameter_in_chain);
      if (!chain_collected.has_value()) {
        VLOG(5) << "Skipping " << instr->name()
                << " because didn't find compatible slice of parameter";
        continue;
      }
      move_infos_.push_back(
          WhileMoveInfo{instr, nullptr, std::move(*chain_collected), -1, -1});
    }
    if (move_infos_.size() >= max_pipelining_per_loop_) {
      break;
    }
  }
  if (direction != CollectivePipeliner::PipeliningDirection::kForward) {
    return;
  }
  dus_index_map_.clear();
  for (auto& to_move : move_infos_) {
    HloInstruction* dus_index = to_move.dynamic_update_slice->mutable_operand(
        to_move.dynamic_update_slice->first_index_operand_number() +
        to_move.sliced_idx);
    auto it = dus_index_map_.find(dus_index);
    int64_t dus_index_tuple_position = dus_index_map_.size();
    if (it != dus_index_map_.end()) {
      dus_index_tuple_position = it->second;
    } else {
      dus_index_map_[dus_index] = dus_index_tuple_position;
    }
  }
}

std::optional<ConstantValue> WhileLoopAnalysis::GetLoopIterationCount() const {
  return loop_iteration_count_;
}

std::optional<ConstantValue> WhileLoopAnalysis::GetLoopStart() const {
  return loop_start_;
}

std::optional<ConstantValue> WhileLoopAnalysis::GetLoopIncrement() const {
  return loop_increment_;
}

const std::vector<WhileMoveInfo>& WhileLoopAnalysis::GetMoveInfos() const {
  return move_infos_;
}

// Simple loop invariant check. If the data doesn't depend in any way from the
// input tuple consider it loop invariant.
// TODO: Extract something more complete in a separate file. This is current
// quite custom to the transformation here.
bool IsLoopInvariant(
    const HloInstruction* instr,
    absl::flat_hash_map<const HloInstruction*, bool>& invariant_cache) {
  auto it = invariant_cache.find(instr);
  if (it != invariant_cache.end()) {
    return it->second;
  }
  // This performs a post order iteration of the graph. First element is the
  // current HLO in the stack and the second parameter is the number of operands
  // to still visit before visiting the HLO itself.
  std::vector<std::pair<const HloInstruction*, int>> stack(
      1, std::make_pair(instr, 0));
  absl::flat_hash_set<const HloInstruction*> visited;
  while (!stack.empty()) {
    auto& current = stack.back();
    invariant_cache[std::get<0>(current)] = true;
    if (std::get<0>(current)->HasSideEffect() ||
        std::get<0>(current)->opcode() == HloOpcode::kParameter) {
      invariant_cache[std::get<0>(current)] = false;
      stack.pop_back();
      continue;
    }
    if (std::get<0>(current)->operands().empty()) {
      invariant_cache[std::get<0>(current)] = true;
      stack.pop_back();
      continue;
    }
    if (std::get<1>(current) > 0) {
      auto* current_operand =
          std::get<0>(current)->operand(std::get<1>(current) - 1);
      auto cop_it = invariant_cache.find(current_operand);
      CHECK(cop_it != invariant_cache.end())
          << "Entry expected to be populated";
      if (!cop_it->second) {
        invariant_cache[std::get<0>(current)] = false;
        stack.pop_back();
        continue;
      }
    }
    if (std::get<0>(current)->operand_count() == std::get<1>(current)) {
      stack.pop_back();
      continue;
    }
    auto* next_operand = std::get<0>(current)->operand(std::get<1>(current)++);
    auto op_it = invariant_cache.find(next_operand);
    if (op_it == invariant_cache.end()) {
      stack.push_back(std::make_pair(next_operand, 0));
    } else if (!op_it->second) {
      invariant_cache[next_operand] &= op_it->second;
    }
  }
  it = invariant_cache.find(instr);
  CHECK(it != invariant_cache.end())
      << "We should have computed \"instr\" value";
  return it->second;
}

// Compute a shape that can hold a concatenation of tensors of shape base_shape.
Shape ComputeFullOutputShape(const WhileMoveInfo& move_info,
                             const Shape& base_shape) {
  return ShapeUtil::PrependMajorDimension(
      move_info.dynamic_update_slice->operand(0)
          ->shape()
          .dimensions()[move_info.sliced_idx],
      base_shape);
}

// Create zero of base type ptype and broadcast it to shape.
HloInstruction* CreateZero(HloComputation* comp, const Shape& shape,
                           PrimitiveType ptype) {
  if (shape.dimensions_size() == 0) {
    return comp->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(ptype)));
  }
  HloInstruction* zero_constant =
      comp->AddInstruction(HloInstruction::CreateBroadcast(
          shape,
          comp->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::Zero(ptype))),
          {}));
  return zero_constant;
}

}  // namespace

// Function that does the work of pushing forward instructions that have been
// determined that can be pipelined. Rough transformation: while (i < LAYERS) {
//   p0 = param(0)
//   p1 = param(1)
//   x = computation(p0)
//   xg = all-reduce(x)
//   y = computation(p1)
//   yg = all-reduce(y)
// }
//
// to
//
// x_prev = computation(p0)
// y_prev = computation(p1)
// i = i + 1
// while (i < LAYERS, x_prev, y_prev) {
//   p0 = param(0)
//   p1 = param(1)
//   xg = all-reduce(x_prev)
//   yg = all-reduce(y_prev)
//   x = computation(p0)
//   y = computation(p1)
//   x_prev = x
//   y_prev = y
// }
// xg_last = all-reduce(x)
// yg_last = all-reduce(y)
Status TransformLoopForward(const WhileLoopAnalysis& loop_analysis,
                            bool insert_non_alias_custom_call,
                            int64_t level_to_operate_on, bool pipeline_use_tree,
                            bool process_different_sized_ops,
                            HloPredicate should_process,
                            HloPredicate acceptable_formatting,
                            HloPredicate reuse_output_buffer,
                            int64_t& next_channel_id) {
  // Defining some maps/sets to keep track of instructions duplicated.
  InstructionMap while_body_to_peeled;
  absl::flat_hash_set<HloInstruction*> to_skip_set;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> formatting_map;
  absl::flat_hash_map<HloInstruction*, int64_t> is_output_instruction;
  std::vector<int64_t> moves_requiring_special_output;
  int64_t count = 0;
  // Add all-reduces to duplicate into a set.
  for (auto& to_move : loop_analysis.GetMoveInfos()) {
    to_skip_set.insert(to_move.collective_to_move);
    if (!to_move.formatting_ops.empty()) {
      formatting_map[to_move.formatting_ops.back()] =
          to_move.collective_to_move;
    }
    const Shape& output_shape = to_move.formatting_ops.empty()
                                    ? to_move.collective_to_move->shape()
                                    : to_move.formatting_ops.back()->shape();
    if (!reuse_output_buffer(to_move.collective_to_move) ||
        output_shape != to_move.collective_to_move->operand(0)->shape()) {
      moves_requiring_special_output.push_back(count);
      to_skip_set.insert(to_move.dynamic_update_slice);
    }
    ++count;
  }
  // Map get-tuple-elements() inside of the loop with elements passed to the
  // tuple that is the "init" of the loop.
  HloInstruction* while_loop = loop_analysis.while_loop_instruction();
  HloComputation* while_body = while_loop->while_body();
  CHECK_EQ(while_body->parameter_instructions().size(), 1)
      << "Expected only one parameter";
  HloInstruction* loop_parameter = while_body->parameter_instructions()[0];
  HloInstruction* loop_init = while_loop->mutable_operand(0);
  const int64_t initial_inputs = loop_init->operand_count();
  while_body_to_peeled[loop_parameter] = loop_init;
  for (auto* user : loop_parameter->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement)
        << "Expected only get-tuple-elements as users";
    while_body_to_peeled[user] =
        loop_init->mutable_operand(user->tuple_index());
  }
  CHECK_EQ(while_body->root_instruction()->opcode(), HloOpcode::kTuple);
  for (int i = 0; i < while_body->root_instruction()->operand_count(); ++i) {
    is_output_instruction[while_body->root_instruction()->mutable_operand(i)] =
        i;
  }

  // Collect the new parameter shapes with the additional state for the indices
  // and construct new operand vectors for the init of the new loop and its root
  // instruction.
  HloComputation* loop_computation = while_loop->parent();
  std::vector<HloInstruction*> new_init_operands;
  std::vector<Shape> new_parameter_shapes;
  std::vector<HloInstruction*> new_root_operands;
  const int64_t operands_indices_count =
      loop_init->operand_count() + loop_analysis.GetUniqueDUSIndices();
  const int64_t new_loop_tuple_operand_count =
      operands_indices_count + moves_requiring_special_output.size();
  new_parameter_shapes.resize(new_loop_tuple_operand_count);
  new_root_operands.resize(new_loop_tuple_operand_count);
  new_init_operands.resize(new_loop_tuple_operand_count);
  for (int i = 0; i < loop_parameter->shape().tuple_shapes().size(); ++i) {
    new_parameter_shapes[i] = loop_parameter->shape().tuple_shapes(i);
    new_root_operands[i] = while_body->root_instruction()->mutable_operand(i);
    new_init_operands[i] = loop_init->mutable_operand(i);
  }

  // Duplicate the loop body into the loop parent computation, so that the first
  // iteration happens there.
  for (auto* instr : while_body->MakeInstructionPostOrder()) {
    if (instr == loop_parameter) {
      continue;
    }
    if (ContainsKey(to_skip_set, instr)) {
      auto it = while_body_to_peeled.find(instr->operand(0));
      CHECK(it != while_body_to_peeled.end());
      HloInstruction* passthrough_operand = it->second;
      while_body_to_peeled[instr] = passthrough_operand;
      continue;
    }
    auto formatting_it = formatting_map.find(instr);
    if (formatting_it != formatting_map.end()) {
      auto it = while_body_to_peeled.find(formatting_it->second);
      CHECK(it != while_body_to_peeled.end());
      HloInstruction* passthrough_operand = it->second;
      while_body_to_peeled[instr] = passthrough_operand;
      continue;
    }
    std::vector<HloInstruction*> new_operands =
        MapNewOperands(instr->operands(), while_body_to_peeled);
    HloInstruction* cloned_instr = loop_computation->AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands));
    TF_RETURN_IF_ERROR(
        UpdateControlDependencies(instr, cloned_instr, while_body_to_peeled));
    UpdateInstructionChannelId(cloned_instr, next_channel_id);
    while_body_to_peeled[instr] = cloned_instr;
    auto output_it = is_output_instruction.find(instr);
    if (output_it != is_output_instruction.end()) {
      new_init_operands[output_it->second] = cloned_instr;
    }
  }

  // Add indices to access the slices for the previous iteration to the
  // loop state. Indices used multiple times for multiple slices have been
  // deduped.
  for (auto& dus : loop_analysis.GetDUSIndices()) {
    new_parameter_shapes[dus.second + initial_inputs] = dus.first->shape();
    new_root_operands[dus.second + initial_inputs] = dus.first;
    new_init_operands[dus.second + initial_inputs] =
        while_body_to_peeled[dus.first];
  }
  absl::flat_hash_map<int64_t, int64_t> moves_requiring_special_output_to_idx;
  for (int i = 0; i < moves_requiring_special_output.size(); ++i) {
    HloInstruction* collective =
        loop_analysis.GetMoveInfos()[moves_requiring_special_output[i]]
            .collective_to_move;
    moves_requiring_special_output_to_idx[moves_requiring_special_output[i]] =
        operands_indices_count + i;
    new_parameter_shapes[operands_indices_count + i] =
        collective->operand(0)->shape();
    new_root_operands[operands_indices_count + i] =
        collective->mutable_operand(0);
    new_init_operands[operands_indices_count + i] =
        while_body_to_peeled[collective->mutable_operand(0)];
  }

  for (auto& move_info : loop_analysis.GetMoveInfos()) {
    auto pipelined_instrs = CollectDependenciesToPipeline(
        move_info.collective_to_move, absl::MakeSpan(move_info.formatting_ops));
    for (auto* pipelined : pipelined_instrs) {
      is_output_instruction[pipelined] = new_init_operands.size();
      new_parameter_shapes.push_back(pipelined->shape());
      new_root_operands.push_back(pipelined);
      new_init_operands.push_back(while_body_to_peeled[pipelined]);
    }
  }
  // Clone new loop computations (cond and body) and create the new loop
  // instruction and connect it to the users/operands of the old loop.
  Shape loop_state_shape = ShapeUtil::MakeTupleShape(new_parameter_shapes);
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  InstructionMap pipelined_values_map_inloop;
  InstructionMap pipelined_values_map_outloop;
  replacements[loop_parameter] = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape(new_parameter_shapes), "loop_peel_param");
  replacements[while_loop->while_condition()->parameter_instructions()[0]] =
      HloInstruction::CreateParameter(
          0, ShapeUtil::MakeTupleShape(new_parameter_shapes),
          "loop_peel_cond_param");
  replacements[while_body->root_instruction()] =
      HloInstruction::CreateTuple(new_root_operands);
  HloComputation* new_while_condition =
      loop_computation->parent()->AddEmbeddedComputation(
          while_loop->while_condition()->CloneWithReplacements(&replacements));
  HloComputation* new_while_body =
      loop_computation->parent()->AddEmbeddedComputation(
          while_body->CloneWithReplacements(&replacements));
  HloInstruction* new_init = loop_computation->AddInstruction(
      HloInstruction::CreateTuple(new_init_operands));
  while_body_to_peeled[while_body->root_instruction()] = new_init;
  TF_RETURN_IF_ERROR(UpdateControlDependencies(while_body->root_instruction(),
                                               new_init, while_body_to_peeled));
  HloInstruction* new_while_loop =
      loop_computation->AddInstruction(HloInstruction::CreateWhile(
          loop_state_shape, new_while_condition, new_while_body, new_init));
  TF_RETURN_IF_ERROR(
      while_loop->ReplaceAllUsesWithDifferentShape(new_while_loop));
  TF_RETURN_IF_ERROR(
      loop_computation->RemoveInstructionAndUnusedOperands(while_loop));
  // Run WhileLoopAnalysis again on the new loop to collect the position of the
  // all-reduces in the new cloned loop as they aren't the same of the old.
  // Loop analysis should result exactly the same, because the loop is the same
  // except some new scalar unused parameters added at the end.
  WhileLoopAnalysis new_loop_analysis(
      new_while_loop, loop_analysis.GetMaxPipeliningPerLoop(),
      pipeline_use_tree, process_different_sized_ops,
      loop_analysis.GetLoopStart()->add(*loop_analysis.GetLoopIncrement()));
  new_loop_analysis.ComputeLoopStatistics();
  new_loop_analysis.CollectCollectivesToMove(
      level_to_operate_on, CollectivePipeliner::PipeliningDirection::kForward,
      should_process, acceptable_formatting);
  CHECK_EQ(new_loop_analysis.GetMoveInfos().size(),
           loop_analysis.GetMoveInfos().size());
  for (int64_t i = new_loop_tuple_operand_count;
       i < new_parameter_shapes.size(); ++i) {
    HloInstruction* pipelined_value_load_inloop =
        new_while_body->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_while_body->parameter_instruction(0), i));
    HloInstruction* pipelined_value_load_outloop =
        loop_computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(new_while_loop, i));
    pipelined_values_map_inloop[new_while_body->root_instruction()->operand(
        i)] = pipelined_value_load_inloop;
    pipelined_values_map_outloop[new_while_body->root_instruction()->operand(
        i)] = pipelined_value_load_outloop;
  }
  auto insert_slice = [](HloInstruction* to_insert, int64_t index_position,
                         int64_t num_indices, HloInstruction* dus_index,
                         HloInstruction* base) {
    HloComputation* computation = to_insert->parent();
    HloInstruction* zero =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(dus_index->shape().element_type())));
    std::vector<HloInstruction*> indices(num_indices, zero);
    indices[index_position] = dus_index;
    return computation->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
        base->shape(), base, to_insert, indices));
  };
  auto process_slice =
      [&next_channel_id, insert_non_alias_custom_call, level_to_operate_on](
          HloInstruction* stacked_data,
          const InstructionMap& pipelined_values_map,
          const WhileMoveInfo& move_info) -> absl::StatusOr<HloInstruction*> {
    HloInstruction* processed = stacked_data->parent()->AddInstruction(
        move_info.collective_to_move->CloneWithNewOperands(
            move_info.collective_to_move->shape(), {stacked_data}));
    UpdateInstructionChannelId(processed, next_channel_id);
    if (insert_non_alias_custom_call) {
      HloInstruction* level =
          stacked_data->parent()->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0(level_to_operate_on + 1)));
      processed = stacked_data->parent()->AddInstruction(
          HloInstruction::CreateCustomCall(
              processed->shape(), {processed, level},
              CollectivePipeliner::kInsertedByPreviousStep));
    }

    InstructionMap cloned_map = pipelined_values_map;
    cloned_map[move_info.collective_to_move] = processed;
    for (auto* formatting_op : move_info.formatting_ops) {
      auto new_operands = MapNewOperands(formatting_op->operands(), cloned_map);
      processed = stacked_data->parent()->AddInstruction(
          formatting_op->CloneWithNewOperands(formatting_op->shape(),
                                              new_operands));
      cloned_map[formatting_op] = processed;
    }
    return processed;
  };
  auto extract_and_process_slice =
      [&process_slice](
          HloInstruction* stacked_data, HloInstruction* data_to_slice,
          const WhileMoveInfo& move_info,
          const InstructionMap& pipelined_values_map,
          HloInstruction* dus_index) -> absl::StatusOr<HloInstruction*> {
    HloComputation* computation = stacked_data->parent();
    const Shape& slice_target_shape =
        move_info.collective_to_move->operand(0)->shape();
    HloInstruction* sliced_data = data_to_slice;
    PrimitiveType element_type =
        move_info.dynamic_update_slice
            ->operand(
                move_info.dynamic_update_slice->first_index_operand_number() +
                move_info.sliced_idx)
            ->shape()
            .element_type();
    HloInstruction* zero = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(element_type)));
    std::vector<HloInstruction*> indices(
        move_info.dynamic_update_slice->operand_count() -
            move_info.dynamic_update_slice->first_index_operand_number(),
        zero);
    indices[move_info.sliced_idx] = dus_index;
    if (slice_target_shape != data_to_slice->shape()) {
      // Slice matrix.
      absl::InlinedVector<int64_t, 4> dynamic_slice_sizes;
      dynamic_slice_sizes.reserve(slice_target_shape.dimensions_size());
      for (int i = 0; i < slice_target_shape.dimensions_size(); ++i) {
        dynamic_slice_sizes.push_back(slice_target_shape.dimensions(i));
      }
      sliced_data =
          computation->AddInstruction(HloInstruction::CreateDynamicSlice(
              slice_target_shape, data_to_slice, indices, dynamic_slice_sizes));
    }
    TF_ASSIGN_OR_RETURN(
        sliced_data,
        process_slice(sliced_data, pipelined_values_map, move_info));
    return computation->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
        move_info.dynamic_update_slice->shape(), stacked_data, sliced_data,
        indices));
  };
  for (int i = 0; i < new_loop_analysis.GetMoveInfos().size(); ++i) {
    auto& move_info = new_loop_analysis.GetMoveInfos()[i];
    std::vector<HloInstruction*> loop_output_to_replace;
    HloInstruction* parameter_instr =
        new_while_body->parameter_instructions()[0];
    for (auto* user : new_while_loop->users()) {
      if (user->tuple_index() != move_info.output_idx) {
        continue;
      }
      loop_output_to_replace.push_back(user);
    }
    const HloInstruction* dus_index_curr_iteration =
        move_info.dynamic_update_slice->operand(
            move_info.dynamic_update_slice->first_index_operand_number() +
            move_info.sliced_idx);
    const int64_t offset_for_index =
        new_loop_analysis.GetDUSIndex(dus_index_curr_iteration) +
        initial_inputs;
    Shape index_shape = dus_index_curr_iteration->shape();
    HloInstruction* input_dus_idx =
        new_while_body->AddInstruction(HloInstruction::CreateGetTupleElement(
            index_shape, parameter_instr, offset_for_index));
    if (insert_non_alias_custom_call) {
      HloInstruction* level =
          new_while_body->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0(level_to_operate_on + 1)));
      input_dus_idx =
          new_while_body->AddInstruction(HloInstruction::CreateCustomCall(
              index_shape, {input_dus_idx, level},
              CollectivePipeliner::kInsertedByPreviousStep));
    }
    HloInstruction* output_dus_idx =
        loop_computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            index_shape, new_while_loop, offset_for_index));
    HloInstruction* input_stacked_data =
        move_info.dynamic_update_slice->mutable_operand(0);
    HloInstruction* output_stacked_data =
        loop_computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            move_info.dynamic_update_slice->shape(), new_while_loop,
            move_info.output_idx));
    HloInstruction* input_data_to_slice = input_stacked_data;
    HloInstruction* output_data_to_slice = output_stacked_data;
    auto it = moves_requiring_special_output_to_idx.find(i);
    if (it != moves_requiring_special_output_to_idx.end()) {
      input_data_to_slice =
          new_while_body->AddInstruction(HloInstruction::CreateGetTupleElement(
              move_info.collective_to_move->operand(0)->shape(),
              parameter_instr, it->second));
      output_data_to_slice = loop_computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(
              move_info.collective_to_move->operand(0)->shape(), new_while_loop,
              it->second));
    }
    TF_ASSIGN_OR_RETURN(input_stacked_data,
                        extract_and_process_slice(
                            input_stacked_data, input_data_to_slice, move_info,
                            pipelined_values_map_inloop, input_dus_idx));
    TF_ASSIGN_OR_RETURN(
        output_stacked_data,
        extract_and_process_slice(output_stacked_data, output_data_to_slice,
                                  move_info, pipelined_values_map_outloop,
                                  output_dus_idx));
    auto replace_instructions_with =
        [](absl::Span<HloInstruction*> to_replace_instrs,
           HloInstruction* new_instr) {
          for (auto* to_replace : to_replace_instrs) {
            HloComputation* computation = to_replace->parent();
            TF_RETURN_IF_ERROR(to_replace->ReplaceAllUsesWith(new_instr));
            TF_RETURN_IF_ERROR(
                computation->RemoveInstructionAndUnusedOperands(to_replace));
          }
          return OkStatus();
        };
    auto* new_peeled_dus = input_stacked_data;
    if (it == moves_requiring_special_output_to_idx.end()) {
      new_peeled_dus = insert_slice(
          move_info.collective_to_move->mutable_operand(0),
          move_info.sliced_idx,
          move_info.dynamic_update_slice->operand_count() -
              move_info.dynamic_update_slice->first_index_operand_number(),
          move_info.dynamic_update_slice->mutable_operand(
              move_info.dynamic_update_slice->first_index_operand_number() +
              move_info.sliced_idx),
          input_stacked_data);
    }
    TF_RETURN_IF_ERROR(
        move_info.dynamic_update_slice->ReplaceAllUsesWith(new_peeled_dus));
    TF_RETURN_IF_ERROR(new_while_body->RemoveInstructionAndUnusedOperands(
        move_info.dynamic_update_slice));
    TF_RETURN_IF_ERROR(replace_instructions_with(
        absl::MakeSpan(loop_output_to_replace), output_stacked_data));
  }
  TF_RETURN_IF_ERROR(loop_computation->parent()->RemoveUnusedComputations());
  return OkStatus();
}

// Function that does the work of sinking all-reduces the output of which are
// concatenated after the loop. Rough transformation: while (i < LAYERS) {
//   p0 = param(0)
//   p1 = param(1)
//   x = computation(p0)
//   xg = all-reduce(x)
//   y = computation(p1)
//   yg = all-reduce(y)
// }
//
// to
//
// x_prev = computation(p0)
// y_prev = computation(p1)
// i = i + 1
// while (i < LAYERS, x_all, y_all) {
//   p0 = param(0)
//   p1 = param(1)
//   x = computation(p0)
//   y = computation(p1)
//   x_all = append(x)
//   y_all = append(y)
// }
// xg_all = all-reduce(x_all)
// yg_all = all-reduce(y_all)
Status TransformLoopForwardSink(const WhileLoopAnalysis& loop_analysis,
                                bool insert_non_alias_custom_call,
                                int64_t level_to_operate_on,
                                bool pipeline_use_tree,
                                bool process_different_sized_ops,
                                HloPredicate should_process,
                                int64_t& next_channel_id) {
  // Defining some maps/sets to keep track of instructions duplicated.
  absl::flat_hash_map<HloInstruction*, int64_t> is_output_instruction;
  absl::flat_hash_map<const HloInstruction*, bool> invariant_cache;

  // Map get-tuple-elements() inside of the loop with elements passed to the
  // tuple that is the "init" of the loop.
  HloInstruction* while_loop = loop_analysis.while_loop_instruction();
  HloComputation* while_body = while_loop->while_body();
  CHECK_EQ(while_body->parameter_instructions().size(), 1)
      << "Expected only one parameter";
  HloInstruction* loop_parameter = while_body->parameter_instructions()[0];
  HloInstruction* loop_init = while_loop->mutable_operand(0);
  CHECK_EQ(while_body->root_instruction()->opcode(), HloOpcode::kTuple);
  for (int i = 0; i < while_body->root_instruction()->operand_count(); ++i) {
    is_output_instruction[while_body->root_instruction()->mutable_operand(i)] =
        i;
  }

  // Collect the new parameter shapes with the additional state for the indices
  // and construct new operand vectors for the init of the new loop and its root
  // instruction.
  HloComputation* loop_computation = while_loop->parent();
  HloComputation* body_computation = while_loop->while_body();
  std::vector<HloInstruction*> new_init_operands;
  std::vector<Shape> new_parameter_shapes;
  std::vector<HloInstruction*> new_root_operands;
  absl::flat_hash_set<int64_t> indices_to_insert;
  const int64_t operands_indices_count = loop_init->operand_count();
  const int64_t new_loop_tuple_operand_count = operands_indices_count;
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  new_parameter_shapes.resize(new_loop_tuple_operand_count);
  new_root_operands.resize(new_loop_tuple_operand_count);
  new_init_operands.resize(new_loop_tuple_operand_count);
  absl::flat_hash_set<int64_t> original_to_move_indices;
  // Initialize data structures with information about the outputs that need to
  // be sunk.
  for (auto& to_move : loop_analysis.GetMoveInfos()) {
    HloInstruction* collective = to_move.collective_to_move;
    Shape shape =
        ComputeFullOutputShape(to_move, collective->operand(0)->shape());
    new_init_operands[to_move.output_idx] =
        CreateZero(loop_computation, shape, shape.element_type());
    new_parameter_shapes[to_move.output_idx] = shape;
    original_to_move_indices.insert(to_move.output_idx);
    indices_to_insert.insert(to_move.output_idx);
    new_root_operands[to_move.output_idx] = collective->mutable_operand(0);
  }
  // Initialize the data structures for output indices that aren't modified.
  for (int i = 0; i < loop_parameter->shape().tuple_shapes().size(); ++i) {
    if (original_to_move_indices.contains(i)) {
      continue;
    }
    new_parameter_shapes[i] = loop_parameter->shape().tuple_shapes(i);
    new_init_operands[i] = loop_init->mutable_operand(i);
    new_root_operands[i] = while_body->root_instruction()->mutable_operand(i);
  }

  // Collect instructions that are necessary for the execution of the sunk
  // instructions. If they are loop invariant they are stored as is, otherwise
  // the version for each iteration is accumulated in a buffer.
  for (auto& move_info : loop_analysis.GetMoveInfos()) {
    auto pipelined_instrs = CollectDependenciesToPipeline(
        move_info.collective_to_move, absl::MakeSpan(move_info.formatting_ops));
    for (auto* pipelined : pipelined_instrs) {
      if (pipelined->opcode() == HloOpcode::kConstant) {
        continue;
      }
      const bool is_loop_invariant =
          IsLoopInvariant(pipelined, invariant_cache);
      is_output_instruction[pipelined] = new_init_operands.size();
      if (is_loop_invariant) {
        new_parameter_shapes.push_back(pipelined->shape());
        new_init_operands.push_back(
            CreateZero(loop_computation, pipelined->shape(),
                       pipelined->shape().element_type()));
        new_root_operands.push_back(pipelined);
        continue;
      }
      Shape expanded_shape =
          ComputeFullOutputShape(move_info, pipelined->shape());
      new_parameter_shapes.push_back(expanded_shape);
      new_init_operands.push_back(CreateZero(loop_computation, expanded_shape,
                                             expanded_shape.element_type()));
      indices_to_insert.insert(new_root_operands.size());
      Shape extra_trivial_dim_shape =
          ShapeUtil::PrependMajorDimension(1, pipelined->shape());
      HloInstruction* reshaped = body_computation->AddInstruction(
          HloInstruction::CreateReshape(extra_trivial_dim_shape, pipelined));
      std::vector<HloInstruction*> indices(
          expanded_shape.dimensions_size(),
          CreateZero(body_computation,
                     move_info.dynamic_update_slice->index_shapes()[0],
                     move_info.dynamic_update_slice->index_shapes()[0]
                         .element_type()));
      indices[0] = move_info.dynamic_update_slice->index_operands()[0];
      HloInstruction* input =
          body_computation->AddInstruction(HloInstruction::CreateCustomCall(
              expanded_shape,
              {body_computation->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0((int32_t)new_root_operands.size())))},
              "PlaceHolder"));
      reshaped = body_computation->AddInstruction(
          HloInstruction::CreateDynamicUpdateSlice(expanded_shape, input,
                                                   reshaped, indices));
      new_root_operands.push_back(reshaped);
    }
  }
  std::unique_ptr<HloInstruction> new_parameter =
      HloInstruction::CreateParameter(
          0, ShapeUtil::MakeTupleShape(new_parameter_shapes),
          absl::StrCat("sink_", loop_parameter->name()));
  // Insert inputs to the collective we are sinking in slices for the loop.
  for (auto& to_move : loop_analysis.GetMoveInfos()) {
    if (!indices_to_insert.contains(to_move.output_idx)) {
      continue;
    }
    HloInstruction* to_insert =
        body_computation->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::PrependMajorDimension(
                1, new_root_operands[to_move.output_idx]->shape()),
            new_root_operands[to_move.output_idx]));
    Shape expanded_shape = ComputeFullOutputShape(
        to_move, new_root_operands[to_move.output_idx]->shape());
    HloInstruction* input =
        body_computation->AddInstruction(HloInstruction::CreateCustomCall(
            expanded_shape,
            {body_computation->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0((int32_t)to_move.output_idx)))},
            "PlaceHolder"));
    std::vector<HloInstruction*> indices(
        expanded_shape.dimensions_size(),
        CreateZero(
            body_computation, to_move.dynamic_update_slice->index_shapes()[0],
            to_move.dynamic_update_slice->index_shapes()[0].element_type()));
    indices[0] = to_move.dynamic_update_slice->index_operands()[0];
    to_insert = body_computation->AddInstruction(
        HloInstruction::CreateDynamicUpdateSlice(expanded_shape, input,
                                                 to_insert, indices));
    new_root_operands[to_move.output_idx] = to_insert;
  }
  std::unique_ptr<HloInstruction> new_root_instr =
      HloInstruction::CreateTuple(new_root_operands);
  // Mark for removal (by setting replacement entry to nullptr) the users of the
  // old parameters we are replacing for the loops. All the computation tree
  // for those should be not used in the new loop.
  for (auto* p_user : body_computation->parameter_instructions()[0]->users()) {
    CHECK_EQ(p_user->opcode(), HloOpcode::kGetTupleElement);
    const int64_t tuple_idx = p_user->tuple_index();
    if (!indices_to_insert.contains(tuple_idx)) {
      continue;
    }
    replacements[p_user] =
        HloInstruction::CreateGetTupleElement(new_parameter.get(), tuple_idx);
    std::vector<HloInstruction*> stack(p_user->users().begin(),
                                       p_user->users().end());
    while (!stack.empty()) {
      auto* u = stack.back();
      stack.pop_back();
      replacements[u] = nullptr;
      for (auto* user : u->users()) {
        if (user == body_computation->root_instruction()) {
          continue;
        }
        stack.push_back(user);
      }
    }
  }
  replacements[body_computation->parameter_instruction(0)] =
      std::move(new_parameter);
  replacements[body_computation->root_instruction()] =
      std::move(new_root_instr);
  replacements[while_loop->while_condition()->parameter_instruction(0)] =
      HloInstruction::CreateParameter(
          0, ShapeUtil::MakeTupleShape(new_parameter_shapes),
          absl::StrCat(
              "sink_",
              while_loop->while_condition()->parameter_instruction(0)->name()));
  // Clone and create new loop.
  HloInstruction* new_init = loop_computation->AddInstruction(
      HloInstruction::CreateTuple(new_init_operands));
  HloComputation* cloned_body =
      body_computation->parent()->AddEmbeddedComputation(
          body_computation->CloneWithReplacements(&replacements));
  HloComputation* cloned_cond =
      body_computation->parent()->AddEmbeddedComputation(
          while_loop->while_condition()->CloneWithReplacements(&replacements));
  for (int64_t i = 0; i < cloned_body->root_instruction()->operand_count();
       ++i) {
    HloInstruction* output =
        cloned_body->root_instruction()->mutable_operand(i);
    if (output->opcode() != HloOpcode::kDynamicUpdateSlice) {
      continue;
    }
    if (!output->operand(0)->IsCustomCall("PlaceHolder")) {
      continue;
    }
    auto idx = Cast<HloConstantInstruction>(output->operand(0)->operand(0))
                   ->literal()
                   .GetFirstInteger();
    auto* new_param =
        cloned_body->AddInstruction(HloInstruction::CreateGetTupleElement(
            output->shape(), cloned_body->parameter_instruction(0), *idx));
    HloInstruction* old_operand_param = output->mutable_operand(0);
    TF_RETURN_IF_ERROR(output->ReplaceOperandWith(0, new_param));
    TF_RETURN_IF_ERROR(
        old_operand_param->parent()->RemoveInstruction(old_operand_param));
    if (insert_non_alias_custom_call && original_to_move_indices.contains(i)) {
      auto* old_operand = output->mutable_operand(1);
      auto* custom_call =
          cloned_body->AddInstruction(HloInstruction::CreateCustomCall(
              old_operand->shape(), {old_operand},
              /*custom_call_target=*/CollectivePipeliner::kSunkByPreviousStep));
      TF_RETURN_IF_ERROR(output->ReplaceOperandWith(1, custom_call));
    }
  }
  HloInstruction* new_while =
      loop_computation->AddInstruction(HloInstruction::CreateWhile(
          new_init->shape(), cloned_cond, cloned_body, new_init));
  std::vector<HloInstruction*> new_output_tuple;
  new_output_tuple.resize(new_root_operands.size(), nullptr);
  // Reproduce computation to the output after the loop on the full shape.
  for (auto& to_move : loop_analysis.GetMoveInfos()) {
    InstructionMap pipelined_map;
    HloInstruction* to_sink = loop_computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(new_while, to_move.output_idx));
    const int64_t new_dim_limit =
        to_move.dynamic_update_slice->shape().dimensions(0);
    pipelined_map[to_move.collective_to_move->mutable_operand(0)] = to_sink;
    auto pipelined_instrs = CollectDependenciesToPipeline(
        to_move.collective_to_move, absl::MakeSpan(to_move.formatting_ops));
    for (auto* original_pipelined : pipelined_instrs) {
      if (original_pipelined->opcode() == HloOpcode::kConstant) {
        continue;
      }
      const bool is_loop_invariant =
          IsLoopInvariant(original_pipelined, invariant_cache);
      CHECK(is_output_instruction.contains(original_pipelined));
      int64_t pipelined_idx = is_output_instruction[original_pipelined];
      HloInstruction* pipelined = loop_computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_while, pipelined_idx));
      // Broadcast loop invariant instructions.
      if (is_loop_invariant) {
        Shape full_shape = ComputeFullOutputShape(to_move, pipelined->shape());
        absl::InlinedVector<int64_t, 4> operand_dims;
        operand_dims.resize(pipelined->shape().dimensions_size());
        absl::c_iota(operand_dims, 1);
        HloInstruction* broadcasted =
            loop_computation->AddInstruction(HloInstruction::CreateBroadcast(
                full_shape, pipelined, operand_dims));
        pipelined_map[original_pipelined] = broadcasted;
      } else {
        pipelined_map[original_pipelined] = pipelined;
      }
    }
    // Cloning the main instruction
    HloInstruction* pipelined_instr_cloned = loop_computation->AddInstruction(
        to_move.collective_to_move->CloneWithNewOperands(
            ComputeFullOutputShape(to_move,
                                   to_move.collective_to_move->shape()),
            {to_sink}));
    UpdateInstructionChannelId(pipelined_instr_cloned, next_channel_id);
    pipelined_map[to_move.collective_to_move] = pipelined_instr_cloned;
    absl::flat_hash_set<HloInstruction*> to_add_batch_set;
    auto collect_operands = [&pipelined_map, &to_add_batch_set,
                             loop_computation,
                             &to_move](HloInstruction* instr) {
      std::vector<HloInstruction*> operands;
      for (auto* operand : instr->mutable_operands()) {
        if (operand->opcode() == HloOpcode::kConstant) {
          HloInstruction* cloned_constant = loop_computation->AddInstruction(
              operand->CloneWithNewOperands(operand->shape(), {}));
          if (!to_add_batch_set.contains(instr)) {
            operands.push_back(cloned_constant);
            continue;
          }
          Shape full_shape =
              ComputeFullOutputShape(to_move, cloned_constant->shape());
          absl::InlinedVector<int64_t, 4> operand_dims;
          operand_dims.resize(cloned_constant->shape().dimensions_size());
          absl::c_iota(operand_dims, 1);
          HloInstruction* broadcasted =
              loop_computation->AddInstruction(HloInstruction::CreateBroadcast(
                  full_shape, cloned_constant, operand_dims));
          operands.push_back(broadcasted);
          continue;
        }
        auto it = pipelined_map.find(operand);
        CHECK(it != pipelined_map.end());
        operands.push_back(it->second);
      }
      return operands;
    };
    absl::flat_hash_set<HloInstruction*> formatting_ops_set(
        to_move.formatting_ops.begin(), to_move.formatting_ops.end());
    std::vector<HloInstruction*> stack(1, to_move.collective_to_move);
    for (auto* current : to_move.formatting_ops) {
      if (IsLoopInvariant(current, invariant_cache)) {
        continue;
      }
      to_add_batch_set.insert(current);
    }
    //  We are adding a batch dimension to the formatting ops, so we need to
    //  specially rewrite each instruction potentially if adding dimensions has
    //  an effect on the instruction itself (like say broadcast, slices ...
    //  etc).
    for (HloInstruction* formatting_op : to_move.formatting_ops) {
      if (!to_add_batch_set.contains(formatting_op) &&
          formatting_op->opcode() != HloOpcode::kBroadcast) {
        HloInstruction* cloned_not_to_batch = loop_computation->AddInstruction(
            formatting_op->CloneWithNewOperands(
                formatting_op->shape(), collect_operands(formatting_op)));
        UpdateInstructionChannelId(cloned_not_to_batch, next_channel_id);
        pipelined_map[formatting_op] = cloned_not_to_batch;
        continue;
      }
      if (formatting_op->IsElementwise() ||
          formatting_op->opcode() == HloOpcode::kReshape ||
          formatting_op->opcode() == HloOpcode::kAllReduce ||
          formatting_op->opcode() == HloOpcode::kConvert ||
          formatting_op->opcode() == HloOpcode::kCollectivePermute) {
        HloInstruction* cloned_elementwise = loop_computation->AddInstruction(
            formatting_op->CloneWithNewOperands(
                ComputeFullOutputShape(to_move, formatting_op->shape()),
                collect_operands(formatting_op)));
        pipelined_map[formatting_op] = cloned_elementwise;
        continue;
      }
      if (formatting_op->opcode() == HloOpcode::kReduce) {
        auto operands = collect_operands(formatting_op);
        std::vector<int64_t> dimensions(formatting_op->dimensions().begin(),
                                        formatting_op->dimensions().end());
        for (auto& dim : dimensions) {
          ++dim;
        }
        // Look through broadcast for reduce init value.
        if (operands[1]->opcode() == HloOpcode::kBroadcast) {
          CHECK(operands[1]->operand(0)->opcode() == HloOpcode::kConstant);
          operands[1] = operands[1]->mutable_operand(0);
        }
        HloInstruction* expanded_reduce =
            loop_computation->AddInstruction(HloInstruction::CreateReduce(
                ComputeFullOutputShape(to_move, formatting_op->shape()),
                operands[0], operands[1], dimensions,
                formatting_op->to_apply()));
        pipelined_map[formatting_op] = expanded_reduce;
        continue;
      }
      if (formatting_op->opcode() == HloOpcode::kBroadcast) {
        CHECK(formatting_op->dimensions().empty());
        auto operands = collect_operands(formatting_op);
        std::vector<int64_t> dimensions(1, 0);
        // Constant scalars don't get expanded ahead of time and are kept
        // scalar.
        if (operands[0]->shape().dimensions_size() == 0) {
          dimensions.clear();
        }
        HloInstruction* expanded_broadcast =
            loop_computation->AddInstruction(HloInstruction::CreateBroadcast(
                ComputeFullOutputShape(to_move, formatting_op->shape()),
                operands[0], dimensions));
        pipelined_map[formatting_op] = expanded_broadcast;
        continue;
      }
      if (formatting_op->opcode() == HloOpcode::kSlice) {
        std::vector<int64_t> slice_start = formatting_op->slice_starts();
        std::vector<int64_t> slice_limits = formatting_op->slice_limits();
        std::vector<int64_t> slice_strides = formatting_op->slice_strides();
        slice_start.insert(slice_start.begin(), 0);
        slice_limits.insert(slice_limits.begin(), new_dim_limit);
        slice_strides.insert(slice_strides.begin(), 1);
        HloInstruction* expanded_slice =
            loop_computation->AddInstruction(HloInstruction::CreateSlice(
                ComputeFullOutputShape(to_move, formatting_op->shape()),
                collect_operands(formatting_op)[0], slice_start, slice_limits,
                slice_strides));
        pipelined_map[formatting_op] = expanded_slice;
        continue;
      }
      if (formatting_op->opcode() == HloOpcode::kDynamicSlice) {
        std::vector<int64_t> dynamic_slice_sizes =
            formatting_op->dynamic_slice_sizes();
        dynamic_slice_sizes.insert(dynamic_slice_sizes.begin(), new_dim_limit);
        HloDynamicSliceInstruction* dynslice =
            Cast<HloDynamicSliceInstruction>(formatting_op);
        HloInstruction* zero = loop_computation->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::Zero(
                formatting_op->operand(dynslice->first_index_operand_number())
                    ->shape()
                    .element_type())));
        std::vector<HloInstruction*> indices(1, zero);
        auto collected_operands = collect_operands(formatting_op);
        indices.insert(indices.end(), std::next(collected_operands.begin()),
                       collected_operands.end());
        HloInstruction* expanded_dynslice =
            loop_computation->AddInstruction(HloInstruction::CreateDynamicSlice(
                ComputeFullOutputShape(to_move, formatting_op->shape()),
                collected_operands[0], indices, dynamic_slice_sizes));
        pipelined_map[formatting_op] = expanded_dynslice;
        continue;
      }
      if (formatting_op->opcode() == HloOpcode::kPad) {
        HloPadInstruction* pad_instruction =
            Cast<HloPadInstruction>(formatting_op);
        PaddingConfig p_config = pad_instruction->padding_config();
        PaddingConfig new_p_config;
        new_p_config.add_dimensions();
        for (auto& dim : p_config.dimensions()) {
          auto* new_dim = new_p_config.add_dimensions();
          *new_dim = dim;
        }
        auto new_operands = collect_operands(formatting_op);
        HloInstruction* expanded_pad =
            loop_computation->AddInstruction(HloInstruction::CreatePad(
                ComputeFullOutputShape(to_move, formatting_op->shape()),
                new_operands[0], new_operands[1], new_p_config));
        pipelined_map[formatting_op] = expanded_pad;
        continue;
      }
      if (formatting_op->opcode() == HloOpcode::kTranspose) {
        HloTransposeInstruction* transpose_instruction =
            Cast<HloTransposeInstruction>(formatting_op);
        std::vector<int64_t> new_dims(
            transpose_instruction->dimensions().begin(),
            transpose_instruction->dimensions().end());
        new_dims.insert(new_dims.begin(), 0);
        for (int64_t& dim : new_dims) {
          ++dim;
        }
        HloInstruction* expanded_transpose =
            loop_computation->AddInstruction(HloInstruction::CreateTranspose(
                ComputeFullOutputShape(to_move, formatting_op->shape()),
                collect_operands(formatting_op)[0], new_dims));
        pipelined_map[formatting_op] = expanded_transpose;
        continue;
      }
      CHECK(false) << "Unsupported instruction " << formatting_op->ToString();
    }
    HloInstruction* inserted_operand =
        to_move.dynamic_update_slice->mutable_operand(1);
    CHECK(pipelined_map.contains(inserted_operand))
        << "Expected to be processed";
    HloInstruction* expanded_inserted = pipelined_map[inserted_operand];
    if (!ShapeUtil::Compatible(expanded_inserted->shape(),
                               to_move.dynamic_update_slice->shape())) {
      expanded_inserted =
          loop_computation->AddInstruction(HloInstruction::CreateReshape(
              to_move.dynamic_update_slice->shape(), expanded_inserted));
    }
    new_output_tuple[to_move.output_idx] = expanded_inserted;
  }
  // Create new loop tuple replacement.
  for (int i = 0; i < new_while->shape().tuple_shapes_size(); ++i) {
    if (new_output_tuple[i] != nullptr) {
      continue;
    }
    new_output_tuple[i] = loop_computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(new_while, i));
  }
  HloInstruction* new_tuple = loop_computation->AddInstruction(
      HloInstruction::CreateTuple(new_output_tuple));
  TF_RETURN_IF_ERROR(while_loop->ReplaceAllUsesWithDifferentShape(new_tuple));
  TF_RETURN_IF_ERROR(
      loop_computation->RemoveInstructionAndUnusedOperands(while_loop));
  TF_RETURN_IF_ERROR(loop_computation->parent()->RemoveUnusedComputations());
  return OkStatus();
}

// Function that does the work of pushing backward instructions that have been
// determined that can be pipelined. Rough transformation:
// while (i < LAYERS) {
//   p0 = param(0)
//   p1 = param(1)
//   p0_ag = all-gather(p0)
//   x = computation(p0_ag)
//   y = computation(p1)
// }
//
// to
//
// x_ag = all-gather(x)
// while (i < LAYERS-1, x_ag) {
//   p0 = param(0)
//   p1 = param(1)
//   p0_ag = param(2)
//   p0_ag_next = all-gather(p0)
//   x = computation(p0_ag)
//   y = computation(p1)
//   x_ag = p0_ag_next
// }
// x_last = computation(p0_ag_next)
static Status TransformLoopBackward(
    const WhileLoopAnalysis& loop_analysis, bool insert_non_alias_custom_call,
    int64_t level_to_operate_on, bool process_different_sized_ops,
    HloPredicate should_process, HloPredicate acceptable_formatting,
    CollectivePipeliner::HloPostprocessor postprocess_peeled,
    CollectivePipeliner::HloPostprocessor postprocess_rotated,
    int64_t& next_channel_id) {
  // Defining some maps/sets to keep track of instructions duplicated.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> while_body_to_peeled;
  absl::flat_hash_map<HloInstruction*, int64_t> collective_to_move_map;
  absl::flat_hash_set<HloInstruction*> is_pipelined_instruction;
  absl::flat_hash_map<HloInstruction*, int64_t> is_output_instruction;
  absl::flat_hash_set<const HloInstruction*> sideeffect_unused_instructions;
  int64_t count = 0;
  // Add instructions to duplicate into a set.
  for (auto& to_move : loop_analysis.GetMoveInfos()) {
    HloInstruction* instr = to_move.collective_to_move;
    collective_to_move_map[instr] = count;
    is_pipelined_instruction.insert(instr);
    is_pipelined_instruction.insert(to_move.formatting_ops.begin(),
                                    to_move.formatting_ops.end());
    ++count;

    // Collect unused instructions with side-effect in the chain, so that we
    // can skip cloning such instructions. This is to work around the fact that
    // we can't have unused Recv instructions to avoid deadlock, and
    // HloModule::RemoveUnusedComputations can't remove unused Recv instructions
    // as they are tagged as has-side-effect. The operand_count check here
    // assumes we only need to collect such instructions when pipelining
    // Recv-done, which may be changed though.
    if (instr->operand_count() == 1) {
      const HloInstruction* opnd = instr->operand(0);
      if (opnd->HasSideEffect() && opnd->user_count() == 1) {
        sideeffect_unused_instructions.insert(opnd);
      }
    }
  }
  HloInstruction* while_loop = loop_analysis.while_loop_instruction();
  HloComputation* while_body = while_loop->while_body();
  CHECK_EQ(while_body->parameter_instructions().size(), 1)
      << "Expected only one parameter";
  HloInstruction* loop_parameter = while_body->parameter_instructions()[0];
  HloInstruction* loop_initial_iteration_idx =
      while_loop->mutable_operand(0)->mutable_operand(
          *loop_analysis.GetLoopIterationIdx());
  // Map loop_parameter to the input tuple for peeling backward.
  while_body_to_peeled[loop_parameter] = while_loop;
  CHECK_EQ(while_body->root_instruction()->opcode(), HloOpcode::kTuple);
  // Record instructions that are part of the output of the loop.
  for (int i = 0; i < while_body->root_instruction()->operand_count(); ++i) {
    is_output_instruction[while_body->root_instruction()->mutable_operand(i)] =
        i;
  }

  // Collect the new parameter shapes with the additional state for the indices
  // and construct new operand vectors for the init of the new loop and its root
  // instruction.
  std::vector<HloInstruction*> new_init_operands;
  std::vector<Shape> new_parameter_shapes;
  std::vector<HloInstruction*> new_root_operands;
  // Number of tuple elements is all the original inputs/outputs to the loop +
  // the pipelined values + the previous iteration loop iteration, which is the
  // only dynamic thing that is allowed to be used by the computation pipelined
  // in the previous iteration.
  const int64_t operands_indices_count =
      while_loop->shape().tuple_shapes_size() +
      loop_analysis.GetMoveInfos().size() + 1;
  new_parameter_shapes.resize(operands_indices_count);
  new_root_operands.resize(operands_indices_count);
  new_init_operands.resize(operands_indices_count);
  // Fill up root and init operands for the new loop.
  for (int i = 0; i < loop_parameter->shape().tuple_shapes_size(); ++i) {
    new_parameter_shapes[i] = loop_parameter->shape().tuple_shapes(i);
    new_root_operands[i] = while_body->root_instruction()->mutable_operand(i);
    new_init_operands[i] = while_loop->mutable_operand(0)->mutable_operand(i);
  }
  // Populating map for cloned instructions in chains pushed backwards.
  // We need a different map, because we want to map the loop iterator
  // differently from the rest of the loop. The whole chain is copied
  // completely, so we don't share anything with the rest of the loop except
  // parameter.
  InstructionMap chain_clone_map;
  chain_clone_map[loop_parameter] = while_loop->mutable_operand(0);
  for (auto* u : loop_parameter->users()) {
    if (IsLoopIterator(u, *loop_analysis.GetLoopIterationIdx())) {
      chain_clone_map[u] = loop_initial_iteration_idx;
    }
  }
  // Add to the rewritten loop the new parameter/output data that is going to be
  // pipelined. Clone chains of pipelined data in the parent computation in the
  // process (they will endup being executed before the loop).
  for (int i = 0; i < loop_analysis.GetMoveInfos().size(); ++i) {
    const int64_t idx = i + loop_parameter->shape().tuple_shapes_size();
    new_parameter_shapes[idx] =
        loop_analysis.GetMoveInfos()[i].collective_to_move->shape();
    new_root_operands[idx] = loop_analysis.GetMoveInfos()[i].collective_to_move;
    TF_ASSIGN_OR_RETURN(
        new_init_operands[idx],
        CloneBackwardChain(*while_loop->parent(),
                           loop_analysis.GetMoveInfos()[i], chain_clone_map,
                           *loop_analysis.GetLoopIterationIdx(),
                           next_channel_id));

    if (postprocess_peeled.has_value()) {
      TF_RETURN_IF_ERROR(postprocess_peeled.value()(new_init_operands[idx]));
    }
  }
  ConstantValue next_loop_iteration =
      loop_analysis.GetLoopStart()->add(*loop_analysis.GetLoopIncrement());
  const Shape& loop_index_shape =
      while_loop->shape().tuple_shapes(*loop_analysis.GetLoopIterationIdx());
  HloInstruction* next_iteration_idx = while_loop->parent()->AddInstruction(
      HloInstruction::CreateConstant(*CreateLiteralOfShape(
          loop_index_shape, next_loop_iteration.GetSignedValue())));
  new_parameter_shapes.back() = loop_parameter->shape().tuple_shapes(
      *loop_analysis.GetLoopIterationIdx());
  new_init_operands.back() = next_iteration_idx;
  auto body_builder = HloComputation::Builder(while_body->name());
  HloInstruction* new_loop_param =
      body_builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeTupleShape(new_parameter_shapes), "param"));
  HloInstruction* loop_iterator_for_pipelined_instrs =
      body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          new_loop_param, new_init_operands.size() - 1));
  InstructionMap while_body_replacement_map;
  while_body_replacement_map[loop_parameter] = new_loop_param;
  InstructionMap collective_to_move_clone_map;
  collective_to_move_clone_map[loop_parameter] = new_loop_param;
  for (auto* u : loop_parameter->users()) {
    if (IsLoopIterator(u, *loop_analysis.GetLoopIterationIdx())) {
      collective_to_move_clone_map[u] = loop_iterator_for_pipelined_instrs;
    }
  }
  // Record the loop variant parameters used in the backward chain.
  LoopVariantParameterInfo loop_variant_parameter_info;
  // Clone loop in the body of the new loop. We change some things like
  // input/output shapes and how we connect loop iterator to the original
  // chains that we are pipelining.
  for (auto* instr : while_body->MakeInstructionPostOrder()) {
    if (instr == loop_parameter || instr == while_body->root_instruction() ||
        sideeffect_unused_instructions.contains(instr)) {
      continue;
    }
    HloInstruction* cloned_instr = nullptr;
    auto it = collective_to_move_map.find(instr);
    if (it != collective_to_move_map.end()) {
      TF_ASSIGN_OR_RETURN(
          cloned_instr,
          CloneBackwardChain(body_builder,
                             loop_analysis.GetMoveInfos()[it->second],
                             collective_to_move_clone_map,
                             *loop_analysis.GetLoopIterationIdx(),
                             next_channel_id, &loop_variant_parameter_info));

      if (postprocess_rotated.has_value()) {
        TF_RETURN_IF_ERROR(postprocess_rotated.value()(cloned_instr));
      }
    } else {
      auto new_operands =
          MapNewOperands(instr->operands(), while_body_replacement_map);
      cloned_instr = body_builder.AddInstruction(
          instr->CloneWithNewOperands(instr->shape(), new_operands));
      TF_RETURN_IF_ERROR(UpdateControlDependencies(instr, cloned_instr,
                                                   while_body_replacement_map));
      UpdateInstructionChannelId(cloned_instr, next_channel_id);
    }
    if (it != collective_to_move_map.end()) {
      const int64_t tuple_idx =
          while_loop->shape().tuple_shapes_size() + it->second;
      HloInstruction* pipelined_value = body_builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(new_loop_param, tuple_idx));
      while_body_replacement_map[instr] = pipelined_value;
      new_root_operands[tuple_idx] = cloned_instr;
      continue;
    }
    while_body_replacement_map[instr] = cloned_instr;
  }
  // For each loop variant parameter used in the backward chain, we temporarily
  // use a newly added loop parameter in the cloned loop. We now need to replace
  // this temporary value with an element in the loop output tuple. The index
  // of the element in the tuple is the same as the index of the loop variant
  // parameter before we pipeline the loop.
  for (const auto& [idx, value] : loop_variant_parameter_info) {
    auto it = while_body_replacement_map.find(new_root_operands[idx]);
    CHECK(it != while_body_replacement_map.end())
        << new_root_operands[idx]->ToString() << " not present in map";
    TF_RETURN_IF_ERROR(value->ReplaceAllUsesWith(it->second));
  }

  new_root_operands.back() =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          loop_index_shape, HloOpcode::kAdd,
          while_body_replacement_map
              [new_root_operands[*loop_analysis.GetLoopIterationIdx()]],
          body_builder.AddInstruction(
              HloInstruction::CreateConstant(*CreateLiteralOfShape(
                  loop_index_shape, next_loop_iteration.GetSignedValue())))));
  HloInstruction* new_loop_root =
      body_builder.AddInstruction(HloInstruction::CreateTuple(
          MapNewOperands(new_root_operands, while_body_replacement_map,
                         /*allow_unmapped=*/true)));
  while_body_replacement_map[while_body->root_instruction()] = new_loop_root;
  HloComputation* new_while_body =
      while_loop->GetModule()->AddEmbeddedComputation(
          body_builder.Build(new_loop_root));
  TF_RETURN_IF_ERROR(UpdateControlDependencies(while_body->root_instruction(),
                                               new_loop_root,
                                               while_body_replacement_map));
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      loop_cond_replacements;
  auto cond_builder =
      HloComputation::Builder(while_loop->while_condition()->name());
  HloInstruction* new_cond_param =
      cond_builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeTupleShape(new_parameter_shapes), "cond_param"));

  // Update the loop bound of the loop to iterate one iteration less.
  // The updated bound is loop_start + (num_iterations-1) * loop_increment.
  HloInstruction* loop_bound = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(*CreateLiteralOfShape(
          loop_initial_iteration_idx->shape(),
          loop_analysis.GetLoopStart()
              ->add(loop_analysis.GetLoopIterationCount()
                        ->sub(ConstantValue::GetOne(
                            loop_analysis.GetLoopStart()->GetBitwidth(),
                            loop_analysis.GetLoopStart()->IsSigned()))
                        .mul(*loop_analysis.GetLoopIncrement()))
              .GetSignedValue())));
  // Construct the new loop condition computation.
  ComparisonDirection cd =
      loop_analysis.GetLoopIncrement()->GetSignedValue() > 0
          ? ComparisonDirection::kLt
          : ComparisonDirection::kGt;
  HloInstruction* loop_iterator =
      cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          new_cond_param, *loop_analysis.GetLoopIterationIdx()));
  HloInstruction* comparison =
      cond_builder.AddInstruction(HloInstruction::CreateCompare(
          while_loop->while_condition()->root_instruction()->shape(),
          loop_iterator, loop_bound, cd));

  HloComputation* new_while_condition =
      while_loop->GetModule()->AddEmbeddedComputation(
          cond_builder.Build(comparison));
  HloInstruction* new_loop_init = while_loop->parent()->AddInstruction(
      HloInstruction::CreateTuple(new_init_operands));
  TF_RETURN_IF_ERROR(UpdateControlDependencies(while_body->root_instruction(),
                                               new_loop_init, chain_clone_map));
  // Create the new loop.
  HloInstruction* new_while_loop =
      while_loop->parent()->AddInstruction(HloInstruction::CreateWhile(
          new_while_body->root_instruction()->shape(), new_while_condition,
          new_while_body, new_loop_init));
  // Clone the loop body in the parent computation of the loop. This is the
  // peeled computation that happens after the loop happened to handle the
  // computation that we peeled away.
  while_body_replacement_map.clear();
  while_body_replacement_map[loop_parameter] = new_while_loop;
  std::vector<HloInstruction*> output_tuple_instructions(
      while_loop->shape().tuple_shapes_size(), nullptr);
  for (auto* instr : while_body->MakeInstructionPostOrder()) {
    if (instr == loop_parameter || instr == while_body->root_instruction() ||
        sideeffect_unused_instructions.contains(instr)) {
      continue;
    }
    auto instruction_is_output_it = is_output_instruction.find(instr);
    auto it = collective_to_move_map.find(instr);
    if (it != collective_to_move_map.end()) {
      const int64_t tuple_idx =
          while_loop->shape().tuple_shapes_size() + it->second;
      HloInstruction* pipelined_value = while_loop->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_while_loop, tuple_idx));
      while_body_replacement_map[instr] = pipelined_value;
      if (instruction_is_output_it != is_output_instruction.end()) {
        output_tuple_instructions[instruction_is_output_it->second] =
            pipelined_value;
      }
      continue;
    }
    auto new_operands =
        MapNewOperands(instr->operands(), while_body_replacement_map);
    HloInstruction* cloned_instr = while_loop->parent()->AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands));
    TF_RETURN_IF_ERROR(UpdateControlDependencies(instr, cloned_instr,
                                                 while_body_replacement_map));
    UpdateInstructionChannelId(cloned_instr, next_channel_id);
    while_body_replacement_map[instr] = cloned_instr;
    if (instruction_is_output_it != is_output_instruction.end()) {
      output_tuple_instructions[instruction_is_output_it->second] =
          cloned_instr;
    }
  }
  // Substitute old loop with the result of the last peeled iteration.
  HloInstruction* final_loop_output = while_loop->parent()->AddInstruction(
      HloInstruction::CreateTuple(output_tuple_instructions));
  HloComputation* loop_computation = while_loop->parent();
  TF_RETURN_IF_ERROR(
      while_loop->ReplaceAllUsesWithDifferentShape(final_loop_output));
  TF_RETURN_IF_ERROR(
      loop_computation->RemoveInstructionAndUnusedOperands(while_loop));
  TF_RETURN_IF_ERROR(loop_computation->parent()->RemoveUnusedComputations());
  return OkStatus();
}

absl::StatusOr<bool> CollectivePipeliner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  CHECK(config_.acceptable_formatting);
  CHECK(config_.should_process);
  bool changed = false;
  std::vector<HloInstruction*> while_loop_instructions;
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        while_loop_instructions.push_back(instruction);
      }
    }
  }
  int64_t transformed_loops = 0;
  int64_t transformed_instructions = 0;
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  VLOG(1) << "Pipelining on direction: "
          << GetPipelineDirectionString(config_.pipelining_direction);
  for (HloInstruction* instruction : while_loop_instructions) {
    VLOG(1) << "While: " << instruction->ToString();
    WhileLoopAnalysis loop_analysis(
        instruction, config_.max_pipelining_per_loop, config_.pipeline_use_tree,
        config_.process_different_sized_ops);
    loop_analysis.ComputeLoopStatistics();
    if (!loop_analysis.GetLoopIterationCount() ||
        loop_analysis.GetLoopIterationCount()->GetUnsignedValue() == 0) {
      continue;
    }
    VLOG(1) << "While iterations: "
            << loop_analysis.GetLoopIterationCount()->ToString();
    loop_analysis.CollectCollectivesToMove(
        config_.level_to_operate_on, config_.pipelining_direction,
        config_.should_process, config_.acceptable_formatting,
        config_.should_allow_loop_variant_parameter_in_chain);
    if (loop_analysis.GetMoveInfos().empty()) {
      continue;
    }
    transformed_instructions += loop_analysis.GetMoveInfos().size();
    VLOG(1) << "Found Collectives to optimize";
    if (VLOG_IS_ON(1)) {
      for (auto& to_move : loop_analysis.GetMoveInfos()) {
        VLOG(1) << "\t" << to_move.collective_to_move->ToString();
        if (to_move.dynamic_update_slice) {
          VLOG(1) << "\t" << to_move.dynamic_update_slice->ToString();
        }
        VLOG(1) << "\t" << to_move.output_idx;
      }
    }
    if (config_.pipelining_direction == PipeliningDirection::kForward) {
      CHECK(config_.reuse_pipelined_op_buffer);
      TF_RETURN_IF_ERROR(TransformLoopForward(
          loop_analysis, !config_.last_run, config_.level_to_operate_on,
          config_.pipeline_use_tree, config_.process_different_sized_ops,
          config_.should_process, config_.acceptable_formatting,
          config_.reuse_pipelined_op_buffer, next_channel_id));
    } else if (config_.pipelining_direction ==
               PipeliningDirection::kForwardSink) {
      TF_RETURN_IF_ERROR(TransformLoopForwardSink(
          loop_analysis, !config_.last_run, config_.level_to_operate_on,
          config_.pipeline_use_tree, config_.process_different_sized_ops,
          config_.should_process, next_channel_id));
    } else {
      CHECK_EQ(config_.pipelining_direction, PipeliningDirection::kBackward);
      TF_RETURN_IF_ERROR(TransformLoopBackward(
          loop_analysis, !config_.last_run, config_.level_to_operate_on,
          config_.process_different_sized_ops, config_.should_process,
          config_.acceptable_formatting, config_.postprocess_backward_peeled_op,
          config_.postprocess_backward_rorated_op, next_channel_id));
    }
    ++transformed_loops;
    changed = true;
  }
  // If this is the last expected run then remove all the custom-calls that we
  // inserted as they shouldn't reach the backend.
  if (config_.last_run) {
    std::vector<HloInstruction*> to_remove;
    for (HloComputation* computation : module->MakeComputationPostOrder()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->IsCustomCall(
                CollectivePipeliner::kInsertedByPreviousStep)) {
          to_remove.push_back(instruction);
          TF_RETURN_IF_ERROR(
              instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
          changed = true;
        }
      }
    }
    for (auto* instruction : to_remove) {
      TF_RETURN_IF_ERROR(
          instruction->parent()->RemoveInstructionAndUnusedOperands(
              instruction));
    }
  }
  VLOG(1) << "Transformed loops: " << transformed_loops
          << " and transformed instructions: " << transformed_instructions
          << " for pipelining direction: "
          << GetPipelineDirectionString(config_.pipelining_direction);
  // Run necessary cleanup to make sure unused code doesn't trigger HloVerifier.
  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());
  }

  return changed;
}

}  // namespace xla
