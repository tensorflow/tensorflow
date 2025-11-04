/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/while_loop_expensive_invariant_code_motion.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/map_util.h"
#include "xla/service/while_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace {
using absl::flat_hash_map;
using absl::flat_hash_set;
using absl::InlinedVector;

struct InvariantInfo {
  explicit InvariantInfo(int64_t user_count)
      : remaining_user_count(user_count) {}
  // The transitive input size of all input operands, traced up to the while
  // loop parameter or leaf invariant ops.
  int64_t transitive_input_size = 0;
  // The remaining users count that remain in the body after all hoistable
  // invariant users are hoisted. This number excludes the root instruction.
  int64_t remaining_user_count;
  // If this instruction is hoisted, this stores the copy outside the body.
  HloInstruction* hoisted_copy = nullptr;
  // Hoistable instructions depending on this op to be hoisted.
  InlinedVector<HloInstruction*, 2> blocked_users;
};
}  // namespace

absl::StatusOr<bool> WhileLoopExpensiveInvariantCodeMotion::
    TryHoistingInvariantInstructionsFromWhileBody(HloInstruction* while_instr) {
  auto print_no_metadata = HloPrintOptions{}.set_print_metadata(false);

  if (!while_instr->shape().IsTuple()) {
    // This restriction leaves one interesting pattern on the table:
    //
    //  while_body(f32[1024, 1024] %param) {
    //    %value = expensive_op(%param)
    //    outfeed(%value)
    //    ROOT = %param
    //  }
    //
    // If we see that pattern in the while, instead of generalizing this
    // algorithm to work with non-tuples, we should instead add a pass that
    // canonicalizes while loops like the above to use a tuple state.
    return false;
  }

  std::string while_instr_name = while_instr->ToString(print_no_metadata);
  VLOG(2) << "Trying to hoist from " << while_instr_name;

  auto maybe_upper_bound = ComputeWhileLoopTripCountUpperBound(while_instr);
  if (maybe_upper_bound && *maybe_upper_bound <= 1) {
    VLOG(2) << "Loop has a trip count of at most 1, skipping.";
    return false;
  }

  HloComputation* while_body = while_instr->while_body();

  // Contains the information for all invariant instructions that can be legally
  // hoisted. When we hoist an instruction in this set, we set its hoisted_copy
  // field to the hoisted instruction.
  flat_hash_map<HloInstruction*, InvariantInfo> invariant_instructions;

  // Map from an invariant instruction to the number of remaining unresolved
  // operands, i.e. operands used by unvisited instructions. If all these
  // operands are used by other invariant instructions, then hoisting out that
  // operand won't leave a copy of itself in the body and it's free to hoist.
  flat_hash_map<HloInstruction*, int64_t> to_hoist_when_ready;

  // Identify invariant GTE instructions so that we can identify its users that
  // are also invariants.
  for (auto* instr : WhileUtil::GetInvariantGTEsForWhileBody(*while_body)) {
    // TODO(b/79147885): We should try to generalize this to tuples for
    // uniformity's sake, if nothing else.
    if (instr->shape().IsArray()) {
      // We subtract 1 from user_count because we know one of the users is root.
      auto emplace_result = invariant_instructions.emplace(
          instr, InvariantInfo(/*user_count=*/instr->user_count() - 1));
      CHECK(emplace_result.second);
      InvariantInfo& info = emplace_result.first->second;
      info.transitive_input_size = shape_size_function_(instr->shape());
    }
  }

  // LICM in the presence of domain instructions is complex, bail.
  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kDomain ||
        instruction->IsCustomCall("SPMDFullToShardShape") ||
        instruction->IsCustomCall("SPMDShardShapeToFull")) {
      return false;
    }
  }

  // instructions_to_replace[i] is hoisted into a loop invariant instruction
  // replacement_instructions[i].
  std::vector<HloInstruction*> instructions_to_replace;
  std::vector<HloInstruction*> replacement_instructions;

  auto hoist = [&](HloInstruction* instruction, InvariantInfo& info) {
    if (info.hoisted_copy) {
      // Already hoisted.
      return;
    }
    VLOG(2) << "Hoisting " << instruction->ToString(print_no_metadata);

    HloInstruction* to_hoist = instruction;
    auto is_hoisted = [&](HloInstruction* instr) {
      return FindOrDie(invariant_instructions, instr).hoisted_copy != nullptr;
    };
    auto get_hoisted = [&](HloInstruction* instr) {
      return FindOrDie(invariant_instructions, instr).hoisted_copy;
    };
    auto set_hoisted = [&](HloInstruction* old_instr,
                           HloInstruction* new_instr) {
      FindOrDie(invariant_instructions, old_instr).hoisted_copy = new_instr;
    };
    WhileUtil::CreateLoopInvariantCopy(to_hoist, while_instr, is_hoisted,
                                       get_hoisted, set_hoisted);

    instructions_to_replace.push_back(instruction);
    replacement_instructions.push_back(info.hoisted_copy);
  };

  // Temporary helper container for marking a operand as checked when
  // decrementing its remaining_user_count counter. Cleared after each
  // iteration.
  flat_hash_set<HloInstruction*> checked_operands;

  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    if (instruction->HasSideEffect() ||
        instruction->opcode() == HloOpcode::kParameter ||
        !instruction->control_predecessors().empty() ||
        !instruction->control_successors().empty() ||
        instruction == while_body->root_instruction()) {
      continue;
    }

    auto is_invariant = [&](HloInstruction* op) {
      return invariant_instructions.find(op) != invariant_instructions.end();
    };

    if (!absl::c_all_of(instruction->operands(), is_invariant)) {
      continue;
    }

    auto emplace_result = invariant_instructions.emplace(
        instruction, InvariantInfo(/*user_count=*/instruction->user_count()));
    CHECK(emplace_result.second);
    InvariantInfo& instr_info = emplace_result.first->second;
    // If root is a users of it, subtract 1 from remaining user count as we
    // don't want root to be blocking other users from being hoisted. Note that
    // for invariant parameter GTEs, they will skip the iteration because their
    // operand parameter(0) is not invariant, and they are put into
    // invariant_instructions before this loop.
    for (auto* user : instruction->users()) {
      if (user == while_body->root_instruction()) {
        --instr_info.remaining_user_count;
        break;
      }
    }

    int64_t num_blocking_operands = 0;
    // Check that hoisting the instruction doesn't cause a significant memory
    // blow-up. LICM extends the live-range of the output of the hoisted
    // instruction to be the entire while loop, which may be problematic on
    // platforms where memory is limited. This can be especially harmful if
    // the instruction has a significantly larger output than its input, e.g.
    // kIota, kBroadcast or kConstant.
    int64_t output_size = 0;

    for (auto* operand : instruction->operands()) {
      auto& operand_info = invariant_instructions.at(operand);
      if (!checked_operands.contains(operand)) {
        instr_info.transitive_input_size += operand_info.transitive_input_size;
        --operand_info.remaining_user_count;
        checked_operands.insert(operand);
      }
      if (operand_info.remaining_user_count == 0) {
        // All users are hoistable invariants, unblock held off users.
        for (auto* user : operand_info.blocked_users) {
          auto it = to_hoist_when_ready.find(user);
          if (it != to_hoist_when_ready.end()) {
            auto& num_blocking = it->second;
            CHECK_GT(num_blocking, 0);
            --num_blocking;
            // Hoist a previously held off instruction now that there are no
            // more blocking operands.
            if (num_blocking == 0) {
              hoist(user, invariant_instructions.at(user));
              to_hoist_when_ready.erase(it);
            }
          }
        }
        operand_info.blocked_users.clear();
      } else if (operand_info.remaining_user_count > 0) {
        ++num_blocking_operands;
        if (operand_info.blocked_users.empty() ||
            operand_info.blocked_users.back() != instruction) {
          operand_info.blocked_users.push_back(instruction);
        }
      } else {
        LOG(FATAL)
            << "An instruction should not have number of negative users.";
      }
    }
    checked_operands.erase(checked_operands.begin(), checked_operands.end());
    ShapeUtil::ForEachSubshape(
        instruction->shape(),
        [&output_size, this](const Shape& subshape,
                             const ShapeIndex& /*index*/) {
          if (subshape.IsArray()) {
            output_size += shape_size_function_(subshape);
          }
        });
    // If it is size-inflating, we leave it as is and potentially will still
    // hoist it out if we later found a group of ops that are worth hoisting
    // as a whole.
    if (output_size > instr_info.transitive_input_size) {
      continue;
    }

    if (!worth_hoisting_individually_(instruction)) {
      continue;
    }

    // Need to wait until we inspected the users of some operands until we can
    // finally decide whether to hoist this instruction.
    if (num_blocking_operands > 0) {
      to_hoist_when_ready.emplace(instruction, num_blocking_operands);
      continue;
    }

    hoist(instruction, instr_info);
  }

  if (instructions_to_replace.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      WhileUtil::MakeInstructionsLiveInResult live_in_instructions_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr, replacement_instructions));

  HloComputation* new_while_body =
      live_in_instructions_result.new_while_instr->while_body();

  for (int i = 0; i < instructions_to_replace.size(); i++) {
    HloInstruction* instruction_to_replace_in_new_while =
        FindOrDie(live_in_instructions_result.while_body_instruction_map,
                  instructions_to_replace[i]);
    TF_RETURN_IF_ERROR(new_while_body->ReplaceInstruction(
        instruction_to_replace_in_new_while,
        live_in_instructions_result.while_body_live_in_values[i]));
  }

  VLOG(1) << "Hoisted " << instructions_to_replace.size()
          << " instructions from " << while_instr_name;

  return true;
}

absl::StatusOr<bool> WhileLoopExpensiveInvariantCodeMotion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(3) << "HLO module before WhileLoopExpensiveInvariantCodeMotion:";
  XLA_VLOG_LINES(3, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->computations(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }

  // Currently, if a loop body that is used by multiple while
  // ops contains an op that can be hoisted, we will make a new computation for
  // each of the while ops, instead of using one shared new computation. This is
  // probably fine, but we may want to improve it in the future if we decide to
  // double-down on shared while bodies.
  for (HloInstruction* while_instr : while_instrs) {
    // Right now we only hoist computations from the while body, but
    // TryHoistingInvariantInstructionsFromWhileBody can be generalized to
    // optimize the condition computation too, if needed.
    //
    // The transform we do here is a pessimization for while loops that execute
    // zero times*, but at this time we expect those to be rare.  If this
    // becomes a problem we can consider using the conditional HLO to avoid
    // doing extra work for while loops with zero trip count.
    //
    // * We delete while loops that have a zero trip count, so this would have
    //   to be a while loop with a somewhat opaque condition expression.

    if (while_instr->frontend_attributes().map().contains(
            "_xla_disable_loop_instr_hoisting")) {
      // If this frontend attr is present, we have knowledge from the framework
      // to disable hoisting from this loop.
      auto print_no_metadata = HloPrintOptions{}.set_print_metadata(false);
      std::string while_instr_name = while_instr->ToString(print_no_metadata);
      VLOG(2) << "Skipping hoisting from: " << while_instr_name
              << " because it is disabled via xla metadata.";
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        bool result,
        TryHoistingInvariantInstructionsFromWhileBody(while_instr));
    changed |= result;
  }

  if (changed) {
    VLOG(3) << "HLO module after WhileLoopExpensiveInvariantCodeMotion:";
    XLA_VLOG_LINES(3, module->ToString());
  } else {
    VLOG(3)
        << "HLO module unchanged after WhileLoopExpensiveInvariantCodeMotion";
  }

  return changed;
}
}  // namespace xla
