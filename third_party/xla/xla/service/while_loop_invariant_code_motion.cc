/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/while_loop_invariant_code_motion.h"

#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/map_util.h"
#include "xla/service/compile_time_cap.h"
#include "xla/service/while_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

using absl::flat_hash_map;
using absl::flat_hash_set;
using absl::InlinedVector;

// Copies `to_hoist` to the computation containing `while_instr`, hoisting its
// operands as needed.  All of its transitive operands are expected to be either
// in `hoisted_instructions` or `unhoisted_invariant_instructions`.  This
// function hoists the operands in `unhoisted_invariant_instructions` and moves
// them into `hoisted_instructions`.
static void CreateLoopInvariantCopy(
    flat_hash_map<HloInstruction*, HloInstruction*>* hoisted_instructions,
    flat_hash_set<HloInstruction*>* unhoisted_invariant_instructions,
    HloInstruction* while_instr, HloInstruction* to_hoist) {
  HloComputation* parent_of_while = while_instr->parent();
  HloComputation* while_body = while_instr->while_body();

  struct DFSFrame {
    HloInstruction* instruction;
    int64_t operand_index;
  };

  InlinedVector<DFSFrame, 8> dfs_stack;
  dfs_stack.push_back({to_hoist, 0});

  HloInstruction* while_body_param = while_body->parameter_instruction(0);
  HloInstruction* while_operand = while_instr->mutable_operand(0);

  do {
    DFSFrame* frame = &dfs_stack.back();
    if (frame->operand_index == frame->instruction->operand_count()) {
      HloInstruction* old_instruction = frame->instruction;

      // All of the operands for old_instruction have been cloned, so it is
      // time to clone old_instruction itself.

      auto get_new_operand = [&](HloInstruction* old_operand) {
        return old_operand == while_body_param
                   ? while_operand
                   : FindOrDie(*hoisted_instructions, old_operand);
      };

      InlinedVector<HloInstruction*, 4> new_operands;
      absl::c_transform(old_instruction->operands(),
                        std::back_inserter(new_operands), get_new_operand);

      HloInstruction* new_instruction =
          parent_of_while->AddInstruction(old_instruction->CloneWithNewOperands(
              old_instruction->shape(), new_operands));

      InsertOrDie(hoisted_instructions, old_instruction, new_instruction);

      // Approximately half of the instructions that would normally be present
      // in unhoisted_invariant_instructions are constants.  We save a bit of
      // compile time by not putting these in the hashtable.
      CHECK_EQ(unhoisted_invariant_instructions->erase(old_instruction),
               to_hoist != old_instruction &&
                   old_instruction->opcode() != HloOpcode::kConstant);
      dfs_stack.pop_back();
      continue;
    }

    HloInstruction* next_operand =
        frame->instruction->mutable_operand(frame->operand_index++);
    if (hoisted_instructions->contains(next_operand) ||
        next_operand == while_body_param) {
      continue;
    }

    dfs_stack.push_back({next_operand, 0});
  } while (!dfs_stack.empty());
}

// Returns true if `instruction` is worth hoisting only if it lets us hoist some
// instruction using it. The rationale is that hoisting these instructions will
// prevent simplification, fusion, and sharding annotation in the while body.
bool WhileLoopInvariantCodeMotion::NotWorthHoistingIndividually(
    const HloInstruction& instruction) {
  if (instruction.IsCustomCall("Sharding")) {
    return true;
  }

  switch (instruction.opcode()) {
    default:
      return false;

    case HloOpcode::kConstant:
      return !hoist_constants_;

    case HloOpcode::kReshape:
      return !hoist_reshapes_;

    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kIota:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return true;
  }
}

absl::StatusOr<bool>
WhileLoopInvariantCodeMotion::TryHoistingInvariantInstructionsFromWhileBody(
    HloInstruction* while_instr, BoundNonLinearCompilerAnalysis* allowance) {
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

  // Maps instructions in the while body to instructions hoisted outside the
  // while that compute the same value.
  flat_hash_map<HloInstruction*, HloInstruction*> hoisted_instructions;

  // Contains instructions that can be legally hoisted, but were deemed to be
  // unprofitable to be hoisted alone by NotWorthHoistingIndividually.  When we
  // hoist an instruction in this set, we move it from
  // unhoisted_invariant_instructions to hoisted_instructions.
  flat_hash_set<HloInstruction*> unhoisted_invariant_instructions;

  // Invariant GTE's axiomatically satisfy the constraints for
  // unhoisted_invariant_instructions -- they can be legally hoisted, but there
  // is no benefit to hoisting them unless something that uses it is also
  // hoisted.
  for (auto* instr : WhileUtil::GetInvariantGTEsForWhileBody(*while_body)) {
    if (instr->shape().IsArray()) {
      // TODO(b/79147885): We should try to generalize this to tuples for
      // uniformity's sake, if nothing else.
      InsertOrDie(&unhoisted_invariant_instructions, instr);
    }
  }

  if (unhoisted_invariant_instructions.empty() && !hoist_constants_) {
    // There are no obviously loop invariant elements in the state being
    // threaded through the while loop so give up.  In theory this precondition
    // is too strong -- we could have code that e.g. permutes the elements in
    // the while state but uses a select to pick the same value on every
    // iteration.
    //
    // If we were asked to hoist constants, we need to scan the while body for
    // constants even if we didn't find any loop invariant values in the while
    // state tuple.
    return false;
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

  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    allowance->DeductCost(1);
    if (!allowance->ContinueAnalysis()) {
      return false;
    }

    if (instruction->HasSideEffect() ||
        instruction->opcode() == HloOpcode::kAfterAll ||
        instruction->opcode() == HloOpcode::kParameter ||
        !instruction->control_predecessors().empty() ||
        !instruction->control_successors().empty()) {
      continue;
    }

    if (!hoist_other_ && instruction->opcode() != HloOpcode::kConstant &&
        instruction->opcode() != HloOpcode::kReshape) {
      continue;
    }
    // Constants don't inflate, so size inflation check doesn't make sense for
    // constants.
    if (hoist_size_inflation_ratio_ &&
        instruction->opcode() != HloOpcode::kConstant) {
      // Check that hoisting the instruction doesn't cause a significant memory
      // blow-up. LICM extends the live-range of the output of the hoisted
      // instruction to be the entire while loop, which may be problematic on
      // platforms where memory is limited. This can be especially harmful if
      // the instruction has a significantly larger output than its input, e.g.
      // kIota, kBroadcast or kConstant.
      int64_t input_size = 0, output_size = 0;

      for (auto* operand : instruction->operands()) {
        ShapeUtil::ForEachSubshape(
            operand->shape(), [&input_size, this](const Shape& subshape,
                                                  const ShapeIndex& /*index*/) {
              if (subshape.IsArray()) {
                input_size += shape_size_function_(subshape);
              }
            });
      }
      ShapeUtil::ForEachSubshape(
          instruction->shape(),
          [&output_size, this](const Shape& subshape,
                               const ShapeIndex& /*index*/) {
            if (subshape.IsArray()) {
              output_size += shape_size_function_(subshape);
            }
          });

      if (output_size > input_size * *hoist_size_inflation_ratio_) {
        continue;
      }
    }

    auto is_invariant = [&](HloInstruction* op) {
      return hoisted_instructions.find(op) != hoisted_instructions.end() ||
             unhoisted_invariant_instructions.contains(op) ||
             op->opcode() == HloOpcode::kConstant;
    };

    if (!absl::c_all_of(instruction->operands(), is_invariant)) {
      continue;
    }

    if (NotWorthHoistingIndividually(*instruction)) {
      VLOG(2) << "Adding " << instruction->ToString(print_no_metadata)
              << " to unhoisted invariant set.";
      // Approximately half of the instructions that reach this point are
      // constants.  We save a bit of compile time by not putting these in the
      // hashtable.
      if (instruction->opcode() != HloOpcode::kConstant) {
        InsertOrDie(&unhoisted_invariant_instructions, instruction);
      }
      continue;
    }

    VLOG(2) << "Hoisting " << instruction->ToString(print_no_metadata);

    CreateLoopInvariantCopy(&hoisted_instructions,
                            &unhoisted_invariant_instructions, while_instr,
                            instruction);

    instructions_to_replace.push_back(instruction);
    replacement_instructions.push_back(
        FindOrDie(hoisted_instructions, instruction));
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

absl::StatusOr<bool> WhileLoopInvariantCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before WhileLoopInvariantCodeMotion:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  BoundNonLinearCompilerAnalysis allowance(module, name(), 10);

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

    if (!allowance.ContinueAnalysis()) {
      break;
    }
    TF_ASSIGN_OR_RETURN(
        bool result,
        TryHoistingInvariantInstructionsFromWhileBody(while_instr, &allowance));
    changed |= result;
  }

  if (changed) {
    // Run DCE if changed. This pass may create new while loops with new
    // computations and if we don't delete the old ones, we can have spurious
    // verification failures (e.g., the verifier may see multiple channel
    // instructions that have the same channel ids).
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }

  if (changed) {
    VLOG(2) << "HLO module after WhileLoopInvariantCodeMotion:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after WhileLoopInvariantCodeMotion";
  }

  return changed;
}
}  // namespace xla
