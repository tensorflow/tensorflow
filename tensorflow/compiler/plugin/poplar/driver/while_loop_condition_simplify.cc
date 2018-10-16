/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/while_loop_condition_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/while_loop_util.h"

#include "tensorflow/compiler/xla/literal_util.h"

#include <map>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<bool> TrySimplifyLoopCondition(HloInstruction* while_inst) {
  HloComputation* while_condition = while_inst->while_condition();
  HloComputation* while_body = while_inst->while_body();

  // Make sure root instruction of while condition is an AND
  if (while_condition->root_instruction()->opcode() != HloOpcode::kAnd) {
    return false;
  }

  // Find all LT instructions, AND predicates and GTEs from parameters
  std::set<HloInstruction*> lt_instructions;
  std::set<HloInstruction*> and_instructions;
  std::map<int64, HloInstruction*> while_condition_GTEs;

  // When looking for LTs, also find the smallest constant
  HloInstruction* smallest_lt;
  int64 smallest_constant_value = INT64_MAX;

  for (HloInstruction* inst : while_condition->MakeInstructionPostOrder()) {
    // Ignore dead instructions
    if (inst->user_count() == 0 && !inst->HasSideEffect() &&
        inst != while_condition->root_instruction()) {
      continue;
    }

    switch (inst->opcode()) {
      case HloOpcode::kGetTupleElement: {
        // Make sure that GTEs are from parameter 0 and only have one user and
        // are integral and that they are unique
        const bool is_GTE_from_param_0 =
            WhileLoopUtil::IsGTEFromParamIndex(inst, 0);
        if (!is_GTE_from_param_0) {
          return false;
        }

        const bool has_one_user = inst->user_count() == 1;
        if (!has_one_user) {
          return false;
        }

        const bool is_integral = ShapeUtil::ElementIsIntegral(inst->shape());
        if (!is_integral) {
          return false;
        }

        const bool unique_GTE =
            while_condition_GTEs.find(inst->tuple_index()) ==
            while_condition_GTEs.end();
        if (unique_GTE) {
          while_condition_GTEs[inst->tuple_index()] = inst;
        } else {
          return false;
        }
        break;
      }
      case HloOpcode::kLt: {
        // Make sure LHS of LT is a GTE from the parameter tuple and RHS is a
        // constant which is integral
        const bool lhs_is_GTE_from_param =
            WhileLoopUtil::IsGTEFromParamIndex(inst->operand(0), 0);
        const bool rhs_is_integral_const =
            WhileLoopUtil::IsIntegralConstant(inst->operand(1));
        if (lhs_is_GTE_from_param && rhs_is_integral_const) {
          lt_instructions.insert(inst);

          // Test if the value is the smallest
          int64 val;
          TF_ASSIGN_OR_RETURN(
              val, LiteralScalarInt64toInt64(inst->operand(1)->literal()));
          if (val < smallest_constant_value) {
            smallest_lt = inst;
            smallest_constant_value = val;
          }
        } else {
          return false;
        }
        break;
      }
      case HloOpcode::kAnd: {
        // Make sure that LHS and RHS of AND is an LT which are unique
        const bool lhs_is_lt = lt_instructions.count(inst->mutable_operand(0));
        const bool rhs_is_lt = lt_instructions.count(inst->mutable_operand(1));
        if (lhs_is_lt && rhs_is_lt && (inst->operand(0) != inst->operand(1))) {
          and_instructions.insert(inst);
        } else {
          return false;
        }
        break;
      }
      case HloOpcode::kConstant:
      case HloOpcode::kParameter:
        // Constants and Parameters are permitted
        break;
      default:
        // Other OPs are not supported in while condition
        return false;
    }
  }

  // Make sure root instruction is in the and_instructions
  if (and_instructions.find(while_condition->root_instruction()) ==
      and_instructions.end()) {
    return false;
  }
  // Limit this to 2 LTs and 1 AND
  if (lt_instructions.size() != 2 && and_instructions.size() != 1) {
    return false;
  }

  // Make sure that the initial value of tuple elements for GTE was a constant 0
  for (auto it : while_condition_GTEs) {
    const HloInstruction* init_val = while_inst->operand(0)->operand(it.first);
    bool is_zero;
    TF_ASSIGN_OR_RETURN(is_zero,
                        WhileLoopUtil::IsIntegralConstantOfValue(init_val, 0));
    if (!is_zero) {
      return false;
    }
  }

  // Find all GTEs from the parameter tuple in the while loop body
  std::map<int64, HloInstruction*> while_body_GTEs;
  for (HloInstruction* inst : while_body->MakeInstructionPostOrder()) {
    const bool is_GTE_from_param_0 =
        WhileLoopUtil::IsGTEFromParamIndex(inst, 0);
    if (is_GTE_from_param_0) {
      while_body_GTEs[inst->tuple_index()] = inst;
    }
  }

  // Map the GTEs from the while loop conditional to GTE
  // instruction in the while loop body
  std::map<HloInstruction*, HloInstruction*> cond_to_body;
  for (auto it_cond : while_condition_GTEs) {
    auto it_body = while_body_GTEs.find(it_cond.first);
    if (it_body == while_body_GTEs.end()) {
      return false;
    }
    cond_to_body[it_cond.second] = it_body->second;
  }

  // Check that all mapped GTE instructions are incremented by 1 and that the
  // resulting increment is *only* used in the output tuple of the while body in
  // the same index
  std::map<HloInstruction*, HloInstruction*> GTE_to_increment;
  for (auto pair : cond_to_body) {
    HloInstruction* body_GTE = pair.second;
    std::vector<HloInstruction*> matching_increments;
    TF_ASSIGN_OR_RETURN(matching_increments,
                        WhileLoopUtil::FindMatchingGTEIncrementsInsideBody(
                            body_GTE, while_body));
    if (matching_increments.size() == 1) {
      GTE_to_increment[body_GTE] = matching_increments[0];
    } else {
      return false;
    }
  }

  // Clean up the while conditional
  HloInstruction* old_root = while_condition->root_instruction();
  // Set the smallest_lt as new root
  while_condition->set_root_instruction(smallest_lt);
  // Remove unused instructions from the old root
  TF_CHECK_OK(while_condition->RemoveInstructionAndUnusedOperands(old_root));

  // Clean up the while body by replacing GTE results/increments with the
  // corresponding values for smallest_lt
  HloInstruction* gte_to_keep = cond_to_body[smallest_lt->mutable_operand(0)];
  HloInstruction* increment_to_keep = GTE_to_increment[gte_to_keep];
  std::vector<HloInstruction*> instructions_to_remove;
  for (auto it : GTE_to_increment) {
    HloInstruction* gte_to_delete = it.first;
    HloInstruction* increment_to_delete = it.second;
    if (gte_to_keep == gte_to_delete) continue;

    TF_CHECK_OK(increment_to_delete->ReplaceAllUsesWith(increment_to_keep));
    // Iterate over all uses of gte_to_delete and replace them with gte_to_keep,
    // except for the increment instruction
    for (HloInstruction* user : gte_to_delete->users()) {
      if (user == increment_to_delete) continue;
      for (int64 op_idx = 0; op_idx < user->operand_count(); op_idx++) {
        if (user->operand(op_idx) == gte_to_delete) {
          TF_CHECK_OK(user->ReplaceOperandWith(op_idx, gte_to_keep));
        }
      }
    }
    instructions_to_remove.push_back(increment_to_delete);
  }
  // Remove the increment instructions and all their dependents
  for (HloInstruction* inst : instructions_to_remove) {
    VLOG(1) << "Removing " << inst->ToString() << " from while loop";
    TF_CHECK_OK(while_body->RemoveInstructionAndUnusedOperands(inst));
  }

  return true;
}
}

WhileLoopConditionSimplify::WhileLoopConditionSimplify() {}

StatusOr<bool> WhileLoopConditionSimplify::Run(HloModule* module) {
  bool changed = false;
  std::vector<HloInstruction*> while_insts;
  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        while_insts.push_back(instr);
      }
    }
  }
  for (HloInstruction* while_inst : while_insts) {
    // Try to simplify the loop condition if it has 2 conditionals which both
    // have some constant upper bound
    TF_ASSIGN_OR_RETURN(bool result, TrySimplifyLoopCondition(while_inst));
    changed |= result;
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
