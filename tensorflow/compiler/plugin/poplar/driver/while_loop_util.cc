#include "tensorflow/compiler/plugin/poplar/driver/while_loop_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include <map>
#include <set>

namespace xla {
namespace poplarplugin {

bool WhileLoopUtil::IsGTEFromParamIndex(const HloInstruction* inst,
                                        int64 param_index) {
  return inst->opcode() == HloOpcode::kGetTupleElement &&
         inst->operand(0)->opcode() == HloOpcode::kParameter &&
         inst->operand(0)->parameter_number() == param_index;
}
bool WhileLoopUtil::IsIntegralConstant(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kConstant &&
         ShapeUtil::ElementIsIntegral(inst->shape());
}

StatusOr<bool> WhileLoopUtil::IsIntegralConstantOfValue(
    const HloInstruction* inst, const int64 value) {
  if (!WhileLoopUtil::IsIntegralConstant(inst)) return false;
  int64 const_value;
  TF_ASSIGN_OR_RETURN(const_value, LiteralScalarInt64toInt64(inst->literal()));
  return const_value == value;
}

StatusOr<std::vector<HloInstruction*>>
WhileLoopUtil::FindMatchingGTEIncrementsInsideBody(
    const HloInstruction* inst, const HloComputation* while_body) {
  CHECK_EQ(inst->opcode(), HloOpcode::kGetTupleElement);
  std::vector<HloInstruction*> ret;
  const int64 tuple_index_cond = inst->tuple_index();
  for (HloInstruction* user : inst->users()) {
    // Check that the user is an Add
    if (user->opcode() != HloOpcode::kAdd) continue;
    // And that one side of the Add is a constant
    if (!(user->operand(0)->opcode() == HloOpcode::kConstant ||
          user->operand(1)->opcode() == HloOpcode::kConstant))
      continue;
    HloInstruction* constant =
        user->operand(0)->opcode() == HloOpcode::kConstant
            ? user->mutable_operand(0)
            : user->mutable_operand(1);
    // check that constant is an integral 1
    bool is_one;
    TF_ASSIGN_OR_RETURN(is_one,
                        WhileLoopUtil::IsIntegralConstantOfValue(constant, 1));
    if (!is_one) continue;

    // check that the user is only used once
    if (user->user_count() == 1) {
      HloInstruction* inst = user->users()[0];
      // check that inst is a root Tuple instruction with the user in the
      // right index
      if (inst->opcode() == HloOpcode::kTuple &&
          inst == while_body->root_instruction() &&
          inst->operands()[tuple_index_cond] == user) {
        ret.push_back(user);
      }
    }
  }
  return ret;
}

StatusOr<int64> WhileLoopUtil::CanConvertWhileToRepeat(
    const HloInstruction* while_inst) {
  HloComputation* while_condition = while_inst->while_condition();
  HloComputation* while_body = while_inst->while_body();
  // Make sure that this is a while loop with a single conditional of form
  // "cond < const"
  std::vector<HloInstruction*> lt_instructions;
  for (HloInstruction* inst : while_condition->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kLt) {
      lt_instructions.push_back(inst);
    }
  }
  // Make sure there is a single conditional
  if (lt_instructions.size() != 1) {
    return xla::FailedPrecondition("Unable to convert this while loop");
  }
  HloInstruction* lt_inst = lt_instructions[0];

  // Make sure that for the LT:
  // * LHS is an integral GTE from from parameter 0
  // * RHS is a constant which is integral
  // * LT is a root instruction
  {
    const bool lhs_is_GTE_param_from_param_0 =
        WhileLoopUtil::IsGTEFromParamIndex(lt_inst->operand(0), 0);
    if (!lhs_is_GTE_param_from_param_0) {
      return xla::FailedPrecondition("Unable to convert this while loop");
    }

    const bool lhs_GTE_is_integral =
        ShapeUtil::ElementIsIntegral(lt_inst->operand(0)->shape());
    if (!lhs_GTE_is_integral) {
      return xla::FailedPrecondition("Unable to convert this while loop");
    }

    const bool rhs_is_integral_const =
        WhileLoopUtil::IsIntegralConstant(lt_inst->operand(1));
    if (!rhs_is_integral_const) {
      return xla::FailedPrecondition("Unable to convert this while loop");
    }

    const bool is_root = while_condition->root_instruction() == lt_inst;
    if (!is_root) {
      return xla::FailedPrecondition("Unable to convert this while loop");
    }
  }

  HloInstruction* comp_GTE = lt_inst->mutable_operand(0);
  int64 cond_tuple_index = comp_GTE->tuple_index();

  int64 repeat_count;
  TF_ASSIGN_OR_RETURN(
      repeat_count, LiteralScalarInt64toInt64(lt_inst->operand(1)->literal()));

  // Make sure that the initial value of the tuple element for GTE is a constant
  // 0 integral
  {
    const HloInstruction* init_val =
        while_inst->operand(0)->operand(cond_tuple_index);
    bool is_zero;
    TF_ASSIGN_OR_RETURN(is_zero,
                        WhileLoopUtil::IsIntegralConstantOfValue(init_val, 0));
    if (!is_zero) {
      return xla::FailedPrecondition("Unable to convert this while loop");
    }
  }

  // Find corresponding GTE in the body
  HloInstruction* body_GTE;
  int64 matching_GTEs = 0;
  for (HloInstruction* inst : while_body->MakeInstructionPostOrder()) {
    const bool is_GTE_from_param_0 =
        WhileLoopUtil::IsGTEFromParamIndex(inst, 0);
    if (!is_GTE_from_param_0) continue;
    if (inst->tuple_index() == cond_tuple_index) {
      body_GTE = inst;
      matching_GTEs++;
    }
  }
  // Make sure there is only one
  if (matching_GTEs != 1) {
    return xla::FailedPrecondition("Unable to convert this while loop");
  }

  // Check that the mapped GTE instruction is incremented by 1 and that the
  // resulting increment is *only* used in the output tuple of the while body in
  // the same index
  std::vector<HloInstruction*> matching_increments;
  TF_ASSIGN_OR_RETURN(
      matching_increments,
      WhileLoopUtil::FindMatchingGTEIncrementsInsideBody(body_GTE, while_body));

  if (matching_increments.size() == 1) {
    return repeat_count;
  } else {
    return xla::FailedPrecondition("Unable to convert this while loop");
  }
}

}  // namespace poplarplugin
}  // namespace xla