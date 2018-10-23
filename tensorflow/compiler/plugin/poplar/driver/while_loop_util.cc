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
    const HloInstruction* inst, const HloComputation* while_body,
    HloOpcode opcode) {
  CHECK_EQ(inst->opcode(), HloOpcode::kGetTupleElement);
  std::vector<HloInstruction*> ret;
  const int64 tuple_index_cond = inst->tuple_index();
  for (HloInstruction* user : inst->users()) {
    // Check that the user is a comparison matching the expected opcode
    if (user->opcode() != opcode) continue;
    // And that one side of the comparison is a constant
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

static const char* err_msg = "Unable to convert this while loop";

StatusOr<int64> WhileLoopUtil::CanConvertWhileToRepeat(
    const HloInstruction* while_inst) {
  HloComputation* while_condition = while_inst->while_condition();
  HloComputation* while_body = while_inst->while_body();
  // Make sure that this is a while loop with a single conditional of form
  // "cond COMP const". There must be only 4 instructions which prevents
  // detached stateful instructions from being excluded from execution.
  if (while_condition->instruction_count() != 4) {
    return xla::FailedPrecondition(err_msg);
  }

  // The root instruction must be the comparison
  HloInstruction* c_inst = while_condition->root_instruction();
  switch (c_inst->opcode()) {
    case HloOpcode::kLt:
    case HloOpcode::kLe:
    case HloOpcode::kGt:
    case HloOpcode::kGe:
      break;
    default:
      return xla::FailedPrecondition(err_msg);
  }

  // Make sure that for the comparison instruction:
  // * LHS is an integral GTE from from parameter 0
  // * RHS is a constant which is integral
  {
    const bool lhs_is_GTE_param_from_param_0 =
        WhileLoopUtil::IsGTEFromParamIndex(c_inst->operand(0), 0);
    if (!lhs_is_GTE_param_from_param_0) {
      return xla::FailedPrecondition(err_msg);
    }

    const bool rhs_is_integral_const =
        WhileLoopUtil::IsIntegralConstant(c_inst->operand(1));
    if (!rhs_is_integral_const) {
      return xla::FailedPrecondition(err_msg);
    }
  }

  HloInstruction* comp_GTE = c_inst->mutable_operand(0);
  int64 cond_tuple_index = comp_GTE->tuple_index();

  const HloInstruction* init_inst =
      while_inst->operand(0)->operand(cond_tuple_index);

  if (init_inst->opcode() != HloOpcode::kConstant) {
    return xla::FailedPrecondition(err_msg);
  }

  int64 initial_value;
  TF_ASSIGN_OR_RETURN(initial_value,
                      LiteralScalarInt64toInt64(init_inst->literal()));

  const HloInstruction* limit_inst = c_inst->operand(1);

  if (limit_inst->opcode() != HloOpcode::kConstant) {
    return xla::FailedPrecondition(err_msg);
  }

  int64 compare_value;
  TF_ASSIGN_OR_RETURN(compare_value,
                      LiteralScalarInt64toInt64(limit_inst->literal()));

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
    return xla::FailedPrecondition(err_msg);
  }

  // Check that the mapped GTE instruction is modified by 1 and that the
  // resulting increment is *only* used in the output tuple of the while body in
  // the same index
  HloOpcode delta_op;
  switch (c_inst->opcode()) {
    case HloOpcode::kLt:
    case HloOpcode::kLe:
      delta_op = HloOpcode::kAdd;
      break;
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    default:
      delta_op = HloOpcode::kSubtract;
      break;
  }

  std::vector<HloInstruction*> matching_increments;
  TF_ASSIGN_OR_RETURN(matching_increments,
                      WhileLoopUtil::FindMatchingGTEIncrementsInsideBody(
                          body_GTE, while_body, delta_op));

  // Calculate and return the number of iterations
  if (matching_increments.size() == 1) {
    switch (c_inst->opcode()) {
      case HloOpcode::kLt:
        return compare_value - initial_value;
      case HloOpcode::kLe:
        return compare_value - initial_value + 1;
      case HloOpcode::kGe:
        return initial_value - compare_value + 1;
      case HloOpcode::kGt:
      default:
        return initial_value - compare_value;
    }
  } else {
    return xla::FailedPrecondition(err_msg);
  }
}

}  // namespace poplarplugin
}  // namespace xla