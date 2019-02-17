#include "tensorflow/compiler/plugin/poplar/driver/tools/while_loop_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include <map>
#include <set>

namespace xla {
namespace poplarplugin {
namespace {

StatusOr<int64> GetLoopDelta(const HloInstruction* delta) {
  if (delta->opcode() == HloOpcode::kConstant) {
    if (WhileLoopUtil::Is32BitsOrLessIntegerConstant(delta)) {
      TF_ASSIGN_OR_RETURN(int64 delta_value,
                          LiteralScalarToNativeType<int64>(delta->literal()));
      if (std::llabs(delta_value) == 1) {
        return delta_value;
      }
    }

  } else if (delta->opcode() == HloOpcode::kNegate &&
             delta->operand(0)->opcode() == HloOpcode::kConstant) {
    const HloInstruction* constant = delta->operand(0);
    if (WhileLoopUtil::Is32BitsOrLessIntegerConstant(constant)) {
      TF_ASSIGN_OR_RETURN(int64 delta_value, LiteralScalarToNativeType<int64>(
                                                 constant->literal()));
      if (std::llabs(delta_value) == 1) {
        return -delta_value;
      }
    }
  }
  return xla::FailedPrecondition("Not a loop delta.");
}

// Returns a the delta counter if this instruction is incremented/decremented
// by (-)1 - simplify it so that it's always an add - for example subtracting 1
// is the same as adding -1.
absl::optional<int64> GetLoopCounter(const HloInstruction* loop_counter) {
  // Loop delta cases
  if (loop_counter->opcode() == HloOpcode::kAdd) {
    // For addition, a loop delta can be either LHS or RHS
    auto lhs_statusor = GetLoopDelta(loop_counter->operand(0));
    if (lhs_statusor.ok()) {
      return lhs_statusor.ValueOrDie();
    }
    auto rhs_statusor = GetLoopDelta(loop_counter->operand(1));
    if (rhs_statusor.ok()) {
      return rhs_statusor.ValueOrDie();
    }
  } else if (loop_counter->opcode() == HloOpcode::kSubtract) {
    // For subtract, a loop delta can be only on RHS
    auto rhs_statusor = GetLoopDelta(loop_counter->operand(1));
    if (rhs_statusor.ok()) {
      return -rhs_statusor.ValueOrDie();
    }
  }
  return absl::nullopt;
}
}  // namespace

bool WhileLoopUtil::IsGTEFromParamIndex(const HloInstruction* inst,
                                        int64 param_index) {
  return inst->opcode() == HloOpcode::kGetTupleElement &&
         inst->operand(0)->opcode() == HloOpcode::kParameter &&
         inst->operand(0)->parameter_number() == param_index;
}
bool WhileLoopUtil::Is32BitsOrLessIntegerConstant(const HloInstruction* inst) {
  return IsScalarConstant(inst) &&
         (ShapeUtil::ElementIsIntegralWithBits(inst->shape(), 8) ||
          ShapeUtil::ElementIsIntegralWithBits(inst->shape(), 16) ||
          ShapeUtil::ElementIsIntegralWithBits(inst->shape(), 32));
}

std::vector<std::pair<HloInstruction*, int64>>
WhileLoopUtil::FindMatchingLoopDeltasInsideBody(
    const HloInstruction* inst, const HloComputation* while_body) {
  CHECK_EQ(inst->opcode(), HloOpcode::kGetTupleElement);
  std::vector<std::pair<HloInstruction*, int64>> ret;

  // Check that the GTE is on a tuple that is only used by GTEs.
  const HloInstruction* tuple = inst->operand(0);
  if (absl::c_any_of(tuple->users(), [](HloInstruction* user) {
        return user->opcode() != HloOpcode::kGetTupleElement;
      })) {
    return ret;
  }

  const int64 tuple_index_cond = inst->tuple_index();
  for (HloInstruction* user : inst->users()) {
    // Check whether this is a loop counter.
    auto optional_loop_counter = GetLoopCounter(user);
    if (!optional_loop_counter) {
      continue;
    }

    // check that the user is only used once
    if (user->user_count() == 1) {
      HloInstruction* inst = user->users()[0];
      // check that inst is a root Tuple instruction with the user in the
      // right index
      if (inst->opcode() == HloOpcode::kTuple &&
          inst == while_body->root_instruction() &&
          inst->operands()[tuple_index_cond] == user) {
        ret.push_back({user, *optional_loop_counter});
      }
    }
  }
  return ret;
}

}  // namespace poplarplugin
}  // namespace xla