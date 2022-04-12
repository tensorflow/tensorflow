/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

// Checks whether instr is or transitively contains an instruction that we
// shouldn't fold.
//
// Specifically, we don't fold kRng or kAfterAll instructions:
//
//  - kRng is already marked as side-effecting and so is skipped elsewhere, but
//    we check for it here.  Even kRng weren't side-effecting and took an
//    explicit seed, we *still* wouldn't want to constant-fold it, because the
//    evaluator's handling of rng is not guaranteed to be identical to any
//    particular backend's rng.
//
//  - kAfterAll needs to be skipped because a kAfterAll op with no args can
//    currently materialize a token "out of thin air".  TODO(b/110532604):
//    Remove this check once AfterAll requires at least one operand, in which
//    case constant folding will be impossible.
static bool IsOrContainsIllegalInstr(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kAfterAll ||
      instr->opcode() == HloOpcode::kRng) {
    return true;
  }
  for (const HloComputation* c : instr->called_computations()) {
    if (absl::c_any_of(c->instructions(), IsOrContainsIllegalInstr)) {
      return true;
    }
  }
  return false;
}

/*static*/ std::atomic<int64_t> HloConstantFolding::slow_op_counter_{0};

StatusOr<bool> HloConstantFolding::Run(HloModule* module) {
  // Limit the constant folding to 0 iterations to skip folding loops. This
  // retains the behavior from before while loop support in HloEvaluator and may
  // be revised.
  auto evaluator = absl::make_unique<HloEvaluator>(/*max_loop_iterations=*/0);
  // fast-path lets us e.g. use Eigen for matmuls.
  evaluator->set_use_fast_path(true);

  bool changed = false;

  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // Skip dead code.
      if (instruction->IsDead()) {
        continue;
      }

      // We only handle instructions where
      //
      //  - at least one operand is a constant, and
      //  - all other operands are either constants or broadcast(constant).
      //
      // Why this particular set of rules around broadcasts?
      //
      //  - We don't want to fold broadcast(constant) on its own, because in
      //    general it's "simpler" to remember that it's a broadcast.  Also,
      //    algsimp will fold an all-one-value constant into a broadcast, so
      //    we'd just end up fighting with it.
      //
      //  - We don't want to fold an op where all operands are broadcasts of
      //    constants, because algsimp will transform op(broadcast(constant) =>
      //    broadcast(op(constant)).  Then we can constant-fold the smaller op.
      //
      //  - So the only remaining case is where some but not all operands are
      //    broadcasts of constants, e.g. op(constant, broadcast(constant)).
      //
      if (!absl::c_any_of(instruction->operands(),
                          [](const HloInstruction* operand) {
                            return operand->opcode() == HloOpcode::kConstant;
                          }) ||
          !absl::c_all_of(
              instruction->operands(), [](const HloInstruction* operand) {
                return operand->opcode() == HloOpcode::kConstant ||
                       (operand->opcode() == HloOpcode::kBroadcast &&
                        operand->operand(0)->opcode() == HloOpcode::kConstant);
              })) {
        continue;
      }

      // Don't fold Constant, Parameter, and Tuple instructions.  Tuple
      // constants are not directly supported by any backends, hence folding
      // Tuple is not useful and would in fact be expanded back into kTuple by
      // Algebraic Simplifier.
      //
      // (We do allow folding subcomputations that contain these instructions.)
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->opcode() == HloOpcode::kConstant ||
          instruction->opcode() == HloOpcode::kTuple) {
        continue;
      }

      // Broadcasts dramatically increase the size of constants, which is often
      // detrimental to performance and memory capacity, so do not fold
      // broadcasts.
      if (instruction->opcode() == HloOpcode::kBroadcast ||
          instruction->opcode() == HloOpcode::kIota) {
        continue;
      }

      // Check for instructions that we can't fold even if they appear inside of
      // a subcomputation (e.g. a kCall).
      if (IsOrContainsIllegalInstr(instruction)) {
        continue;
      }

      // Don't constant-fold side-effecting instructions or instructions which
      // contain side-effecting instructions.
      if (instruction->HasSideEffect()) {
        continue;
      }

      // Don't constant fold unless it's a net positive or the output is small.
      if (instruction->shape().IsArray()) {
        int64_t elements_in_removed_operands = 0;
        for (HloInstruction* operand : instruction->operands()) {
          if (operand->user_count() == 1 && operand->shape().IsArray()) {
            elements_in_removed_operands +=
                ShapeUtil::ElementsIn(operand->shape());
          }
        }
        int64_t elements_in_constant =
            ShapeUtil::ElementsIn(instruction->shape());

        static const int64_t kMaximumConstantSizeElements = 45 * 1000 * 1000;
        if (elements_in_constant > elements_in_removed_operands &&
            elements_in_constant > kMaximumConstantSizeElements) {
          continue;
        }
      }

      VLOG(5) << "Constant folding: " << instruction->ToString();

      absl::Duration slow_timeout =
          absl::Seconds(uint64_t{1} << slow_op_counter_.load());
      SlowOperationAlarm slow_alarm(slow_timeout, [instruction, slow_timeout] {
        const bool ndebug =
#if NDEBUG
            true;
#else
            false;
#endif
        absl::string_view explanation_msg =
            ndebug
                ? "This isn't necessarily a bug; constant-folding is "
                  "inherently a trade-off between compilation time and speed "
                  "at runtime.  XLA has some guards that attempt to keep "
                  "constant folding from taking too long, but fundamentally "
                  "you'll always be able to come up with an input program that "
                  "takes a long time.\n\n"
                  "If you'd like to file a bug, run with envvar "
                  "XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results."
                : "XLA was built without compiler optimizations, which can be "
                  "slow.  Try rebuilding with -c opt.";
        return absl::StrFormat(
            "Constant folding an instruction is taking > %s:\n\n"
            "  %s\n\n"  // instruction->ToString()
            "%s",       // explanation_msg
            absl::FormatDuration(slow_timeout), instruction->ToString(),
            explanation_msg);
      });

      // Currently we skip unimplemented operations.
      // TODO(b/35975797): Fold constant computations for more operations.
      Literal result;
      if (!evaluator->TryEvaluate(
              instruction, &result,
              /*recursively_evaluate_nonconstant_operands=*/true)) {
        VLOG(2) << "Constant folding failed for instruction: "
                << instruction->ToString();
        continue;
      }

      slow_alarm.cancel();
      if (slow_alarm.fired()) {
        slow_op_counter_++;
      }

      VLOG(4) << "Constant folded: " << instruction->ToString();

      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          instruction, HloInstruction::CreateConstant(std::move(result))));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
