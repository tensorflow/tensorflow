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
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

StatusOr<bool> HloConstantFolding::Run(HloModule* module) {
  // Limit the constant folding to 0 iterations to skip folding loops. This
  // retains the behavior from before while loop support in HloEvaluator and may
  // be revised.
  auto evaluator = absl::make_unique<HloEvaluator>(/*max_loop_iterations=*/0);

  XLA_VLOG_LINES(2,
                 "HloConstantFolding::Run(), before:\n" + module->ToString());
  bool changed = false;

  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // Skip dead code.
      if (instruction->user_count() == 0 &&
          computation->root_instruction() != instruction) {
        continue;
      }
      // Skip Constant, Parameter, Tuple, AfterAll, Rng operations.
      // Tuple constants are not directly supported by any backends, hence
      // folding Tuple is not useful and would in fact be expanded back into
      // kTuple by Algebraic Simplifier.
      // TODO(b/110532604): Enable AfterAll once AfterAll requires at least one
      // operand in which case constant folding will be impossible and this
      // special case is not necessary.
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->opcode() == HloOpcode::kConstant ||
          instruction->opcode() == HloOpcode::kTuple ||
          instruction->opcode() == HloOpcode::kAfterAll ||
          instruction->opcode() == HloOpcode::kRng) {
        continue;
      }

      // Skip instructions with non-constant operands.
      if (!hlo_query::AllOperandsAreConstants(*instruction)) {
        continue;
      }

      // Broadcasts dramatically increase the size of constants, which is often
      // detrimental to performance and memory capacity, so do not fold
      // broadcasts.
      if (instruction->opcode() == HloOpcode::kBroadcast ||
          instruction->opcode() == HloOpcode::kIota) {
        continue;
      }

      // Don't constant fold unless it's a net positive or the output is small.
      if (instruction->shape().IsArray()) {
        int64 elements_in_removed_operands = 0;
        for (HloInstruction* operand : instruction->operands()) {
          if (operand->user_count() == 1 && operand->shape().IsArray()) {
            elements_in_removed_operands +=
                ShapeUtil::ElementsIn(operand->shape());
          }
        }
        int64 elements_in_constant =
            ShapeUtil::ElementsIn(instruction->shape());

        static const int64 kMaximumConstantSizeElements = 2 * 1000 * 1000;
        if (elements_in_constant > elements_in_removed_operands &&
            elements_in_constant > kMaximumConstantSizeElements) {
          continue;
        }
      }

      Literal result;
      // Currently we skip unimplemented operations.
      // TODO(b/35975797): Fold constant computations for more operations.
      if (!evaluator->TryEvaluate(instruction, &result)) {
        VLOG(2) << "Constant folding failed for instruction: "
                << instruction->ToString();
        continue;
      }
      VLOG(4) << "Constant folded: " << instruction->ToString();

      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          instruction, HloInstruction::CreateConstant(std::move(result))));
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "HloConstantFolding::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
