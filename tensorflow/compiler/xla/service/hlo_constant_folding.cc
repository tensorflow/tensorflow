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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
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
  auto evaluator = MakeUnique<HloEvaluator>();

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
      // Skip Constant, Parameter, Reduce operation.
      // TODO(b/35975797): Enable Reduce operation once arbitrary computation
      // are supported by the evaluator.
      // TODO(b/64407269): Enable Tuple once the timeout issue is resolved.
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->opcode() == HloOpcode::kConstant ||
          instruction->opcode() == HloOpcode::kTuple ||
          instruction->opcode() == HloOpcode::kReduce) {
        continue;
      }
      // Skip instructions with non-constant operands.
      if (!hlo_query::AllOperandsAreConstants(*instruction)) {
        continue;
      }

      // Broadcasts dramatically increase the size of constants, which is often
      // detrimental to performance and memory capacity, so do not fold
      // broadcasts.
      if (instruction->opcode() == HloOpcode::kBroadcast) {
        continue;
      }

      std::unique_ptr<Literal> result = evaluator->TryEvaluate(instruction);
      // Currently we skip unimplemented operations.
      // TODO(b/35975797): Fold constant computations for more operations.
      if (result == nullptr) {
        VLOG(2) << "Constant folding failed for instruction: "
                << instruction->ToString();
        continue;
      }

      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          instruction, HloInstruction::CreateConstant(std::move(result))));
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "HloConstantFolding::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
