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

#include "tensorflow/compiler/plugin/poplar/driver/passes/constant_slice_folding.h"

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

StatusOr<bool> ConstantSliceFolding::Run(HloModule* module) {
  auto evaluator = absl::make_unique<HloEvaluator>(/*max_loop_iterations=*/0);

  bool changed = false;

  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kSlice &&
          instruction->opcode() != HloOpcode::kReshape) {
        continue;
      }

      if (!hlo_query::AllOperandsAreConstants(*instruction)) {
        continue;
      }

      if (ShapeUtil::ElementsIn(instruction->shape()) != 1) {
        continue;
      }

      Literal result;
      if (!evaluator->TryEvaluate(instruction, &result)) {
        VLOG(2) << "ConstantSliceFolding folding failed for instruction: "
                << instruction->ToString();
        continue;
      }

      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          instruction, HloInstruction::CreateConstant(std::move(result))));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
