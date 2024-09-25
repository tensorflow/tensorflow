/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/zero_sized_hlo_elimination.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> ZeroSizedHloElimination::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : comp->MakeInstructionPostOrder()) {
      if (instruction->HasSideEffect() || !instruction->shape().IsArray() ||
          instruction->opcode() == HloOpcode::kConstant) {
        continue;
      }
      if (comp->IsSafelyRemovable(instruction) &&
          ShapeUtil::IsZeroElementArray(instruction->shape()) &&
          instruction->shape().is_static()) {
        // If the instruction doesn't have a layout, use a default layout for
        // the literal.
        Shape shape = instruction->shape();
        if (!LayoutUtil::HasLayout(shape)) {
          LayoutUtil::SetToDefaultLayout(&shape);
        }
        TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
            instruction,
            HloInstruction::CreateConstant(Literal::CreateFromShape(shape))));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
