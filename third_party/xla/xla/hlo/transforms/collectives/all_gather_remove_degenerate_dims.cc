/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/all_gather_remove_degenerate_dims.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> AllGatherRemoveDegenerateDims::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAllGather) {
        continue;
      }
      // Skip all-gathers we can't change.
      auto* all_gather = Cast<HloAllGatherInstruction>(inst);
      if (all_gather->constrain_layout() ||
          !ShapeUtil::HasDegenerateDimensions(inst->shape())) {
        continue;
      }

      int64_t all_gather_dim = all_gather->all_gather_dimension();
      Shape new_operand_shape = inst->operand(0)->shape();
      Shape new_all_gather_shape = inst->shape();
      for (int i = inst->shape().dimensions().size() - 1; i >= 0; --i) {
        if (i != all_gather_dim && new_all_gather_shape.dimensions(i) == 1) {
          new_operand_shape.DeleteDimension(i);
          new_all_gather_shape.DeleteDimension(i);
          if (i < all_gather_dim) {
            --all_gather_dim;
          }
        }
      }

      // If this was a degenerate all-gather without any other degenerate
      // dimensions, the operand shape will be unchanged.
      if (new_operand_shape == inst->operand(0)->shape()) {
        continue;
      }

      auto* reshaped_operand =
          computation->AddInstruction(HloInstruction::CreateReshape(
              new_operand_shape, inst->mutable_operand(0)));
      auto* new_all_gather = Cast<HloAllGatherInstruction>(
          computation->AddInstruction(all_gather->CloneWithNewOperands(
              new_all_gather_shape, {reshaped_operand})));
      new_all_gather->set_all_gather_dimension(all_gather_dim);

      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          inst, HloInstruction::CreateReshape(inst->shape(), new_all_gather)));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
