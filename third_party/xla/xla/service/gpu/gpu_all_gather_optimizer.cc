/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_all_gather_optimizer.h"

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> AllGatherOptimizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (!HloOpcodeIsBinaryCommutative(instruction->opcode())) {
        continue;
      }

      HloInstruction* left_op = instruction->mutable_operand(0);
      HloInstruction* right_op = instruction->mutable_operand(1);

      if (right_op->opcode() != HloOpcode::kAllGather ||
          left_op->opcode() != HloOpcode::kAllGather) {
        VLOG(2) << "Binary op's operands are not all-gather deduced types.";
        continue;
      }

      auto* left_all_gather = Cast<HloAllGatherInstruction>(left_op);
      auto* right_all_gather = Cast<HloAllGatherInstruction>(right_op);

      if (right_all_gather->constrain_layout() !=
              left_all_gather->constrain_layout() ||
          right_all_gather->use_global_device_ids() !=
              left_all_gather->use_global_device_ids() ||
          !ReplicaGroupsEqual(right_all_gather->replica_groups(),
                              left_all_gather->replica_groups())) {
        VLOG(2) << "The right and left all-gather ops are not compatible "
                   "to merge. ";
        continue;
      }

      if (!ShapeUtil::Equal(left_all_gather->operand(0)->shape(),
                            right_all_gather->operand(0)->shape())) {
        VLOG(2) << "all-gather operands have different shapes";
        continue;
      }

      if (right_all_gather->user_count() != 1 ||
          left_all_gather->user_count() != 1) {
        VLOG(2) << "all-gather user_count > 1 ";
        continue;
      }
      auto index_in_full_shape =
          computation->AddInstruction(HloInstruction::CreateBinary(
              right_all_gather->operand(0)->shape(), instruction->opcode(),
              left_all_gather->mutable_operand(0),
              right_all_gather->mutable_operand(0)));

      int64_t all_gather_dimension =
          Cast<HloAllGatherInstruction>(right_all_gather)
              ->all_gather_dimension();

      auto combined = HloInstruction::CreateAllGather(
          left_all_gather->shape(), {index_in_full_shape}, all_gather_dimension,
          left_all_gather->replica_groups(),
          /*constrain_layout=*/false, left_all_gather->channel_id(),
          Cast<HloAllGatherInstruction>(left_all_gather)
              ->use_global_device_ids());

      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          instruction, std::move(combined)));
      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
