/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/command_buffer_scheduling.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

StatusOr<bool> CommandBufferScheduling::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  std::vector<HloInstruction*> instructions = entry->MakeInstructionPostOrder();

  // TODO(anlunx): Add support for multiple fusion instructions.
  // TODO(anlunx): Add support for conditionals and while loops.
  for (HloInstruction* instruction : instructions) {
    if (instruction->opcode() != HloOpcode::kFusion) {
      continue;
    }

    auto* fusion = static_cast<HloFusionInstruction*>(instruction);
    auto builder = HloComputation::Builder("command_buffer");

    // Create parameters to the command buffer computation.
    std::vector<HloInstruction*> parameters;
    for (int64_t i = 0; i < fusion->operand_count(); i++) {
      const HloInstruction* operand = fusion->operand(i);
      TF_ASSIGN_OR_RETURN(HloInstruction * parameter,
                          builder.AddParameter(HloInstruction::CreateParameter(
                              i, operand->shape(), "param")));
      parameters.push_back(parameter);
    }

    // Create the fusion instruction inside the command buffer.
    builder.AddInstruction(HloInstruction::CreateFusion(
        fusion->shape(), fusion->fusion_kind(), parameters,
        fusion->fused_instructions_computation()));

    HloComputation* command_buffer =
        module->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                  /*is_entry=*/false);

    // Replace the fusion instruction with a call to the command buffer.
    HloInstruction* call_command_buffer =
        entry->AddInstruction(HloInstruction::CreateCall(
            fusion->shape(), fusion->operands(), command_buffer));
    TF_RETURN_IF_ERROR(call_command_buffer->CopyAllControlDepsFrom(fusion));
    TF_RETURN_IF_ERROR(fusion->DropAllControlDeps());
    TF_RETURN_IF_ERROR(entry->ReplaceInstruction(fusion, call_command_buffer));
  }

  return true;
}

}  // namespace xla::gpu
