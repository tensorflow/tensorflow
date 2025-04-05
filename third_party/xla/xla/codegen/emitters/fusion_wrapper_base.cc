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
#include "xla/codegen/emitters/fusion_wrapper_base.h"

#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace emitters {

absl::StatusOr<bool> FusionWrapperBase::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto instructions = module->entry_computation()->MakeInstructionPostOrder();
  bool changed = false;

  std::function<absl::Status(HloInstruction*)> handle_instruction;
  handle_instruction = [&](HloInstruction* instruction) -> absl::Status {
    const HloOpcode opcode = instruction->opcode();
    if (opcode == HloOpcode::kConditional || opcode == HloOpcode::kWhile ||
        opcode == HloOpcode::kCall || opcode == HloOpcode::kAsyncStart) {
      for (auto* computation : instruction->called_computations()) {
        for (auto* inner_instruction :
             computation->MakeInstructionPostOrder()) {
          TF_RETURN_IF_ERROR(handle_instruction(inner_instruction));
        }
      }
      return absl::OkStatus();
    }
    if (!MustWrapInstruction(opcode)) {
      return absl::OkStatus();
    }
    auto* computation = instruction->parent();
    auto* fusion_instruction =
        computation->AddInstruction(HloInstruction::CreateFusion(
            instruction->shape(), ChooseFusionKind(*instruction, *instruction),
            instruction));
    const absl::string_view wrapped_opcode =
        HloOpcodeString(instruction->opcode());
    module->SetAndUniquifyInstrName(fusion_instruction,
                                    absl::StrCat("wrapped_", wrapped_opcode));
    module->SetAndUniquifyComputationName(
        fusion_instruction->fused_instructions_computation(),
        absl::StrCat("wrapped_", wrapped_opcode, "_computation"));
    if (module->has_schedule()) {
      module->schedule().replace_instruction(computation, instruction,
                                             fusion_instruction);
    }
    TF_RETURN_IF_ERROR(fusion_instruction->CopyAllControlDepsFrom(instruction));
    TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
    TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(fusion_instruction));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
    changed = true;
    return absl::OkStatus();
  };

  for (auto* instruction : instructions) {
    TF_RETURN_IF_ERROR(handle_instruction(instruction));
  }
  return changed;
}

}  // namespace emitters
}  // namespace xla
