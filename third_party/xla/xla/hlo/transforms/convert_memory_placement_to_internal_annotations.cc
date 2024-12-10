/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/convert_memory_placement_to_internal_annotations.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/side_effect_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> ConvertMemoryPlacementToInternalAnnotations::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* c : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : c->MakeInstructionPostOrder()) {
      if (instruction->IsCustomCall(
              host_memory_offload_annotations::kDevicePlacement)) {
        const auto& frontend_attributes = instruction->frontend_attributes();
        const auto it = frontend_attributes.map().find(kXlaBufferPlacementAttr);
        if (it == frontend_attributes.map().end()) {
          continue;
        }
        // XLA currently does not differentiate between pinned and unpinned host
        // memory.
        const bool is_to_host_case =
            (it->second ==
                 host_memory_offload_annotations::kMemoryTargetPinnedHost ||
             it->second ==
                 host_memory_offload_annotations::kMemoryTargetUnpinnedHost);
        const bool is_to_device_case =
            (it->second ==
             host_memory_offload_annotations::kMemoryTargetDevice);
        if (!is_to_host_case && !is_to_device_case) {
          continue;
        }
        if (is_to_host_case) {
          VLOG(1) << "Process forward case: " << instruction->ToString();
          if (instruction->operand_count() != 1) {
            return Internal(
                "Custom calls with target %s must have exactly one operand. %s "
                "has %d.",
                host_memory_offload_annotations::kDevicePlacement,
                instruction->name(), instruction->operand_count());
          }
          HloInstruction* input = instruction->mutable_operand(0);
          HloInstruction* move_to_host_custom_call =
              c->AddInstruction(HloInstruction::CreateCustomCall(
                  input->shape(), {input},
                  host_memory_offload_annotations::
                      kMoveToHostCustomCallTarget));
          if (instruction->has_sharding()) {
            move_to_host_custom_call->set_sharding(instruction->sharding());
          }
          TF_RETURN_IF_ERROR(
              instruction->ReplaceAllUsesWith(move_to_host_custom_call));
          TF_RETURN_IF_ERROR(
              c->RemoveInstructionAndUnusedOperands(instruction));
          changed = true;
        } else if (is_to_device_case) {
          VLOG(1) << "Process backward case: " << instruction->ToString();
          HloInstruction* custom_call_operand = instruction->mutable_operand(0);
          HloInstruction* new_result =
              c->AddInstruction(HloInstruction::CreateCustomCall(
                  custom_call_operand->shape(), {custom_call_operand},
                  host_memory_offload_annotations::
                      kMoveToDeviceCustomCallTarget));
          TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(new_result));
          TF_RETURN_IF_ERROR(
              c->RemoveInstructionAndUnusedOperands(instruction));
          changed = true;
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
