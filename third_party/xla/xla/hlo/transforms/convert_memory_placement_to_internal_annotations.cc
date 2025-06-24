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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/memory_annotations.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
absl::StatusOr<absl::string_view> GetCustomCallTarget(
    absl::string_view external_annotation) {
  if (external_annotation == memory_annotations::kMemoryTargetPinnedHost ||
      external_annotation == memory_annotations::kMemoryTargetUnpinnedHost) {
    return memory_annotations::kMoveToHostCustomCallTarget;
  }
  if (external_annotation == memory_annotations::kMemoryTargetDevice) {
    return memory_annotations::kMoveToDeviceCustomCallTarget;
  }
  if (external_annotation == memory_annotations::kMemoryTargetDeviceSram) {
    return memory_annotations::kPinToDeviceSramCustomCallTarget;
  }
  if (external_annotation == memory_annotations::kMemoryTargetPinnedDevice) {
    return memory_annotations::kPinToDeviceCustomCallTarget;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid external annotation: ", external_annotation));
}

absl::StatusOr<bool>
ConvertCustomCallWithExternalAnnotationToInternalAnnotation(
    HloComputation* c, HloInstruction* instruction) {
  const auto& frontend_attributes = instruction->frontend_attributes();
  const auto it = frontend_attributes.map().find(kXlaBufferPlacementAttr);
  if (it == frontend_attributes.map().end()) {
    return false;
  }
  // XLA currently does not differentiate between pinned and unpinned host
  // memory.
  const bool is_to_host_case =
      (it->second == memory_annotations::kMemoryTargetPinnedHost ||
       it->second == memory_annotations::kMemoryTargetUnpinnedHost);
  const bool is_to_device_case =
      (it->second == memory_annotations::kMemoryTargetDevice ||
       it->second == memory_annotations::kMemoryTargetDeviceSram ||
       it->second == memory_annotations::kMemoryTargetPinnedDevice);
  if (!is_to_host_case && !is_to_device_case) {
    return false;
  }
  const absl::StatusOr<absl::string_view> custom_call_target =
      GetCustomCallTarget(it->second);
  TF_RETURN_IF_ERROR(custom_call_target.status());
  if (is_to_host_case) {
    VLOG(1) << "Process forward case: " << instruction->ToString();
    if (instruction->operand_count() != 1) {
      return Internal(
          "Custom calls with target %s must have exactly one operand. %s "
          "has %d.",
          memory_annotations::kDevicePlacement, instruction->name(),
          instruction->operand_count());
    }
    HloInstruction* input = instruction->mutable_operand(0);
    HloInstruction* move_to_host_custom_call =
        c->AddInstruction(HloInstruction::CreateCustomCall(
            input->shape(), {input}, *custom_call_target));
    if (instruction->has_sharding()) {
      move_to_host_custom_call->set_sharding(instruction->sharding());
    }
    TF_RETURN_IF_ERROR(
        instruction->ReplaceAllUsesWith(move_to_host_custom_call));
    TF_RETURN_IF_ERROR(c->RemoveInstructionAndUnusedOperands(instruction));
    return true;
  } else if (is_to_device_case) {
    VLOG(1) << "Process backward case: " << instruction->ToString();
    HloInstruction* custom_call_operand = instruction->mutable_operand(0);
    HloInstruction* new_result =
        c->AddInstruction(HloInstruction::CreateCustomCall(
            custom_call_operand->shape(), {custom_call_operand},
            *custom_call_target));
    TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(new_result));
    TF_RETURN_IF_ERROR(c->RemoveInstructionAndUnusedOperands(instruction));
    return true;
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> ConvertMemoryPlacementToInternalAnnotations::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* c : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : c->MakeInstructionPostOrder()) {
      if (instruction->IsCustomCall(memory_annotations::kDevicePlacement)) {
        TF_ASSIGN_OR_RETURN(
            auto result,
            ConvertCustomCallWithExternalAnnotationToInternalAnnotation(
                c, instruction));
        changed |= result;
      }
    }
  }
  return changed;
}

}  // namespace xla
