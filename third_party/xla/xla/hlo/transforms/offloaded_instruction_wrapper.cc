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

#include "xla/hlo/transforms/offloaded_instruction_wrapper.h"

#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla::offloader_util {

namespace {

absl::Status ClearComputeTypeFrontendAttribute(HloInstruction* instr) {
  FrontendAttributes copy_of_frontend_attributes = instr->frontend_attributes();
  copy_of_frontend_attributes.mutable_map()->erase(kXlaComputeTypeAttr);
  instr->set_frontend_attributes(copy_of_frontend_attributes);
  return absl::OkStatus();
}

void ClearSideEffects(HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kCustomCall) {
    static_cast<HloCustomCallInstruction*>(instr)
        ->set_custom_call_has_side_effect(false);
  }
}

}  // namespace

absl::Status RecursivelyClearComputeTypeFrontendAttribute(
    HloComputation* computation) {
  for (HloInstruction* instruction : computation->instructions()) {
    TF_RETURN_IF_ERROR(ClearComputeTypeFrontendAttribute(instruction));
    for (HloComputation* called_computation :
         instruction->called_computations()) {
      TF_RETURN_IF_ERROR(
          RecursivelyClearComputeTypeFrontendAttribute(called_computation));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::pair<HloInstruction*, HloCallInstruction*>>>
FindAndWrapOffloadedComputations(
    HloComputation& computation,
    absl::FunctionRef<bool(const HloInstruction*)> should_offload,
    absl::FunctionRef<bool(const HloInstruction&, const HloInstruction&)>
        should_fuse,
    absl::FunctionRef<absl::Status(HloInstruction*)>
        clear_backend_config_device_type,
    absl::string_view new_call_name_prefix) {
  // If a constant is used on TC and offloaded, clear offload annotations and
  // only materialize it on TC. This simplifies the dependency chain.
  for (HloInstruction* instr : computation.instructions()) {
    if (instr->IsConstant() && should_offload(instr)) {
      TF_RETURN_IF_ERROR(clear_backend_config_device_type(instr));
    }
  }

  std::vector<std::pair<HloInstruction*, HloCallInstruction*>>
      offloaded_instructions_and_calls;
  // On each iteration of the outer loop, try to create one offloaded
  // computation out of a connected set of offloaded instructions.
  while (true) {
    HloInstruction* offloaded_instr = nullptr;
    HloCallInstruction* offloaded_call_instr = nullptr;
    // Stores all the ancestor instructions of offloaded_call_instr which were
    // not added to the current offloaded computation.
    absl::flat_hash_set<HloInstruction*> unmerged_ancestors;
    std::vector<HloInstruction*> post_order =
        computation.MakeInstructionPostOrder();
    for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
      HloInstruction* instr = *it;
      // The current instruction is not an offload instruction.
      if (!should_offload(instr)) {
        if (absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
              return user == offloaded_call_instr ||
                     unmerged_ancestors.contains(user);
            })) {
          unmerged_ancestors.insert(instr);
        }
        continue;
      }

      VLOG(2) << "Offloading instruction: " << instr->ToString();

      // The current instruction is the root of a new offloaded computation.
      if (offloaded_call_instr == nullptr) {
        VLOG(2) << instr->name()
                << " is the root of a new offloaded computation";

        HloInstruction* call_instr;
        if (instr->opcode() == HloOpcode::kCall) {
          call_instr = instr;
        } else {
          call_instr = computation.CreateCallInstruction({instr});
          call_instr->SetAndSanitizeName(new_call_name_prefix);
          call_instr->UniquifyName(computation.parent());
          call_instr->set_frontend_attributes(instr->frontend_attributes());
        }
        offloaded_call_instr = absl::down_cast<HloCallInstruction*>(call_instr);
        CHECK_NE(offloaded_call_instr, nullptr);
        TF_RETURN_IF_ERROR(
            clear_backend_config_device_type(offloaded_call_instr));
        TF_RETURN_IF_ERROR(
            ClearComputeTypeFrontendAttribute(offloaded_call_instr));
        ClearSideEffects(instr);
        offloaded_instr = instr;
        continue;
      }

      // If the current instruction is indirectly connected to the current
      // offloaded computation, it must go in a separate offload computation.
      if (absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
            return unmerged_ancestors.contains(user);
          })) {
        VLOG(2) << instr->name()
                << " is indirectly connected to the current offloaded "
                   "computation, it must go in a separate offload computation";

        unmerged_ancestors.insert(instr);
        continue;
      }

      // The current instruction is directly connected to the current offloaded
      // computation.
      if (offloaded_call_instr->IsUserOf(instr)) {
        VLOG(2)
            << instr->name()
            << " is directly connected to the current offloaded computation";
        if (should_fuse(*offloaded_call_instr, *instr)) {
          bool instr_escapes_offloaded_computation =
              absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
                return user != offloaded_call_instr && !should_offload(user);
              });

          VLOG(3) << instr->name() << " fusing into existing computation";
          VLOG(6) << "instr_escapes_offloaded_computation: "
                  << instr_escapes_offloaded_computation;

          offloaded_call_instr->AppendInstructionIntoCalledComputation(
              instr, /*add_output=*/instr_escapes_offloaded_computation);
          ClearSideEffects(instr);
          offloaded_instr = instr;
        } else {
          unmerged_ancestors.insert(instr);
        }
        continue;
      }

      if (should_fuse(*offloaded_call_instr, *instr)) {
        VLOG(2) << instr->ToString()
                << " current instruction is disconnected from the current "
                   "offload computation";
        // If the current instruction is disconnected from the current offload
        // computation, include it anyway.
        offloaded_call_instr->AppendInstructionIntoCalledComputation(
            instr, /*add_output=*/true);
        ClearSideEffects(instr);
        offloaded_instr = instr;
      } else {
        unmerged_ancestors.insert(instr);
      }
    }

    // If there are no offload instructions left in the computation, stop
    // looking.
    if (offloaded_call_instr == nullptr) {
      break;
    }
    offloaded_instructions_and_calls.push_back(
        std::pair(offloaded_instr, offloaded_call_instr));

    for (HloInstruction* instr : computation.instructions()) {
      // If an offloaded instruction is a Sharding custom call or has control
      // dependencies (such as those around elided copies), remove it
      // explicitly since it won't be removed by HloDCE.
      if (instr->IsDead() && (instr->IsCustomCall("Sharding") ||
                              (instr->HasControlDependencies() &&
                               !instr->HasSuccessorControlDependencies()))) {
        TF_RETURN_IF_ERROR(instr->SafelyDropAllControlDependencies());
        TF_RETURN_IF_ERROR(computation.RemoveInstruction(instr));
      }
    }

    // DCE any offloaded instructions that have no remaining un-wrapped uses.
    TF_RETURN_IF_ERROR(HloDCE().Run(computation.parent()).status());

    VLOG(6) << "After offloading computation after DCE:";
    XLA_VLOG_LINES(6, computation.parent()->ToString());
  }
  return offloaded_instructions_and_calls;
}

}  // namespace xla::offloader_util
