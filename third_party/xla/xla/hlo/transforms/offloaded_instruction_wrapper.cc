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
#include "xla/tsl/platform/statusor.h"
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

absl::Status ClearConstantsComputeType(
    HloComputation& computation,
    absl::FunctionRef<bool(const HloInstruction*)> should_offload,
    absl::FunctionRef<absl::Status(HloInstruction*)>
        clear_backend_config_device_type) {
  // If a constant is used on TC and offloaded, clear offload annotations and
  // only materialize it on TC. This simplifies the dependency chain.
  for (HloInstruction* instr : computation.instructions()) {
    if (instr->IsConstant() && should_offload(instr)) {
      TF_RETURN_IF_ERROR(clear_backend_config_device_type(instr));
    }
  }
  return absl::OkStatus();
}

absl::Status RemoveDeadOffloadedInstructions(HloComputation& computation) {
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
  return absl::OkStatus();
}

// Stateful helper for finding and wrapping a single offloaded computation.
class SingleOffloadedComputationWrapper {
 public:
  SingleOffloadedComputationWrapper(
      HloComputation& computation,
      absl::FunctionRef<bool(const HloInstruction*)> should_offload,
      absl::FunctionRef<bool(const HloInstruction&, const HloInstruction&)>
          should_fuse,
      absl::FunctionRef<absl::Status(HloInstruction*)>
          clear_backend_config_device_type,
      absl::string_view new_call_name_prefix)
      : computation_(computation),
        should_offload_(should_offload),
        should_fuse_(should_fuse),
        clear_backend_config_device_type_(clear_backend_config_device_type),
        new_call_name_prefix_(new_call_name_prefix) {}

  absl::StatusOr<std::pair<HloInstruction*, HloCallInstruction*>> WrapNext() {
    std::vector<HloInstruction*> post_order =
        computation_.MakeInstructionPostOrder();
    for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
      if (absl::Status status = ProcessInstruction(*it); !status.ok()) {
        return status;
      }
    }

    if (offloaded_call_instr_ == nullptr) {
      return std::pair<HloInstruction*, HloCallInstruction*>{nullptr, nullptr};
    }
    return std::pair(offloaded_instr_, offloaded_call_instr_);
  }

 private:
  absl::Status ProcessInstruction(HloInstruction* instr) {
    if (!should_offload_(instr)) {
      if (IsAncestor(instr)) {
        unmerged_ancestors_.insert(instr);
      }
      return absl::OkStatus();
    }

    VLOG(2) << "Offloading instruction: " << instr->ToString();

    if (offloaded_call_instr_ == nullptr) {
      return CreateNewOffloadedComputation(instr);
    }

    if (absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
          return unmerged_ancestors_.contains(user);
        })) {
      VLOG(2) << instr->name()
              << " is indirectly connected to the current offloaded "
                 "computation, it must go in a separate offload computation";
      unmerged_ancestors_.insert(instr);
      return absl::OkStatus();
    }

    if (offloaded_call_instr_->IsUserOf(instr)) {
      return FuseDirectlyConnectedInstruction(instr);
    }

    return FuseDisconnectedInstruction(instr);
  }

  bool IsAncestor(HloInstruction* instr) {
    return absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
      return user == offloaded_call_instr_ ||
             unmerged_ancestors_.contains(user);
    });
  }

  absl::Status CreateNewOffloadedComputation(HloInstruction* instr) {
    VLOG(2) << instr->name() << " is the root of a new offloaded computation";

    HloInstruction* call_instr;
    if (instr->opcode() == HloOpcode::kCall) {
      call_instr = instr;
    } else {
      call_instr = computation_.CreateCallInstruction({instr});
      call_instr->SetAndSanitizeName(new_call_name_prefix_);
      call_instr->UniquifyName(computation_.parent());
      call_instr->set_frontend_attributes(instr->frontend_attributes());
    }
    offloaded_call_instr_ = tsl::down_cast<HloCallInstruction*>(call_instr);
    CHECK_NE(offloaded_call_instr_, nullptr);
    TF_RETURN_IF_ERROR(
        clear_backend_config_device_type_(offloaded_call_instr_));
    TF_RETURN_IF_ERROR(
        ClearComputeTypeFrontendAttribute(offloaded_call_instr_));
    ClearSideEffects(instr);
    offloaded_instr_ = instr;
    return absl::OkStatus();
  }

  absl::Status FuseDirectlyConnectedInstruction(HloInstruction* instr) {
    VLOG(2) << instr->name()
            << " is directly connected to the current offloaded computation";
    if (should_fuse_(*offloaded_call_instr_, *instr)) {
      bool instr_escapes_offloaded_computation =
          absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
            return user != offloaded_call_instr_ && !should_offload_(user);
          });

      VLOG(3) << instr->name() << " fusing into existing computation";
      VLOG(6) << "instr_escapes_offloaded_computation: "
              << instr_escapes_offloaded_computation;

      offloaded_call_instr_->AppendInstructionIntoCalledComputation(
          instr, /*add_output=*/instr_escapes_offloaded_computation);
      ClearSideEffects(instr);
      offloaded_instr_ = instr;
    } else {
      unmerged_ancestors_.insert(instr);
    }
    return absl::OkStatus();
  }

  absl::Status FuseDisconnectedInstruction(HloInstruction* instr) {
    if (should_fuse_(*offloaded_call_instr_, *instr)) {
      VLOG(2) << instr->ToString()
              << " current instruction is disconnected from the current "
                 "offload computation";
      offloaded_call_instr_->AppendInstructionIntoCalledComputation(
          instr, /*add_output=*/true);
      ClearSideEffects(instr);
      offloaded_instr_ = instr;
    } else {
      unmerged_ancestors_.insert(instr);
    }
    return absl::OkStatus();
  }

  HloComputation& computation_;
  absl::FunctionRef<bool(const HloInstruction*)> should_offload_;
  absl::FunctionRef<bool(const HloInstruction&, const HloInstruction&)>
      should_fuse_;
  absl::FunctionRef<absl::Status(HloInstruction*)>
      clear_backend_config_device_type_;
  absl::string_view new_call_name_prefix_;

  HloInstruction* offloaded_instr_ = nullptr;
  HloCallInstruction* offloaded_call_instr_ = nullptr;
  absl::flat_hash_set<HloInstruction*> unmerged_ancestors_;
};

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
  TF_RETURN_IF_ERROR(ClearConstantsComputeType(
      computation, should_offload, clear_backend_config_device_type));

  std::vector<std::pair<HloInstruction*, HloCallInstruction*>>
      offloaded_instructions_and_calls;

  while (true) {
    SingleOffloadedComputationWrapper wrapper(
        computation, should_offload, should_fuse,
        clear_backend_config_device_type, new_call_name_prefix);
    TF_ASSIGN_OR_RETURN(auto instr_and_call, wrapper.WrapNext());

    if (instr_and_call.second == nullptr) {
      break;
    }

    offloaded_instructions_and_calls.push_back(instr_and_call);

    TF_RETURN_IF_ERROR(RemoveDeadOffloadedInstructions(computation));

    // DCE any offloaded instructions that have no remaining un-wrapped uses.
    TF_RETURN_IF_ERROR(HloDCE().Run(computation.parent()).status());

    VLOG(6) << "After offloading computation after DCE:";
    XLA_VLOG_LINES(6, computation.parent()->ToString());
  }
  return offloaded_instructions_and_calls;
}

}  // namespace xla::offloader_util
