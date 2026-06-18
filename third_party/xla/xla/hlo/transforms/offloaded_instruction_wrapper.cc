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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/side_effect_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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
    ABSL_RETURN_IF_ERROR(ClearComputeTypeFrontendAttribute(instruction));
    for (HloComputation* called_computation :
         instruction->called_computations()) {
      ABSL_RETURN_IF_ERROR(
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
      ABSL_RETURN_IF_ERROR(clear_backend_config_device_type(instr));
    }
  }

  auto collect_offload_candidates = [&]() {
    std::vector<HloInstruction*> post_order =
        computation.MakeInstructionPostOrder();
    std::vector<HloInstruction*> candidates;
    for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
      if (should_offload(*it)) {
        candidates.push_back(*it);
      }
    }
    return candidates;
  };

  std::vector<HloInstruction*> offload_candidates =
      collect_offload_candidates();
  size_t candidate_idx = 0;

  std::vector<std::pair<HloInstruction*, int64_t>>
      offloaded_instructions_and_calls;
  // On each iteration of the outer loop, try to create one offloaded
  // computation out of a connected set of offloaded instructions.
  while (candidate_idx < offload_candidates.size()) {
    HloInstruction* root_instr = offload_candidates[candidate_idx++];
    if (root_instr->parent() == nullptr || !should_offload(root_instr)) {
      continue;
    }

    VLOG(2) << "Offloading instruction: " << root_instr->ToString();
    VLOG(2) << root_instr->name()
            << " is the root of a new offloaded computation";

    HloInstruction* call_instr;
    if (root_instr->opcode() == HloOpcode::kCall) {
      call_instr = root_instr;
    } else {
      call_instr = computation.CreateCallInstruction({root_instr});
      call_instr->SetAndSanitizeName(new_call_name_prefix);
      call_instr->UniquifyName(computation.parent());
      call_instr->set_frontend_attributes(root_instr->frontend_attributes());
    }
    HloCallInstruction* offloaded_call_instr =
        absl::down_cast<HloCallInstruction*>(call_instr);
    CHECK_NE(offloaded_call_instr, nullptr);
    ABSL_RETURN_IF_ERROR(clear_backend_config_device_type(offloaded_call_instr));
    ABSL_RETURN_IF_ERROR(ClearComputeTypeFrontendAttribute(offloaded_call_instr));
    ClearSideEffects(root_instr);
    HloInstruction* offloaded_instr = root_instr;

    const bool can_fuse_any = std::any_of(
        offload_candidates.begin() + candidate_idx, offload_candidates.end(),
        [&](HloInstruction* cand) {
          return cand->parent() != nullptr && should_offload(cand) &&
                 should_fuse(*offloaded_call_instr, *cand);
        });

    bool fused = false;
    bool has_unmerged_offload_instr = true;
    if (can_fuse_any) {
      // Stores all the ancestor instructions of offloaded_call_instr which were
      // not added to the current offloaded computation.
      absl::flat_hash_set<HloInstruction*> unmerged_ancestors;
      auto add_unmerged_ancestor = [&](HloInstruction* ancestor) {
        if (!unmerged_ancestors.insert(ancestor).second) {
          return;
        }
        std::vector<HloInstruction*> worklist = {ancestor};
        while (!worklist.empty()) {
          HloInstruction* curr = worklist.back();
          worklist.pop_back();
          for (HloInstruction* op : curr->operands()) {
            if (unmerged_ancestors.insert(op).second) {
              worklist.push_back(op);
            }
          }
        }
      };
      auto add_non_offload_call_operands = [&]() {
        for (HloInstruction* op : offloaded_call_instr->operands()) {
          if (!should_offload(op)) {
            add_unmerged_ancestor(op);
          }
        }
      };
      add_non_offload_call_operands();

      has_unmerged_offload_instr = false;
      for (size_t i = candidate_idx; i < offload_candidates.size(); ++i) {
        HloInstruction* instr = offload_candidates[i];
        if (instr->parent() == nullptr || !should_offload(instr)) {
          continue;
        }

        if (!unmerged_ancestors.contains(instr) &&
            should_fuse(*offloaded_call_instr, *instr)) {
          bool add_output =
              !offloaded_call_instr->IsUserOf(instr) ||
              absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
                return user != offloaded_call_instr && !should_offload(user);
              });
          offloaded_call_instr->AppendInstructionIntoCalledComputation(
              instr, add_output);
          ClearSideEffects(instr);
          if (instr->IsDead() && !instr->HasSuccessorControlDependencies()) {
            ABSL_RETURN_IF_ERROR(instr->SafelyDropAllControlDependencies());
            ABSL_RETURN_IF_ERROR(computation.RemoveInstruction(instr));
          }
          offloaded_instr = instr;
          fused = true;
          add_non_offload_call_operands();
        } else {
          add_unmerged_ancestor(instr);
          has_unmerged_offload_instr = true;
        }
      }
    }

    offloaded_instructions_and_calls.push_back(
        {offloaded_instr, offloaded_call_instr->unique_id()});
    if (!has_unmerged_offload_instr) {
      break;
    }
    if (fused) {
      offload_candidates = collect_offload_candidates();
      candidate_idx = 0;
    }
  }

  std::vector<std::pair<HloInstruction*, HloCallInstruction*>> alive_calls;
  if (offloaded_instructions_and_calls.empty()) {
    return alive_calls;
  }

  for (HloInstruction* instr : computation.instructions()) {
    // If an offloaded instruction is a Sharding custom call or has control
    // dependencies (such as those around elided copies), remove it
    // explicitly since it won't be removed by HloDCE.
    if (instr->IsDead() && (instr->IsCustomCall("Sharding") ||
                            (instr->HasControlDependencies() &&
                             !instr->HasSuccessorControlDependencies()))) {
      ABSL_RETURN_IF_ERROR(instr->SafelyDropAllControlDependencies());
      ABSL_RETURN_IF_ERROR(computation.RemoveInstruction(instr));
    }
  }

  // DCE any offloaded instructions that have no remaining un-wrapped uses.
  ABSL_RETURN_IF_ERROR(HloDCE::RunOnComputation(&computation).status());
  ABSL_RETURN_IF_ERROR(computation.parent()->RemoveUnusedComputations());
  if (computation.parent()->has_schedule()) {
    ABSL_RETURN_IF_ERROR(computation.parent()->schedule().Update());
  }

  VLOG(6) << "After offloading computation after DCE:";
  XLA_VLOG_LINES(6, computation.parent()->ToString());

  // Filter out any offloaded call instructions that were removed by HloDCE.
  // Since HloDCE deletes instructions and subsequent iterations of the while
  // loop may allocate new ones reusing the same pointer address, we must use
  // unique_id() to safely check if the instruction is still in the computation.
  absl::flat_hash_map<int64_t, HloInstruction*> alive_instructions;
  for (HloInstruction* instr : computation.instructions()) {
    alive_instructions[instr->unique_id()] = instr;
  }

  alive_calls.reserve(offloaded_instructions_and_calls.size());
  for (const auto& pair : offloaded_instructions_and_calls) {
    if (auto it = alive_instructions.find(pair.second);
        it != alive_instructions.end()) {
      alive_calls.push_back(
          {pair.first, absl::down_cast<HloCallInstruction*>(it->second)});
    }
  }
  return alive_calls;
}

}  // namespace xla::offloader_util
