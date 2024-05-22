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

#include "xla/service/host_offloading_prepare.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using xla::host_memory_offload_annotations::kMoveToHostCustomCallTarget;

bool IsHostAsyncStart(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kAsyncStart &&
         instruction->async_execution_thread() == HloInstruction::kHostThread &&
         instruction->async_wrapped_instruction()->opcode() == HloOpcode::kCall;
}

absl::StatusOr<bool> RemoveSurroundingMoveCustomCalls(
    HloInstruction* async_start) {
  // If any of the operands are a MoveToHost custom call, remove them.
  bool removed = false;
  for (HloInstruction* operand : async_start->operands()) {
    // TODO(b/338463228): It could be the case that custom-calls are on the
    // other side of a bitcast or tuple.
    if (operand->IsCustomCall(kMoveToHostCustomCallTarget)) {
      CHECK_EQ(operand->operands().size(), 1);
      VLOG(1) << "Replacing " << operand->ToString() << " with "
              << operand->operands().at(0)->ToString();
      TF_RETURN_IF_ERROR(
          operand->ReplaceAllUsesWith(operand->mutable_operand(0)));
      TF_RETURN_IF_ERROR(async_start->parent()->RemoveInstruction(operand));
      removed = true;
    }
  }
  return removed;
}

absl::StatusOr<bool> ElideMoveCustomCalls(HloModule* module) {
  bool changed = false;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (HloComputation* computation : module->computations()) {
    if (computation->execution_thread() != HloInstruction::kHostThread) {
      continue;
    }
    // This is a computation to be offloaded to the host.
    std::vector<HloInstruction*> callers =
        call_graph->GetComputationCallers(computation);
    for (HloInstruction* caller : callers) {
      VLOG(2) << "Hlo computation " << computation->name()
              << " is offloaded to host and has caller " << caller->ToString();
      if (caller->parent()->execution_thread() == HloInstruction::kHostThread) {
        VLOG(3) << "Nested host computation, must be a async-wrapper";
        continue;
      }
      VLOG(2) << "Going to adjust before and after " << caller->name();
    }
  }
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsHostAsyncStart(instruction)) {
        VLOG(2) << "Found async start of host computation: "
                << instruction->ToString() << " done must be "
                << instruction->users().at(0)->ToString();
        TF_ASSIGN_OR_RETURN(bool removed,
                            RemoveSurroundingMoveCustomCalls(instruction));
        changed = changed || removed;
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> ConvertToCustomCall(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsHostAsyncStart(instruction)) {
        auto* call_start = Cast<HloAsyncInstruction>(instruction);
        auto* call = call_start->async_wrapped_instruction();

        // Create a custom call from the original call instruction.
        auto custom_call = HloInstruction::CreateCustomCall(
            call->shape(), call->operands(), call->called_computations().at(0),
            "HostExecute");
        custom_call->set_output_to_operand_aliasing(
            call->output_operand_aliasing());

        // Replace async computation root with the custom call.
        HloComputation* async_computation =
            call_start->async_wrapped_computation();
        async_computation->set_root_instruction(
            async_computation->AddInstruction(std::move(custom_call)));
        TF_RETURN_IF_ERROR(async_computation->RemoveInstruction(call));

        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> HostOffloadingPrepare::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  switch (rewrite_) {
    case Rewrite::kElideMoveToHost:
      return ElideMoveCustomCalls(module);
    case Rewrite::kConvertToCustomCall:
      return ConvertToCustomCall(module);
  }
}

}  // namespace xla
