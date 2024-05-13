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
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<bool> HostOffloadingPrepare::RemoveSurroundingMoveCustomCalls(
    HloInstruction* async_start) {
  // If any of the operands are a MoveToHost custom call, remove them.
  bool removed = false;
  for (HloInstruction* operand : async_start->operands()) {
    // TODO(b/338463228): It could be the case that custom-calls are on the
    // otherside of a bitcast or tuple.
    if (operand->IsCustomCall(
            host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
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

absl::StatusOr<bool> HostOffloadingPrepare::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
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
      if (instruction->opcode() == HloOpcode::kAsyncStart &&
          instruction->async_execution_thread() ==
              HloInstruction::kHostThread) {
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

}  // namespace xla
