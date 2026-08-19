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

#include "xla/hlo/transforms/host_offloading_prepare.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/memory_annotations.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

using xla::memory_annotations::kMoveToDeviceCustomCallTarget;
using xla::memory_annotations::kMoveToHostCustomCallTarget;

bool IsHostAsyncStart(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kAsyncStart &&
         instruction->async_execution_thread() == HloInstruction::kHostThread &&
         instruction->async_wrapped_instruction()->opcode() == HloOpcode::kCall;
}

absl::StatusOr<bool> RemoveSurroundingMoveCustomCalls(
    HloInstruction* async_start) {
  bool removed = false;
  // If any input operand traces back to a MoveToHost custom call (even through
  // transparent/shape-preserving operations like optimization barriers,
  // bitcasts, or reshapes), remove it since host offloading handles data
  // placement directly without requiring explicit memory copy instructions.
  for (HloInstruction* operand : async_start->operands()) {
    // Traverse transparent unary operations backwards to find the source.
    while (operand->opcode() == HloOpcode::kOptimizationBarrier ||
           operand->opcode() == HloOpcode::kBitcast ||
           operand->opcode() == HloOpcode::kReshape) {
      if (operand->operand_count() == 0) {
        break;
      }
      operand = operand->mutable_operand(0);
    }
    if (operand->IsCustomCall(kMoveToHostCustomCallTarget)) {
      CHECK_EQ(operand->operands().size(), 1);
      VLOG(1) << "Replacing " << operand->ToString() << " with "
              << operand->operands().at(0)->ToString();
      HloComputation* parent = operand->parent();
      ABSL_RETURN_IF_ERROR(operand->ReplaceAllUsesWith(operand->mutable_operand(0)));
      ABSL_RETURN_IF_ERROR(parent->RemoveInstruction(operand));
      removed = true;
    }
  }

  // If any users of async_done are MoveToDevice custom calls, remove them and
  // propagate the target memory space directly onto the async_done and
  // async_start output shapes to preserve layout annotations.
  for (HloInstruction* user : async_start->users()) {
    if (user->opcode() == HloOpcode::kAsyncDone) {
      std::vector<HloInstruction*> move_to_device_users;
      for (HloInstruction* done_user : user->users()) {
        if (done_user->IsCustomCall(kMoveToDeviceCustomCallTarget)) {
          move_to_device_users.push_back(done_user);
        }
      }
      for (HloInstruction* move_to_device : move_to_device_users) {
        CHECK_EQ(move_to_device->operands().size(), 1);
        VLOG(1) << "Replacing " << move_to_device->ToString() << " with "
                << user->ToString();
        // Preserve target memory space annotations from MoveToDevice on the
        // async output tuple and async_done shapes.
        if (move_to_device->shape().has_layout()) {
          user->mutable_shape()->mutable_layout()->set_memory_space(
              move_to_device->shape().layout().memory_space());
          *ShapeUtil::GetMutableSubshape(async_start->mutable_shape(), {1}) =
              user->shape();
        }
        ABSL_RETURN_IF_ERROR(move_to_device->ReplaceAllUsesWith(user));
        ABSL_RETURN_IF_ERROR(
            async_start->parent()->RemoveInstruction(move_to_device));
        removed = true;
      }
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
        ABSL_ASSIGN_OR_RETURN(bool removed,
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
        HloAsyncInstruction* call_start =
            Cast<HloAsyncInstruction>(instruction);
        HloInstruction* call = call_start->async_wrapped_instruction();

        HloComputation* inner_comp = call->to_apply();
        // Create a custom call from the original call instruction.
        std::unique_ptr<HloInstruction> custom_call =
            HloInstruction::CreateCustomCall(call->shape(), call->operands(),
                                             inner_comp, "HostExecute");
        // Propagate frontend attributes (e.g., latency_metadata) from inner
        // custom calls to the new custom call.
        for (const HloInstruction* inner_instr : inner_comp->instructions()) {
          if (inner_instr->opcode() == HloOpcode::kCustomCall &&
              inner_instr->has_frontend_attributes()) {
            custom_call->set_frontend_attributes(
                inner_instr->frontend_attributes());
            break;
          }
        }
        custom_call->set_output_to_operand_aliasing(
            call->output_operand_aliasing());

        // Replace async computation root with the custom call.
        HloComputation* async_computation =
            call_start->async_wrapped_computation();
        async_computation->set_root_instruction(
            async_computation->AddInstruction(std::move(custom_call)));
        ABSL_RETURN_IF_ERROR(async_computation->RemoveInstruction(call));

        changed = true;
      }
    }
  }
  if (changed && module->has_schedule()) {
    ABSL_RETURN_IF_ERROR(module->schedule().Update());
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> HostOffloadingPrepare::RunImpl(
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
