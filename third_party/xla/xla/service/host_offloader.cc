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

#include "xla/service/host_offloader.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

void SetMemorySpace(Shape* shape, int64_t memory_space_color) {
  CHECK(shape->has_layout());
  shape->mutable_layout()->set_memory_space(memory_space_color);
}

StatusOr<bool> DuplicateBroadcastForEachUse(HloModule* module) {
  bool split_at_least_one = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kBroadcast ||
          !instruction->HasConstantOperand()) {
        continue;
      }
      absl::InlinedVector<HloUse, 8> uses;
      for (HloInstruction* user : instruction->users()) {
        for (int64_t i = 0; i < user->operand_count(); ++i) {
          if (user->operand(i) != instruction) {
            continue;
          }
          uses.push_back(HloUse{user, i, /*operand_index=*/{}});
        }
      }

      if (uses.size() <= 1) {
        continue;
      }

      VLOG(1) << "Splitting broadcast " << instruction->ToString()
              << " which has " << uses.size() << " uses";
      split_at_least_one = true;
      // Don't create a new broadcast for the first use; we can still use the
      // original.
      for (int i = 1; i < uses.size(); ++i) {
        const HloUse& use = uses[i];
        HloInstruction* new_broadcast =
            instruction->parent()->AddInstruction(instruction->Clone());
        VLOG(2) << "New broadcast " << new_broadcast->ToString();
        TF_RETURN_IF_ERROR(use.instruction->ReplaceOperandWith(
            use.operand_number, new_broadcast));
      }
    }
  }
  return split_at_least_one;
}

}  // namespace

Status HostOffloader::HandlePipelineForwardCustomCall(
    HloInstruction* custom_call) {
  // Save a pointer to this custom call for when we want to remove it later.
  custom_calls_to_remove_.emplace(custom_call);

  // We expect that the DUS is the only user of this custom call.
  if (custom_call->user_count() != 1) {
    return FailedPrecondition(
        "Expecting custom call %s to only have 1 user; it has %d users: [%s]",
        custom_call->name(), custom_call->user_count(),
        absl::StrJoin(custom_call->users(), ", ",
                      [](std::string* out, const HloInstruction* user) {
                        out->append(user->name());
                      }));
  }
  HloInstruction* dynamic_update_slice = custom_call->users()[0];

  // Skip past any bitcasts.
  // TODO(b/319167527): Update this to be a bit more generic and safe.
  while (dynamic_update_slice->opcode() == HloOpcode::kBitcast) {
    VLOG(1) << "Skipping bitcast " << dynamic_update_slice->ToString();
    dynamic_update_slice = dynamic_update_slice->users()[0];
  }
  if (dynamic_update_slice->opcode() != HloOpcode::kDynamicUpdateSlice) {
    return InternalError(
        "Expecting only bitcasts between custom call (%s) and dynamic update "
        "slice (%s)",
        custom_call->name(), dynamic_update_slice->name());
  }

  // Get the buffer for this DUS.
  const HloBuffer& unique_buffer =
      alias_analysis_->GetUniqueBufferAt(dynamic_update_slice);

  // Look at the positions of this DUS:
  // TODO(b/319167527):
  //  Add kCopy to the list after ensuring that it is always safe to
  //  do so.
  constexpr std::array kAllowedPositionOpcodes = {
      HloOpcode::kTuple,
      HloOpcode::kGetTupleElement,
      HloOpcode::kDynamicUpdateSlice,
      HloOpcode::kBroadcast,
      HloOpcode::kWhile,
      HloOpcode::kParameter,
      HloOpcode::kOptimizationBarrier};
  for (const HloValue* value : unique_buffer.values()) {
    for (const HloPosition& position : value->positions()) {
      // Check if this position is of an allowed type.
      if (absl::c_find(kAllowedPositionOpcodes,
                       position.instruction->opcode()) ==
          kAllowedPositionOpcodes.end()) {
        return InternalError(
            "DynamicUpdateSlice %s's position %s is not supported. Not going "
            "to offload this one",
            dynamic_update_slice->name(), position.instruction->name());
      }
    }
  }

  // Check if there is a broadcast which creates this buffer.
  // For now, we only offload if the original destination of DUS is created by a
  // broadcast.
  HloInstruction* broadcast_instruction = nullptr;
  for (const HloValue* val : unique_buffer.values()) {
    HloInstruction* defining_instruction = val->defining_position().instruction;
    if (defining_instruction->opcode() == HloOpcode::kBroadcast) {
      VLOG(1) << "Found a broadcast instruction "
              << defining_instruction->ToString();
      broadcast_instruction = defining_instruction;
      break;
    }
  }

  if (broadcast_instruction == nullptr) {
    return InternalError(
        "The destination buffer of %s was not created by a broadcast; cannot "
        "offload. Has defining position(s) [%s]",
        dynamic_update_slice->name(),
        absl::StrJoin(unique_buffer.values(), ", ",
                      [](std::string* str, const HloValue* value) {
                        str->append(
                            value->defining_position().instruction->name());
                      }));
  }

  // TODO(b/319681297): Check that all uses of the broadcast are preceded by a
  // host copy.

  // Save the broadcast to later be replaced with a
  // custom-call("AllocateBuffer")
  broadcasts_to_replace_.emplace(broadcast_instruction);
  buffers_to_move_to_host_memory_.emplace(&unique_buffer);
  return OkStatus();
}

void HostOffloader::HandlePipelineBackwardCustomCall(
    HloInstruction* custom_call) {
  // Save a pointer to this custom call for later removal.
  custom_calls_to_remove_.emplace(custom_call);
}

Status HostOffloader::DynamifySlice(HloInstruction* slice) {
  VLOG(3) << "Dynamifying slice " << slice->ToString();
  std::vector<HloInstruction*> start_constants;
  for (int64_t start : slice->slice_starts()) {
    HloInstruction* constant = slice->parent()->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(start)));
    start_constants.push_back(constant);
  }
  std::vector<int64_t> slice_sizes;
  slice_sizes.reserve(slice->slice_limits().size());
  for (int i = 0; i < slice->slice_limits().size(); ++i) {
    slice_sizes.push_back(slice->slice_limits()[i] - slice->slice_starts()[i]);
  }
  HloInstruction* new_ds =
      slice->parent()->AddInstruction(HloInstruction::CreateDynamicSlice(
          slice->shape(), slice->mutable_operand(0), start_constants,
          slice_sizes));
  VLOG(3) << "Newly created dynamic slice: " << new_ds->name();
  TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(new_ds));
  TF_RETURN_IF_ERROR(slice->parent()->RemoveInstruction(slice));
  return OkStatus();
}

StatusOr<bool> HostOffloader::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Split broadcasts so that each HloUse of a broadcast instruction will get
  // its own copy.
  // TODO(b/319293925): Do not blindly duplicate all broadcasts, instead do it
  // only when necessary.
  TF_ASSIGN_OR_RETURN(bool duplicated_at_least_one_broadcast,
                      DuplicateBroadcastForEachUse(module));
  if (duplicated_at_least_one_broadcast) {
    changed = true;
  }

  // Run HloAliasAnalysis on module.
  TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module));

  // Iterate over all instructions and look for XLA host offload annoations.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }
      if (instruction->custom_call_target() == kPipelineForwardTarget) {
        TF_RETURN_IF_ERROR(HandlePipelineForwardCustomCall(instruction));
      } else if (instruction->custom_call_target() == kPipelineBackwardTarget) {
        HandlePipelineBackwardCustomCall(instruction);
      }
    }
  }

  absl::flat_hash_set<HloInstruction*> slices_to_dynamify;
  // Change the memory space of these buffers to the host memory space.
  for (const HloBuffer* buffer : buffers_to_move_to_host_memory_) {
    for (const HloValue* value : buffer->values()) {
      for (const HloPosition& position : value->positions()) {
        for (HloInstruction* user : position.instruction->users()) {
          // If a user of this position is a slice, change it to be a
          // dynamic-slice.
          if (user->opcode() == HloOpcode::kSlice) {
            slices_to_dynamify.emplace(user);
          }
        }
        Shape* shape_to_change = ShapeUtil::GetMutableSubshape(
            position.instruction->mutable_shape(), position.index);
        VLOG(2) << "Setting instruction to have host memory space: "
                << position.instruction->name();
        SetMemorySpace(shape_to_change, kHostMemorySpaceColor);
        changed = true;
      }
    }
  }

  for (HloInstruction* user : slices_to_dynamify) {
    TF_RETURN_IF_ERROR(DynamifySlice(user));
  }

  // Replace these broadcasts with AllocateBuffer instructions for host memory.
  for (HloInstruction* broadcast : broadcasts_to_replace_) {
    HloInstruction* allocate_buffer =
        broadcast->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            broadcast->shape(), {}, "AllocateBuffer"));
    VLOG(2) << "Replacing broadcast " << broadcast->name()
            << " with AllocateBuffer " << allocate_buffer->ToString();
    SetMemorySpace(allocate_buffer->mutable_shape(), kHostMemorySpaceColor);
    CHECK_OK(broadcast->ReplaceAllUsesWith(allocate_buffer));
    TF_RETURN_IF_ERROR(broadcast->parent()->RemoveInstruction(broadcast));
    changed = true;
  }

  // Remove these custom-calls that were previously used for annotation.
  for (HloInstruction* custom_call : custom_calls_to_remove_) {
    CHECK_EQ(custom_call->operand_count(), 1);
    HloInstruction* operand = custom_call->operands()[0];

    if (custom_call->shape().layout() != operand->shape().layout()) {
      // LayoutAssignment might change the layout of the operand but leave the
      // custom call layout unchanged. In that case, insert a copy.
      // TODO(b/319686942): Once LayoutAssignment propagates the layout through
      // this specific custom call, remove this insertion of a copy.
      TF_RETURN_IF_ERROR(custom_call->ReplaceAllUsesWith(
          custom_call->parent()->AddInstruction(HloInstruction::CreateUnary(
              custom_call->shape(), HloOpcode::kCopy, operand))));
    } else {
      CHECK_OK(custom_call->ReplaceAllUsesWith(operand));
    }

    TF_RETURN_IF_ERROR(custom_call->parent()->RemoveInstruction(custom_call));
    changed = true;
  }

  return changed;
}

}  // namespace xla
