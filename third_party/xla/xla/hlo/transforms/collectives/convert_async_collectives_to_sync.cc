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

#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<HloInstruction*> CreateSyncVariant(HloInstruction* async_start,
                                                  HloInstruction* async_done) {
  HloInstruction* sync_instruction = nullptr;
  HloComputation* computation = async_start->parent();

  const HloOpcode async_start_op = async_start->opcode();
  switch (async_start_op) {
    case HloOpcode::kAllReduceStart: {
      auto* async_ar = Cast<HloAllReduceInstruction>(async_start);
      sync_instruction =
          computation->AddInstruction(HloInstruction::CreateAllReduce(
              async_done->shape(), async_ar->operands(), async_ar->to_apply(),
              async_ar->device_list(), async_ar->constrain_layout(),
              async_ar->channel_id(), async_ar->use_global_device_ids()));
      break;
    }
    case HloOpcode::kAllGatherStart: {
      auto* async_ag = Cast<HloAllGatherInstruction>(async_start);
      sync_instruction =
          computation->AddInstruction(HloInstruction::CreateAllGather(
              async_done->shape(), async_ag->operands(),
              async_ag->all_gather_dimension(), async_ag->device_list(),
              async_ag->constrain_layout(), async_ag->channel_id(),
              async_ag->use_global_device_ids()));
      break;
    }
    case HloOpcode::kCollectivePermuteStart: {
      auto* async_cp = Cast<HloCollectivePermuteInstruction>(async_start);
      TF_RET_CHECK(async_cp->operand_count() == 1);
      sync_instruction =
          computation->AddInstruction(HloInstruction::CreateCollectivePermute(
              async_done->shape(), async_cp->mutable_operand(0),
              async_cp->source_target_pairs(), async_cp->channel_id()));
      break;
    }
    case HloOpcode::kAsyncStart: {
      auto* as_start = Cast<HloAsyncInstruction>(async_start);
      HloInstruction* wrapped = as_start->async_wrapped_instruction();
      sync_instruction =
          computation->AddInstruction(wrapped->CloneWithNewOperands(
              async_done->shape(), as_start->operands()));
      break;
    }
    default:
      return Internal("Unexpected async start op %s",
                           HloOpcodeString(async_start->opcode()));
  }

  sync_instruction->set_metadata(async_start->metadata());
  sync_instruction->CopyBackendConfigFrom(async_start);

  TF_RETURN_IF_ERROR(async_done->ReplaceAllUsesWith(sync_instruction));

  // Collectives may have control dependencies due to passes like collective
  // schedule linearizer. Since we are running post scheduling, we can safely
  // ignore these control dependencies. Drop them to prepare for removal of the
  // async-start/done.
  TF_RETURN_IF_ERROR(async_start->DropAllControlDeps());
  TF_RETURN_IF_ERROR(async_done->DropAllControlDeps());

  // When we remove the async-done (and its unused operands), in most cases,
  // the async-start may not be deleted if its considered as having side effects
  // but in some cases it will be (e.g., the generic HLO kAsyncStart). Track its
  // removal and remove it if it was not removed when async-done is removed.
  bool is_async_start_removed = false;
  auto track_async_start_removed = [&](const HloInstruction* instr) {
    is_async_start_removed |= instr == async_start;
  };
  TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
      async_done, track_async_start_removed));
  if (!is_async_start_removed) {
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(async_start));
  }
  return sync_instruction;
}

/*static*/ absl::Status
ConvertAsyncCollectivesToSync::ReplaceAsyncInstructionsWithSync(
    HloComputation* computation,
    absl::Span<const std::pair<HloInstruction*, HloInstruction*>> async_pairs) {
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replaced_ops;
  for (auto& [async_start, async_done] : async_pairs) {
    TF_ASSIGN_OR_RETURN(HloInstruction * sync,
                        CreateSyncVariant(async_start, async_done));
    // Remember name of async instruction for profile usability.
    FrontendAttributes attributes;
    auto& map = *attributes.mutable_map();
    map[kAsyncCollectiveNameAttributeName] = async_start->name();
    sync->add_frontend_attributes(std::move(attributes));

    replaced_ops[async_start] = nullptr;
    replaced_ops[async_done] = sync;
  }

  // Update schedule.
  HloModule* module = computation->parent();
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  std::vector<HloInstruction*> new_sequence;
  new_sequence.reserve(sequence.size());
  for (HloInstruction* instr : sequence.instructions()) {
    auto it = replaced_ops.find(instr);
    if (it != replaced_ops.end()) {
      if (it->second != nullptr) {
        new_sequence.push_back(it->second);
      }
    } else {
      new_sequence.push_back(instr);
    }
  }
  module->schedule().set_sequence(computation, new_sequence);
  return absl::OkStatus();
}

absl::StatusOr<bool> ConvertAsyncCollectivesToSync::RunOnComputation(
    HloComputation* computation) {
  HloModule* module = computation->parent();
  std::vector<std::pair<HloInstruction*, HloInstruction*>> async_pairs;

  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);

  // Set of async-start ops that are currently in flight, i.e., their done not
  // yet seen.
  absl::flat_hash_set<HloInstruction*> in_flight_ops;

  for (HloInstruction* instruction : sequence.instructions()) {
    if (hlo_query::IsAsyncCollectiveStartOp(instruction)) {
      in_flight_ops.insert(instruction);
      VLOG(3) << "Found async start " << instruction->ToString();
    } else if (hlo_query::IsAsyncCollectiveDoneOp(instruction)) {
      // If this done is matching with the previous start and all intervening
      // ops are nops (i.e., prev_async_start was not reset to null), then we
      // were unable to schedule an independent op to overlap with this async
      // collective, so convert it to sync.
      VLOG(3) << "Found async done " << instruction->ToString();

      // All async-done ops are unary ops.
      TF_RET_CHECK(instruction->operand_count() == 1);
      HloInstruction* matching_async_start = instruction->mutable_operand(0);

      // Find if corresponding async-start is in the set of in-flight ops and
      // erase it (since it cannot be paired with any other async-done).
      if (in_flight_ops.erase(matching_async_start) == 1) {
        async_pairs.push_back({matching_async_start, instruction});
        VLOG(3) << "Added pair: {" << matching_async_start->name() << ", "
                << instruction->name();
      }
    } else if (!in_flight_ops.empty() && (!is_nop_ || !is_nop_(instruction))) {
      VLOG(3) << "Found intervening non-NOP instruction "
              << instruction->ToString();
      in_flight_ops.clear();
    }
  }

  if (async_pairs.empty()) {
    return false;
  }

  TF_RETURN_IF_ERROR(ConvertAsyncInstructionsToSync(computation, async_pairs));
  return true;
}

absl::StatusOr<bool> ConvertAsyncCollectivesToSync::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) {
    VLOG(3) << "Skipping as module is not scheduled";
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!module->schedule().is_computation_scheduled(computation)) {
      VLOG(3) << "Skipping computation" << computation->name()
              << " as it is not scheduled";
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        RunOnComputation(computation));
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace xla
