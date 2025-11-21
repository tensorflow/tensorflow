/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_rematerialization.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/bind_front.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/tuple_points_to_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_data_structures.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_memory_tracker.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_options.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout_util.h"
#include "xla/map_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/numbers.h"

namespace xla {

namespace {

using ::tsl::strings::HumanReadableNumBytes;
using RematAlgorithm = HloRematerializationOptions::RematAlgorithm;

// Performs the rematerialization of all items in `best_items` and returns the
// number of net instructions added.
absl::StatusOr<int64_t> RematerializeInstructions(
    HloRematerializationMemoryTracker* memory_tracker,
    std::vector<HloRematItem*>* best_items,
    absl::flat_hash_set<const HloInstruction*>* remat_move_instructions,
    HloRematInstructionList* instruction_list, HloSchedule* schedule,
    HloRematerialization* rematerialization,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map) {
  int64_t net_instructions_added = 0;
  std::vector<std::string> instruction_names(best_items->size());
  // Rematerialize the block of instructions in the reverse order to account for
  // dependencies between instructions in best_items.
  for (int i = best_items->size() - 1; i >= 0; --i) {
    HloRematItem* best_item = (*best_items)[i];
    HloInstruction* best = best_item->instruction;
    instruction_names[i] = best->name();
    HloComputation* computation = best->parent();

    // If the item to remat has no unplaced users, then skip the
    // rematerialization. Such an instruction can appear in best_items because
    // it is part of a good block, but does not itself add any benefit.
    if (!memory_tracker->HasUnplacedUsers(best_item)) {
      continue;
    }

    HloCloneContext context(computation->parent());
    HloInstruction* remat =
        computation->AddInstruction(best->Clone(/*suffix=*/"remat", &context));
    // Call the callback on the original and rematerialized instruction.
    TF_RETURN_IF_ERROR(rematerialization->on_rematerialized(best, remat));
    for (auto& cloned_computation_pair : context.cloned_computations()) {
      if (!schedule->is_computation_scheduled(cloned_computation_pair.first)) {
        continue;
      }
      HloInstructionSequence& sequence =
          schedule->GetOrCreateSequence(cloned_computation_pair.second);
      HloInstructionSequence& old_sequence =
          schedule->GetOrCreateSequence(cloned_computation_pair.first);
      for (HloInstruction* instr : old_sequence.instructions()) {
        sequence.push_back(instr);
      }
    }
    // Increment channel_id on channel instructions with a channel id.
    if (DynCast<HloChannelInstruction>(best) &&
        DynCast<HloChannelInstruction>(best)->channel_id()) {
      remat->set_channel_id(rematerialization->NextChannelId());
    }

    // Add control dependencies to the new operation.
    TF_RETURN_IF_ERROR(remat->CopyAllControlDepsFrom(best));

    HloRematItem* remat_item = instruction_list->CreateItem(remat);
    // Peak priority specific optimization. Any recomputed instruction
    // should not be rematerialized again.
    if (rematerialization->remat_algorithm() == RematAlgorithm::kPeakPriority) {
      (*rematerializable_map)[remat] = false;
    }

    // Replace each remaining use of 'best' with the rematerialization.
    absl::InlinedVector<HloRematItem*, 4> indirect_users;
    absl::flat_hash_map<int64_t, HloInstruction*> gte_cache;
    for (auto& user : memory_tracker->GetItemUses(best_item)) {
      if (!memory_tracker->IsInstructionPlaced(user.user->instruction)) {
        VLOG(2) << "  Replacing use of " << best->name() << " in "
                << user.user->instruction->name() << " with " << remat->name();
        HloInstruction* remat_use = remat;
        HloInstruction* const user_operand =
            user.user->instruction->mutable_operand(user.operand_number);
        if (remat_use == user_operand) {
          continue;
        }
        // If the output of a multi-output fusion node is forwarded to one of
        // its users as is, all the element buffers are also treated as uses
        // by that user, which need to be skipped.
        if (user.index && remat_use->shape() != user_operand->shape()) {
          auto cached_gte = gte_cache.find(*user.index);
          if (cached_gte == gte_cache.end()) {
            remat_use = computation->AddInstruction(
                HloInstruction::CreateGetTupleElement(
                    ShapeUtil::GetTupleElementShape(remat_use->shape(),
                                                    *user.index),
                    remat_use, *user.index),
                /*new_name=*/"gte.remat");
            indirect_users.push_back(instruction_list->CreateItem(remat_use));
            gte_cache[*user.index] = remat_use;
          } else {
            remat_use = cached_gte->second;
          }
        }
        if (user_operand->shape() != remat_use->shape()) {
          remat_use = computation->AddInstruction(
              HloInstruction::CreateBitcast(user_operand->shape(), remat_use),
              /*new_name=*/"bitcast.remat");
          indirect_users.push_back(instruction_list->CreateItem(remat_use));
        }
        TF_RETURN_IF_ERROR(user.user->instruction->ReplaceOperandWith(
            user.operand_number, remat_use));
        // Peak priority specific optimization. Any recomputed instruction
        // should not be rematerialized again.
        if (remat_use != remat && rematerialization->remat_algorithm() ==
                                      RematAlgorithm::kPeakPriority) {
          (*rematerializable_map)[remat_use] = false;
        }
        TF_RETURN_IF_ERROR(
            rematerialization->on_rematerialized(user_operand, remat_use));
      }
    }

    // Update memory tracker buffer calculations now, we're about to use them
    // to place `remat_item` in a good spot.
    TF_RETURN_IF_ERROR(
        memory_tracker->AddRecomputeInstructionToBufferCalculations(
            best_item, remat_item, absl::MakeSpan(indirect_users)));

    // Insert rematerialized instruction right before the earliest unplaced
    // use of the instruction *and* the earliest unplaced last use of any
    // operands of remat. Unplaced uses of the remat's operands are included
    // because we don't want to extend the live range of remat's operands as
    // this could increase memory usage.
    ItemList place_before;
    const absl::flat_hash_set<HloRematItem*> indirect_users_set(
        indirect_users.begin(), indirect_users.end());
    for (auto user : remat->users()) {
      if (!indirect_users_set.contains(instruction_list->GetItem(user))) {
        place_before.push_back(instruction_list->GetItem(user));
      }
    }
    for (auto* indirect_user : indirect_users) {
      for (auto user : indirect_user->instruction->users()) {
        if (!indirect_users_set.contains(instruction_list->GetItem(user))) {
          place_before.push_back(instruction_list->GetItem(user));
        }
      }
    }
    for (auto* operand : remat->operands()) {
      for (auto* operand_user : operand->users()) {
        if (operand_user != remat) {
          if (!memory_tracker->IsInstructionPlaced(operand_user)) {
            // If an operand is smaller than the remat instruction, placing
            // its last user first will reduce the peak memory usage in between
            // it and the remat instruction's user. This is because in that
            // interval you have to extend the live range of either the remat
            // instruction or the operand. Whichever is smaller will cause the
            // smallest the peak memory usage. This optimization is only applied
            // when using the peak priority rematerialization algorithm but can
            // be extended to other algorithms in the future.
            if (rematerialization->remat_algorithm() ==
                RematAlgorithm::kPeakPriority) {
              int64_t operand_size = memory_tracker->BytesUsedByBuffers(
                  instruction_list->GetItem(operand), false);
              int64_t remat_size =
                  memory_tracker->BytesUsedByBuffers(remat_item, false);
              if (operand_size < remat_size) {
                VLOG(2) << "Skipping operand_user: " << operand_user->name()
                        << " of operand: " << operand->name()
                        << " when choosing to place remat instruction before "
                           "it. Remat instruction: "
                        << remat->name()
                        << ", remat instruction size: " << remat_size
                        << ", operand size: " << operand_size;
                continue;
              }
            }
            HloRematItem* operand_user_item =
                instruction_list->GetItem(operand_user);
            place_before.push_back(operand_user_item);
          }
        }
      }
    }
    // Insert rematerialized instruction before any of its successors to
    // preserve ordering regarding control dependency.
    for (auto successor : remat->control_successors()) {
      HloRematItem* successor_item = instruction_list->GetItem(successor);
      // Assert to make sure we never remat an operation with control
      // successor already placed.
      CHECK(!memory_tracker->IsInstructionPlaced(successor))
          << successor->name();
      place_before.push_back(successor_item);
    }
    instruction_list->InsertBeforeInstructions(remat_item, place_before);
    for (auto* bitcast : indirect_users) {
      instruction_list->InsertBeforeInstructions(bitcast, place_before);
    }
    // Have the memory tracker make the exact same update to its instruction
    // ordering.
    HloRematerializationMemoryTracker::NewItemAndSuccessor
        recompute_and_successor;
    recompute_and_successor.new_item = remat_item;
    recompute_and_successor.successor = instruction_list->next(remat_item);

    TF_RETURN_IF_ERROR(
        memory_tracker->AddRecomputeInstructionToInstructionOrdering(
            best_item, recompute_and_successor,
            absl::MakeSpan(indirect_users)));

    // Helper function that looks through indirect users when determining if
    // there is an active user for an HloInstruction.
    std::function<bool(HloInstruction*)> uses_empty = [&](HloInstruction* i) {
      for (auto* u : i->users()) {
        if (!IsSupportedIndirectUser(u) || !uses_empty(u)) {
          return false;
        }
      }
      return true;
    };
    // If the rematerialized instruction is dead then rematerialization is
    // essentially a move. Don't delete the instruction now because we don't
    // want duplicate HloInstruction* values during the course of the
    // transformation because we keep maps with HloInstruction* values as
    // keys.
    if (uses_empty(best)) {
      VLOG(2) << best->name() << " is now dead";
      if (ContainsKey(*remat_move_instructions, best)) {
        // Previously, 'best' was a rematerialization which killed the
        // instruction it was a copying of. Now 'remat' is a rematerialization
        // of 'best' and kills 'best'. Stop rematerializing this instruction
        // to avoid an infinite loop.
        instruction_list->Denylist(remat);
      }
      remat_move_instructions->insert(remat);
      net_instructions_added += indirect_users.size();
    } else {
      net_instructions_added += indirect_users.size() + 1;
    }
    for (auto* indirect_user : indirect_users) {
      instruction_list->Denylist(indirect_user->instruction);
    }
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(best->opcode()) ||
        HloDataflowAnalysis::IsAsynchronousOperationDone(best->opcode())) {
      VLOG(2) << "The old instruction " << best->name()
              << " is an async op. Removing to maintain one start to one done "
                 "invariant to keep the HLO valid.";
      // We need to remove all control dependencies from best before removing it
      // from the computation.  Its control dependencies were previously copied
      // to the remat instruction.
      TF_RETURN_IF_ERROR(best->DropAllControlDeps());
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(best));
    }
  }
  return net_instructions_added;
}

// Performs rematerialization of `best_item` via the compression strategy.
// Returns the net number of instructions added.
absl::StatusOr<int64_t> CompressInstruction(
    HloRematerializationMemoryTracker* memory_tracker, HloRematItem* best_item,
    const Shape& compact_shape, HloRematInstructionList* instruction_list) {
  HloInstruction* best = best_item->instruction;
  VLOG(5) << "Transposing instruction " << best->name() << " (saving "
          << HumanReadableNumBytes(memory_tracker->MemoryReducedIfCompressed(
                 best_item, compact_shape))
          << ") to" << compact_shape.ToString(true);

  HloComputation* computation = best->parent();
  HloInstruction* compressed = computation->AddInstruction(
      HloInstruction::CreateUnary(compact_shape, HloOpcode::kCopy, best),
      /*new_name=*/absl::StrCat(best->name(), ".remat_compressed"));

  HloInstruction* uncompressed = computation->AddInstruction(
      HloInstruction::CreateUnary(best->shape(), HloOpcode::kCopy, compressed),
      /*new_name=*/absl::StrCat(best->name(), ".remat_uncompressed"));

  HloRematItem* compressed_item = instruction_list->CreateItem(compressed);
  compressed_item->placed = true;

  HloRematItem* uncompressed_item = instruction_list->CreateItem(uncompressed);

  // Replace each remaining use of 'best' with the uncompressed.
  std::vector<HloInstruction*> best_users_copy = best->users();
  for (HloInstruction* user : best_users_copy) {
    if (!memory_tracker->IsInstructionPlaced(user)) {
      VLOG(5) << "  Replacing use of " << best->name() << " in " << user->name()
              << " with " << uncompressed->name();
      TF_RETURN_IF_ERROR(best->ReplaceUseWith(user, uncompressed));
    }
  }

  // Update memory tracker buffer calculations now, we're about to use them
  // to place uncompressed in a good spot.
  TF_RETURN_IF_ERROR(
      memory_tracker->AddCompressInstructionsToBufferCalculations(
          best_item, compressed_item, uncompressed_item));

  // Insert rematerialized instruction right before the earliest unplaced
  // use of the instruction.
  ItemList place_before;
  for (auto user : uncompressed->users()) {
    place_before.push_back(instruction_list->GetItem(user));
  }

  instruction_list->Denylist(compressed_item->instruction);
  instruction_list->Denylist(uncompressed_item->instruction);
  instruction_list->InsertBeforeInstructions(uncompressed_item, place_before);
  instruction_list->InsertAfterInstructions(compressed_item, {best_item});

  // Have the memory tracker make the exact same update to its instruction
  // ordering.
  HloRematerializationMemoryTracker::NewItemAndPredecessor
      compressed_and_predecessor;
  compressed_and_predecessor.new_item = compressed_item;
  compressed_and_predecessor.predecessor = best_item;
  HloRematerializationMemoryTracker::NewItemAndSuccessor
      uncompressed_and_successor;
  uncompressed_and_successor.new_item = uncompressed_item;
  uncompressed_and_successor.successor =
      instruction_list->next(uncompressed_item);
  TF_RETURN_IF_ERROR(
      memory_tracker->AddCompressInstructionsToInstructionOrdering(
          best_item, compressed_and_predecessor, uncompressed_and_successor));

  // Net two instructions added.
  return 2;
}

// Performs rematerialization of `best_item` via the host offload strategy.
// Returns the net number of instructions added.
absl::StatusOr<int64_t> OffloadInstruction(
    HloRematerializationMemoryTracker* memory_tracker, HloRematItem* best_item,
    HloRematInstructionList* instruction_list) {
  HloInstruction* best_instruction = best_item->instruction;
  HloComputation* computation = best_instruction->parent();
  VLOG(2) << "Best_instruction's users: "
          << absl::StrJoin(best_instruction->users(), ", ",
                           [](std::string* str, const auto* x) {
                             return str->append(x->name());
                           });

  // Set up shapes for different memory locations.
  Shape instruction_shape_device = best_instruction->shape();
  Shape instruction_shape_host = best_instruction->shape();
  instruction_shape_host.mutable_layout()->set_memory_space(
      memory_tracker->options().host_memory_offload_config->host_memory_space);
  Shape context_shape = ShapeUtil::MakeShape(U32, {});

  // Create copy instructions to and from host memory.
  HloInstruction* copy_start_to_host =
      computation->AddInstruction(HloInstruction::CreateCopyStart(
          ShapeUtil::MakeTupleShape({instruction_shape_host,
                                     instruction_shape_device, context_shape}),
          best_instruction));
  HloInstruction* copy_done_to_host =
      computation->AddInstruction(HloInstruction::CreateUnary(
          instruction_shape_host, HloOpcode::kCopyDone, copy_start_to_host));

  HloInstruction* copy_start_to_device =
      computation->AddInstruction(HloInstruction::CreateCopyStart(
          ShapeUtil::MakeTupleShape({instruction_shape_device,
                                     instruction_shape_host, context_shape}),
          copy_done_to_host));
  HloInstruction* copy_done_to_device = computation->AddInstruction(
      HloInstruction::CreateUnary(instruction_shape_device,
                                  HloOpcode::kCopyDone, copy_start_to_device));
  VLOG(3) << "Created copy_start_to_host instr: "
          << copy_start_to_host->ToString();
  VLOG(3) << "Created copy_done_to_host instr: "
          << copy_done_to_host->ToString();
  VLOG(3) << "Created copy_start_to_device instr: "
          << copy_start_to_device->ToString();
  VLOG(3) << "Created copy_done_to_device instr: "
          << copy_done_to_device->ToString();

  // Update the HloCostAnalysis with the new instructions.
  TF_RETURN_IF_ERROR(
      copy_start_to_host->Visit(&memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(
      copy_done_to_host->Visit(&memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(copy_start_to_device->Visit(
      &memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(
      copy_done_to_device->Visit(&memory_tracker->options().hlo_cost_analysis));

  // Create an HloRematItem for each instruction. These items will be inserted
  // into the HloRematInstructionList, which is essentially our schedule.
  HloRematItem* copy_start_to_host_item =
      instruction_list->CreateItem(copy_start_to_host);
  HloRematItem* copy_done_to_host_item =
      instruction_list->CreateItem(copy_done_to_host);
  HloRematItem* copy_start_to_device_item =
      instruction_list->CreateItem(copy_start_to_device);
  HloRematItem* copy_done_to_device_item =
      instruction_list->CreateItem(copy_done_to_device);

  // Add the newly created instructions to the deny list to prevent them from
  // becoming rematerialized later.
  instruction_list->Denylist(copy_start_to_host);
  instruction_list->Denylist(copy_done_to_host);
  instruction_list->Denylist(copy_start_to_device);
  instruction_list->Denylist(copy_done_to_device);

  HloRematItem* place_before{nullptr};
  // Find the first item that we need to place our final copy-done before. That
  // will be the first unplaced user of best_instruction.
  {
    ItemList place_before_list;
    for (auto user : best_instruction->users()) {
      if (user == copy_start_to_host) {
        // Skip the copy that we just added.
        continue;
      }
      auto item_of_user = instruction_list->GetItem(user);
      if (memory_tracker->IsInstructionPlaced(user)) {
        // Skip placed items.
        continue;
      }
      place_before_list.push_back(item_of_user);
    }
    CHECK(!place_before_list.empty()) << "Have nothing to place this before!";
    for (auto* item = instruction_list->first(); item != nullptr;
         item = instruction_list->next(item)) {
      if (absl::c_linear_search(place_before_list, item)) {
        place_before = item;
        break;
      }
    }
  }
  CHECK_NE(place_before, nullptr)
      << "Could not find an item to place this before.";

  // This function walks along the instruction list (schedule) and returns first
  // instruction which will be executed after `time_spent_on_copy` seconds of
  // compute has elapsed. Returns a result in the range [start_item, end_item).
  auto get_first_item_after_compute_time = [&](HloRematItem* start_item,
                                               HloRematItem* end_item,
                                               auto successor_func,
                                               float time_spent_on_copy) {
    // Do not count the computation time of the first item.
    // In the case of iterating forward in time, it is the output of this
    // item which we want to offload. In the case of iterating backward in
    // time, this buffer is a dependency of that start item.
    float time_so_far = 0.0;
    auto* current_item = start_item;
    // Walk the instruction list and accumulate the computation time.
    while (time_so_far < time_spent_on_copy) {
      auto next_item = successor_func(current_item);
      if (next_item == end_item) {
        // TODO(b/293323448): This is a bad thing, but not an error. Previously,
        // when evaluating whether or not to host offload this instruction we
        // checked how much compute there was between uses. We found that there
        // was enough total compute to cover the time required to copy the data
        // to the host and back. However, that check does not necessarily
        // guarantee that the compute is split in such a way that it will give
        // us enough compute to hide both copies in series. For example lets say
        // that the copies in total take this long:
        // | <-------------  Copies take this long --------------> |
        // Lets say the two copies take the same amount of time:
        // | <----- Copy to host -----> <---- Copy to device ----> |
        //
        // And you have a compute sequence that looks like this:
        // +-----------+ +-----------+   +-----------+ +-----------+
        // | Compute-1 | | Compute-2 |   | Compute-3 | | Compute-4 |
        // +-----------+ +-----------+   +-----------+ +-----------+
        // It would make sense to insert the copy-start/done instructions
        // as follows:
        // ^ Copy-start to host
        //          Copy-done to host ^
        //        Copy-start to device ^
        //                                     Copy-done to device ^
        //
        // However, if the compute sequence is not even, like this:
        // +-----------------------------------------+ +-----------+
        // |                Compute-1                | | Compute-2 |
        // +-----------------------------------------+ +-----------+
        // Then we would find enough compute to hide our copy on the forward
        // pass, but on the backward pass, there wouldn't be enough compute
        // remaining, even though we originally calculated that there was enough
        // total compute for the two copies.
        LOG(WARNING) << "Didn't find enough computation before end of window";
        break;
      }
      current_item = next_item;
      CHECK_NE(current_item, nullptr) << "current_item is null";
      CHECK_NE(current_item->instruction, nullptr)
          << "current_item's instruction is null";
      // TODO(b/293321321): HloCostAnalysis has no knowledge of any newly
      // rematerialized instructions via recompute or compression strategies.
      // This should be fixed.
      time_so_far += std::max(
          0.0f, memory_tracker->options().hlo_cost_analysis.optimal_seconds(
                    *current_item->instruction));
    }
    return current_item;
  };

  // Figure out how much time these copies will take.
  const int64_t bytes_used_by_buffers = memory_tracker->BytesUsedByBuffers(
      best_item, /*only_count_unplaced_users=*/false);
  const float copy_to_host_time_seconds =
      bytes_used_by_buffers /
      memory_tracker->options()
          .host_memory_offload_config->bandwidth_to_host_bytes_per_second;
  const float copy_from_host_time_seconds =
      bytes_used_by_buffers /
      memory_tracker->options()
          .host_memory_offload_config->bandwidth_from_host_bytes_per_second;
  VLOG(2) << "HloRematItem uses " << bytes_used_by_buffers << "B and will take "
          << copy_to_host_time_seconds << "s to copy to host and "
          << copy_from_host_time_seconds << "s to copy from host.";

  // Place the copy-start to host as early as possible.
  VLOG(2) << "Inserting " << copy_start_to_host_item->instruction->name()
          << " immediately after " << best_item->instruction->name();
  instruction_list->InsertAfterInstructions(copy_start_to_host_item,
                                            {best_item});

  // Place the copy-done to device as late as possible.
  VLOG(2) << "Inserting " << copy_done_to_device_item->instruction->name()
          << " immediately before " << place_before->instruction->name();
  instruction_list->InsertBeforeInstructions(copy_done_to_device_item,
                                             {place_before});

  // Place the first copy-done after enough runtime after the first copy-start
  // to hide the memory transfer.
  auto first_item_after_to_host_copy = get_first_item_after_compute_time(
      copy_start_to_host_item, copy_done_to_device_item,
      [&instruction_list](HloRematItem* item) {
        return instruction_list->next(item);
      },
      copy_to_host_time_seconds);
  VLOG(2) << "Inserting " << copy_done_to_host_item->instruction->name()
          << " immediately after "
          << first_item_after_to_host_copy->instruction->name();
  instruction_list->InsertAfterInstructions(copy_done_to_host_item,
                                            {first_item_after_to_host_copy});

  // Place the second copy-start early enough so that there is enough
  // runtime to hide the memory transfer before the second copy-done.
  auto first_item_before_from_host_copy = get_first_item_after_compute_time(
      copy_done_to_device_item, copy_done_to_host_item,
      [&instruction_list](HloRematItem* item) {
        return instruction_list->prev(item);
      },
      copy_from_host_time_seconds);
  VLOG(2) << "Inserting " << copy_start_to_device_item->instruction->name()
          << " immediately before "
          << first_item_before_from_host_copy->instruction->name();
  instruction_list->InsertBeforeInstructions(
      copy_start_to_device_item, {first_item_before_from_host_copy});

  // Once all of the items are in the proper place in the instruction list, mark
  // them as placed or not depending on which item is the current item in the
  // memory tracker.
  {
    auto item = instruction_list->first();
    while (item != nullptr) {
      if (item == copy_start_to_host_item || item == copy_done_to_host_item ||
          item == copy_start_to_device_item ||
          item == copy_done_to_device_item) {
        item->placed = true;
      } else if (memory_tracker->IsInProgressItem(item)) {
        // Our newly added items are defaulted as not placed, so breaking here
        // gives us our desired result.
        break;
      }
      item = instruction_list->next(item);
    }
  }

  // It is critical to only update the users after items have been marked as
  // placed, since we will only want to update non-placed items.

  // Replace uses of best_instruction with copy_done_to_device.
  // Note that items must be created before this point.
  std::vector<HloInstruction*> best_users_copy = best_instruction->users();
  for (HloInstruction* user : best_users_copy) {
    if (!memory_tracker->IsInstructionPlaced(user)) {
      VLOG(3) << "  Replacing use of " << best_instruction->name() << " in "
              << user->name() << " with " << copy_done_to_device->name();
      TF_RETURN_IF_ERROR(
          best_instruction->ReplaceUseWith(user, copy_done_to_device));
    } else {
      VLOG(3) << user->name() << " is placed, not going to update";
    }
  }

  // Finally, update the HloRematerializationSweepMemoryTracker. This will
  // update the tracking of buffer creations and uses.
  TF_RETURN_IF_ERROR(memory_tracker->AddOffloadInstructionsToBufferCalculations(
      best_item, copy_start_to_host_item, copy_done_to_host_item,
      copy_start_to_device_item, copy_done_to_device_item));
  HloRematerializationMemoryTracker::NewItemAndPredecessor
      copy_start_to_host_and_predecessor;
  copy_start_to_host_and_predecessor.new_item = copy_start_to_host_item;
  copy_start_to_host_and_predecessor.predecessor = best_item;
  HloRematerializationMemoryTracker::NewItemAndPredecessor
      copy_done_to_host_and_predecessor;
  copy_done_to_host_and_predecessor.new_item = copy_done_to_host_item;
  copy_done_to_host_and_predecessor.predecessor = first_item_after_to_host_copy;
  HloRematerializationMemoryTracker::NewItemAndSuccessor
      copy_start_to_device_and_successor;
  copy_start_to_device_and_successor.new_item = copy_start_to_device_item;
  copy_start_to_device_and_successor.successor =
      first_item_before_from_host_copy;
  HloRematerializationMemoryTracker::NewItemAndSuccessor
      copy_done_to_device_and_successor;
  copy_done_to_device_and_successor.new_item = copy_done_to_device_item;
  copy_done_to_device_and_successor.successor = place_before;
  TF_RETURN_IF_ERROR(
      memory_tracker->AddOffloadInstructionsToInstructionOrdering(
          best_item, copy_start_to_host_and_predecessor,
          copy_done_to_host_and_predecessor, copy_start_to_device_and_successor,
          copy_done_to_device_and_successor));

  // Net four instructions added.
  return 4;
}

// A simple struct to encapsulate the number of instructions added during
// rematerialization.
struct InstructionsAdded {
  // Total count of instructions rematerialized.
  int remat_count;
  // Total count of instructions rematerialized minus number of original
  // instructions that are now dead.
  int net_instructions_added;
  // Amount of effort expended to find the instructions to rematerialize.
  int effort;
};

// Rematerializes the best block of instructions of size between min_block_size
// and max_block_size (both inclusive) if at least one candidate block of
// instructions can be found. Returns number of instructions rematerialized.
absl::StatusOr<InstructionsAdded> RematerializeBestBlock(
    int min_block_size, int max_block_size,
    HloRematerializationMemoryTracker* memory_tracker,
    HloRematInstructionList* instruction_list, HloSchedule* schedule,
    int64_t memory_limit_bytes,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
    absl::flat_hash_set<const HloInstruction*>* remat_move_instructions,
    HloRematerialization* rematerialization) {
  CHECK(min_block_size > 0) << "Negative block size.";

  std::vector<HloRematItem*> best_items;
  RematStrategy best_strategy;
  int effort;
  TF_ASSIGN_OR_RETURN(int64_t peak_memory,
                      rematerialization->GetComputationPeakMemory(
                          memory_tracker->computation()));
  std::tie(best_items, best_strategy, effort) =
      memory_tracker->PickRematerializationCandidates(
          *instruction_list, memory_limit_bytes, rematerializable_map,
          min_block_size, max_block_size, peak_memory);
  InstructionsAdded num_instructions_added;
  num_instructions_added.remat_count = best_items.size();
  num_instructions_added.effort = effort;
  if (best_items.empty()) {
    VLOG(5) << "No items found when trying to rematerializing block.";
    num_instructions_added.net_instructions_added = 0;
    return num_instructions_added;
  }

  if (best_strategy.kind == RematStrategy::kCompress) {
    CHECK(best_items.size() == 1)
        << "More than one instruction compressed simultaneously.";
    HloInstruction* best = best_items[0]->instruction;
    VLOG(1) << "Remat via compression: " << best->name() << " (saving "
            << HumanReadableNumBytes(memory_tracker->MemoryReducedIfCompressed(
                   best_items[0], best_strategy.compact_shape))
            << ")";

    TF_ASSIGN_OR_RETURN(
        num_instructions_added.net_instructions_added,
        CompressInstruction(memory_tracker, best_items[0],
                            best_strategy.compact_shape, instruction_list));

  } else if (best_strategy.kind == RematStrategy::kHostOffload) {
    CHECK_EQ(best_items.size(), 1)
        << "More than one buffer offloaded simultaneously.";
    VLOG(1) << "Remat via offload: " << best_items[0]->instruction->name();
    TF_ASSIGN_OR_RETURN(
        num_instructions_added.net_instructions_added,
        OffloadInstruction(memory_tracker, best_items[0], instruction_list));
    VLOG(4) << "Offload done, hlo computation:\n"
            << memory_tracker->computation()->ToString();
    VLOG(6) << "Memory tracker:\n" << memory_tracker->ToString();
  } else {
    CHECK_EQ(best_strategy.kind, RematStrategy::kRecompute)
        << "Expecting strategy to be Recompute";
    VLOG(1) << "Remat via recomputation: {"
            << absl::StrJoin(best_items, ", ",
                             [](std::string* out, HloRematItem* item) {
                               absl::StrAppend(out, item->instruction->name());
                             })
            << '}';
    TF_ASSIGN_OR_RETURN(num_instructions_added.net_instructions_added,
                        RematerializeInstructions(
                            memory_tracker, &best_items,
                            remat_move_instructions, instruction_list, schedule,
                            rematerialization, rematerializable_map));
  }
  return num_instructions_added;
}
}  // namespace

absl::StatusOr<int64_t> HloRematerialization::ComputePeakMemory(
    const HloComputation* computation, const HloInstructionSequence& order,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  TF_ASSIGN_OR_RETURN(
      auto peak_memory_result,
      ComputePeakMemoryAndInstruction(computation, order, execution_threads));
  return peak_memory_result.memory_usage;
}

absl::StatusOr<MemoryUsageAndInstruction>
HloRematerialization::ComputePeakMemoryAndInstruction(
    const HloComputation* computation, const HloInstructionSequence& order,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  HloRematInstructionList instruction_list(order);
  std::unique_ptr<HloRematerializationSweepMemoryTracker> tracker =
      HloRematerializationSweepMemoryTracker::CreateTracker(
          options_, computation, *points_to_analysis_, instruction_list);
  int64_t peak_memory = tracker->memory_usage();
  const HloInstruction* peak_instruction =
      instruction_list.first() == nullptr
          ? nullptr
          : instruction_list.first()->instruction;
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_RETURN_IF_ERROR(tracker->BeginInstruction(item));
    TF_ASSIGN_OR_RETURN(
        int64_t callee_usage,
        CalledComputationsMemoryUsage(instruction, execution_threads));
    int64_t memory_at_instruction = tracker->memory_usage() + callee_usage;
    if (memory_at_instruction > peak_memory) {
      peak_memory = memory_at_instruction;
      peak_instruction = instruction;
    }
    TF_RETURN_IF_ERROR(tracker->EndInstruction());
  }
  VLOG(1) << "Peak memory for " << computation->name() << ": "
          << HumanReadableNumBytes(peak_memory);
  return MemoryUsageAndInstruction({peak_memory, peak_instruction});
}

absl::StatusOr<int64_t> HloRematerialization::CalledComputationsMemoryUsage(
    const HloInstruction* instruction,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  const CallSite* callsite =
      call_graph_->GetNode(instruction->parent()).GetCallSite(instruction);
  if (callsite == nullptr || callsite->context() == CallContext::kEmbedded) {
    return 0;
  }
  int64_t callee_usage = 0;
  for (const HloComputation* computation : callsite->called_computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads)) {
      continue;
    }
    TF_RET_CHECK(ContainsKey(computation_peak_memory_, computation));
    callee_usage += computation_peak_memory_.at(computation);
  }
  return callee_usage;
}

absl::StatusOr<int64_t> HloRematerialization::GetComputationPeakMemory(
    const HloComputation* computation) {
  if (options_.remat_algorithm == RematAlgorithm::kAlwaysRemat) {
    return computation_peak_memory_.at(computation);
  }
  if (options_.remat_algorithm == RematAlgorithm::kPeakPriority) {
    TF_ASSIGN_OR_RETURN(MemoryUsageAndInstruction memory_usage_and_instruction,
                        computation_peak_memory_tracker_.at(computation)
                            ->ComputePeakMemoryUsageAndInstruction());
    return memory_usage_and_instruction.memory_usage;
  }
  return absl::InvalidArgumentError("Unsupported rematerialization algorithm.");
}

absl::Status HloRematerialization::UpdateScheduleFromSequence(
    HloComputation* computation, HloSchedule* schedule,
    const HloInstructionSequence& sequence,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  schedule->set_sequence(computation, sequence);

  // TODO(b/398843357): This is expensive and we shouldn't need to recompute
  // the points_to_analysis_ after rematerializing each computation. Recompute
  // points_to_analysis_ since the older analysis does not include
  // rematerialized instructions.
  TF_ASSIGN_OR_RETURN(points_to_analysis_,
                      TuplePointsToAnalysis::Run(computation->parent()));
  TF_ASSIGN_OR_RETURN(
      computation_peak_memory_[computation],
      ComputePeakMemory(computation, schedule->sequence(computation),
                        execution_threads));

  return absl::OkStatus();
}

absl::StatusOr<bool>
HloRematerialization::RematerializeCalledComputationsPeakPriority(
    const CallSite* callsite, int64_t memory_tracker_memory_usage,
    HloSchedule* schedule, int64_t memory_limit_bytes, int64_t min_remat_size,
    int64_t cost_estimate_memory_limit_bytes,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool module_changed_in_this_pass = false;
  for (HloComputation* called_computation : callsite->called_computations()) {
    if (!ContainsKey(rematerialized_computations_, called_computation) &&
        HloInstruction::IsThreadIncluded(called_computation->execution_thread(),
                                         execution_threads)) {
      // Memory limit for the subcomputation is the memory limit less the
      // amount of memory used at this point in the computation.
      int64_t subcomputation_memory_limit_bytes = std::max<int64_t>(
          0, memory_limit_bytes - memory_tracker_memory_usage);

      TF_ASSIGN_OR_RETURN(
          bool subcomputation_changed,
          RematerializeComputationPeakPriority(
              called_computation, schedule, subcomputation_memory_limit_bytes,
              min_remat_size, execution_threads));
      module_changed_in_this_pass |= subcomputation_changed;
    }
  }
  return module_changed_in_this_pass;
}

// Rematerializes the given peak memory instruction until the peak memory is
// reduced or the effort is exhausted. Returns whether the module was changed.
absl::StatusOr<HloRematerialization::RematerializationStepResult>
RematPeakAggressively(
    const HloInstruction* peak_instruction,
    HloRematerialization::RematerializationStateData& state,
    HloRematerialization* remat,
    HloRematerializationPeakMemoryTracker& memory_tracker,
    int64_t peak_memory_during_remat, int64_t memory_limit_bytes,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Initialize max_block_size to 1 so that only
  // single instruction rematerialization is considered first.
  bool module_changed_in_this_pass = false;
  int max_block_size = 1;
  int net_instructions_added = 0;
  int remat_count = 0;
  // Rematerialize until the peak usage is brought down.
  TF_ASSIGN_OR_RETURN(int64_t current_memory_usage,
                      memory_tracker.GetMemoryUsage());
  while (current_memory_usage >= peak_memory_during_remat) {
    VLOG(2) << "Highest peak memory at instruction " << peak_instruction->name()
            << ", using " << HumanReadableNumBytes(current_memory_usage)
            << " (peak memory "
            << HumanReadableNumBytes(peak_memory_during_remat) << ") "
            << ", limit is " << HumanReadableNumBytes(memory_limit_bytes);

    TF_ASSIGN_OR_RETURN(
        InstructionsAdded instructions_added,
        RematerializeBestBlock(
            /*min_block_size=*/1, max_block_size, &memory_tracker,
            state.instruction_list, state.schedule,
            state.cost_estimate_memory_limit_bytes, state.rematerializable_map,
            state.remat_move_instructions, remat));
    net_instructions_added += instructions_added.net_instructions_added;
    remat_count += instructions_added.remat_count;
    TF_ASSIGN_OR_RETURN(current_memory_usage, memory_tracker.GetMemoryUsage());
    if (instructions_added.net_instructions_added > 0) {
      VLOG(1) << "memory_usage after rematerialization = "
              << HumanReadableNumBytes(current_memory_usage);
      VLOG(1) << "net_instructions_added = "
              << instructions_added.net_instructions_added
              << " remat_count = " << instructions_added.remat_count;
    }
    if (instructions_added.remat_count == 0) {
      // Unable to find a block to rematerialize.
      // Consider doubling the block size.
      max_block_size = 2 * max_block_size;
      VLOG(3) << "Unable to find a block to rematerialize, increasing "
                 "block size to "
              << max_block_size;
    }
    if (instructions_added.remat_count > 0) {
      VLOG(2) << "Instructions were rematerialized";
      // Found a valid block. Reset to start looking for single
      // instructions again.
      remat->UpdateMaxRematerializedBlockSize(max_block_size);
      module_changed_in_this_pass = true;
      max_block_size = 1;
    }
    if (max_block_size > remat->GetBlockSizeLimit()) {
      VLOG(2) << "Block size limit reached. Block size = " << max_block_size;
      break;
    }
  }
  return HloRematerialization::RematerializationStepResult{
      module_changed_in_this_pass,
      net_instructions_added,
      remat_count,
  };
}

absl::StatusOr<HloRematerialization::RematSubpassResult>
HloRematerialization::PeakPrioritySubPass(
    HloRematerialization::RematerializationStateData& state,
    HloComputation* computation, const CallGraphNode& call_graph_node,
    int64_t min_remat_size, int64_t peak_memory_during_remat,
    int64_t memory_limit_bytes,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(3) << "Loading memory tracker for rematerialization on "
          << computation->name() << " instruction list size "
          << state.instruction_list->size();
  TF_ASSIGN_OR_RETURN(points_to_analysis_,
                      TuplePointsToAnalysis::Run(computation->parent()));
  HloRematerializationPeakMemoryTracker* memory_tracker =
      computation_peak_memory_tracker_.at(computation).get();
  state.instruction_list->PromoteNodesToSkip([&](HloRematItem* item) {
    return memory_tracker->AllocatedSize(item) >= min_remat_size;
  });
  bool module_changed_in_this_subpass = false;

  // The peak memory of the computation at any point in the instruction
  // sequence.
  TF_ASSIGN_OR_RETURN(const MemoryUsageAndInstruction
                          peak_memory_and_instruction_prior_to_subpass,
                      memory_tracker->ComputePeakMemoryUsageAndInstruction());
  const int64_t peak_memory_prior_to_subpass =
      peak_memory_and_instruction_prior_to_subpass.memory_usage;
  const HloInstruction* peak_memory_instruction_prior_to_subpass =
      peak_memory_and_instruction_prior_to_subpass.instruction;

  // Total count of instructions rematerialized.
  int64_t remat_count = 0;
  // Total count of clones created minus number of original rematerialized
  // instructions which are dead.
  int64_t net_instructions_added = 0;

  // Jump straight to the peak memory instruction.
  TF_RETURN_IF_ERROR(memory_tracker->JumpToInstruction(
      peak_memory_instruction_prior_to_subpass));

  // Rematerialize until the peak usage is brought down.
  TF_ASSIGN_OR_RETURN(
      HloRematerialization::RematerializationStepResult remat_step_result,
      RematPeakAggressively(peak_memory_instruction_prior_to_subpass, state,
                            this, *memory_tracker, peak_memory_during_remat,
                            memory_limit_bytes, execution_threads));
  remat_count += remat_step_result.remat_instructions_count;
  net_instructions_added += remat_step_result.net_instructions_added;
  module_changed_in_this_subpass |= remat_step_result.module_changed;
  VLOG(2) << "Remat was "
          << (remat_step_result.module_changed ? "able" : "not able")
          << " to rematerialize the peak memory instant";

  TF_ASSIGN_OR_RETURN(const int64_t updated_memory_usage,
                      memory_tracker->GetMemoryUsage());
  VLOG(2) << "Finished block rematerialization, peak memory = "
          << HumanReadableNumBytes(updated_memory_usage)
          << " with peak memory at instruction "
          << peak_memory_instruction_prior_to_subpass->name();
  const CallSite* callsite =
      call_graph_node.GetCallSite(peak_memory_instruction_prior_to_subpass);
  if (callsite != nullptr && callsite->context() == CallContext::kControlFlow &&
      updated_memory_usage > memory_limit_bytes) {
    // Memory usage exceeds the limit. Try to rematerialize any
    // subcomputation(s) that this instruction calls.
    VLOG(1) << "Memory usage still over the limit (" << updated_memory_usage
            << " > " << memory_limit_bytes
            << "). Rematerializing computations called by "
            << peak_memory_instruction_prior_to_subpass->name();

    TF_ASSIGN_OR_RETURN(
        const int64_t updated_memory_usage_without_callees,
        memory_tracker->GetMemoryUsageWithoutCalledComputations());
    TF_ASSIGN_OR_RETURN(
        bool callee_usage_changed_sub_module,
        RematerializeCalledComputationsPeakPriority(
            callsite, updated_memory_usage_without_callees, state.schedule,
            memory_limit_bytes, min_remat_size,
            state.cost_estimate_memory_limit_bytes, execution_threads));
    module_changed_in_this_subpass |= callee_usage_changed_sub_module;

    // Update memory tracker to account for any rematerialization performed
    // in the callee computations.
    for (const HloComputation* computation : callsite->called_computations()) {
      TF_RETURN_IF_ERROR(
          memory_tracker->CalleeComputationWasUpdated(computation));
    }
  }

  if (module_changed_in_this_subpass) {
    VLOG(2) << "Module changed in this pass, updating peak memory stats.";
    TF_ASSIGN_OR_RETURN(
        MemoryUsageAndInstruction new_peak_memory_and_instruction,
        PeakPriorityUpdateVariables(*state.instruction_list, computation,
                                    state.schedule, execution_threads));
    VLOG(3) << "Updating peak_memory_during_remat, OLD: "
            << peak_memory_during_remat
            << " NEW: " << new_peak_memory_and_instruction.memory_usage;
    VLOG(3) << "Updating peak_memory_instruction, OLD: "
            << peak_memory_instruction_prior_to_subpass->name()
            << " NEW: " << new_peak_memory_and_instruction.instruction->name();
    peak_memory_during_remat = new_peak_memory_and_instruction.memory_usage;
  } else {
    VLOG(2)
        << "No instructions were rematerialized, stopping remat inner loop, "
           "peak memory = "
        << HumanReadableNumBytes(peak_memory_during_remat);
  }

  bool over_memory_limit = peak_memory_during_remat > memory_limit_bytes;

  VLOG(1) << "In computation " << computation->name() << " rematerialized "
          << remat_count << " instructions; " << net_instructions_added
          << " net instructions added";
  VLOG(1) << "  peak memory usage now "
          << HumanReadableNumBytes(peak_memory_during_remat) << " (was "
          << HumanReadableNumBytes(peak_memory_prior_to_subpass) << ")";
  VLOG(2) << "Should stop remat of computation: "
          << (!over_memory_limit || !module_changed_in_this_subpass)
          << "(peak memory_during_remat = " << peak_memory_during_remat
          << ", memory_limit_bytes = " << memory_limit_bytes
          << ", changed = " << module_changed_in_this_subpass << ")";

  RematSubpassResult remat_subpass_result{
      // NOLINTNEXTLINE (-Wpre-c++20-compat-pedantic)
      .status = RematSubpassStatus::kUnchanged,
      .peak_memory_during_remat = peak_memory_during_remat,
  };
  if (module_changed_in_this_subpass && over_memory_limit) {
    remat_subpass_result.status =
        RematSubpassStatus::kChangedButOverMemoryLimit;
  }
  if (module_changed_in_this_subpass && !over_memory_limit) {
    remat_subpass_result.status =
        RematSubpassStatus::kChangedAndUnderMemoryLimit;
  }
  return remat_subpass_result;
}

absl::StatusOr<bool> HloRematerialization::RematerializeComputationPeakPriority(
    HloComputation* computation, HloSchedule* schedule,
    int64_t memory_limit_bytes, int64_t min_remat_size,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Rematerializing Using Peak Priority";
  // If memory limit is zero, cost savings estimates don't work because the cost
  // is defined as memory_limit_bytes / memory_reduced. Bounds it to a large
  // enough value for cost differences to be comparable.
  int64_t cost_estimate_memory_limit_bytes =
      std::max(kMinimumCostEstimateMemoryLimitBytes, memory_limit_bytes);

  TF_ASSIGN_OR_RETURN(MemoryUsageAndInstruction peak_memory_result,
                      computation_peak_memory_tracker_.at(computation)
                          ->ComputePeakMemoryUsageAndInstruction());
  int64_t peak_memory_during_remat = peak_memory_result.memory_usage;
  if (peak_memory_during_remat <= memory_limit_bytes) {
    // Nothing to do.
    VLOG(1) << "Asked to rematerialize computation of size "
            << peak_memory_during_remat
            << " but it already fits within the given memory limit ("
            << memory_limit_bytes << ")";
    return false;
  }
  VLOG(1) << "Rematerializing computation " << computation->name()
          << " with limit " << HumanReadableNumBytes(memory_limit_bytes);
  VLOG(1) << "peak memory usage is "
          << HumanReadableNumBytes(peak_memory_during_remat);
  CHECK(!ContainsKey(rematerialized_computations_, computation));

  // If the rematerialization makes the source instruction dead, then the
  // rematerialization is added to 'remat_move_instructions' (the
  // rematerialization is essentially a move). If the next rematerialization
  // of the instruction is also a move then the rematerialization should be
  // added to the denylist instead.
  absl::flat_hash_set<const HloInstruction*> remat_move_instructions;

  // The map from instructions to their rematerializable status.
  absl::flat_hash_map<const HloInstruction*, bool> rematerializable_map;
  const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);
  // HloRematInstructionList instruction_list(schedule->sequence(computation));

  HloRematerialization::RematerializationStateData rematerialization_state = {
      computation_instruction_list_.at(computation).get(),
      computation,
      schedule,
      memory_limit_bytes,
      cost_estimate_memory_limit_bytes,
      &rematerializable_map,
      &remat_move_instructions,
      &execution_threads};

  RematSubpassStatus remat_subpass_status;
  bool changed = false;
  do {
    TF_ASSIGN_OR_RETURN(
        RematSubpassResult remat_subpass_result,
        PeakPrioritySubPass(rematerialization_state, computation,
                            call_graph_node, min_remat_size,
                            peak_memory_during_remat, memory_limit_bytes,
                            execution_threads));
    changed |= (remat_subpass_result.status != RematSubpassStatus::kUnchanged);
    remat_subpass_status = remat_subpass_result.status;
    peak_memory_during_remat = remat_subpass_result.peak_memory_during_remat;
  } while (remat_subpass_status ==
           RematSubpassStatus::kChangedButOverMemoryLimit);

  rematerialized_computations_.insert(computation);
  return changed;
}

absl::StatusOr<MemoryUsageAndInstruction>
HloRematerialization::PeakPriorityUpdateVariables(
    const HloRematInstructionList& instruction_list,
    HloComputation* computation, HloSchedule* schedule,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // No need to update instruction_list, we kept it updated as we
  // added instructions.
  HloInstructionSequence sequence_from_list;
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    sequence_from_list.push_back(item->instruction);
  }
  TF_RETURN_IF_ERROR(HloRematerialization::UpdateScheduleFromSequence(
      computation, schedule, sequence_from_list, execution_threads));
  VLOG(2) << "Schedule updated";
  // Update peak memory.
  return computation_peak_memory_tracker_.at(computation)
      ->ComputePeakMemoryUsageAndInstruction();
}

absl::StatusOr<bool> HloRematerialization::RematerializeComputation(
    HloComputation* computation, HloSchedule* schedule,
    int64_t memory_limit_bytes, int64_t min_remat_size,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto peak_memory_usage = computation_peak_memory_.at(computation);
  if (peak_memory_usage <= memory_limit_bytes) {
    // Nothing to do.
    VLOG(1) << "Asked to rematerialize computation of size "
            << peak_memory_usage
            << " but it already fits within the given memory limit ("
            << memory_limit_bytes << ")";
    return false;
  }
  VLOG(1) << "Rematerializing computation " << computation->name()
          << " with limit " << HumanReadableNumBytes(memory_limit_bytes);
  VLOG(1) << "peak memory usage is "
          << HumanReadableNumBytes(peak_memory_usage);
  CHECK(!ContainsKey(rematerialized_computations_, computation));

  HloRematInstructionList instruction_list(schedule->sequence(computation));
  std::unique_ptr<HloRematerializationSweepMemoryTracker> memory_tracker =
      HloRematerializationSweepMemoryTracker::CreateTracker(
          options_, computation, *points_to_analysis_, instruction_list);

  instruction_list.PromoteNodesToSkip([&](HloRematItem* item) {
    return memory_tracker->AllocatedSize(item) >= min_remat_size;
  });
  bool changed = false;

  // If the rematerialization makes the source instruction dead, then the
  // rematerialization is added to 'remat_move_instructions' (the
  // rematerialization is essentially a move). If the next rematerialization of
  // the instruction is also a move then the rematerialization is added to the
  // denylist.
  absl::flat_hash_set<const HloInstruction*> remat_move_instructions;

  // The map from instructions to their rematerializable status.
  // In Peak priority remat, it also stores instructions that were at any point
  // recomputed by this pass for optimization reasons.
  absl::flat_hash_map<const HloInstruction*, bool> rematerializable_map;

  // The peak memory of the computation at any point in the instruction
  // sequence.
  int64_t peak_memory = memory_tracker->memory_usage();

  // Total count of instructions rematerialized.
  int64_t remat_count = 0;
  // Total count of clones created minus number of original rematerialized
  // instructions which are dead.
  int64_t net_instructions_added = 0;

  const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);

  // Iterate through all instructions in the sequence. At each instruction
  // (program point) if memory_usage exceeds the specified limit then
  // rematerialize HLO instructions until memory_usage is reduced.
  int64_t instruction_index = 0;
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_ASSIGN_OR_RETURN(
        int64_t callee_usage,
        CalledComputationsMemoryUsage(instruction, execution_threads));
    TF_RETURN_IF_ERROR(memory_tracker->BeginInstruction(item));

    VLOG(2) << "Program point at " << instruction->name()
            << ", memory usage = " << memory_tracker->memory_usage()
            << ", callee usage = " << callee_usage << ", [" << instruction_index
            << "/" << instruction_list.size() << "]";
    instruction_index++;

    // Initialize both min_block_size and max_block_size to 1 so that only
    // single instruction rematerialization is considered first.
    int min_block_size = 1;
    int max_block_size = 1;
    // Only trigger rematerialization when the memory usage changes.
    if (memory_tracker->AllocatedSize(item) + callee_usage > 0) {
      // Finding larger blocks of instructions to rematerialize can be time
      // consuming. To limit the amount of time spent attempting to find such
      // large blocks, count the amount of effort expended to find single
      // instructions to rematerialize and then limit the total amount of effort
      // to at most a factor of block_rematerialization_factor_ more.
      bool is_first_phase = true;
      int64_t first_phase_effort = 0;
      int64_t second_phase_effort = 0;
      while (memory_tracker->memory_usage() + callee_usage >
             memory_limit_bytes) {
        VLOG(2) << "Over memory limit at instruction " << instruction->name()
                << ", using "
                << HumanReadableNumBytes(memory_tracker->memory_usage() +
                                         callee_usage)
                << ", limit is " << HumanReadableNumBytes(memory_limit_bytes);

        TF_ASSIGN_OR_RETURN(
            InstructionsAdded instructions_added,
            RematerializeBestBlock(
                min_block_size, max_block_size, memory_tracker.get(),
                &instruction_list, schedule, memory_limit_bytes,
                &rematerializable_map, &remat_move_instructions, this));
        net_instructions_added += instructions_added.net_instructions_added;
        remat_count += instructions_added.remat_count;
        if (is_first_phase) {
          first_phase_effort += instructions_added.effort;
        } else {
          second_phase_effort += instructions_added.effort;
        }
        if (instructions_added.net_instructions_added > 0) {
          VLOG(1) << "memory_usage after rematerialization = "
                  << HumanReadableNumBytes(memory_tracker->memory_usage());
        }
        if (instructions_added.remat_count == 0) {
          // Unable to find a block to rematerialize.
          // Consider doubling the block size.
          min_block_size = max_block_size + 1;
          max_block_size = 2 * max_block_size;
          is_first_phase = false;
        } else {
          // Found a valid block. Reset to start looking for single instructions
          // again.
          UpdateMaxRematerializedBlockSize(max_block_size);
          changed = true;
          min_block_size = 1;
          max_block_size = 1;
        }
        if (max_block_size > options_.block_size_limit ||
            second_phase_effort >
                options_.block_rematerialization_factor * first_phase_effort) {
          break;
        }
      }
    }
    const CallSite* callsite = call_graph_node.GetCallSite(instruction);
    if (callsite != nullptr &&
        callsite->context() == CallContext::kControlFlow &&
        memory_tracker->memory_usage() + callee_usage > memory_limit_bytes) {
      // Memory usage exceeds the limit. Try to rematerialize any
      // subcomputation(s) that this instruction calls.
      VLOG(1) << "Memory usage still over the limit ("
              << (memory_tracker->memory_usage() + callee_usage) << " > "
              << memory_limit_bytes
              << "). Rematerializing computations called by "
              << instruction->name();

      // Recompute callee usage to account for any rematerialization performed
      // in the callee computations.
      for (HloComputation* called_computation :
           callsite->called_computations()) {
        if (!ContainsKey(rematerialized_computations_, called_computation) &&
            HloInstruction::IsThreadIncluded(
                called_computation->execution_thread(), execution_threads)) {
          // Memory limit for the subcomputation is the memory limit less the
          // amount of memory used at this point in the computation.
          int64_t subcomputation_memory_limit_bytes = std::max<int64_t>(
              0, memory_limit_bytes - memory_tracker->memory_usage());
          TF_ASSIGN_OR_RETURN(
              bool subcomputation_changed,
              RematerializeComputation(called_computation, schedule,
                                       subcomputation_memory_limit_bytes,
                                       min_remat_size, execution_threads));
          changed |= subcomputation_changed;
        }
      }

      TF_ASSIGN_OR_RETURN(callee_usage, CalledComputationsMemoryUsage(
                                            instruction, execution_threads));
    }

    peak_memory = std::max<int64_t>(
        peak_memory, memory_tracker->memory_usage() + callee_usage);
    VLOG(3) << "peak memory usage = " << HumanReadableNumBytes(peak_memory);

    TF_RETURN_IF_ERROR(memory_tracker->EndInstruction());
  }

  // Verify some invariants on the memory tracker.
  for (auto* instruction : computation->instructions()) {
    CHECK(memory_tracker->IsInstructionPlaced(instruction))
        << instruction->name();
  }

  VLOG(1) << "In computation " << computation->name() << " rematerialized "
          << remat_count << " instructions; " << net_instructions_added
          << " net instructions added";
  VLOG(1) << "  peak memory usage now " << HumanReadableNumBytes(peak_memory)
          << " (was "
          << HumanReadableNumBytes(computation_peak_memory_.at(computation))
          << ")";

  // Update peak memory used by computation.
  computation_peak_memory_.at(computation) = peak_memory;

  // Update order to include rematerialized instructions.
  HloInstructionSequence& sequence = schedule->GetOrCreateSequence(computation);
  sequence.clear();
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    HloInstruction* instruction = item->instruction;
    if (instruction->parent() == nullptr) {
      // TODO(b/446297799): Stop it before it reaches this point.
      VLOG(2) << "Instruction " << instruction->name()
              << " is not in a computation. Ignoring";
      continue;
    }
    sequence.push_back(instruction);
  }
  rematerialized_computations_.insert(computation);

  instructions_rematerialized_ += remat_count;
  net_instructions_added_ += net_instructions_added;

  return changed;
}

absl::StatusOr<RematAlgorithmFunction>
HloRematerialization::GetRematAlgorithmFunction(
    HloRematerializationOptions::RematAlgorithm remat_algorithm) {
  switch (remat_algorithm) {
    case HloRematerializationOptions::RematAlgorithm::kAlwaysRemat:
      return absl::bind_front(&HloRematerialization::RematerializeComputation,
                              this);
    case HloRematerializationOptions::RematAlgorithm::kPeakPriority:
      return absl::bind_front(
          &HloRematerialization::RematerializeComputationPeakPriority, this);
    default:
      return absl::InvalidArgumentError(
          "Unsupported rematerialization algorithm.");
  }
}

absl::StatusOr<bool> HloRematerialization::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (options_.remat_mode_config.host_offload) {
    CHECK(options_.host_memory_offload_config.has_value())
        << "Host memory config is required when host memory offload strategy "
           "is specified";
  }
  VLOG(1) << "HloRematerialization() with memory limit of "
          << HumanReadableNumBytes(options_.memory_limit_bytes);
  if (!options_.remat_mode_config.compress &&
      !options_.remat_mode_config.recompute &&
      !options_.remat_mode_config.host_offload) {
    // All rematerialization strategies are disabled; nothing to do.
    VLOG(1) << "All rematerialization strategies are disabled. Skipping.";
    return false;
  }
  VLOG(2) << "HloRemat mode: compress: " << options_.remat_mode_config.compress
          << ", host_offload: " << options_.remat_mode_config.host_offload
          << ", recompute: " << options_.remat_mode_config.recompute;
  XLA_VLOG_LINES(3, "Before HloRematerialization:\n" + module->ToString());

  // Initialize pass object state.
  computation_peak_memory_.clear();
  computation_instruction_list_.clear();
  computation_peak_memory_tracker_.clear();
  rematerialized_computations_.clear();
  instructions_rematerialized_ = 0;
  net_instructions_added_ = 0;

  TF_RET_CHECK(module->has_schedule());
  TF_ASSIGN_OR_RETURN(points_to_analysis_, TuplePointsToAnalysis::Run(module));
  next_channel_id_ = hlo_query::NextChannelId(*module);

  // Adjust memory limit to account for the output of the entry
  // computation. This is necessary because the per-computation accounting in
  // HloRematerializationMemoryTracker do not include output as these are
  // typically allocated by the caller.
  int64_t module_output_size = 0;
  ShapeUtil::ForEachSubshape(
      module->result_shape(),
      [&module_output_size, this](const Shape& subshape,
                                  const ShapeIndex& output_index) {
        module_output_size += options_.hlo_cost_analysis.GetShapeSize(subshape);
      });

  int64_t adjusted_memory_limit_bytes =
      std::max<int64_t>(0, options_.memory_limit_bytes - module_output_size);
  VLOG(1) << "Adjusted memory limit accounting for output ("
          << HumanReadableNumBytes(module_output_size)
          << "): " << HumanReadableNumBytes(adjusted_memory_limit_bytes);

  call_graph_ = CallGraph::Build(module);

  // Buffer assignment allocates a single stack for all asynchronous
  // computations of the same thread, which persists for the entire duration of
  // the program. We need to account for this by adjusting the memory limit.
  int64_t total_async_peak_memory = 0;
  if (!options_.async_computation_parallelism.empty()) {
    // We cannot compute memory usage for both the main and asynchronous threads
    // at the same time, as that will cause the asynchronous callee usage to be
    // added to the main thread callers usage. The callee's memory is
    // preallocated, so the caller doesn't pay for it.
    absl::flat_hash_set<absl::string_view> async_threads;
    for (const auto& [computation, _] :
         options_.async_computation_parallelism) {
      async_threads.insert(computation->execution_thread());
    }
    TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
        [this, module,
         &async_threads](const CallGraphNode& node) -> absl::Status {
          auto callee_thread = node.computation()->execution_thread();
          if (node.context() == CallContext::kControlFlow &&
              HloInstruction::IsThreadIncluded(callee_thread, async_threads)) {
            // Temporary, so just use a sweep tracker even for kPeakPriority.
            TF_ASSIGN_OR_RETURN(computation_peak_memory_[node.computation()],
                                ComputePeakMemory(node.computation(),
                                                  module->schedule().sequence(
                                                      node.computation()),
                                                  {callee_thread}));
          }
          return absl::OkStatus();
        },
        /*visit_unreachable_nodes=*/false));

    int64_t async_peak_memory = 0;
    // Only consider asynchronous computations invoked from the main thread.
    for (const auto [entry_computation, parallel_threads] :
         options_.async_computation_parallelism) {
      const int64_t peak_memory =
          computation_peak_memory_.at(entry_computation);
      // Adjust memory usage for parallel execution of the same computation
      // on different devices.
      const int64_t parallel_peak_memory = peak_memory * parallel_threads;
      async_peak_memory = std::max(async_peak_memory, parallel_peak_memory);
    }
    adjusted_memory_limit_bytes =
        std::max<int64_t>(0, adjusted_memory_limit_bytes - async_peak_memory);
    total_async_peak_memory += async_peak_memory;
    VLOG(1) << "Adjusted memory limit accounting for async computations ("
            << HumanReadableNumBytes(async_peak_memory)
            << "): " << HumanReadableNumBytes(adjusted_memory_limit_bytes);

    // Reset back to a clean state, since we don't expect to utilize the
    // async computation memory usage anymore.
    computation_peak_memory_.clear();
  }
  // Compute peak memory usage of all computations in the module called in a
  // sequential context.
  TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
      [this, module,
       &execution_threads](const CallGraphNode& node) -> absl::Status {
        if (node.context() == CallContext::kControlFlow &&
            HloInstruction::IsThreadIncluded(
                node.computation()->execution_thread(), execution_threads)) {
          if (options_.remat_algorithm == RematAlgorithm::kAlwaysRemat) {
            TF_ASSIGN_OR_RETURN(computation_peak_memory_[node.computation()],
                                ComputePeakMemory(node.computation(),
                                                  module->schedule().sequence(
                                                      node.computation()),
                                                  execution_threads));
          } else if (options_.remat_algorithm ==
                     RematAlgorithm::kPeakPriority) {
            computation_instruction_list_[node.computation()] =
                std::make_unique<HloRematInstructionList>(
                    module->schedule().sequence(node.computation()));
            TF_ASSIGN_OR_RETURN(
                computation_peak_memory_tracker_[node.computation()],
                HloRematerializationPeakMemoryTracker::CreateTracker(
                    options_, node.computation(), *points_to_analysis_,
                    *computation_instruction_list_[node.computation()],
                    call_graph_.get(), execution_threads,
                    computation_peak_memory_tracker_));
          } else {
            return absl::InvalidArgumentError(
                "Unsupported rematerialization algorithm.");
          }
        }
        return absl::OkStatus();
      },
      /*visit_unreachable_nodes=*/false));

  // The peak memory usage of the module equals the peak memory use of the entry
  // computation plus the output size of the computation plus memory use of
  // asynchronous computations. This is because the peak memory for a
  // computation does not include the output as this is typically accounted for
  // in the caller.
  TF_ASSIGN_OR_RETURN(const int64_t entry_computation_peak_memory,
                      GetComputationPeakMemory(module->entry_computation()));
  const int64_t before_peak_memory = entry_computation_peak_memory +
                                     module_output_size +
                                     total_async_peak_memory;
  VLOG(1) << "Peak memory usage of module (before): "
          << HumanReadableNumBytes(before_peak_memory);

  // Initialize the HloCostAnalysis on this computation.
  for (auto* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&options_.hlo_cost_analysis));
  }

  TF_ASSIGN_OR_RETURN(RematAlgorithmFunction remat_algorithm_func,
                      GetRematAlgorithmFunction(options_.remat_algorithm));

  // Subcomputations called by the entry computation will also be
  // rematerialized.
  TF_ASSIGN_OR_RETURN(
      bool changed,
      remat_algorithm_func(module->entry_computation(), &module->schedule(),
                           adjusted_memory_limit_bytes, options_.min_remat_size,
                           execution_threads));
  // Rematerialization can introduce dead code. This occurs if all uses of an
  // instruction are replaced with rematerializations of the instruction.

  // Stash away the schedule during copy insertion, to avoid validation failures
  // while the module is in flux.
  HloSchedule saved_schedule = module->schedule();
  module->clear_schedule();
  TF_ASSIGN_OR_RETURN(bool dead_code_removed, HloPassFix<HloDCE>().Run(module));
  changed |= dead_code_removed;

  // After DCE, the module sequence may include instructions which no longer
  // exist. Update the schedule and restore it.
  TF_RETURN_IF_ERROR(saved_schedule.Update(execution_threads));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(saved_schedule)));
  VLOG(1) << "Rematerialized " << instructions_rematerialized_
          << " instructions in module " << module->name() << "; "
          << net_instructions_added_ << " net instructions added";
  const int64_t current_peak_memory = entry_computation_peak_memory +
                                      module_output_size +
                                      total_async_peak_memory;
  VLOG(1) << "Peak memory usage of module now "
          << HumanReadableNumBytes(current_peak_memory) << " ("
          << current_peak_memory << " bytes), was "
          << HumanReadableNumBytes(before_peak_memory) << " ("
          << before_peak_memory << " bytes)";
  const int64_t reduced_peak_memory = before_peak_memory - current_peak_memory;
  VLOG(1) << "Reduced peak memory by "
          << HumanReadableNumBytes(reduced_peak_memory) << " ("
          << reduced_peak_memory << " bytes)";

  sizes_.before_bytes = before_peak_memory;
  sizes_.after_bytes = current_peak_memory;

  XLA_VLOG_LINES(5, "After HloRematerialization:\n" + module->ToString());

  if (current_peak_memory > options_.memory_limit_bytes) {
    LOG(WARNING) << absl::StrFormat(
        "Can't reduce memory use below %s (%d bytes) by rematerialization; "
        "only reduced to %s (%d bytes), down from %s (%d bytes) originally",
        HumanReadableNumBytes(options_.memory_limit_bytes),
        options_.memory_limit_bytes, HumanReadableNumBytes(current_peak_memory),
        current_peak_memory, HumanReadableNumBytes(before_peak_memory),
        before_peak_memory);
  }
  return changed;
}

}  // namespace xla
