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

#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_memory_tracker.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/tuple_points_to_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_data_structures.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_options.h"
#include "xla/map_util.h"
#include "xla/service/call_graph.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::tsl::strings::HumanReadableNumBytes;

// Potential optimizations:
// . Avoid N^2 behavior by keeping a priority queue of candidates.
// . Cache IsRematerializable in HloRematItem?  Only correct if control
//   predecessors and successors don't change.

bool IsRematerializable(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kCopy) {
    if (LayoutUtil::Equal(instruction->shape().layout(),
                          instruction->operand(0)->shape().layout())) {
      // Don't rematerialize copies added by copy insertion (layout doesn't
      // change).
      return false;
    }
  }

  if (auto collective = DynCast<HloCollectiveInstruction>(instruction)) {
    return !collective->constrain_layout();
  }

  // Don't rematerialize instructions with side effects or instructions which
  // cannot be cloned safely.
  switch (instruction->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConstant:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kParameter:
    case HloOpcode::kWhile:
      return false;
    default:
      return !instruction->HasSideEffect();
  }
}

// Checks whether an instruction can be rematerialized, by looking up the
// cache before, and eventually calling the IsRematerializable() API.
bool CanBeRematerialized(
    const HloInstruction* instruction,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map) {
  auto it = rematerializable_map->find(instruction);
  if (it != rematerializable_map->end()) {
    return it->second;
  }
  bool rematerializable = IsRematerializable(instruction);
  (*rematerializable_map)[instruction] = rematerializable;
  return rematerializable;
}

// Whether any instruction in `block` is denylisted or non-rematerializable.
bool AnyDenylistedOrNonRematerializable(
    const std::vector<HloRematItem*>& block,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map) {
  for (auto* item : block) {
    if (item->denylisted) {
      return true;
    }
    if (!CanBeRematerialized(item->instruction, rematerializable_map)) {
      return true;
    }
  }
  return false;
}

}  // namespace

HloRematerializationBufferAnalyzer
HloRematerializationBufferAnalyzer::CreateAnalyzer(
    const HloRematerializationOptions& options,
    const HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloRematInstructionList& instruction_list) {
  HloRematerializationBufferAnalyzer analyzer(options, instruction_list);

  PointsToSet::BufferSet live_out_set =
      points_to_analysis.GetPointsToSet(computation->root_instruction())
          .CreateFlattenedSet();
  absl::flat_hash_map<const LogicalBuffer*, BufferId>
      logical_buffer_to_buffer_id;
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* const instruction = item->instruction;
    for (const LogicalBuffer* logical_buffer :
         points_to_analysis.GetBuffersDefinedByInstruction(instruction)) {
      Buffer* buffer;
      if (instruction->opcode() == HloOpcode::kWhile) {
        // The while instruction defines no new buffers. Instead it reuses the
        // buffers of its operand. Find the Buffer of its operand at the
        // proper ShapeIndex.
        const PointsToSet& operand_points_to =
            points_to_analysis.GetPointsToSet(instruction->operand(0));
        CHECK_EQ(operand_points_to.element(logical_buffer->index()).size(), 1);
        const LogicalBuffer* source_logical_buffer =
            operand_points_to.element(logical_buffer->index())[0];
        buffer = &analyzer.buffers_.at(
            logical_buffer_to_buffer_id.at(source_logical_buffer));

        // Mark buffer as has indirect use and live out.
        buffer->has_indirect_uses = true;
        buffer->live_out =
            buffer->live_out || ContainsKey(live_out_set, logical_buffer);

        // Add users of while to Buffer users.
        bool unused;
        for (HloRematItemUse& user_item :
             analyzer.GetUsers(instruction_list, logical_buffer,
                               points_to_analysis, &unused)) {
          auto existing_user_it =
              absl::c_find_if(buffer->users, [&](const HloRematItemUse& use) {
                return user_item.user == use.user;
              });
          if (existing_user_it == buffer->users.end()) {
            buffer->unfinished_user_count++;
            user_item.user->buffers_used.push_back(buffer->id);
            buffer->users.push_back(user_item);
          }
        }
      } else {
        buffer = &analyzer.CreateBufferFromLogicalBuffer(
            logical_buffer, points_to_analysis,
            ContainsKey(live_out_set, logical_buffer));
        item->buffers_defined.push_back(buffer->id);
        for (HloRematItemUse& user : buffer->users) {
          if (!absl::c_linear_search(user.user->buffers_used, buffer->id)) {
            user.user->buffers_used.push_back(buffer->id);
          }
        }
      }

      logical_buffer_to_buffer_id[logical_buffer] = buffer->id;
    }

    // Trace the output of each instruction. This is so that we can properly
    // track which outputs does GTEs have.
    for (const LogicalBuffer* logical_buffer :
         points_to_analysis.GetPointsToSet(instruction).CreateFlattenedSet()) {
      item->buffers_output.push_back(
          logical_buffer_to_buffer_id[logical_buffer]);
    }
  }

  return analyzer;
}

UsesList HloRematerializationBufferAnalyzer::GetUsers(
    const HloRematInstructionList& instruction_list,
    const LogicalBuffer* logical_buffer,
    const TuplePointsToAnalysis& points_to_analysis, bool* has_indirect_users) {
  UsesList users;
  // To identify uses iterate through all HloInstruction users of the
  // BufferAliases of the logical buffer.
  *has_indirect_users = false;
  for (const BufferAlias& buffer_alias :
       points_to_analysis.GetBufferAliases(*logical_buffer)) {
    for (const HloInstruction* user : buffer_alias.instruction()->users()) {
      if (points_to_analysis.DoesNotUseOperandBuffer(
              buffer_alias.instruction(), buffer_alias.index(), user)) {
        // The alias may be an operand of 'user', but the LogicalBuffer cannot
        // possibly be used by the instruction so ignore 'user'. This is the
        // case, for example, for the tuple element buffers in a GetTupleElement
        // instruction (the GTE instruction only uses the pointer vector).
        continue;
      }
      if (buffer_alias.instruction() != logical_buffer->instruction() &&
          !IsSupportedIndirectUser(buffer_alias.instruction())) {
        *has_indirect_users = true;
      }
      // A buffer may be used by the instruction via more than one alias. For
      // example, a buffer which appears in more than one element of a tuple.
      HloRematItem* user_item = instruction_list.GetItem(user);
      std::optional<int64_t> user_index =
          logical_buffer->index().size() != 1
              ? std::nullopt
              : std::make_optional(logical_buffer->index().back());
      for (int64_t op_idx : user->OperandIndices(buffer_alias.instruction())) {
        if (!absl::c_linear_search(
                users, HloRematItemUse{user_item, static_cast<int>(op_idx),
                                       user_index})) {
          users.push_back(
              HloRematItemUse{user_item, static_cast<int>(op_idx), user_index});
        }
      }
    }
  }
  return users;
}

bool HloRematerializationMemoryTracker::IsInstructionPlaced(
    const HloInstruction* instruction) const {
  return IsItemPlaced(GetItem(instruction));
}

bool HloRematerializationMemoryTracker::IsInProgressItem(
    HloRematItem* item) const {
  return item == in_progress_item();
}

int64_t HloRematerializationMemoryTracker::BytesUsedByBuffers(
    const HloRematItem* item, bool only_count_unplaced_users) const {
  int64_t bytes_used_by_buffers = 0;
  for (const auto& buffer_id : item->buffers_defined) {
    VLOG(3) << "  buffer " << buffer_id << "'s users are "
            << absl::StrJoin(buffer_analyzer()->buffers_.at(buffer_id).users,
                             ", ", [](std::string* str, const auto& use) {
                               str->append(use.user->instruction->name());
                             });
    for (const auto& use : buffer_analyzer()->buffers_.at(buffer_id).users) {
      if (!only_count_unplaced_users || !IsItemPlaced(use.user)) {
        // Found a non-placed user
        bytes_used_by_buffers += buffer_analyzer()->AllocatedSize(buffer_id);
        // Don't count uses of this buffer multiple times.
        break;
      }
    }
  }
  return bytes_used_by_buffers;
}

std::tuple<UsesList, UsesList>
HloRematerializationMemoryTracker::GetPlacedAndUnplacedUsers(
    const UsesList& uses) const {
  UsesList placed_users, unplaced_users;
  for (const HloRematItemUse& use : uses) {
    if (IsItemPlaced(use.user)) {
      DCHECK(IsItemFinished(use.user)) << use.user->instruction->name();
      placed_users.push_back(use);
    } else {
      unplaced_users.push_back(use);
    }
  }
  return {placed_users, unplaced_users};
}

UsesList HloRematerializationMemoryTracker::GetItemUses(
    HloRematItem* item) const {
  UsesList combined_users;
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffer_analyzer()->buffers_.at(buffer_id);
    for (const HloRematItemUse& user : buffer.users) {
      combined_users.push_back(user);
    }
  }
  return combined_users;
}

std::tuple<std::vector<HloRematItem*>, RematStrategy, int>
HloRematerializationMemoryTracker::PickRematerializationCandidates(
    const HloRematInstructionList& instruction_list, int64_t memory_limit_bytes,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
    int min_block_size, int max_block_size, int64_t peak_memory_bytes) {
  // Keep track of the cost of each rematerialization option.
  // This cost is defined as:
  //
  // memory_limit_bytes / memory_reduced
  //
  // The idea is to choose the operation that will save the most memory for
  // rematerialization and do not worry about how much the compute costs since
  // running out of memory is more harmful than taking longer to get the answer.
  std::vector<HloRematItem*> best_items;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  RematStrategy best_strategy;
  int effort = 0;
  VLOG(5) << "Picking candidate block with size in [" << min_block_size << ", "
          << max_block_size << "]";

  for (auto* start_item = instruction_list.first_skip_node();
       start_item != nullptr;
       start_item = instruction_list.next_skip_node(start_item)) {
    std::vector<HloRematItem*> block =
        GetInitialBlock(instruction_list, start_item, min_block_size);
    if (block.size() < min_block_size) {
      VLOG(5) << "No more blocks of size at least " << min_block_size;
      break;
    }

    // If any item in the starting block are denylisted or non-rematable, then
    // break and move on to next start_item (we can actually move to the last
    // invalid item in this block, but let's ignore that optimization for now).
    // In peak priority mode, the map includes all instructions that were
    // recomputed so far to improve compilation time and remat efficiency. Such
    // optimization could be applied to other modes as well.
    if (AnyDenylistedOrNonRematerializable(block, rematerializable_map)) {
      VLOG(5) << "Block contains denylisted or non-rematerializable items, "
                 "skipping.";
      continue;
    }

    // First, calculate the cost of compression rematerialization for this
    // instruction.
    if (options().remat_mode_config.compress && block.size() == 1) {
      auto cost =
          GetCostOfCompression(block[0], memory_limit_bytes, peak_memory_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kCompress on block of size " << block.size();
        best_strategy.kind = RematStrategy::kCompress;
        // TODO(b/293323448): This `best_strategy.compact_shape` is already
        // computed inside GetCostOfCompression, should we get it from there? Or
        // is it ok to recompute?
        best_strategy.compact_shape =
            *buffer_analyzer()->GetCompactShape(block[0]->instruction).value();
        best_items = block;
        best_cost = *cost;
      }
    }

    // Second, calculate the cost of host offload rematerialization for this
    // instruction.
    if (options().remat_mode_config.host_offload && block.size() == 1) {
      auto cost = GetCostOfHostOffload(block[0], memory_limit_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kHostOffload on block of size "
                << block.size();
        best_strategy.kind = RematStrategy::kHostOffload;
        best_items = block;
        best_cost = *cost;
      }
    }

    // Finally, calculate the cost of recompute rematerialization for this
    // instruction block. There is one difference between this rematerialization
    // strategy and the other two: recompute can rematerialize more than one
    // instruction at a time. Evaluate the cost of rematerializing the current
    // block, add the next instruction to the block, and then repeat until we
    // reach the configured max block size.
    if (!options().remat_mode_config.recompute) {
      // Recompute is not enabled, nothing else to do for this block.
      continue;
    }

    while (block.size() <= max_block_size) {
      auto cost = GetCostOfRecompute(block, memory_limit_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kRecompute on block of size "
                << block.size();
        best_strategy.kind = RematStrategy::kRecompute;
        best_items = block;
        best_cost = *cost;
      }

      // Try to add the next instruction to this block to evaluate as a possibly
      // better candidate for rematerialization.
      auto* last_item = block[block.size() - 1];
      auto* next_item = instruction_list.next(last_item);
      if (next_item == nullptr || next_item->denylisted || !next_item->placed ||
          next_item == in_progress_item() ||
          !CanBeRematerialized(next_item->instruction, rematerializable_map)) {
        break;
      }
      block.push_back(next_item);
    }
  }
  return {best_items, best_strategy, effort};
}

std::vector<HloRematItem*> HloRematerializationMemoryTracker::GetInitialBlock(
    const HloRematInstructionList& instruction_list, HloRematItem* start_item,
    int min_block_size) const {
  std::vector<HloRematItem*> item_block;
  HloRematItem* curr_item = start_item;
  for (int i = 0; i < min_block_size; ++i) {
    if (curr_item == nullptr || !IsItemPlaced(curr_item) ||
        IsInProgressItem(curr_item)) {
      break;
    }
    item_block.push_back(curr_item);
    curr_item = instruction_list.next(curr_item);
  }
  return item_block;
}

bool HloRematerializationMemoryTracker::HasUnplacedUsers(
    HloRematItem* item) const {
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffer_analyzer()->buffers_.at(buffer_id);
    for (const HloRematItemUse& user : buffer.users) {
      if (!IsItemPlaced(user.user)) {
        return true;
      }
    }
  }
  return false;
}

bool HloRematerializationMemoryTracker::IsCurrentlyRecomputable(
    absl::Span<const HloRematItem* const> items) const {
  CHECK_NE(in_progress_item(), nullptr);
  for (const HloRematItem* item : items) {
    if (!IsItemFinished(item)) {
      LOG(WARNING) << "Unplaced item or in progress item being checked for "
                      "recomputation.";
      return false;
    }
    if (std::any_of(item->instruction->control_successors().begin(),
                    item->instruction->control_successors().end(),
                    [this](const HloInstruction* instruction) {
                      return IsInstructionPlaced(instruction);
                    })) {
      // If any of the candidate's control successor has been placed, we need
      // to skip this candidate. Otherwise we will violate control dependency.
      return false;
    }
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffer_analyzer()->buffers_.at(buffer_id);
      // Avoid recomputing instructions with indirect uses as it is
      // difficult to reason about liveness after recomputing the
      // instruction.
      // Avoid recomputing instructions with live out buffers.
      // Avoid recomputing buffers that are in nested tuples.
      // TODO(mpurohit): Check why live_out buffers are an issue here.
      if (buffer.has_indirect_uses || buffer.live_out ||
          buffer.index.size() > 1) {
        return 0;
      }
      if (IsBufferInUse(buffer_id)) {
        return 0;
      }
    }
  }
  return true;
}

bool HloRematerializationMemoryTracker::IsCurrentlyCompressible(
    const HloRematItem* item) const {
  CHECK_NE(in_progress_item(), nullptr);
  if (!IsItemFinished(item)) {
    return false;
  }

  const Shape& original_shape = item->instruction->shape();
  if (!original_shape.IsArray()) {
    return false;
  }

  // Only consider compressing single output instruction.
  if (item->buffers_output.size() != 1) {
    // TODO(b/291824123): Currently only handling single output buffers.
    VLOG(2) << "  " << item->instruction->name()
            << " has more than one output buffer; cannot offload to host.";
    return false;
  }

  BufferId buffer_id = item->buffers_output[0];
  const Buffer& output_buffer = buffer_analyzer()->buffers_.at(buffer_id);
  if (output_buffer.live_out) {
    return false;
  }
  return IsBufferLive(buffer_id) && !IsBufferInUse(buffer_id) &&
         IsInstructionCurrentlyLive(item);
}

bool HloRematerializationMemoryTracker::IsCurrentlyOffloadable(
    const HloRematItem* item) const {
  CHECK_NE(in_progress_item(), nullptr);

  // TODO(b/291823800): Bitcasts and tuples complicate things. Skip for now.
  for (auto buffer_id : item->buffers_defined) {
    for (auto use : buffer_analyzer()->buffers_.at(buffer_id).users) {
      if (use.user->instruction->opcode() == HloOpcode::kBitcast) {
        VLOG(3) << "  " << item->instruction->name()
                << " has a user which is a bitcast instruction("
                << use.user->instruction->name()
                << "); cannot offload "
                   "to host.";
        return {};
      }
      if (use.user->instruction->opcode() == HloOpcode::kTuple) {
        VLOG(3) << "  " << item->instruction->name()
                << " has a user which is a tuple instruction("
                << use.user->instruction->name()
                << "); cannot offload "
                   "to host.";
        return {};
      }
    }
  }

  // Only consider offloading single output instructions.
  if (item->buffers_output.size() != 1) {
    // TODO(b/291824123): Currently only handling single output buffers.
    VLOG(2) << "  " << item->instruction->name()
            << " has more than one output buffer; cannot offload to host.";
    return false;
  }
  BufferId buffer_id = item->buffers_output[0];
  const Buffer& output_buffer = buffer_analyzer()->buffers_.at(buffer_id);
  if (!IsItemFinished(item) || output_buffer.live_out) {
    VLOG(2) << "  " << item->instruction->name()
            << " is not yet placed, is in progress, or is \"live_out\"; cannot "
               "offload to host.";
    return {};
  }

  if (IsBufferInUse(buffer_id)) {
    VLOG(2) << "  " << item->instruction->name()
            << " is used by the current instruction in mem tracker ("
            << in_progress_item()->instruction->name()
            << "); cannot offload to host.";
    return false;
  }

  return true;
}

int64_t HloRematerializationMemoryTracker::MemoryReducedIfRecomputed(
    absl::Span<const HloRematItem* const> items) const {
  int64_t memory_reduced = 0;
  absl::flat_hash_set<const HloRematItem*> recompute_candidates;

  for (const HloRematItem* item : items) {
    // Compute the amount of memory reduced (if any) by recomputing
    // 'item->instruction'. The LogicalBuffers defined by 'item->instruction'
    // will no longer be live at this program point, so initially set
    // memory_reduced to the size of its defined values.
    for (BufferId buffer_id : item->buffers_defined) {
      if (IsBufferLive(buffer_id)) {
        memory_reduced += buffer_analyzer()->AllocatedSize(buffer_id);
      }
    }

    // Account for any logical buffers whose live range must be extended across
    // this program point.
    for (BufferId buffer_id : item->buffers_used) {
      if (!IsBufferLive(buffer_id)) {
        // This logical buffer is used by 'item->instruction' but is not live at
        // this program point. Recomputing 'item->instruction' will extend
        // the buffer's live range across this program point unless it is
        // defined by an instruction that is also being recomputed.
        HloRematItem* defining_instruction =
            buffer_analyzer()->buffers_.at(buffer_id).defining_instruction;
        if (!recompute_candidates.contains(defining_instruction)) {
          memory_reduced -= buffer_analyzer()->AllocatedSize(buffer_id);
        }
      }
    }
    recompute_candidates.insert(item);
  }

  return memory_reduced;
}

int64_t HloRematerializationMemoryTracker::MemoryReducedIfCompressed(
    const HloRematItem* item, const Shape& compact_shape) const {
  BufferId buffer_id = item->buffers_output[0];
  const Buffer& buffer = buffer_analyzer()->buffers_.at(buffer_id);
  int64_t compact_shape_size =
      options_.hlo_cost_analysis.GetShapeSize(compact_shape);
  // Account for buffers that are compressed after instruction.
  return buffer.size - compact_shape_size;
}

int64_t HloRematerializationMemoryTracker::MemoryReducedIfOffloaded(
    const HloRematItem* item) const {
  return BytesUsedByBuffers(item, /*only_count_unplaced_users=*/true);
}

std::optional<int64_t> HloRematerializationMemoryTracker::GetCostOfRecompute(
    absl::Span<const HloRematItem* const> candidate_items,
    int64_t memory_limit_bytes) const {
  // Evaluate this block as a candidate for recompute rematerialization.
  if (!IsCurrentlyRecomputable(candidate_items)) {
    return {};
  }
  const int64_t memory_reduced = MemoryReducedIfRecomputed(candidate_items);
  if (memory_reduced <= 0) {
    return {};
  }

  // If none of the users of any 'item' have been placed in the
  // sequence (as tracked by this memory tracker), then rematerialization of
  // 'item' is a zero-cost move of 'item->instruction' in the sequence.
  if (NoUserPlaced(candidate_items)) {
    return 0;
  }

  // Return the inverse of the benefit of rematerialization.
  return memory_limit_bytes / memory_reduced;
}

std::optional<int64_t> HloRematerializationMemoryTracker::GetCostOfCompression(
    const HloRematItem* candidate_item, int64_t memory_limit_bytes,
    int64_t peak_memory_bytes) {
  CHECK_NE(candidate_item, nullptr);
  if (!IsCurrentlyCompressible(candidate_item)) {
    return {};
  }

  const Shape* compact_shape =
      buffer_analyzer()->GetCompactShape(candidate_item->instruction).value();
  const int64_t memory_reduced =
      MemoryReducedIfCompressed(candidate_item, *compact_shape);
  // Since the compressed and uncompressed buffers need to be alive
  // while performing the compression/uncompression, only perform
  // the compression if the sum of the two sizes is less than the
  // peak memory.
  const int64_t size = options_.hlo_cost_analysis.GetShapeSize(
      candidate_item->instruction->shape());
  const int64_t reduced_size =
      options_.hlo_cost_analysis.GetShapeSize(*compact_shape);
  // TODO(victorstone): I don't think this size check is right.
  if (memory_reduced > 0 && size + reduced_size < peak_memory_bytes) {
    return memory_limit_bytes / memory_reduced;
  }
  return {};
}

std::optional<int64_t> HloRematerializationMemoryTracker::GetCostOfHostOffload(
    const HloRematItem* candidate_item, int64_t memory_limit_bytes) const {
  CHECK(candidate_item != nullptr);
  HloInstruction* candidate_instruction = candidate_item->instruction;

  VLOG(2)
      << "Considering host offload as an option for remat. looking at instr "
      << candidate_instruction->name();

  if (!IsCurrentlyOffloadable(candidate_item)) {
    return {};
  }
  const int64_t memory_reduced = MemoryReducedIfOffloaded(candidate_item);

  if (memory_reduced == 0) {
    VLOG(2) << "  " << candidate_instruction->name()
            << " consumes no memory; no point in offloading.";
    return {};
  }

  // How much compute is between this candidate's last placed user and its first
  // non-placed user?
  const Buffer& output_buffer =
      buffer_analyzer()->buffers_.at(candidate_item->buffers_output[0]);
  const auto [placed_uses, unplaced_uses] =
      GetPlacedAndUnplacedUsers(output_buffer.users);
  const HloRematItem* last_placed_user = nullptr;
  const HloRematItem* first_unplaced_user = nullptr;
  for (const auto* item = buffer_analyzer()->instruction_list_.first();
       item != nullptr;
       item = buffer_analyzer()->instruction_list_.next(item)) {
    if (absl::c_find_if(placed_uses, [&](const auto& use) {
          return use.user == item;
        }) != placed_uses.end()) {
      last_placed_user = item;
    }
    if (first_unplaced_user == nullptr &&
        absl::c_find_if(unplaced_uses, [&](const auto& use) {
          return use.user == item;
        }) != unplaced_uses.end()) {
      first_unplaced_user = item;
      break;
    }
  }

  if (last_placed_user == nullptr) {
    VLOG(3) << "  " << candidate_instruction->name()
            << " has no placed users, starting search at self.";
    last_placed_user = candidate_item;
  }
  CHECK(first_unplaced_user != nullptr)
      << "Didn't find any unplaced user for instruction \""
      << candidate_instruction->name()
      << "\". There must be a "
         "bug in how we calculate how much memory this item uses.";

  float time_spent_before_next_use = 0.0;
  for (auto* item = last_placed_user; item != first_unplaced_user;
       item = buffer_analyzer()->instruction_list_.next(item)) {
    time_spent_before_next_use += std::max(
        0.0f, options_.hlo_cost_analysis.optimal_seconds(*item->instruction));
  }

  if (time_spent_before_next_use <= 0.0) {
    // Instructions between take no time.
    return {};
  }

  const float time_spent_on_copies =
      memory_reduced / options_.host_memory_offload_config
                           ->bandwidth_to_host_bytes_per_second +
      memory_reduced / options_.host_memory_offload_config
                           ->bandwidth_from_host_bytes_per_second;
  if (time_spent_before_next_use < time_spent_on_copies) {
    // Host offload only considers cases where we can completely hide the copy
    // times. In this case, there is not enough compute time to hide offloading
    // and copying the data back in.
    return {};
  }
  VLOG(3) << "  " << candidate_instruction->name() << " has enough time ("
          << time_spent_before_next_use
          << ") between itself and next use. The memcpy out and back will take "
          << time_spent_on_copies << "s";
  // TODO(b/293323448): Properly calculate a cost; this cost metric is not
  // useful.
  return memory_limit_bytes / memory_reduced;
}

int64_t HloRematerializationMemoryTracker::AllocatedSize(
    HloRematItem* item) const {
  int64_t size = 0;
  for (auto buffer_id : item->buffers_defined) {
    size += buffer_analyzer()->AllocatedSize(buffer_id);
  }
  return size;
}

bool HloRematerializationMemoryTracker::IsInstructionCurrentlyLive(
    const HloRematItem* item) const {
  // If the instruction has not started yet, it is not alive.
  if (!IsItemPlaced(item)) {
    return false;
  }
  for (const HloInstruction* user : item->instruction->users()) {
    if (!IsItemPlaced(GetItem(user))) {
      // If there is an unplaced user, consider this instruction currently
      // live.
      return true;
    }
  }
  return false;
}

bool HloRematerializationMemoryTracker::IsBufferInUse(
    BufferId buffer_id) const {
  if (in_progress_item() == nullptr) {
    return false;
  }
  const BufferIdList& in_progress_uses = in_progress_item()->buffers_used;
  return absl::c_linear_search(in_progress_uses, buffer_id);
}

bool HloRematerializationMemoryTracker::NoUserPlaced(
    absl::Span<const HloRematItem* const> items) const {
  for (const HloRematItem* item : items) {
    const HloInstruction* instruction = item->instruction;
    if (absl::c_any_of(instruction->users(),
                       [this](const HloInstruction* instruction) {
                         return IsInstructionPlaced(instruction);
                       })) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<HloRematerializationSweepMemoryTracker>
HloRematerializationSweepMemoryTracker::CreateTracker(
    const HloRematerializationOptions& options,
    const HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloRematInstructionList& instruction_list) {
  HloRematerializationBufferAnalyzer buffer_analyzer =
      HloRematerializationBufferAnalyzer::CreateAnalyzer(
          options, computation, points_to_analysis, instruction_list);

  std::unique_ptr<HloRematerializationSweepMemoryTracker> sweep_memory_tracker(
      new HloRematerializationSweepMemoryTracker(
          options, computation, instruction_list, std::move(buffer_analyzer)));

  XLA_VLOG_LINES(10, sweep_memory_tracker->ToString());
  DCHECK(sweep_memory_tracker->Check());

  return sweep_memory_tracker;
}

absl::Status HloRematerializationSweepMemoryTracker::BeginInstruction(
    HloRematItem* item) {
  const HloInstruction* instruction = item->instruction;
  VLOG(3) << "BeginInstruction " << instruction->name();
  TF_RET_CHECK(in_progress_item_ == nullptr);
  in_progress_item_ = item;

  item->placed = true;

  CountAllocatedMemory(item);

  // TODO(b/37686934): Elementwise instructions can share the buffer of a (dead)
  // operand. Account for this potential reuse here.

  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();

  if (VLOG_IS_ON(1)) {
    DCHECK(Check());
  }
  return absl::OkStatus();
}

absl::Status HloRematerializationSweepMemoryTracker::EndInstruction() {
  TF_RET_CHECK(in_progress_item_ != nullptr);
  VLOG(3) << "EndInstruction " << in_progress_item_->instruction->name();

  TF_RETURN_IF_ERROR(CountFreedMemory(in_progress_item_));

  in_progress_item_ = nullptr;

  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();

  if (VLOG_IS_ON(1)) {
    DCHECK(Check());
  }
  return absl::OkStatus();
}

absl::Status HloRematerializationSweepMemoryTracker::
    AddCompressInstructionsToBufferCalculations(
        HloRematItem* original_item, HloRematItem* compressed_item,
        HloRematItem* uncompressed_item) {
  CHECK(IsItemPlaced(original_item))
      << "Compressing instruction, but the original is not yet placed.";
  CHECK_EQ(original_item->buffers_output.size(), 1)
      << "Only compressing items which have a single output buffer";

  // Update the memory usage by replacing the old instruction with the new one.
  // Original buffer is now dead.
  memory_usage_ -= options().hlo_cost_analysis.GetShapeSize(
      original_item->instruction->shape());
  // Compressed buffer is now alive.
  memory_usage_ += options().hlo_cost_analysis.GetShapeSize(
      compressed_item->instruction->shape());

  // Update the original item's only output buffer.
  BufferId original_buffer_id = original_item->buffers_output[0];
  Buffer& original_buffer = buffer_analyzer_->buffers_.at(original_buffer_id);
  auto [placed_users, unplaced_users] =
      GetPlacedAndUnplacedUsers(original_buffer.users);
  // Update the list of users to only be placed_users.
  original_buffer.users = std::move(placed_users);
  // Update to reflect that all users are finished, since any user after this
  // point will be using the uncompressed version.
  original_buffer.unfinished_user_count = 0;
  // Add the new compression instruction as a user of the original instruction.
  original_buffer.users.push_back(
      HloRematItemUse{compressed_item, 0, std::nullopt});

  // We are reallocating the vector containing the buffers potentially,
  // invalidating the original_buffer reference, so copy the index that we need
  // across NewBuffer calls.
  ShapeIndex copied_index = original_buffer.index;

  // Create a new buffer which is the one that the new compress instruction will
  // define.
  Buffer& compressed_buffer = buffer_analyzer_->NewBuffer(
      compressed_item, compressed_item->instruction->shape(), copied_index,
      {HloRematItemUse{uncompressed_item, 0, std::nullopt}},
      /*live_out=*/false, /*has_indirect_uses=*/false);
  // Update the compress item to only use the output buffer of the original
  // item.
  compressed_item->buffers_used = original_item->buffers_output;
  // Update the compress item to define & output this newly created buffer.
  compressed_item->buffers_output = {compressed_buffer.id};
  compressed_item->buffers_defined.push_back(compressed_buffer.id);

  // Create a new buffer which is the one that the new uncompress instruction
  // will define.
  Buffer& uncompressed_buffer = buffer_analyzer_->NewBuffer(
      uncompressed_item, uncompressed_item->instruction->shape(), copied_index,
      std::move(unplaced_users), /*live_out=*/false,
      /*has_indirect_uses=*/false);
  // Update the uncompressed item to only use the output buffer of the compress
  // item.
  uncompressed_item->buffers_used = {compressed_item->buffers_output[0]};
  // Update the uncompressed item to define & output this newly created buffer.
  uncompressed_item->buffers_output = {uncompressed_buffer.id};
  uncompressed_item->buffers_defined = {uncompressed_buffer.id};

  // uncompressed_buffer inherited its users as the unplaced users of the
  // original instruction. In each of these uses, replace the use of the
  // original buffer with the newly created final buffer.
  buffer_analyzer_->ReplaceUsesInUsersOfBuffer(uncompressed_buffer,
                                               original_buffer_id);

  return absl::OkStatus();
}

absl::Status HloRematerializationSweepMemoryTracker::
    AddRecomputeInstructionToBufferCalculations(
        HloRematItem* original_item, HloRematItem* recompute_item,
        absl::Span<HloRematItem*> indirect_users) {
  VLOG(3) << "AddRecomputeInstruction: original_instruction = "
          << original_item->instruction->name() << ", recompute_instruction = "
          << recompute_item->instruction->name();

  TF_RET_CHECK(in_progress_item_ != nullptr);
  TF_RET_CHECK(original_item->placed) << original_item->instruction->name();
  TF_RET_CHECK(!recompute_item->placed) << recompute_item->instruction->name();

  // Construct the list of buffers used and defined by the recompute.
  recompute_item->buffers_used = original_item->buffers_used;

  // Account for the additional buffer uses created by the new recompute
  // instruction. Update memory usage if the recomputation makes a dead
  // buffer live again.
  for (BufferId buffer_id : original_item->buffers_used) {
    Buffer& buffer = buffer_analyzer_->buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      // Buffer used by this instruction was dead, now is alive.
      memory_usage_ += buffer_analyzer_->AllocatedSize(buffer.id);
    }
    buffer.unfinished_user_count++;
    absl::InlinedVector<HloRematItemUse, 2> filtered_users;
    std::copy_if(
        buffer.users.begin(), buffer.users.end(),
        std::back_inserter(filtered_users),
        [&](const HloRematItemUse& iu) { return iu.user == original_item; });
    for (HloRematItemUse& u : filtered_users) {
      buffer.users.push_back(
          HloRematItemUse{recompute_item, u.operand_number, u.index});
    }
  }

  const absl::flat_hash_set<HloRematItem*> indirect_users_set(
      indirect_users.begin(), indirect_users.end());
  // Create a new set of Buffers defined by the new recompute
  // instruction. Update the internal data structures and memory use to account
  // for them.
  for (BufferId old_buffer_id : original_item->buffers_defined) {
    Buffer& old_buffer = buffer_analyzer_->buffers_.at(old_buffer_id);

    UsesList placed_users;
    UsesList unplaced_users;
    for (HloRematItemUse& user : old_buffer.users) {
      if (user.user->placed) {
        placed_users.push_back(user);
      } else {
        // We keep only the indirect users that are in the provided list.
        // We consider all the other dead and remove any buffer use they might
        // perform and remove it from the buffer user list.
        if (!IsSupportedIndirectUser(user.user->instruction) ||
            indirect_users_set.contains(user.user)) {
          unplaced_users.push_back(user);
        } else {
          CHECK(user.user->buffers_defined.empty())
              << "Buffers defined expected to be empty for use passthrough "
                 "instructions";
          user.user->buffers_output.clear();
          user.user->buffers_used.clear();
        }
      }
    }
    old_buffer.users = std::move(placed_users);
    old_buffer.unfinished_user_count = 0;

    // Buffer is now dead.
    memory_usage_ -= buffer_analyzer_->AllocatedSize(old_buffer.id);

    // Sweep Memory Tracker keeps placements accurate and can ask for a check.
    Buffer& new_buffer = buffer_analyzer_->RematerializeBuffer(
        old_buffer, recompute_item, std::move(unplaced_users),
        /*check_placement=*/true);

    recompute_item->buffers_defined.push_back(new_buffer.id);
    recompute_item->buffers_output.push_back(new_buffer.id);
    auto update_buffers = [old_buffer_id, new_buffer_id = new_buffer.id](
                              BufferIdList& to_update) {
      std::replace(to_update.begin(), to_update.end(), old_buffer_id,
                   new_buffer_id);
    };
    // Update users with the id of the new buffer.
    for (HloRematItemUse& user : new_buffer.users) {
      update_buffers(user.user->buffers_used);
      update_buffers(user.user->buffers_output);
    }
  }

  // Update the indirect users with the id of the new buffers.
  for (HloRematItem* indirect_user : indirect_users) {
    // Source of the buffers that are gonna be passthrough.
    const HloRematItem* source_item =
        instruction_list_.GetItem(indirect_user->instruction->operand(0));
    switch (indirect_user->instruction->opcode()) {
      case HloOpcode::kBitcast: {
        // If the source is another indirect user then copy the output
        // in the used and output lists of the bitcast as they don't define any
        // buffer.
        if (IsSupportedIndirectUser(source_item->instruction)) {
          indirect_user->buffers_used = source_item->buffers_output;
          indirect_user->buffers_output = source_item->buffers_output;
        } else {
          // If it's a real instruction producing a buffer then copy the defined
          // buffers into used and output.
          indirect_user->buffers_used = source_item->buffers_defined;
          indirect_user->buffers_output = source_item->buffers_defined;
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        // GTEs just use the tuple buffer and output the buffer they actually
        // extract from the tuple.
        const HloGetTupleElementInstruction* gte =
            Cast<HloGetTupleElementInstruction>(indirect_user->instruction);
        for (BufferId buffer_id : source_item->buffers_defined) {
          const Buffer& def_buffer = buffer_analyzer_->buffers_.at(buffer_id);
          if (def_buffer.index == ShapeIndex{gte->tuple_index()}) {
            indirect_user->buffers_output.push_back(buffer_id);
          }
          // This is the tuple buffer.
          if (def_buffer.index.empty()) {
            indirect_user->buffers_used.push_back(buffer_id);
          }
        }
        break;
      }
      default: {
        LOG(FATAL) << "Unsupported indirect instruction with opcode "
                   << indirect_user->instruction->opcode();
        break;
      }
    }
    // Fixup buffer users for the indirect instructions. For GTEs is only the
    // tuple buffer, while for bitcast is the buffer they pass through.
    for (BufferId buffer_id : indirect_user->buffers_used) {
      Buffer& buffer = buffer_analyzer_->buffers_.at(buffer_id);
      buffer.unfinished_user_count++;
      buffer.users.push_back(HloRematItemUse{indirect_user, 0, std::nullopt});
    }
  }

  VLOG(3) << "  memory usage = " << memory_usage_;
  XLA_VLOG_LINES(10, ToString());

  DCHECK(Check());

  return absl::OkStatus();
}

absl::Status HloRematerializationSweepMemoryTracker::
    AddOffloadInstructionsToBufferCalculations(
        HloRematItem* original_item, HloRematItem* copy_start_to_host_item,
        HloRematItem* copy_done_to_host_item,
        HloRematItem* copy_start_to_device_item,
        HloRematItem* copy_done_to_device_item) {
  CHECK_EQ(original_item->buffers_defined.size(), 1);

  // Split up the users of the original instruction into placed and unplaced.
  CHECK_EQ(original_item->buffers_output.size(), 1);
  BufferId original_buffer_id = original_item->buffers_output[0];
  Buffer& original_buffer = buffer_analyzer_->buffers_.at(original_buffer_id);
  auto [placed_users, unplaced_users] =
      GetPlacedAndUnplacedUsers(original_buffer.users);

  // Update the original item's buffer's users to be:
  //  1. The placed_users only.
  //  2. The newly created copy_start_to_host.
  original_buffer.users = std::move(placed_users);
  original_buffer.users.emplace_back(copy_start_to_host_item, 0, std::nullopt);
  // Set the only unfinished user as the newly created copy_to_host instruction.
  // We will later determine if that user is finished or not and update this
  // value if so.
  original_buffer.unfinished_user_count = 1;

  // Create new buffers for all of the newly created instructions.
  CHECK_EQ(copy_start_to_host_item->instruction->shape().tuple_shapes().size(),
           3)
      << "copy_start_to_host_item's shape is "
      << copy_start_to_host_item->instruction->shape().ToString();
  CHECK_EQ(
      copy_start_to_device_item->instruction->shape().tuple_shapes().size(), 3)
      << "copy_start_to_device_item's shape is "
      << copy_start_to_device_item->instruction->shape().ToString();

  // The first copy-start is a tuple of 3 elements: (host_buffer, device_buffer,
  // context). Since we're not tracking host memory, we'll only create buffers
  // for the other two.
  BufferId copy_start_to_host_device_buffer_id =
      buffer_analyzer_
          ->NewBuffer(
              copy_start_to_host_item,
              copy_start_to_host_item->instruction->shape().tuple_shapes(1),
              ShapeIndex(),
              UsesList{
                  HloRematItemUse{copy_done_to_host_item, 0, std::nullopt}},
              /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;
  BufferId copy_start_to_host_context_buffer_id =
      buffer_analyzer_
          ->NewBuffer(
              copy_start_to_host_item,
              copy_start_to_host_item->instruction->shape().tuple_shapes(2),
              ShapeIndex(),
              UsesList{
                  HloRematItemUse{copy_done_to_host_item, 0, std::nullopt}},
              /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;

  // The second copy-start is a tuple of 3 elements: (device_buffer,
  // host_buffer, context). Since we're not tracking host memory, we'll only
  // create buffers for the other two.
  BufferId copy_start_to_device_device_buffer_id =
      buffer_analyzer_
          ->NewBuffer(
              copy_start_to_device_item,
              copy_start_to_device_item->instruction->shape().tuple_shapes(0),
              ShapeIndex(),
              UsesList{
                  HloRematItemUse{copy_done_to_device_item, 0, std::nullopt}},
              /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;
  BufferId copy_start_to_device_context_buffer_id =
      buffer_analyzer_
          ->NewBuffer(
              copy_start_to_device_item,
              copy_start_to_device_item->instruction->shape().tuple_shapes(2),
              ShapeIndex(),
              UsesList{
                  HloRematItemUse{copy_done_to_device_item, 0, std::nullopt}},
              /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;

  // The final copy-done outputs the final device buffer that is the
  // rematerialized original buffer.
  BufferId copy_done_to_device_buffer_id =
      buffer_analyzer_
          ->NewBuffer(copy_done_to_device_item,
                      copy_done_to_device_item->instruction->shape(),
                      ShapeIndex(), std::move(unplaced_users),
                      /*live_out=*/false,
                      /*has_indirect_uses=*/false)
          .id;

  // Update items of the newly created instructions to reference the newly
  // created buffers.
  copy_start_to_host_item->buffers_used = original_item->buffers_output;
  copy_start_to_host_item->buffers_output = {
      copy_start_to_host_device_buffer_id,
      copy_start_to_host_context_buffer_id};
  copy_start_to_host_item->buffers_defined = {
      copy_start_to_host_device_buffer_id,
      copy_start_to_host_context_buffer_id};

  copy_done_to_host_item->buffers_used =
      copy_start_to_host_item->buffers_output;
  // The only buffer that copy_done_to_host defines is a host buffer. Since
  // we're not tracking host memory, we're not going to bother with that buffer
  // for now.
  copy_done_to_host_item->buffers_output = {};
  copy_done_to_host_item->buffers_defined = {};

  copy_start_to_device_item->buffers_used =
      copy_done_to_host_item->buffers_output;
  copy_start_to_device_item->buffers_output = {
      copy_start_to_device_device_buffer_id,
      copy_start_to_device_context_buffer_id};
  copy_start_to_device_item->buffers_defined = {
      copy_start_to_device_device_buffer_id,
      copy_start_to_device_context_buffer_id};

  copy_done_to_device_item->buffers_used =
      copy_start_to_device_item->buffers_output;
  copy_done_to_device_item->buffers_output = {copy_done_to_device_buffer_id};
  copy_done_to_device_item->buffers_defined = {copy_done_to_device_buffer_id};

  Buffer& copy_done_to_device_buffer =
      buffer_analyzer_->buffers_.at(copy_done_to_device_buffer_id);
  buffer_analyzer_->ReplaceUsesInUsersOfBuffer(copy_done_to_device_buffer,
                                               original_buffer_id);

  // We know that the 4 newly created instructions are not in progress, so if
  // they're marked as placed, we can count the allocation and deallocation of
  // buffers. Calling these functions also does some user accounting. Since
  // these instructions have a strict order, if one isn't placed, the following
  // ones won't be either.
  if (copy_start_to_host_item->placed) {
    CountAllocatedMemory(copy_start_to_host_item);
    TF_RETURN_IF_ERROR(CountFreedMemory(copy_start_to_host_item));
    // This will account for the freed memory that is defined by the original
    // item.

    if (copy_done_to_host_item->placed) {
      CountAllocatedMemory(copy_done_to_host_item);
      TF_RETURN_IF_ERROR(CountFreedMemory(copy_done_to_host_item));

      if (copy_start_to_device_item->placed) {
        CountAllocatedMemory(copy_start_to_device_item);
        TF_RETURN_IF_ERROR(CountFreedMemory(copy_start_to_device_item));

        if (copy_done_to_device_item->placed) {
          CountAllocatedMemory(copy_done_to_device_item);
          TF_RETURN_IF_ERROR(CountFreedMemory(copy_done_to_device_item));
        }
      }
    }
  }

  return absl::OkStatus();
}

bool HloRematerializationSweepMemoryTracker::Check() const {
  auto elements_are_unique = [](const BufferIdList& vec) {
    return vec.size() == std::set<BufferId>(vec.begin(), vec.end()).size();
  };

  // Verify buffers_defined per instruction.
  for (auto* instruction : computation()->instructions()) {
    const BufferIdList& defined_buffers =
        instruction_list_.GetItem(instruction)->buffers_defined;
    CHECK(elements_are_unique(defined_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique defined buffers: "
        << absl::StrJoin(
               defined_buffers, ", ",
               [this](std::string* out, BufferId buffer_id) {
                 absl::StrAppend(
                     out, buffer_analyzer_->buffers_.at(buffer_id).ToString());
               });

    for (const Buffer& buffer : buffer_analyzer_->buffers_) {
      if (buffer.defining_instruction->instruction == instruction) {
        CHECK(absl::c_linear_search(defined_buffers, buffer.id))
            << "Instruction " << instruction->name()
            << " defined buffers is missing: " << buffer.ToString();
      }
    }
  }

  // Verify buffers_used per instruction.
  for (auto* instruction : computation()->instructions()) {
    const BufferIdList& used_buffers =
        instruction_list_.GetItem(instruction)->buffers_used;
    CHECK(elements_are_unique(used_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique used buffers: "
        << absl::StrJoin(
               used_buffers, ", ",
               [this](std::string* out, BufferId buffer_id) {
                 absl::StrAppend(
                     out, buffer_analyzer_->buffers_.at(buffer_id).ToString());
               });
  }
  for (const Buffer& buffer : buffer_analyzer_->buffers_) {
    int64_t unfinished_uses = 0;
    absl::flat_hash_set<HloRematItem*> already_counted_user;
    for (const HloRematItemUse& user : buffer.users) {
      const BufferIdList& used_buffers = user.user->buffers_used;
      CHECK(absl::c_linear_search(used_buffers, buffer.id))
          << "Instruction " << user.user->instruction->name()
          << " used buffers is missing " << buffer.ToString();
      if (!IsItemFinished(user.user) &&
          already_counted_user.insert(user.user).second) {
        unfinished_uses++;
      }
    }
    CHECK_EQ(buffer.unfinished_user_count, unfinished_uses)
        << "Incorrect unplaced use count for " << buffer.ToString();
  }
  return true;
}

std::string HloRematerializationSweepMemoryTracker::ToString() const {
  std::string output =
      absl::StrCat("HloRematerializationSweepMemoryTracker for ",
                   computation()->name(), "\n");
  absl::StrAppend(&output,
                  "Memory usage: ", HumanReadableNumBytes(memory_usage_), " (",
                  memory_usage_, " bytes)");
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    const HloInstruction* instruction = item->instruction;
    absl::string_view inprogress =
        item == in_progress_item_ ? " in-progress" : "";
    absl::string_view placed = item->placed ? " placed" : "";
    absl::StrAppend(&output, "  ", instruction->name(), inprogress, placed,
                    "\n    Defines:\n");
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffer_analyzer_->buffers_[buffer_id];
      absl::string_view live = IsBufferLive(buffer_id) ? " live" : "";
      absl::StrAppend(&output, "      ", buffer.ToString(), live, ", ",
                      buffer.unfinished_user_count, " unfinished uses\n");
    }
    absl::StrAppend(&output, "    Outputs:\n");
    for (BufferId buffer_id : item->buffers_output) {
      absl::StrAppend(&output, "      ",
                      buffer_analyzer_->buffers_[buffer_id].ToString(), "\n");
    }
    absl::StrAppend(&output, "    Uses:\n");
    for (BufferId buffer_id : item->buffers_used) {
      absl::StrAppend(&output, "      ",
                      buffer_analyzer_->buffers_[buffer_id].ToString(), "\n");
    }
  }
  return output;
}

bool HloRematerializationSweepMemoryTracker::IsItemPlaced(
    const HloRematItem* item) const {
  return item->placed;
}

bool HloRematerializationSweepMemoryTracker::IsItemFinished(
    const HloRematItem* item) const {
  return IsItemPlaced(item) && item != in_progress_item_;
}

bool HloRematerializationSweepMemoryTracker::IsBufferLive(
    BufferId buffer_id) const {
  const Buffer& buffer = buffer_analyzer_->buffers_[buffer_id];
  return (IsItemPlaced(buffer.defining_instruction) &&
          buffer.unfinished_user_count > 0);
}

void HloRematerializationSweepMemoryTracker::CountAllocatedMemory(
    HloRematItem* item) {
  // All buffers defined by this instruction need memory.
  for (BufferId buffer_id : item->buffers_defined) {
    VLOG(3) << "  Buffer "
            << buffer_analyzer_->buffers_.at(buffer_id).ToString()
            << " is now live.";
    memory_usage_ += buffer_analyzer_->AllocatedSize(buffer_id);
  }
}

absl::Status HloRematerializationSweepMemoryTracker::CountFreedMemory(
    HloRematItem* item) {
  for (BufferId buffer_id : item->buffers_used) {
    Buffer& buffer = buffer_analyzer_->buffers_.at(buffer_id);
    buffer.unfinished_user_count--;
    TF_RET_CHECK(buffer.unfinished_user_count >= 0)
        << buffer.ToString() << " has negative unfinished user count.";
    if (buffer.unfinished_user_count == 0) {
      // Buffer is now dead.
      VLOG(3) << "  " << buffer.ToString() << " is now dead.";
      memory_usage_ -= buffer_analyzer_->AllocatedSize(buffer_id);
      // The memory usage can become negative inside the computation as we can
      // free up the parameter space and reuse it for other tensors.
    }
  }

  // If any buffer defined by this instruction has no uses, then memory can be
  // reclaimed immediately.
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffer_analyzer_->buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      VLOG(3) << "  " << buffer.ToString() << " is immediately dead.";
      memory_usage_ -= buffer_analyzer_->AllocatedSize(buffer_id);
      // The memory usage can become negative inside the computation as we can
      // free up the parameter space and reuse it for other tensors.
    }
  }
  return absl::OkStatus();
}

void HloRematerializationBufferAnalyzer::ReplaceUsesInUsersOfBuffer(
    Buffer& buffer, BufferId old_id) const {
  // Loop over the users of this buffer. For each of those users look at their
  // buffers used. If that buffer Id matches the passed in old_id, then replace
  // it with the Id of this current buffer.
  for (HloRematItemUse& use : buffer.users) {
    BufferIdList& buffers_used = use.user->buffers_used;
    absl::c_replace(buffers_used, old_id, buffer.id);
  }
}

absl::StatusOr<const Shape*>
HloRematerializationBufferAnalyzer::GetCompactShape(const HloInstruction* hlo) {
  auto it = compact_shape_.find(hlo);
  if (it != compact_shape_.end()) {
    return &it->second;
  }
  const Shape& original_shape = hlo->shape();
  TF_ASSIGN_OR_RETURN(Shape min_shape,
                      options_.compact_shape_function(original_shape));
  return &compact_shape_.emplace(hlo, min_shape).first->second;
}

absl::StatusOr<std::unique_ptr<HloRematerializationPeakMemoryTracker>>
HloRematerializationPeakMemoryTracker::CreateTracker(
    const HloRematerializationOptions& options,
    const HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloRematInstructionList& instruction_list,
    const CallGraph* call_graph,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    const absl::flat_hash_map<
        const HloComputation*,
        std::unique_ptr<HloRematerializationPeakMemoryTracker>>&
        computation_peak_memory_tracker) {
  HloRematerializationBufferAnalyzer buffer_analyzer =
      HloRematerializationBufferAnalyzer::CreateAnalyzer(
          options, computation, points_to_analysis, instruction_list);

  std::unique_ptr<HloRematerializationPeakMemoryTracker> peak_memory_tracker(
      new HloRematerializationPeakMemoryTracker(
          options, computation, instruction_list, call_graph, execution_threads,
          computation_peak_memory_tracker, std::move(buffer_analyzer)));

  TF_RETURN_IF_ERROR(peak_memory_tracker->InitializeSegmentTree());
  TF_RETURN_IF_ERROR(
      peak_memory_tracker->InitializeSleatorDietzOrderMaintenance());
  TF_RETURN_IF_ERROR(peak_memory_tracker->InitializeFinalUser());

  XLA_VLOG_LINES(10, peak_memory_tracker->ToString());
  DCHECK(peak_memory_tracker->Check());

  return peak_memory_tracker;
}

absl::Status HloRematerializationPeakMemoryTracker::JumpToInstruction(
    const HloInstruction* instruction) {
  HloRematItem* item = instruction_list_.GetItem(instruction);
  VLOG(3) << "JumpToInstruction " << instruction->name();

  // Try to lookup `item` in our segment tree.
  TF_ASSIGN_OR_RETURN(
      MemoryUsageAndInstruction current_memory_usage_and_instruction,
      segtree_->Query(instruction, instruction));
  in_progress_item_ = item;

  VLOG(3) << "  memory usage = "
          << current_memory_usage_and_instruction.memory_usage;
  VLOG(10) << ToString();

  return absl::OkStatus();
}

absl::StatusOr<MemoryUsageAndInstruction>
HloRematerializationPeakMemoryTracker::ComputePeakMemoryUsageAndInstruction()
    const {
  return segtree_->Query();
}

absl::Status HloRematerializationPeakMemoryTracker::
    AddRecomputeInstructionToBufferCalculations(
        HloRematItem* original_item, HloRematItem* recompute_item,
        absl::Span<HloRematItem*> indirect_users) {
  VLOG(3) << "AddRecomputeInstructionToBufferCalculations: "
          << "original_instruction = " << original_item->instruction->name()
          << ", recompute_instruction = "
          << recompute_item->instruction->name();

  TF_RET_CHECK(in_progress_item_ != nullptr);
  TF_RET_CHECK(IsItemPlaced(original_item))
      << original_item->instruction->name();
  TF_RET_CHECK(
      !instruction_ordering_.ContainsInstruction(recompute_item->instruction))
      << recompute_item->instruction->name();

  // Construct the list of buffers used and defined by the recompute.
  recompute_item->buffers_used = original_item->buffers_used;

  // Account for the additional buffer uses created by the new recompute
  // instruction. Update memory usage if the recomputation makes a dead
  // buffer live again.
  for (BufferId buffer_id : original_item->buffers_used) {
    Buffer& buffer = buffer_analyzer_->buffers_.at(buffer_id);
    absl::InlinedVector<HloRematItemUse, 2> filtered_users;
    std::copy_if(
        buffer.users.begin(), buffer.users.end(),
        std::back_inserter(filtered_users),
        [&](const HloRematItemUse& iu) { return iu.user == original_item; });
    for (HloRematItemUse& u : filtered_users) {
      buffer.users.push_back(
          HloRematItemUse{recompute_item, u.operand_number, u.index});
    }
  }

  const absl::flat_hash_set<HloRematItem*> indirect_users_set(
      indirect_users.begin(), indirect_users.end());
  // Create a new set of Buffers defined by the new recompute
  // instruction. Update the internal data structures and memory use to account
  // for them.
  for (BufferId old_buffer_id : original_item->buffers_defined) {
    Buffer& old_buffer = buffer_analyzer_->buffers_.at(old_buffer_id);

    UsesList placed_users;
    UsesList unplaced_users;
    for (HloRematItemUse& user : old_buffer.users) {
      if (IsItemPlaced(user.user)) {
        placed_users.push_back(user);
      } else {
        // We keep only the indirect users that are in the provided list.
        // We consider all the other dead and remove any buffer use they might
        // perform and remove it from the buffer user list.
        if (!IsSupportedIndirectUser(user.user->instruction) ||
            indirect_users_set.contains(user.user)) {
          unplaced_users.push_back(user);
        } else {
          CHECK(user.user->buffers_defined.empty())
              << "Buffers defined expected to be empty for use passthrough "
                 "instructions";
          user.user->buffers_output.clear();
          user.user->buffers_used.clear();
        }
      }
    }
    old_buffer.users = std::move(placed_users);

    // Peak Memory Tracker does not keep placements accurate and cannot ask for
    // a check.
    Buffer& new_buffer = buffer_analyzer_->RematerializeBuffer(
        old_buffer, recompute_item, std::move(unplaced_users),
        /*check_placement=*/false);

    recompute_item->buffers_defined.push_back(new_buffer.id);
    recompute_item->buffers_output.push_back(new_buffer.id);
    auto update_buffers = [old_buffer_id, new_buffer_id = new_buffer.id](
                              BufferIdList& to_update) {
      std::replace(to_update.begin(), to_update.end(), old_buffer_id,
                   new_buffer_id);
    };
    // Update users with the id of the new buffer.
    for (HloRematItemUse& user : new_buffer.users) {
      update_buffers(user.user->buffers_used);
      update_buffers(user.user->buffers_output);
    }
  }

  // Update the indirect users with the id of the new buffers.
  for (HloRematItem* indirect_user : indirect_users) {
    // Source of the buffers that are gonna be passthrough.
    const HloRematItem* source_item =
        instruction_list_.GetItem(indirect_user->instruction->operand(0));
    switch (indirect_user->instruction->opcode()) {
      case HloOpcode::kBitcast: {
        // If the source is another indirect user then copy the output
        // in the used and output lists of the bitcast as they don't define any
        // buffer.
        if (IsSupportedIndirectUser(source_item->instruction)) {
          indirect_user->buffers_used = source_item->buffers_output;
          indirect_user->buffers_output = source_item->buffers_output;
        } else {
          // If it's a real instruction producing a buffer then copy the defined
          // buffers into used and output.
          indirect_user->buffers_used = source_item->buffers_defined;
          indirect_user->buffers_output = source_item->buffers_defined;
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        // GTEs just use the tuple buffer and output the buffer they actually
        // extract from the tuple.
        const HloGetTupleElementInstruction* gte =
            Cast<HloGetTupleElementInstruction>(indirect_user->instruction);
        for (BufferId buffer_id : source_item->buffers_defined) {
          const Buffer& def_buffer = buffer_analyzer_->buffers_.at(buffer_id);
          if (def_buffer.index == ShapeIndex{gte->tuple_index()}) {
            indirect_user->buffers_output.push_back(buffer_id);
          }
          // This is the tuple buffer.
          if (def_buffer.index.empty()) {
            indirect_user->buffers_used.push_back(buffer_id);
          }
        }
        break;
      }
      default: {
        LOG(FATAL) << "Unsupported indirect instruction with opcode "
                   << indirect_user->instruction->opcode();
        break;
      }
    }
    // Fixup buffer users for the indirect instructions. For GTEs is only the
    // tuple buffer, while for bitcast is the buffer they pass through.
    for (BufferId buffer_id : indirect_user->buffers_used) {
      Buffer& buffer = buffer_analyzer_->buffers_.at(buffer_id);
      buffer.users.push_back(HloRematItemUse{indirect_user, 0, std::nullopt});
    }
  }

  return absl::OkStatus();
}

absl::Status HloRematerializationPeakMemoryTracker::
    AddRecomputeInstructionToInstructionOrdering(
        HloRematItem* original_item, NewItemAndSuccessor recompute,
        absl::Span<HloRematItem*> indirect_users) {
  VLOG(3) << "AddRecomputeInstructionToInstructionOrdering: "
          << "original_instruction = " << original_item->instruction->name()
          << ", recompute_instruction = "
          << recompute.new_item->instruction->name()
          << ", recompute_successor_instruction = "
          << recompute.successor->instruction->name();

  TF_ASSIGN_OR_RETURN(
      bool original_before_recompute_successor,
      instruction_ordering_.CompareOrder(original_item->instruction,
                                         recompute.successor->instruction));
  TF_RET_CHECK(original_before_recompute_successor)
      << "Original item must be strictly before the recompute successor.";

  // The recompute instruction is placed immediately before its successor.
  TF_RETURN_IF_ERROR(InsertInstructionBeforeSuccessor(recompute));

  // The indirect users are then, in order, also immediately placed before that
  // same successor.
  for (HloRematItem* indirect_user : indirect_users) {
    NewItemAndSuccessor indirect_user_and_successor;
    indirect_user_and_successor.new_item = indirect_user;
    indirect_user_and_successor.successor = recompute.successor;
    TF_RETURN_IF_ERROR(
        InsertInstructionBeforeSuccessor(indirect_user_and_successor));
  }

  for (BufferId buffer_id : recompute.new_item->buffers_used) {
    // We take advantage of the fact that the final user of this buffer has not
    // been updated yet. If its old final user is before the recompute, then the
    // recompute is the new final user. We need to update both final_user_ and
    // segtree_ appropriately.
    TF_ASSIGN_OR_RETURN(
        bool final_user_is_before_recompute,
        instruction_ordering_.CompareOrder(final_user_[buffer_id]->instruction,
                                           recompute.new_item->instruction));
    if (!final_user_is_before_recompute) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(const HloInstruction* final_user_successor,
                        instruction_ordering_.GetNextInstruction(
                            final_user_[buffer_id]->instruction));
    TF_RETURN_IF_ERROR(
        segtree_->Update(final_user_successor, recompute.new_item->instruction,
                         buffer_analyzer_->AllocatedSize(buffer_id)));
    final_user_[buffer_id] = recompute.new_item;
  }

  for (BufferId old_buffer_id : original_item->buffers_defined) {
    Buffer& old_buffer = buffer_analyzer_->buffers_.at(old_buffer_id);
    // We want to know the new final user of this buffer for the updates we
    // are about to do.
    TF_RETURN_IF_ERROR(UpdateFinalUser(old_buffer_id, old_buffer.users));

    // Buffer is now dead for the interval from the new final user to the
    // recompute, exclusive.
    TF_ASSIGN_OR_RETURN(const HloInstruction* final_user_successor,
                        instruction_ordering_.GetNextInstruction(
                            final_user_[old_buffer_id]->instruction));
    // In the edge case where `final_user` is right before `recompute`, the
    // interval is empty and trying to update it would result in an error.
    if (final_user_successor != recompute.new_item->instruction) {
      TF_ASSIGN_OR_RETURN(const HloInstruction* recompute_predecessor,
                          instruction_ordering_.GetPreviousInstruction(
                              recompute.new_item->instruction));
      TF_RETURN_IF_ERROR(
          segtree_->Update(final_user_successor, recompute_predecessor,
                           -buffer_analyzer_->AllocatedSize(old_buffer_id)));
    }
  }

  for (BufferId new_buffer_id : recompute.new_item->buffers_defined) {
    // We are setting this for the first time. Technically, we know that this
    // will be set to the final user of the corresponding old_buffer_id that we
    // just overwrote in the previous loop, but we recompute to not worry about
    // the correspondence between old and new buffers.
    Buffer& new_buffer = buffer_analyzer_->buffers_.at(new_buffer_id);
    TF_RETURN_IF_ERROR(UpdateFinalUser(new_buffer_id, new_buffer.users));
  }

  if (VLOG_IS_ON(3)) {
    TF_ASSIGN_OR_RETURN(int64_t memory_usage, GetMemoryUsage());
    VLOG(3) << "  memory usage = " << memory_usage;
  }
  XLA_VLOG_LINES(10, ToString());

  return absl::OkStatus();
}

absl::Status HloRematerializationPeakMemoryTracker::
    AddCompressInstructionsToBufferCalculations(
        HloRematItem* original_item, HloRematItem* compressed_item,
        HloRematItem* uncompressed_item) {
  CHECK(IsItemPlaced(original_item))
      << "Compressing instruction, but the original is not yet placed.";
  CHECK_EQ(original_item->buffers_output.size(), 1)
      << "Only compressing items which have a single output buffer";

  // Update the original item's only output buffer.
  BufferId original_buffer_id = original_item->buffers_output[0];
  Buffer& original_buffer = buffer_analyzer_->buffers_.at(original_buffer_id);
  auto [placed_users, unplaced_users] =
      GetPlacedAndUnplacedUsers(original_buffer.users);
  // Update the list of users to only be placed_users.
  original_buffer.users = std::move(placed_users);
  // Add the new compression instruction as a user of the original instruction.
  original_buffer.users.push_back(
      HloRematItemUse{compressed_item, 0, std::nullopt});

  // We are reallocating the vector containing the buffers potentially,
  // invalidating the original_buffer reference, so copy the index that we need
  // across NewBuffer calls.
  ShapeIndex copied_index = original_buffer.index;

  // Create a new buffer which is the one that the new compress instruction will
  // define.
  Buffer& compressed_buffer = buffer_analyzer_->NewBuffer(
      compressed_item, compressed_item->instruction->shape(), copied_index,
      {HloRematItemUse{uncompressed_item, 0, std::nullopt}},
      /*live_out=*/false, /*has_indirect_uses=*/false);
  // Update the compress item to only use the output buffer of the original
  // item.
  compressed_item->buffers_used = original_item->buffers_output;
  // Update the compress item to define & output this newly created buffer.
  compressed_item->buffers_output = {compressed_buffer.id};
  compressed_item->buffers_defined.push_back(compressed_buffer.id);

  // Create a new buffer which is the one that the new uncompress instruction
  // will define.
  Buffer& uncompressed_buffer = buffer_analyzer_->NewBuffer(
      uncompressed_item, uncompressed_item->instruction->shape(), copied_index,
      std::move(unplaced_users), /*live_out=*/false,
      /*has_indirect_uses=*/false);
  // Update the uncompressed item to only use the output buffer of the compress
  // item.
  uncompressed_item->buffers_used = {compressed_item->buffers_output[0]};
  // Update the uncompressed item to define & output this newly created buffer.
  uncompressed_item->buffers_output = {uncompressed_buffer.id};
  uncompressed_item->buffers_defined = {uncompressed_buffer.id};

  // uncompressed_buffer inherited its users as a subset of the users of the
  // original instruction. In each of these uses, replace the use of the
  // original buffer with the newly created final buffer.
  buffer_analyzer_->ReplaceUsesInUsersOfBuffer(uncompressed_buffer,
                                               original_buffer_id);

  return absl::OkStatus();
}

absl::Status HloRematerializationPeakMemoryTracker::
    AddCompressInstructionsToInstructionOrdering(
        HloRematItem* original_item, NewItemAndPredecessor compressed,
        NewItemAndSuccessor uncompressed) {
  TF_ASSIGN_OR_RETURN(
      bool original_after_compressed_precedessor,
      instruction_ordering_.CompareOrder(compressed.predecessor->instruction,
                                         original_item->instruction));
  TF_RET_CHECK(!original_after_compressed_precedessor)
      << "Original item must be weakly before the compressed successor.";
  TF_ASSIGN_OR_RETURN(
      bool compressed_predecessor_before_uncompressed_successor,
      instruction_ordering_.CompareOrder(compressed.predecessor->instruction,
                                         uncompressed.successor->instruction));
  TF_RET_CHECK(compressed_predecessor_before_uncompressed_successor)
      << "Compressed predecessor must be strictly before the uncompressed "
         "successor.";

  TF_RETURN_IF_ERROR(InsertInstructionAfterPredecessor(compressed));
  TF_RETURN_IF_ERROR(InsertInstructionBeforeSuccessor(uncompressed));

  // We perform three range updates on the segment tree.
  // The uncompressed buffer previously lasted from the start of `original_item`
  // to the end of its final user, an interval that includes
  // [`compressed`, `uncompressed`] *inclusive*. Since we compress, it no longer
  // needs to be live over (`compressed`, `uncompressed`) *exclusive*. However,
  // the compressed buffer needs to be live over [`compressed`, `uncompressed`]
  // *inclusive*. We do this by adding `uncompressed_size` to the endpoints and
  // then adjusting the entire range by +`compressed_size` and
  // -`uncompressed_size`.
  const int64_t uncompressed_size = options().hlo_cost_analysis.GetShapeSize(
      original_item->instruction->shape());
  TF_RETURN_IF_ERROR(segtree_->Update(compressed.new_item->instruction,
                                      compressed.new_item->instruction,
                                      uncompressed_size));
  TF_RETURN_IF_ERROR(segtree_->Update(uncompressed.new_item->instruction,
                                      uncompressed.new_item->instruction,
                                      uncompressed_size));
  const int64_t compressed_size = options().hlo_cost_analysis.GetShapeSize(
      compressed.new_item->instruction->shape());
  const int64_t compression_delta = compressed_size - uncompressed_size;
  TF_RETURN_IF_ERROR(segtree_->Update(compressed.new_item->instruction,
                                      uncompressed.new_item->instruction,
                                      compression_delta));

  // Update the final user map to reflect the new dependency chain
  // original -> compress -> uncompress -> (old final user of original).
  BufferId original_buffer_id = original_item->buffers_output[0];
  BufferId compressed_buffer_id = compressed.new_item->buffers_output[0];
  BufferId uncompressed_buffer_id = uncompressed.new_item->buffers_output[0];

  final_user_[uncompressed_buffer_id] = final_user_[original_buffer_id];
  final_user_[compressed_buffer_id] = uncompressed.new_item;
  final_user_[original_buffer_id] = compressed.new_item;

  return absl::OkStatus();
}

absl::Status HloRematerializationPeakMemoryTracker::
    AddOffloadInstructionsToBufferCalculations(
        HloRematItem* original_item, HloRematItem* copy_start_to_host_item,
        HloRematItem* copy_done_to_host_item,
        HloRematItem* copy_start_to_device_item,
        HloRematItem* copy_done_to_device_item) {
  CHECK_EQ(original_item->buffers_defined.size(), 1);

  // Split up the users of the original instruction into placed and unplaced.
  CHECK_EQ(original_item->buffers_output.size(), 1);
  BufferId original_buffer_id = original_item->buffers_output[0];
  Buffer& original_buffer = buffer_analyzer_->buffers_.at(original_buffer_id);
  auto [placed_users, unplaced_users] =
      GetPlacedAndUnplacedUsers(original_buffer.users);

  // Update the original item's buffer's users to be:
  //  1. The `original_users` only.
  //  2. The newly created copy_start_to_host.
  original_buffer.users = std::move(placed_users);
  original_buffer.users.emplace_back(copy_start_to_host_item, 0, std::nullopt);

  // Create new buffers for all of the newly created instructions.
  CHECK_EQ(copy_start_to_host_item->instruction->shape().tuple_shapes().size(),
           3)
      << "copy_start_to_host_item's shape is "
      << copy_start_to_host_item->instruction->shape().ToString();
  CHECK_EQ(
      copy_start_to_device_item->instruction->shape().tuple_shapes().size(), 3)
      << "copy_start_to_device_item's shape is "
      << copy_start_to_device_item->instruction->shape().ToString();

  // The first copy-start is a tuple of 3 elements: (host_buffer, device_buffer,
  // context). Since we're not tracking host memory, we'll only create buffers
  // for the other two.
  BufferId copy_start_to_host_device_buffer_id =
      buffer_analyzer_
          ->NewBuffer(
              copy_start_to_host_item,
              copy_start_to_host_item->instruction->shape().tuple_shapes(1),
              ShapeIndex(),
              UsesList{
                  HloRematItemUse{copy_done_to_host_item, 0, std::nullopt}},
              /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;
  BufferId copy_start_to_host_context_buffer_id =
      buffer_analyzer_
          ->NewBuffer(
              copy_start_to_host_item,
              copy_start_to_host_item->instruction->shape().tuple_shapes(2),
              ShapeIndex(),
              UsesList{
                  HloRematItemUse{copy_done_to_host_item, 0, std::nullopt}},
              /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;

  // The second copy-start is a tuple of 3 elements: (device_buffer,
  // host_buffer, context). Since we're not tracking host memory, we'll only
  // create buffers for the other two.
  BufferId copy_start_to_device_device_buffer_id =
      buffer_analyzer_
          ->NewBuffer(
              copy_start_to_device_item,
              copy_start_to_device_item->instruction->shape().tuple_shapes(0),
              ShapeIndex(),
              UsesList{
                  HloRematItemUse{copy_done_to_device_item, 0, std::nullopt}},
              /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;
  BufferId copy_start_to_device_context_buffer_id =
      buffer_analyzer_
          ->NewBuffer(
              copy_start_to_device_item,
              copy_start_to_device_item->instruction->shape().tuple_shapes(2),
              ShapeIndex(),
              UsesList{
                  HloRematItemUse{copy_done_to_device_item, 0, std::nullopt}},
              /*live_out=*/false, /*has_indirect_uses=*/false)
          .id;

  // The final copy-done outputs the final device buffer that is the
  // rematerialized original buffer.
  BufferId copy_done_to_device_buffer_id =
      buffer_analyzer_
          ->NewBuffer(copy_done_to_device_item,
                      copy_done_to_device_item->instruction->shape(),
                      ShapeIndex(), std::move(unplaced_users),
                      /*live_out=*/false,
                      /*has_indirect_uses=*/false)
          .id;

  // Update items of the newly created instructions to reference the newly
  // created buffers.
  copy_start_to_host_item->buffers_used = original_item->buffers_output;
  copy_start_to_host_item->buffers_output = {
      copy_start_to_host_device_buffer_id,
      copy_start_to_host_context_buffer_id};
  copy_start_to_host_item->buffers_defined = {
      copy_start_to_host_device_buffer_id,
      copy_start_to_host_context_buffer_id};

  copy_done_to_host_item->buffers_used =
      copy_start_to_host_item->buffers_output;
  // The only buffer that copy_done_to_host defines is a host buffer. Since
  // we're not tracking host memory, we're not going to bother with that buffer
  // for now.
  copy_done_to_host_item->buffers_output = {};
  copy_done_to_host_item->buffers_defined = {};

  copy_start_to_device_item->buffers_used =
      copy_done_to_host_item->buffers_output;
  copy_start_to_device_item->buffers_output = {
      copy_start_to_device_device_buffer_id,
      copy_start_to_device_context_buffer_id};
  copy_start_to_device_item->buffers_defined = {
      copy_start_to_device_device_buffer_id,
      copy_start_to_device_context_buffer_id};

  copy_done_to_device_item->buffers_used =
      copy_start_to_device_item->buffers_output;
  copy_done_to_device_item->buffers_output = {copy_done_to_device_buffer_id};
  copy_done_to_device_item->buffers_defined = {copy_done_to_device_buffer_id};

  Buffer& copy_done_to_device_buffer =
      buffer_analyzer_->buffers_.at(copy_done_to_device_buffer_id);
  buffer_analyzer_->ReplaceUsesInUsersOfBuffer(copy_done_to_device_buffer,
                                               original_buffer_id);

  return absl::OkStatus();
}

absl::Status HloRematerializationPeakMemoryTracker::
    AddOffloadInstructionsToInstructionOrdering(
        HloRematItem* original_item, NewItemAndPredecessor copy_start_to_host,
        NewItemAndPredecessor copy_done_to_host,
        NewItemAndSuccessor copy_start_to_device,
        NewItemAndSuccessor copy_done_to_device) {
  TF_ASSIGN_OR_RETURN(bool original_after_copy_start_to_host_predecessor,
                      instruction_ordering_.CompareOrder(
                          copy_start_to_host.predecessor->instruction,
                          original_item->instruction));
  TF_RET_CHECK(!original_after_copy_start_to_host_predecessor)
      << "Original item must be weakly before the copy_start_to_host "
         "predecessor.";
  TF_ASSIGN_OR_RETURN(
      bool copy_start_to_host_predecessor_before_copy_done_to_host_predecessor,
      instruction_ordering_.CompareOrder(
          copy_start_to_host.predecessor->instruction,
          copy_start_to_device.successor->instruction));
  TF_RET_CHECK(
      copy_start_to_host_predecessor_before_copy_done_to_host_predecessor)
      << "copy_start_to_host predecessor must be strictly before the "
         "copy_done_to_host predecessor.";
  TF_ASSIGN_OR_RETURN(
      bool copy_done_to_host_predecessor_before_copy_start_to_device_successor,
      instruction_ordering_.CompareOrder(
          copy_done_to_host.predecessor->instruction,
          copy_start_to_device.successor->instruction));
  TF_RET_CHECK(
      copy_done_to_host_predecessor_before_copy_start_to_device_successor)
      << "copy_done_to_host predecessor must be strictly before the "
         "copy_start_to_device successor.";
  TF_ASSIGN_OR_RETURN(
      bool copy_start_to_device_successor_before_copy_done_to_device_successor,
      instruction_ordering_.CompareOrder(
          copy_start_to_device.successor->instruction,
          copy_done_to_device.successor->instruction));
  TF_RET_CHECK(
      copy_start_to_device_successor_before_copy_done_to_device_successor)
      << "copy_start_to_device successor must be strictly before the "
         "copy_done_to_device successor.";

  TF_RETURN_IF_ERROR(InsertInstructionAfterPredecessor(copy_start_to_host));
  TF_RETURN_IF_ERROR(InsertInstructionAfterPredecessor(copy_done_to_host));
  TF_RETURN_IF_ERROR(InsertInstructionBeforeSuccessor(copy_start_to_device));
  TF_RETURN_IF_ERROR(InsertInstructionBeforeSuccessor(copy_done_to_device));

  // Timeline of changes in memory usage. [] is inclusive. () is exclusive.
  // original
  // |  copy_start_to_host
  // |  |  copy_done_to_host
  // |  |  |  copy_start_to_device
  // |  |  |  |  copy_done_to_device
  // |  |  |  |  |  final_user_[original]
  // |  |  |  |  |  |
  // |  (--------)  |  output is no longer live
  // |  [--]  |  |  |  device buffer and context (to host) are live
  // |  |  |  [--]  |  device buffer and context (to device) are live
  // |  [--------]  |  host copy is live but doesn't affect device memory
  BufferId original_buffer_id = original_item->buffers_output[0];
  final_user_[original_buffer_id] = copy_start_to_host.new_item;
  BufferId copy_start_to_host_device_buffer_id =
      copy_start_to_host.new_item->buffers_output[0];
  BufferId copy_start_to_host_context_buffer_id =
      copy_start_to_host.new_item->buffers_output[1];
  BufferId copy_start_to_device_device_buffer_id =
      copy_start_to_device.new_item->buffers_output[0];
  BufferId copy_start_to_device_context_buffer_id =
      copy_start_to_device.new_item->buffers_output[1];
  BufferId copy_done_to_device_buffer_id =
      copy_done_to_device.new_item->buffers_output[0];

  TF_ASSIGN_OR_RETURN(const HloInstruction* copy_done_to_host_successor,
                      instruction_ordering_.GetNextInstruction(
                          copy_done_to_host.new_item->instruction));
  TF_ASSIGN_OR_RETURN(const HloInstruction* copy_start_to_device_predecessor,
                      instruction_ordering_.GetPreviousInstruction(
                          copy_start_to_device.new_item->instruction));
  TF_RETURN_IF_ERROR(segtree_->Update(
      copy_done_to_host_successor, copy_start_to_device_predecessor,
      -buffer_analyzer_->AllocatedSize(original_buffer_id)));

  TF_RETURN_IF_ERROR(segtree_->Update(
      copy_start_to_host.new_item->instruction,
      copy_done_to_host.new_item->instruction,
      buffer_analyzer_->AllocatedSize(copy_start_to_host_device_buffer_id) +
          buffer_analyzer_->AllocatedSize(
              copy_start_to_host_context_buffer_id)));
  TF_RETURN_IF_ERROR(segtree_->Update(
      copy_start_to_device.new_item->instruction,
      copy_done_to_device.new_item->instruction,
      buffer_analyzer_->AllocatedSize(copy_start_to_device_device_buffer_id) +
          buffer_analyzer_->AllocatedSize(
              copy_start_to_device_context_buffer_id)));

  // Set this before overwriting final_user_[original_buffer_id].
  final_user_[copy_done_to_device_buffer_id] = final_user_[original_buffer_id];
  final_user_[original_buffer_id] = copy_start_to_host.new_item;
  final_user_[copy_start_to_host_device_buffer_id] = copy_done_to_host.new_item;
  final_user_[copy_start_to_host_context_buffer_id] =
      copy_done_to_host.new_item;
  final_user_[copy_start_to_device_device_buffer_id] =
      copy_done_to_device.new_item;
  final_user_[copy_start_to_device_context_buffer_id] =
      copy_done_to_device.new_item;

  return absl::OkStatus();
}

absl::StatusOr<int64_t>
HloRematerializationPeakMemoryTracker::GetMemoryUsage() {
  return GetMemoryUsageDuring(in_progress_item_->instruction);
}

absl::StatusOr<int64_t> HloRematerializationPeakMemoryTracker::
    GetMemoryUsageWithoutCalledComputations() {
  TF_ASSIGN_OR_RETURN(const int64_t total_memory_usage, GetMemoryUsage());
  int64_t total_callee_memory_usage = 0;
  HloInstruction* instruction = in_progress_item_->instruction;
  const CallSite* callsite =
      call_graph_->GetNode(instruction->parent()).GetCallSite(instruction);
  if (callsite == nullptr || callsite->context() == CallContext::kEmbedded) {
    return total_memory_usage;
  }
  for (const HloComputation* computation : callsite->called_computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    total_callee_memory_usage +=
        last_computation_peak_memory_usage_[computation];
  }
  return total_memory_usage - total_callee_memory_usage;
}

absl::Status HloRematerializationPeakMemoryTracker::CalleeComputationWasUpdated(
    const HloComputation* callee_computation) {
  TF_ASSIGN_OR_RETURN(
      MemoryUsageAndInstruction peak_memory_usage_and_instruction,
      computation_peak_memory_tracker_.at(callee_computation)
          ->ComputePeakMemoryUsageAndInstruction());
  const int64_t new_peak_memory_usage =
      peak_memory_usage_and_instruction.memory_usage;
  const int64_t last_peak_memory_usage =
      last_computation_peak_memory_usage_[callee_computation];
  const int64_t delta = new_peak_memory_usage - last_peak_memory_usage;
  const HloInstruction* calling_instruction =
      computation_to_calling_instruction_[callee_computation];
  TF_RETURN_IF_ERROR(
      segtree_->Update(calling_instruction, calling_instruction, delta));
  last_computation_peak_memory_usage_[callee_computation] =
      new_peak_memory_usage;
  return absl::OkStatus();
}

bool HloRematerializationPeakMemoryTracker::Check() const {
  auto elements_are_unique = [](const BufferIdList& vec) {
    return vec.size() == std::set<BufferId>(vec.begin(), vec.end()).size();
  };

  // Verify buffers_defined per instruction.
  for (auto* instruction : computation()->instructions()) {
    const BufferIdList& defined_buffers =
        instruction_list_.GetItem(instruction)->buffers_defined;
    CHECK(elements_are_unique(defined_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique defined buffers: "
        << absl::StrJoin(
               defined_buffers, ", ",
               [this](std::string* out, BufferId buffer_id) {
                 absl::StrAppend(
                     out, buffer_analyzer_->buffers_.at(buffer_id).ToString());
               });

    for (const Buffer& buffer : buffer_analyzer_->buffers_) {
      if (buffer.defining_instruction->instruction == instruction) {
        CHECK(absl::c_linear_search(defined_buffers, buffer.id))
            << "Instruction " << instruction->name()
            << " defined buffers is missing: " << buffer.ToString();
      }
    }
  }

  // Verify buffers_used per instruction.
  for (auto* instruction : computation()->instructions()) {
    const BufferIdList& used_buffers =
        instruction_list_.GetItem(instruction)->buffers_used;
    CHECK(elements_are_unique(used_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique used buffers: "
        << absl::StrJoin(
               used_buffers, ", ",
               [this](std::string* out, BufferId buffer_id) {
                 absl::StrAppend(
                     out, buffer_analyzer_->buffers_.at(buffer_id).ToString());
               });
  }
  return true;
}

std::string HloRematerializationPeakMemoryTracker::ToString() const {
  std::string output =
      absl::StrCat("HloRematerializationPeakMemoryTracker for ",
                   computation()->name(), "\n");

  absl::StatusOr<MemoryUsageAndInstruction> memory_usage_and_instruction =
      ComputePeakMemoryUsageAndInstruction();
  if (memory_usage_and_instruction.ok()) {
    absl::StrAppend(
        &output, "Peak Memory usage: ",
        HumanReadableNumBytes(memory_usage_and_instruction->memory_usage),
        " at instruction ", memory_usage_and_instruction->instruction->name(),
        "\n");
  } else {
    absl::StrAppend(&output, "Peak Memory Error: ",
                    memory_usage_and_instruction.status().ToString(), "\n");
  }

  absl::StrAppend(&output, "Instruction info in order of instruction_list");
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    const HloInstruction* instruction = item->instruction;
    absl::string_view inprogress =
        item == in_progress_item_ ? " in-progress" : "";
    absl::string_view placed = IsItemPlaced(item) ? " placed" : "";
    absl::StrAppend(&output, "  ", instruction->name(), inprogress, placed,
                    "\n    Defines:\n");
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffer_analyzer_->buffers_[buffer_id];
      absl::string_view live = IsBufferLive(buffer_id) ? " live" : "";
      absl::StrAppend(&output, "      ", buffer.ToString(), live, ", ",
                      buffer.unfinished_user_count, " unfinished uses\n");
    }
    absl::StrAppend(&output, "    Outputs:\n");
    for (BufferId buffer_id : item->buffers_output) {
      absl::StrAppend(&output, "      ",
                      buffer_analyzer_->buffers_[buffer_id].ToString(), "\n");
    }
    absl::StrAppend(&output, "    Uses:\n");
    for (BufferId buffer_id : item->buffers_used) {
      absl::StrAppend(&output, "      ",
                      buffer_analyzer_->buffers_[buffer_id].ToString(),
                      " with final user ",
                      final_user_.at(buffer_id)->instruction->name(), "\n");
    }
  }

  absl::StrAppend(&output, "Instructions in order of instruction_ordering:");
  for (const HloInstruction* instruction =
           instruction_ordering_.GetFirstInstruction();
       instruction != nullptr;
       instruction = instruction_ordering_.GetNextInstruction(instruction)
                         .value_or(nullptr)) {
    absl::StrAppend(&output, instruction->name(), "\n");
  }
  return output;
}

bool HloRematerializationPeakMemoryTracker::IsItemPlaced(
    const HloRematItem* item) const {
  if (in_progress_item_ == nullptr) {
    // Nothing has been placed.
    return false;
  }
  // `item` is placed if is weakly before `in_progress_item_`. Since
  // CompareOrder() does strict comparisons, we reverse the direction and negate
  // the answer.
  absl::StatusOr<bool> is_unplaced = instruction_ordering_.CompareOrder(
      in_progress_item_->instruction, item->instruction);
  DCHECK_OK(is_unplaced);
  return is_unplaced.ok() && !is_unplaced.value();
}

bool HloRematerializationPeakMemoryTracker::IsItemFinished(
    const HloRematItem* item) const {
  return IsItemPlaced(item) && item != in_progress_item_;
}

bool HloRematerializationPeakMemoryTracker::IsBufferLive(
    BufferId buffer_id) const {
  const Buffer& buffer = buffer_analyzer_->buffers_[buffer_id];
  const auto final_user = final_user_.find(buffer_id);
  DCHECK(final_user != final_user_.end());

  return (IsItemPlaced(buffer.defining_instruction) &&
          final_user != final_user_.end() &&
          !IsItemFinished(final_user->second));
}

absl::Status HloRematerializationPeakMemoryTracker::InitializeSegmentTree() {
  // To initialize segtree_, we will need to compute the memory usage at every
  // point in time (filling in the following vector).
  std::vector<MemoryUsageAndInstruction> initial_memory_usage_and_instructions;

  int64_t current_memory = 0;
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    const HloInstruction* instruction = item->instruction;
    // We begin `instruction`, allocating space for the buffers defined.
    // TODO(b/37686934): Elementwise instructions can share the buffer of a
    // (dead) operand. Account for this potential reuse here.
    for (BufferId buffer_id : item->buffers_defined) {
      current_memory += buffer_analyzer_->AllocatedSize(buffer_id);
    }

    // We factor in the memory used by any callees as well.
    TF_ASSIGN_OR_RETURN(int64_t callee_usage,
                        CalledComputationsMemoryUsage(instruction));
    initial_memory_usage_and_instructions.push_back(
        {.memory_usage = current_memory + callee_usage,
         .instruction = instruction});

    // We end `instruction`, freeing space for the buffers we finished using.
    for (BufferId buffer_id : item->buffers_used) {
      Buffer& buffer = buffer_analyzer_->buffers_.at(buffer_id);
      buffer.unfinished_user_count--;
      TF_RET_CHECK(buffer.unfinished_user_count >= 0)
          << buffer.ToString() << " has negative unfinished user count.";
      if (buffer.unfinished_user_count == 0) {
        // Buffer is now dead.
        VLOG(3) << "  " << buffer.ToString() << " is now dead.";
        current_memory -= buffer_analyzer_->AllocatedSize(buffer_id);
        // The memory usage can become negative inside the computation as we can
        // free up the parameter space and reuse it for other tensors.
      }
    }

    // If any buffer defined by this instruction has no uses, then memory can be
    // reclaimed immediately.
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffer_analyzer_->buffers_.at(buffer_id);
      if (buffer.unfinished_user_count == 0) {
        VLOG(3) << "  " << buffer.ToString() << " is immediately dead.";
        current_memory -= buffer_analyzer_->AllocatedSize(buffer_id);
        // The memory usage can become negative inside the computation as we can
        // free up the parameter space and reuse it for other tensors.
      }
    }
  }

  segtree_.reset(new AVLLazySegmentTree(initial_memory_usage_and_instructions));
  return absl::OkStatus();
}

absl::Status HloRematerializationPeakMemoryTracker::
    InitializeSleatorDietzOrderMaintenance() {
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    TF_RETURN_IF_ERROR(instruction_ordering_.InsertBeforeInstruction(
        /*old_instruction=*/nullptr,
        /*new_instruction=*/item->instruction));
  }
  return absl::OkStatus();
}

absl::Status HloRematerializationPeakMemoryTracker::InitializeFinalUser() {
  for (BufferId buffer_id = 0; buffer_id < buffer_analyzer_->buffers_.size();
       ++buffer_id) {
    TF_RETURN_IF_ERROR(UpdateFinalUser(
        buffer_id, buffer_analyzer_->buffers_[buffer_id].users));
  }
  return absl::OkStatus();
}

absl::Status HloRematerializationPeakMemoryTracker::UpdateFinalUser(
    BufferId buffer_id, const UsesList& uses) {
  const Buffer& buffer = buffer_analyzer_->buffers_[buffer_id];
  HloRematItem* new_final_user = buffer.defining_instruction;
  for (const HloRematItemUse& use : uses) {
    TF_ASSIGN_OR_RETURN(bool is_before, instruction_ordering_.CompareOrder(
                                            new_final_user->instruction,
                                            use.user->instruction));
    if (is_before) {
      new_final_user = use.user;
    }
  }
  final_user_[buffer_id] = new_final_user;
  return absl::OkStatus();
}

absl::Status
HloRematerializationPeakMemoryTracker::InsertInstructionAfterPredecessor(
    HloRematerializationPeakMemoryTracker::NewItemAndPredecessor
        new_item_and_predecessor) {
  // Figure out the memory usage amount to use before actually editing any
  // data structures.
  TF_ASSIGN_OR_RETURN(
      int64_t memory_usage_after,
      GetMemoryUsageAfter(new_item_and_predecessor.predecessor->instruction));
  TF_RETURN_IF_ERROR(instruction_ordering_.InsertAfterInstruction(
      new_item_and_predecessor.predecessor->instruction,
      new_item_and_predecessor.new_item->instruction));
  MemoryUsageAndInstruction memory_usage_and_instruction = {
      .memory_usage = memory_usage_after,
      .instruction = new_item_and_predecessor.new_item->instruction,
  };
  TF_RETURN_IF_ERROR(segtree_->InsertAfterInstruction(
      new_item_and_predecessor.predecessor->instruction,
      memory_usage_and_instruction));
  return absl::OkStatus();
}

absl::Status
HloRematerializationPeakMemoryTracker::InsertInstructionBeforeSuccessor(
    HloRematerializationPeakMemoryTracker::NewItemAndSuccessor
        new_item_and_successor) {
  // Figure out the memory usage amount to use before actually editing any
  // data structures.
  TF_ASSIGN_OR_RETURN(
      int64_t memory_usage_before,
      GetMemoryUsageBefore(new_item_and_successor.successor->instruction));
  TF_RETURN_IF_ERROR(instruction_ordering_.InsertBeforeInstruction(
      new_item_and_successor.successor->instruction,
      new_item_and_successor.new_item->instruction));
  MemoryUsageAndInstruction memory_usage_and_instruction = {
      .memory_usage = memory_usage_before,
      .instruction = new_item_and_successor.new_item->instruction,
  };
  TF_RETURN_IF_ERROR(segtree_->InsertBeforeInstruction(
      new_item_and_successor.successor->instruction,
      memory_usage_and_instruction));
  return absl::OkStatus();
}

absl::StatusOr<int64_t>
HloRematerializationPeakMemoryTracker::CalledComputationsMemoryUsage(
    const HloInstruction* instruction) {
  const CallSite* callsite =
      call_graph_->GetNode(instruction->parent()).GetCallSite(instruction);
  if (callsite == nullptr || callsite->context() == CallContext::kEmbedded) {
    return 0;
  }
  int64_t total_callee_memory_usage = 0;
  for (const HloComputation* computation : callsite->called_computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    TF_RET_CHECK(ContainsKey(computation_peak_memory_tracker_, computation));
    TF_ASSIGN_OR_RETURN(
        MemoryUsageAndInstruction callee_memory_usage_and_instruction,
        computation_peak_memory_tracker_.at(computation)
            ->ComputePeakMemoryUsageAndInstruction());
    const int64_t callee_memory_usage =
        callee_memory_usage_and_instruction.memory_usage;
    last_computation_peak_memory_usage_[computation] = callee_memory_usage;
    computation_to_calling_instruction_[computation] = instruction;
    total_callee_memory_usage += callee_memory_usage;
  }
  return total_callee_memory_usage;
}

absl::StatusOr<int64_t>
HloRematerializationPeakMemoryTracker::GetMemoryUsageDuring(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(MemoryUsageAndInstruction memory_usage_and_instruction,
                      segtree_->Query(instruction, instruction));
  return memory_usage_and_instruction.memory_usage;
}

absl::StatusOr<int64_t>
HloRematerializationPeakMemoryTracker::GetMemoryUsageAfter(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(int64_t memory_usage, GetMemoryUsageDuring(instruction));

  // We simulate ending `instruction`, freeing space for the buffers we finished
  // using. This is like SweepMemoryTracker::EndInstruction(), except that we
  // do not have accurate `unfinished_user_count` information and depend on
  // `final_user_` instead.
  HloRematItem* item = instruction_list_.GetItem(instruction);
  for (BufferId buffer_id : item->buffers_used) {
    if (instruction == final_user_[buffer_id]->instruction) {
      memory_usage -= buffer_analyzer_->AllocatedSize(buffer_id);
    }
  }
  // If any buffer defined by this instruction has no uses, then memory can be
  // reclaimed immediately.
  for (BufferId buffer_id : item->buffers_defined) {
    if (instruction == final_user_[buffer_id]->instruction) {
      memory_usage -= buffer_analyzer_->AllocatedSize(buffer_id);
    }
  }
  return memory_usage;
}

absl::StatusOr<int64_t>
HloRematerializationPeakMemoryTracker::GetMemoryUsageBefore(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(
      const HloInstruction* previous_instruction,
      instruction_ordering_.GetPreviousInstruction(instruction));
  // If this is the first instruction, then we just use the fact that memory
  // usage begins at zero.
  if (previous_instruction == nullptr) {
    return 0;
  }
  return GetMemoryUsageAfter(previous_instruction);
}

}  // namespace xla
