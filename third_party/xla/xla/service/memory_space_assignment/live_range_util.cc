/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/live_range_util.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"

namespace xla {

namespace {

const HloValue* FindAssociatedValue(HloPosition pos, const HloBuffer& buffer) {
  for (const HloValue* value : buffer.values()) {
    for (const HloPosition& p : value->positions()) {
      if (p == pos) {
        return value;
      }
    }
  }
  LOG(FATAL) << "No associated value found for position " << pos.ToString()
             << " and buffer " << buffer.ToString();
  return nullptr;
}

bool IsTrivialOrForwardedPosition(HloPosition position) {
  return position.instruction->opcode() == HloOpcode::kGetTupleElement ||
         position.instruction->opcode() == HloOpcode::kTuple ||
         position.instruction->opcode() == HloOpcode::kBitcast ||
         (position.instruction->opcode() == HloOpcode::kAsyncStart &&
          !position.index.empty() && position.index.front() == 0);
}

bool IsNonTrivialNonForwardedPosition(HloPosition position) {
  return !IsTrivialOrForwardedPosition(position);
}

HloPosition GetNonTrivialNonForwardedSourcePosition(HloPosition position) {
  while (IsTrivialOrForwardedPosition(position)) {
    if (position.instruction->opcode() == HloOpcode::kGetTupleElement) {
      int64_t tuple_index = position.instruction->tuple_index();
      position.instruction = position.instruction->mutable_operand(0);
      position.index.push_front(tuple_index);
    } else if (position.instruction->opcode() == HloOpcode::kTuple) {
      if (position.index.empty()) {
        return position;
      }
      int64_t tuple_index = position.index.front();
      position.index.pop_front();
      position.instruction = position.instruction->mutable_operand(tuple_index);
    } else if (position.instruction->opcode() == HloOpcode::kBitcast) {
      position.instruction = position.instruction->mutable_operand(0);
    } else if (position.instruction->opcode() == HloOpcode::kAsyncStart &&
               !position.index.empty() && position.index.front() == 0) {
      position.index.pop_front();
      if (position.index.empty()) {
        position.instruction = position.instruction->mutable_operand(0);
      } else {
        position.instruction =
            position.instruction->mutable_operand(position.index.front());
        position.index.pop_front();
      }
    }
  }
  return position;
}

std::string InstructionNameAndComputationName(const HloInstruction* inst) {
  return absl::StrCat("(", inst->name(),
                      " computation: ", inst->parent()->name(), ")");
}

bool IsWhileLoopConditionParameter(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kParameter ||
      inst->parent()->caller_instructions().empty()) {
    return false;
  }
  for (const HloInstruction* caller : inst->parent()->caller_instructions()) {
    if (caller->opcode() == HloOpcode::kWhile &&
        caller->while_condition() == inst->parent()) {
      return true;
    }
  }
  return false;
}

}  // namespace

LiveRangeCalculator::LiveRangeCalculator(
    const HloBuffer& buffer,
    const absl::flat_hash_map<const HloInstruction*, int64_t>& inst_to_index)
    : buffer_(buffer), inst_to_index_(inst_to_index) {
  // NOLINTNEXTLINE
  for (const auto& [inst, index] : inst_to_index) {
    index_to_inst_[index] = inst;
    auto [it, inserted] =
        earliest_inst_time_in_comp_.try_emplace(inst->parent(), index);
    if (!inserted) {
      it->second = std::min(it->second, index);
    }
  }

  for (const HloValue* value : buffer_.values()) {
    for (const HloUse& use : value->GetUses()) {
      if (use.instruction->opcode() == HloOpcode::kAsyncDone) {
        // This needs to be updated when async updates are added.
        HloInstruction* async_start = use.instruction->async_chain_start();
        int64_t start_time = GetEarliestCalledComputationStartTime(async_start);
        async_use_live_ranges_.push_back(
            {start_time, inst_to_index.at(use.instruction)});
      }
      uses_by_value_and_comp_[value][use.instruction->parent()].push_back(&use);
      HloInstruction* operand_instruction =
          use.instruction->mutable_operand(use.operand_number);
      position_to_uses_[GetNonTrivialNonForwardedSourcePosition(HloPosition{
                            operand_instruction, use.operand_index})]
          .push_back(&use);
    }
  }

  async_use_live_ranges_ = SortAndMergeRanges(std::move(async_use_live_ranges_),
                                              /*merge_overlapping_ranges=*/true)
                               .ranges;
}

int64_t LiveRangeCalculator::GetInstructionTime(
    const HloInstruction* inst) const {
  return FindOrDefault(inst_to_index_, inst, -1);
}

int64_t LiveRangeCalculator::GetEarliestInstructionTime(
    const HloComputation* comp) const {
  auto it = earliest_inst_time_in_comp_.find(comp);
  if (it != earliest_inst_time_in_comp_.end()) {
    return it->second;
  }
  // Fallback to lazy evaluation if the computation wasn't in inst_to_index
  int64_t earliest = -1;
  for (const HloInstruction* inst : comp->instructions()) {
    int64_t time = GetInstructionTime(inst);
    if (time != -1) {
      if (earliest == -1 || time < earliest) {
        earliest = time;
      }
    }
  }
  return earliest;
}

std::vector<const HloUse*> LiveRangeCalculator::FindLastUsesInComputation(
    const HloComputation* comp, const HloValue* value) const {
  auto it = uses_by_value_and_comp_.find(value);
  if (it == uses_by_value_and_comp_.end()) {
    return {};
  }
  auto it2 = it->second.find(comp);
  if (it2 == it->second.end()) {
    return {};
  }

  int64_t max_time = -1;
  for (const HloUse* use : it2->second) {
    int64_t time = GetInstructionTime(use->instruction);
    max_time = std::max(max_time, time);
  }
  if (max_time == -1) {
    return {};
  }

  std::vector<const HloUse*> last_uses;
  for (const HloUse* use : it2->second) {
    if (GetInstructionTime(use->instruction) == max_time) {
      last_uses.push_back(use);
    }
  }
  return last_uses;
}

int64_t LiveRangeCalculator::GetEarliestCalledComputationStartTime(
    const HloInstruction* inst) const {
  int64_t earliest_called_computation_start_time =
      std::numeric_limits<int64_t>::max();
  for (const HloComputation* branch_comp : inst->called_computations()) {
    earliest_called_computation_start_time =
        std::min(earliest_called_computation_start_time,
                 GetEarliestInstructionTime(branch_comp));
  }
  return earliest_called_computation_start_time;
}

std::vector<LiveRange> LiveRangeCalculator::DetermineLiveRangesGivenLastUses(
    const std::vector<const HloUse*>& last_uses, const HloValue* value,
    int64_t start_time) {
  CHECK(!last_uses.empty());
  const HloInstruction* last_inst = last_uses[0]->instruction;
  for (const HloUse* last_use : last_uses) {
    CHECK_EQ(last_use->instruction, last_inst);
  }
  HloUse use = *last_uses[0];
  HloInstruction* operand_instruction =
      use.instruction->mutable_operand(use.operand_number);
  HloPosition current_non_trivial_position =
      GetNonTrivialNonForwardedSourcePosition(
          HloPosition{operand_instruction, use.operand_index});
  int64_t last_inst_time = GetInstructionTime(last_inst);
  std::vector<LiveRange> result;
  if (last_inst->opcode() != HloOpcode::kWhile &&
      last_inst->opcode() != HloOpcode::kConditional) {
    LiveRange live_range = {start_time, last_inst_time};
    SetLiveRangeForNonTrivialNonForwardedPosition(current_non_trivial_position,
                                                  live_range, result);
    return result;
  }

  // For while loop parameter positions, the positions inside the condition and
  // body computations are considered separate HloValues. Their live ranges
  // will be computed independently. We do not need any special handling for
  // while loops apart from this common logic.
  int64_t earliest_schedule_flattened_computation_start_time =
      GetEarliestCalledComputationStartTime(last_inst);
  LiveRange live_range = {start_time,
                          earliest_schedule_flattened_computation_start_time};
  SetLiveRangeForNonTrivialNonForwardedPosition(current_non_trivial_position,
                                                live_range, result);

  if (last_inst->opcode() == HloOpcode::kConditional) {
    for (const HloUse* last_use : last_uses) {
      std::vector<LiveRange> ranges =
          DetermineLiveRangesForConditionalUse(last_use, value);
      result.insert(result.end(), ranges.begin(), ranges.end());
    }
  }

  return SortAndMergeRanges(std::move(result)).ranges;
}

std::vector<LiveRange> LiveRangeCalculator::DetermineLiveRangeInComputation(
    HloComputation* comp, const HloValue* value) {
  int64_t start_time = GetEarliestInstructionTime(comp);

  // The HloValue should be live from the position inside the computation till
  // its last use. There can be multiple uses for the last use instruction. When
  // the last use instruction is a special instruction, it can result in
  // disjoint live ranges. For example, if the last instruction is a conditional
  // instruction.
  std::vector<const HloUse*> internal_last_uses =
      FindLastUsesInComputation(comp, value);
  if (internal_last_uses.empty()) {
    return {{start_time, start_time}};
  }

  return DetermineLiveRangesGivenLastUses(internal_last_uses, value,
                                          start_time);
}

std::vector<LiveRange>
LiveRangeCalculator::DetermineLiveRangesForConditionalUse(
    const HloUse* conditional_use, const HloValue* value) {
  const HloInstruction* inst = conditional_use->instruction;
  CHECK(inst->opcode() == HloOpcode::kConditional);
  if (conditional_use->operand_number == 0) {
    // Operand 0 of a conditional is the predicate value, it does not have a
    // corresponding schedule-flattened computation.
    return {};
  }
  return DetermineLiveRangeInComputation(
      inst->branch_computation(conditional_use->operand_number - 1), value);
}

LiveRangeResult LiveRangeCalculator::SortAndMergeRanges(
    std::vector<LiveRange> all_ranges, bool merge_overlapping_ranges) const {
  LiveRangeResult result;
  if (all_ranges.empty()) {
    return result;
  }
  std::sort(all_ranges.begin(), all_ranges.end(),
            [](const LiveRange& a, const LiveRange& b) {
              return a.start_time < b.start_time;
            });
  std::vector<LiveRange> merged;
  for (const auto& r : all_ranges) {
    if (merged.empty()) {
      merged.push_back(r);
      continue;
    }
    if (merged.back().end_time == r.start_time) {
      merged.back().end_time = std::max(merged.back().end_time, r.end_time);
    } else if (merged.back().end_time < r.start_time) {
      merged.push_back(r);
    } else {
      result.overlap_gt_1_detected = true;
      VLOG(1) << "Overlapping live ranges detected: ["
              << merged.back().start_time << ", " << merged.back().end_time
              << "] and [" << r.start_time << ", " << r.end_time << "]\n"
              << "Buffer: " << buffer_.ToString();
      if (merge_overlapping_ranges) {
        merged.back().end_time = std::max(merged.back().end_time, r.end_time);
      } else {
        merged.push_back(r);
      }
    }
  }
  result.ranges = merged;
  return result;
}

std::vector<LiveRange> LiveRangeCalculator::CalculateHloValueLiveRange(
    const HloValue* value) {
  VLOG(3) << "Calculating live range for value: " << value->ToString();
  int64_t start_time = GetInstructionTime(value->defining_instruction());
  // - Loop condition parameter positions are considered new HloValues (as
  //   opposed to forwarded HloPositions).
  // - All the HloValues might not have a use in the condition computation,
  //   but they should be live throughout the computation.
  if (IsWhileLoopConditionParameter(value->defining_instruction())) {
    int64_t end_time = GetInstructionTime(
        value->defining_instruction()->parent()->root_instruction());
    LiveRange live_range = {start_time, end_time};
    std::vector<LiveRange> result;
    SetLiveRangeForNonTrivialNonForwardedPosition(value->defining_position(),
                                                  live_range, result);
    return result;
  }

  std::vector<const HloUse*> uses;
  uses.reserve(value->GetUses().size());
  for (const HloUse& use : value->GetUses()) {
    uses.push_back(&use);
  }
  if (uses.empty()) {
    if (value->defining_instruction()->opcode() == HloOpcode::kFusion) {
      // A fusion output HloValue with no uses is a dummy output, which does not
      // require any memory allocation.
      return {};
    }
    std::vector<LiveRange> result;
    LiveRange live_range = {start_time, start_time};
    SetLiveRangeForNonTrivialNonForwardedPosition(value->defining_position(),
                                                  live_range, result);
    return result;
  }

  std::sort(uses.begin(), uses.end(), [this](const HloUse* a, const HloUse* b) {
    return GetInstructionTime(a->instruction) <
           GetInstructionTime(b->instruction);
  });

  // The HloValue should be live from definition to the last use instruction.
  // There can be multiple uses for the last use instruction. When the last use
  // instruction has called computations, it can result in disjoint live ranges.
  // For example, if the last instruction is a conditional instruction.
  const HloInstruction* last_inst = uses.back()->instruction;
  std::vector<const HloUse*> last_uses;
  for (auto it = uses.rbegin();
       it != uses.rend() && (*it)->instruction == last_inst; ++it) {
    last_uses.push_back(*it);
  }
  return DetermineLiveRangesGivenLastUses(last_uses, value, start_time);
}

LiveRangeResult LiveRangeCalculator::CalculateBufferLiveRange() {
  std::vector<LiveRange> all_ranges;
  all_ranges.reserve(buffer_.values().size());
  for (const HloValue* value : buffer_.values()) {
    std::vector<LiveRange> val_ranges = CalculateHloValueLiveRange(value);
    VLOG(3) << "Live ranges for value " << value->ToString() << ": ";
    for (const auto& range : val_ranges) {
      VLOG(3) << "  " << DetailedLiveRangeDebugString(range);
    }
    all_ranges.insert(all_ranges.end(), val_ranges.begin(), val_ranges.end());
  }
  return SortAndMergeRanges(std::move(all_ranges));
}

void LiveRangeCalculator::SetLiveRangeForNonTrivialNonForwardedPosition(
    HloPosition pos, LiveRange range, std::vector<LiveRange>& live_ranges) {
  CHECK(IsNonTrivialNonForwardedPosition(pos))
      << "pos: " << pos.ToString() << " [" << range.start_time << ", "
      << range.end_time << "]\nbuffer:" << buffer_.ToString();
  std::optional<LiveRange> async_range =
      FindContainingAsyncUseLiveRange(range.start_time);
  if (async_range.has_value()) {
    if (range.end_time <= async_range.value().end_time) {
      VLOG(3) << "Non-trivial position " << pos.ToString() << " live range "
              << DetailedLiveRangeDebugString(range)
              << " is completely within async use live range "
              << DetailedLiveRangeDebugString(async_range.value())
              << " and hence not considered.\n";
      return;
    }
    VLOG(3) << "Non-trivial position " << pos.ToString() << " live range "
            << DetailedLiveRangeDebugString(range)
            << " is partially within async use live range "
            << DetailedLiveRangeDebugString(async_range.value())
            << ". Updating live range and position.";
    range.start_time = std::max(range.start_time, async_range.value().end_time);
    const HloValue* associated_value = FindAssociatedValue(pos, buffer_);

    for (const HloPosition& p : associated_value->positions()) {
      if (GetInstructionTime(p.instruction) == async_range.value().end_time &&
          IsNonTrivialNonForwardedPosition(p)) {
        pos = p;
        break;
      }
    }
    VLOG(3) << "Updated non-trivial position " << pos.ToString()
            << " live range " << DetailedLiveRangeDebugString(range) << "\n";
  }
  auto live_range_it = position_to_live_range_.find(pos);
  if (live_range_it != position_to_live_range_.end()) {
    CHECK_EQ(live_range_it->second.start_time, range.start_time);
    CHECK_EQ(live_range_it->second.end_time, range.end_time);
    return;
  }

  VLOG(3) << "Setting live range for non-trivial position " << pos.ToString()
          << " to " << DetailedLiveRangeDebugString(range) << "\n";
  position_to_live_range_[pos] = range;
  live_ranges.push_back(range);
}

std::optional<LiveRange> LiveRangeCalculator::FindContainingAsyncUseLiveRange(
    int64_t start_time) const {
  auto it = std::lower_bound(async_use_live_ranges_.begin(),
                             async_use_live_ranges_.end(), start_time,
                             [](const LiveRange& element, int64_t value) {
                               return element.end_time < value;
                             });
  if (it != async_use_live_ranges_.end()) {
    const auto& async_range = *it;
    if (async_range.start_time <= start_time &&
        start_time <= async_range.end_time) {
      return async_range;
    }
  }
  return std::nullopt;
}

std::optional<LiveRange>
LiveRangeCalculator::GetLiveRangeForNonTrivialNonForwardedPosition(
    HloPosition pos) const {
  auto it = position_to_live_range_.find(pos);
  if (it != position_to_live_range_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::string LiveRangeCalculator::DetailedLiveRangeDebugString(
    const LiveRange& range) const {
  const HloInstruction* start_inst = index_to_inst_.at(range.start_time);
  const HloInstruction* end_inst = index_to_inst_.at(range.end_time);
  return absl::StrCat(
      range.ToString(),
      " start_inst: ", InstructionNameAndComputationName(start_inst),
      " end_inst: ", InstructionNameAndComputationName(end_inst));
}

}  // namespace xla
