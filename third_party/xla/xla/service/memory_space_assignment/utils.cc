/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/utils.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "re2/re2.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace memory_space_assignment {

bool MemorySpaceAssignmentUtils::IsValueAllowedInAlternateMemory(
    const HloValue* value, int64_t alternate_memory_space) {
  // If the buffer is a tuple, don't use this algorithm for now. The buffers
  // that are pointed to by the tuple will still use this algorithm.  Because
  // tuples are cheap to place in the alternate memory (they are just pointers)
  // we don't need to use prefetch/evict logic.
  if (value->shape().IsTuple()) {
    VLOG(4) << "Keeping value " << value->ToShortString()
            << " in default mem because it is a tuple.";
    return false;
  }

  // Don't place scalars in the alternate memory.
  if (ShapeUtil::IsEffectiveScalar(value->shape())) {
    VLOG(4) << "Keeping value " << value->ToShortString()
            << " in default mem because it is a scalar.";
    return false;
  }

  // TODO(berkin): Not allocating add-dependencies either since they need to be
  // treated specially. We should revisit this later.
  for (const HloPosition& position : value->positions()) {
    if (position.instruction->opcode() == HloOpcode::kAddDependency) {
      VLOG(4) << "Keeping value " << value->ToShortString()
              << " in default mem because it has a "
              << "add-dependency position.";
      return false;
    }
  }

  // Send and Recv HLOs return a request identifier. These should not be
  // allocated in the alternate memory.
  for (const HloPosition& position : value->positions()) {
    if ((position.instruction->opcode() == HloOpcode::kSend ||
         position.instruction->opcode() == HloOpcode::kRecv) &&
        DynCast<HloSendRecvInstruction>(position.instruction)
            ->is_host_transfer()) {
      // TODO(berkin): Host transfers using alternate memory space doesn't seem
      // to work at the moment.
      VLOG(4) << "Keeping value " << value->ToShortString()
              << " in default mem because it is a send/recv buffer used for "
                 "host transfer.";
      return false;
    }

    // If the tensor is pre-colored to a memory space that is neither the
    // default (0) nor the alternate, disallow it from the alternate memory
    // space.
    int64_t memory_space = 0;
    if (position.shape().has_layout()) {
      memory_space = position.shape().layout().memory_space();
    }
    if (memory_space != 0 && memory_space != alternate_memory_space) {
      VLOG(4) << "Value " << value->ToShortString()
              << " not allowed in the alternate memory space due to existing "
                 "memory space: "
              << memory_space;
      return false;
    }
  }

  return true;
}

bool MemorySpaceAssignmentUtils::IsIntervalAllowedInAlternateMemory(
    const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval,
    int64_t alternate_memory_space) {
  return IsValueAllowedInAlternateMemory(interval.buffer,
                                         alternate_memory_space) &&
         absl::c_all_of(interval.colocations,
                        [alternate_memory_space](const HloValue* value) {
                          return IsValueAllowedInAlternateMemory(
                              value, alternate_memory_space);
                        });
}

bool MemorySpaceAssignmentUtils::DoesUseMatchFilter(
    const HloOperandFilter& filter, const HloUse& hlo_use,
    int64_t operand_size) {
  // The order of checks is such that the most expensive checks are done last.
  if (filter.has_size_gte() && operand_size < filter.size_gte()) {
    return false;
  }
  if (filter.has_size_lte() && operand_size > filter.size_lte()) {
    return false;
  }
  if (filter.has_operand_number() &&
      hlo_use.operand_number != filter.operand_number()) {
    return false;
  }
  if (filter.has_tuple_index() &&
      hlo_use.operand_index != ShapeIndex(filter.tuple_index().index().begin(),
                                          filter.tuple_index().index().end())) {
    return false;
  }
  if (filter.has_instruction_name_regex() &&
      !RE2::FullMatch(hlo_use.instruction->name(),
                      filter.instruction_name_regex())) {
    return false;
  }
  if (filter.has_instruction_regex() &&
      !RE2::FullMatch(hlo_use.instruction->ToString(),
                      filter.instruction_regex())) {
    return false;
  }
  return true;
}

bool MemorySpaceAssignmentUtils::DoesPositionMatchFilter(
    const HloPositionMatcher& filter,
    const MsaBufferInterval& buffer_interval) {
  // The order of checks is such that the most expensive checks are done last.
  HloInstruction* instruction = buffer_interval.buffer->instruction();
  if (filter.has_size_gte() && filter.size_gte() > buffer_interval.size) {
    return false;
  }
  if (filter.has_size_lte() && filter.size_lte() < buffer_interval.size) {
    return false;
  }
  if (filter.has_tuple_index() &&
      buffer_interval.buffer->index() !=
          ShapeIndex(filter.tuple_index().index().begin(),
                     filter.tuple_index().index().end())) {
    return false;
  }
  return DoesInstructionMatchFilter(filter, *instruction) &&
         DoesBufferIntervalMatchHloUseFilter(filter, buffer_interval);
}

bool MemorySpaceAssignmentUtils::DoesInstructionMatchFilter(
    const HloPositionMatcher& filter, const HloInstruction& instruction) {
  if (filter.has_instruction_name_regex() &&
      !RE2::FullMatch(instruction.name(), filter.instruction_name_regex())) {
    return false;
  }
  if (filter.has_instruction_regex() &&
      !RE2::FullMatch(instruction.ToString(), filter.instruction_regex())) {
    return false;
  }
  return true;
}

bool MemorySpaceAssignmentUtils::DoesBufferIntervalMatchHloUseFilter(
    const HloPositionMatcher& filter,
    const MsaBufferInterval& buffer_interval) {
  if (!filter.has_hlo_use_filter()) {
    return true;
  }
  for (const HloUse& use : buffer_interval.buffer->GetUses()) {
    if (DoesUseMatchFilter(filter.hlo_use_filter(), use,
                           buffer_interval.size)) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<xla::HloLiveRange::LogicalTime>
MemorySpaceAssignmentUtils::GetScheduleTimeFromInstructionMatcher(
    const HloPositionMatcher& position_matcher,
    const absl::flat_hash_map<const xla::HloInstruction*,
                              xla::HloLiveRange::LogicalTime>& schedule) {
  for (auto schedule_entry : schedule) {
    if (DoesInstructionMatchFilter(position_matcher, *schedule_entry.first)) {
      return schedule_entry.second;
    }
  }
  return NotFound("Reference instruction %s was not found in the schedule.",
                  position_matcher.DebugString());
}

absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetPrefetchTimeByEagerness(
    float prefetch_eagerness, int64_t earliest_prefetch_time,
    int64_t latest_prefetch_time) {
  CHECK_GE(prefetch_eagerness, 0.0);
  CHECK_LE(prefetch_eagerness, 1.0);
  if (earliest_prefetch_time > latest_prefetch_time) {
    return static_cast<std::optional<int64_t>>(std::nullopt);
  }
  return static_cast<std::optional<int64_t>>(
      earliest_prefetch_time +
      (latest_prefetch_time - earliest_prefetch_time) * prefetch_eagerness);
}

absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetPrefetchTimeAfterInstruction(
    const HloPositionMatcher& after_instruction,
    const absl::flat_hash_map<const xla::HloInstruction*,
                              xla::HloLiveRange::LogicalTime>& schedule) {
  TF_ASSIGN_OR_RETURN(
      auto reference_instruction_time,
      GetScheduleTimeFromInstructionMatcher(after_instruction, schedule));
  return static_cast<std::optional<int64_t>>(reference_instruction_time);
}

absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetPrefetchTimeBeforeInstruction(
    const HloPositionMatcher& before_instruction,
    const absl::flat_hash_map<const xla::HloInstruction*,
                              xla::HloLiveRange::LogicalTime>& schedule) {
  TF_ASSIGN_OR_RETURN(
      auto reference_instruction_time,
      GetScheduleTimeFromInstructionMatcher(before_instruction, schedule));
  return static_cast<std::optional<int64_t>>(reference_instruction_time - 1);
}
absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetPrefetchTime(
    const PreferredPrefetchOverrideOptions& override_options,
    int64_t earliest_prefetch_time, int64_t latest_prefetch_time,
    const absl::flat_hash_map<const HloInstruction*, HloLiveRange::LogicalTime>&
        instruction_schedule) {
  switch (override_options.options_case()) {
    case PreferredPrefetchOverrideOptions::kPrefetchEagerness:
      return GetPrefetchTimeByEagerness(override_options.prefetch_eagerness(),
                                        earliest_prefetch_time,
                                        latest_prefetch_time);
    case PreferredPrefetchOverrideOptions::kAfterInstruction:
      return GetPrefetchTimeAfterInstruction(
          override_options.after_instruction(), instruction_schedule);
    case PreferredPrefetchOverrideOptions::kBeforeInstruction:
      return GetPrefetchTimeBeforeInstruction(
          override_options.before_instruction(), instruction_schedule);
    case PreferredPrefetchOverrideOptions::OPTIONS_NOT_SET:
      break;
  }
  return static_cast<absl::StatusOr<std::optional<int64_t>>>(std::nullopt);
}

absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetOverriddenPreferredPrefetchTime(
    const PreferredPrefetchOverrides& preferred_prefetch_overrides,
    int64_t operand_size, const HloUse& hlo_use,
    const absl::flat_hash_map<const HloInstruction*, HloLiveRange::LogicalTime>&
        instruction_schedule,
    int64_t earliest_prefetch_time, int64_t latest_prefetch_time) {
  for (const auto& override : preferred_prefetch_overrides.overrides()) {
    if (!MemorySpaceAssignmentUtils::DoesUseMatchFilter(
            override.hlo_operand_filter(), hlo_use, operand_size)) {
      continue;
    }
    VLOG(3) << "Config match for instruction " << hlo_use.instruction->name()
            << " operand number " << hlo_use.operand_number << " operand index "
            << hlo_use.operand_index.ToString() << " size " << operand_size
            << " live range (" << earliest_prefetch_time << ", "
            << latest_prefetch_time << ")";
    TF_ASSIGN_OR_RETURN(
        auto prefetch_time,
        GetPrefetchTime(override.override_options(), earliest_prefetch_time,
                        latest_prefetch_time, instruction_schedule));
    if (prefetch_time.has_value() &&
        prefetch_time.value() >= earliest_prefetch_time &&
        prefetch_time.value() <= latest_prefetch_time) {
      return prefetch_time;
    }
  }
  return static_cast<absl::StatusOr<std::optional<int64_t>>>(std::nullopt);
}

bool MemorySpaceAssignmentUtils::DoesCrossProgramPrefetchBufferMatchAnyFilter(
    const MsaSortOrderOverrides& sort_order_overrides,
    const MsaBufferInterval& buffer_interval) {
  for (const MsaSortOrderOverride& override :
       sort_order_overrides.overrides()) {
    if (override.has_apply_to_cross_program_prefetches() &&
        override.apply_to_cross_program_prefetches() &&
        MemorySpaceAssignmentUtils::DoesPositionMatchFilter(
            override.hlo_position_matcher(), buffer_interval) &&
        override.override_options().has_assign_first() &&
        override.override_options().assign_first()) {
      VLOG(3) << "Cross program prefetch buffer "
              << buffer_interval.buffer->ToString()
              << " matches sort order override " << override.DebugString();
      return true;
    }
  }
  return false;
}

int64_t MemorySpaceAssignmentUtils::GetBufferIntervalOverridePriority(
    const MsaSortOrderOverrides& msa_sort_order_overrides,
    const MsaBufferInterval& buffer_interval, bool is_cross_program_prefetch) {
  if (msa_sort_order_overrides.overrides_size() == 0) {
    return 0;
  }
  for (int64_t i = 0; i < msa_sort_order_overrides.overrides_size(); ++i) {
    const auto& override = msa_sort_order_overrides.overrides(i);
    if (is_cross_program_prefetch &&
        (!override.has_apply_to_cross_program_prefetches() ||
         !override.apply_to_cross_program_prefetches())) {
      continue;
    }
    if (!MemorySpaceAssignmentUtils::DoesPositionMatchFilter(
            override.hlo_position_matcher(), buffer_interval)) {
      continue;
    }
    VLOG(3) << "Override Sort Order Config " << i << " matches "
            << buffer_interval.buffer->instruction()->ToString();
    switch (override.override_options().options_case()) {
      case MsaSortOrderOverrideOptions::kAssignFirst:
        return std::numeric_limits<int64_t>::lowest() + i;
      case MsaSortOrderOverrideOptions::kAssignLast:
        return std::numeric_limits<int64_t>::max() - i;
      case MsaSortOrderOverrideOptions::OPTIONS_NOT_SET:
        continue;
    }
  }
  return 0;
}
}  // namespace memory_space_assignment
}  // namespace xla
