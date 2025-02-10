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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"

namespace xla {
namespace memory_space_assignment {

using MsaBufferInterval =
    GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval;

// Encapsulates common utility methods for memory space assignment.
class MemorySpaceAssignmentUtils {
 public:
  // Returns true if this buffer is allowed to be placed in the alternate
  // memory.
  static bool IsIntervalAllowedInAlternateMemory(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval,
      int64_t alternate_memory_space);

  // Returns true if the HloValue is allowed to be placed in alternate memory.
  static bool IsValueAllowedInAlternateMemory(const HloValue* value,
                                              int64_t alternate_memory_space);

  static bool DoesUseMatchFilter(const HloOperandFilter& filter,
                                 const HloUse& hlo_use, int64_t operand_size);

  static bool DoesInstructionMatchFilter(const HloPositionMatcher& filter,
                                         const HloInstruction& instruction);

  static bool DoesPositionMatchFilter(const HloPositionMatcher& filter,
                                      const MsaBufferInterval& buffer_interval);

  static absl::StatusOr<xla::HloLiveRange::LogicalTime>
  GetScheduleTimeFromInstructionMatcher(
      const HloPositionMatcher& position_matcher,
      const absl::flat_hash_map<const xla::HloInstruction*,
                                xla::HloLiveRange::LogicalTime>& schedule);

  static absl::StatusOr<std::optional<int64_t>> GetPrefetchTimeByEagerness(
      float prefetch_eagerness, int64_t earliest_prefetch_time,
      int64_t latest_prefetch_time);

  static absl::StatusOr<std::optional<int64_t>> GetPrefetchTimeAfterInstruction(
      const HloPositionMatcher& after_instruction,
      const absl::flat_hash_map<const xla::HloInstruction*,
                                xla::HloLiveRange::LogicalTime>& schedule);

  static absl::StatusOr<std::optional<int64_t>>
  GetPrefetchTimeBeforeInstruction(
      const HloPositionMatcher& before_instruction,
      const absl::flat_hash_map<const xla::HloInstruction*,
                                xla::HloLiveRange::LogicalTime>& schedule);

  static absl::StatusOr<std::optional<int64_t>> GetPrefetchTime(
      const PreferredPrefetchOverrideOptions& override_options,
      int64_t earliest_prefetch_time, int64_t latest_prefetch_time,
      const absl::flat_hash_map<const HloInstruction*,
                                HloLiveRange::LogicalTime>&
          instruction_schedule);

  static absl::StatusOr<std::optional<int64_t>>
  GetOverriddenPreferredPrefetchTime(
      const PreferredPrefetchOverrides& preferred_prefetch_overrides,
      int64_t operand_size, const HloUse& hlo_use,
      const absl::flat_hash_map<const HloInstruction*,
                                HloLiveRange::LogicalTime>&
          instruction_schedule,
      int64_t earliest_prefetch_time, int64_t latest_prefetch_time);

  static bool DoesCrossProgramPrefetchBufferMatchAnyFilter(
      const MsaSortOrderOverrides& sort_order_overrides,
      const MsaBufferInterval& buffer_interval);

  // Returns an integer representing the priority of a MsaBufferInterval during
  // assignment, a smaller number indicates a higher priority.
  static int64_t GetBufferIntervalOverridePriority(
      const MsaSortOrderOverrides& msa_sort_order_overrides,
      const MsaBufferInterval& buffer_interval,
      bool is_cross_program_prefetch = false);

 private:
  static bool DoesBufferIntervalMatchHloUseFilter(
      const HloPositionMatcher& filter,
      const MsaBufferInterval& buffer_interval);
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_
