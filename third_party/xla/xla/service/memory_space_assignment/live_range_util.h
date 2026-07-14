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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_LIVE_RANGE_UTIL_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_LIVE_RANGE_UTIL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"

namespace xla {

// Represents a closed interval [start_time, end_time] of instruction schedule
// indices during which a value or buffer is live.
struct LiveRange {
  int64_t start_time;
  int64_t end_time;

  std::string ToString() const {
    return absl::StrCat("[", start_time, ", ", end_time, "]");
  }
};

struct LiveRangeResult {
  std::vector<LiveRange> ranges;
  // Two live ranges [a, b] and [c, d] are considered overlapping only if b > c,
  // if b == c, they are merged, because one HloPosition can start at the same
  // instuction as another HloPosition ends (e.g., dynamic-update-slice). If
  // overlap_gt_1_detected is true, it only means that overlapping live ranges
  // were detected, it does not mean that overlaps were merged/removed.
  bool overlap_gt_1_detected = false;
};

// Calculates the live ranges of HloValues and HloBuffers based on a flattened
// instruction schedule. It handles schedule-flattened computations (like
// branches in conditionals and bodies of while loops) to accurately determine
// the span of liveness across control flow boundaries. Schedule-flattened
// computations impose constraints regarding the continuity of buffer liveness
// across boundaries. Additionally, async ops, loops, and conditionals can
// introduce additional complexity.
//
// Below we use the term "last uses" of an HloValue, which raises the question,
// how can we have more than 1 last use. There are 2 ways:
// - An HloValue may be used multiple times by the same instruction, e.g.,
//   add(a, a).
// - When dealing with conditionals, each branch has its own last use of an
//   HloValue.
class LiveRangeCalculator {
 public:
  // Constructs a calculator for the given buffer, using the provided
  // instruction-to-index mapping to determine time.
  LiveRangeCalculator(
      const HloBuffer& buffer,
      const absl::flat_hash_map<const HloInstruction*, int64_t>& inst_to_index);

  // Returns a merged and sorted list of live ranges for the entire buffer
  // (union of live ranges of all its values). Also returns whether overlapping
  // live ranges were detected during sorting and merging. Note: Overlapping
  // live ranges are not merged/removed.
  LiveRangeResult CalculateBufferLiveRange();

  // Trivial positions are not eligible for real memory allocations, they are
  // GTE, tuple, bitcast, or forwarded operand positions in an async
  // instruction. If the live range of a non-trivial position of `buffer_` is
  // not a sub-interval of the live range of any other non-trivial position of
  // the same buffer, it is considered an "outer" position, and it requires a
  // real memory allocation. This function returns the live range of the given
  // non-trivial position, if it is an "outer" position and the live range is
  // already calculated. Otherwise, it returns nullopt.
  // NOTE: CalculateBufferLiveRange() (or CalculateHloValueLiveRange()) must be
  // executed prior to querying individual position live ranges.
  std::optional<LiveRange> GetLiveRangeForNonTrivialNonForwardedPosition(
      HloPosition pos) const;

  // Returns a debug string representation of the live range, including
  // the names of the start and end instructions.
  std::string DetailedLiveRangeDebugString(const LiveRange& range) const;

  // Returns the earliest flattened-schedule index of any instruction within the
  // given computation.
  int64_t GetEarliestInstructionTime(const HloComputation* comp) const;

  const absl::flat_hash_map<HloPosition, LiveRange>& position_to_live_range()
      const {
    return position_to_live_range_;
  }

  const absl::flat_hash_map<HloPosition, std::vector<const HloUse*>>&
  position_to_uses() const {
    return position_to_uses_;
  }

  // Sorts the given live ranges by start time and merges overlapping ranges.
  LiveRangeResult SortAndMergeRanges(
      std::vector<LiveRange> all_ranges,
      bool merge_overlapping_ranges = false) const;

  // Returns a merged and sorted list of live ranges for the given HLO value,
  // starting from its definition to its last use(s).
  std::vector<LiveRange> CalculateHloValueLiveRange(const HloValue* value);

  // Returns the schedule index (time) of the given instruction.
  // Returns -1 if the instruction is not in the schedule.
  int64_t GetInstructionTime(const HloInstruction* inst) const;

 private:
  // Returns the last (with highest schedule index) uses of the given value in
  // the given computation. The last uses returned by this function might not be
  // true use sites (where a value is read/used), e.g., when the last use is a
  // conditional. This method is not recursive.
  std::vector<const HloUse*> FindLastUsesInComputation(
      const HloComputation* comp, const HloValue* value) const;

  // For instructions that call computations, that are inlined in the flattened
  // schedule (e.g., While, Conditional), returns the earliest start time, among
  // all schedule-flattened computations. Note: The instruction itself is not
  // considered; it is scheduled after the schedule-flattened computations.
  int64_t GetEarliestCalledComputationStartTime(
      const HloInstruction* inst) const;

  // Determines the live ranges for a value given its last uses.
  // If the last use is a control flow instruction (While/Conditional),
  // it recursively determines liveness within the schedule-flattened
  // computations. Note: kWhile and kConditional instructions are handled
  // differently from other instructions with schedule-flattened computations.
  // REQUIRES: `last_uses` is not empty, and all uses in `last_uses` must belong
  // to the same instruction.
  std::vector<LiveRange> DetermineLiveRangesGivenLastUses(
      const std::vector<const HloUse*>& last_uses, const HloValue* value,
      int64_t start_time);

  // Determines the live range of a value within a specific schedule-flattened
  // computation (e.g., a branch of a conditional or body of a loop).
  std::vector<LiveRange> DetermineLiveRangeInComputation(HloComputation* comp,
                                                         const HloValue* value);

  // Determines the live ranges of a value for a use in a Conditional
  // instruction.
  std::vector<LiveRange> DetermineLiveRangesForConditionalUse(
      const HloUse* conditional_use, const HloValue* value);

  // Associates a calculated live range with a non-trivial position.
  void SetLiveRangeForNonTrivialNonForwardedPosition(
      HloPosition pos, LiveRange range, std::vector<LiveRange>& live_ranges);

  // Returns the async use live range that contains the given start time, if
  // one exists.
  std::optional<LiveRange> FindContainingAsyncUseLiveRange(
      int64_t start_time) const;

  // The buffer for which we are calculating live ranges.
  const HloBuffer& buffer_;

  // A map from instruction to its index in the flattened schedule.
  const absl::flat_hash_map<const HloInstruction*, int64_t>& inst_to_index_;

  // A reverse mapping from instruction index in the flattened schedule to
  // instruction.
  absl::flat_hash_map<int64_t, const HloInstruction*> index_to_inst_;

  // A map to store the earliest instruction time in each computation.
  absl::flat_hash_map<const HloComputation*, int64_t>
      earliest_inst_time_in_comp_;

  // A nested hash map to find all the uses of a value in a computation.
  // Format: uses_by_value_and_comp_[value][computation] = vector_of_uses.
  absl::flat_hash_map<
      const HloValue*,
      absl::flat_hash_map<const HloComputation*, std::vector<const HloUse*>>>
      uses_by_value_and_comp_;

  // A map from non-trivial HloPositions to all the uses of that position.
  // The HloUse objects are owned by HloValue (which in turn are owned by
  // HloAliasAnalysis). This map stores non-owning raw pointers; the calculator
  // must not outlive the HloAliasAnalysis.
  absl::flat_hash_map<HloPosition, std::vector<const HloUse*>>
      position_to_uses_;

  // Contains live ranges of non-trivial positions of the buffer that are not
  // sub-intervals of the live range of any other non-trivial position of the
  // same buffer. From an MSA perspective, these "outer" positions require
  // independent memory allocations, whereas the nested "inner" positions
  // can share/mirror the allocation of their containing position.
  absl::flat_hash_map<HloPosition, LiveRange> position_to_live_range_;

  // Live ranges for which buffer_ is required to have contiguous allocation
  // because of async uses. If an async operand has an aliased use inside the
  // async called computation, it will result in overlapping live ranges because
  // the new HloValue inside the async called computation is live at the same
  // time as the async operand. This is safe to merge and does not qualify as a
  // real overlap. This data structure is used to identify such cases.
  //
  // Example:
  //
  // async_comp (p0: f32[2,2]) -> f32[2,2] {
  //   p2 = f32[2,2] parameter(1)
  //   ROOT custom_call0 = f32[2,2] custom-call(p2) out_to_op_alias={{0}:(0)}
  // }
  //
  // ENTRY entry (p0: f32[2,2], p1: f32[2,2]) -> f32[2,2] {
  //   p0 = f32[2,2] parameter(0)
  //   p1 = f32[2,2] parameter(1)
  //   as_start = {f32[2,2], f32[2,2], s32[]} async-start(p0) calls=async_comp
  //   negate0 = f32[2,2] negate(p1)
  //   async_done = f32[2,2] async-done(as_start)
  //   ROOT add0 = f32[2,2] add(negate0, async_done)
  // }
  //
  // In the example above, the HloPosition as_start {0} is a forwarded
  // position of p0 and requires a contiguous allocation and thus has a live
  // range from as_start to async_done. The async computation
  // has an aliased use of p0, which defines a new HloValue for the same buffer,
  // that has a live range from custom_call0 to add0. This results in an
  // overlap, which is legal and safe to merge.
  std::vector<LiveRange> async_use_live_ranges_;
};

}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_LIVE_RANGE_UTIL_H_
