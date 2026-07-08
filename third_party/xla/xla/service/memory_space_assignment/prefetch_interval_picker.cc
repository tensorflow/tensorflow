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

#include "xla/service/memory_space_assignment/prefetch_interval_picker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace memory_space_assignment {
namespace {

// Each time we retry compilation, increase the preferred eviction end time by
// this amount multiplied by preferred overlap to async copy ratio.
const float kEvictionRetryMultiplier = 2.0;

// The number of decreasing intervals for CostAnalysisPrefetchIntervalPicker to
// return when it runs out of increasing intervals. Increasing this number may
// hurt compilation time.
const int kNumExploredDecreasingIntervals = 100;

}  // namespace

bool InstructionCountPrefetchIntervalPicker::CanAllocateInAlternateMemoryNoCopy(
    int64_t shape_size, int64_t start_time, int64_t end_time) const {
  return end_time - start_time <= max_overlap_count_;
}

int64_t InstructionCountPrefetchIntervalPicker::PreferredEvictionEndTime(
    const Shape& shape, int64_t start_time, int64_t latest_end_time) const {
  return std::min(start_time + min_overlap_count_, latest_end_time);
}

int64_t InstructionCountPrefetchIntervalPicker::LatestPrefetchStartTime(
    int64_t shape_size, int64_t start_time, int64_t end_time,
    const HloUse* use) const {
  return end_time - min_overlap_count_;
}

int64_t InstructionCountPrefetchIntervalPicker::PreferredPrefetchStartTime(
    const Shape& shape, int64_t earliest_prefetch_start_time,
    int64_t latest_prefetch_start_time, int64_t prefetch_end_time) const {
  return std::max(earliest_prefetch_start_time,
                  prefetch_end_time - max_overlap_count_);
}

int64_t InstructionCountPrefetchIntervalPicker::EstimatedPrefetchEndTime(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  // For testing, assume the end time is the estimated prefetch end time.
  return end_time;
}

float InstructionCountPrefetchIntervalPicker::GetLogicalIntervalElapsed(
    int64_t start_time, int64_t end_time) const {
  // For testing, just assume every HLO takes 1 second.
  return static_cast<float>(end_time - start_time - 1);
}

void InstructionCountPrefetchIntervalPicker::Begin(
    const HloUse& use, int64_t start_time, int64_t end_time,
    std::optional<int64_t> preferred_time) {
  end_time_ = end_time;
  const Shape& shape = ShapeUtil::GetSubshape(
      use.instruction->operand(use.operand_number)->shape(), use.operand_index);
  if (preferred_time) {
    current_prefetch_time_ = *preferred_time;
  } else {
    current_prefetch_time_ =
        PreferredPrefetchStartTime(shape, start_time, end_time, end_time);
  }
}

int64_t InstructionCountPrefetchIntervalPicker::Next() {
  CHECK(!Done()) << "Prefetch interval picker's Next() is called even though "
                    "Done() is false";
  return current_prefetch_time_++;
}

bool InstructionCountPrefetchIntervalPicker::Done() const {
  return end_time_ - current_prefetch_time_ <= min_overlap_count_;
}

int64_t InstructionCountPrefetchIntervalPicker::latest_time() const {
  return end_time_ - min_overlap_count_ - 1;
}

std::string InstructionCountPrefetchIntervalPicker::ToDebugString() const {
  return absl::StrCat("Overlapped HLOs = ", end_time_ - current_prefetch_time_);
}

std::string InstructionCountPrefetchIntervalPicker::ToNoCopyDebugString(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  return absl::StrCat("Overlapped HLOs = ", end_time - start_time);
}

CostAnalysisPrefetchIntervalPicker::CostAnalysisPrefetchIntervalPicker(
    const CostAnalysis& cost_analysis, float min_overlap_to_async_copy_ratio,
    float preferred_overlap_to_async_copy_ratio,
    float max_overlap_to_mem_size_async_copy_ratio, int64_t mem_size_bytes,
    std::optional<int64_t> size_override)
    : while_nest_level_(
          cost_analysis.hlo_live_range().instruction_schedule().size() + 1, 0),
      computation_nest_level_(
          cost_analysis.hlo_live_range().instruction_schedule().size() + 1, 0),
      cost_analysis_(cost_analysis),
      min_overlap_to_async_copy_ratio_(min_overlap_to_async_copy_ratio),
      preferred_overlap_to_async_copy_ratio_(
          preferred_overlap_to_async_copy_ratio),
      size_override_(size_override) {
  max_async_copy_elapsed_ =
      cost_analysis_.GetAsyncCopyElapsed(cost_analysis_.GetShapeSizeBytes(
          ShapeUtil::MakeShape(S32, {mem_size_bytes / 4}))) *
      max_overlap_to_mem_size_async_copy_ratio;
  instruction_schedule_ =
      &cost_analysis_.hlo_live_range().instruction_schedule();

  // Create a vector of elapsed times and while nesting levels of HLO
  // instructions. The elapsed times are multiplied by
  // pow(while_execution_count, nest_level) to account for executing the HLOs
  // multiple times in while loops.
  std::vector<float> instructions_elapsed_time(
      instruction_schedule_->size() + 1, 0.0);
  int max_while_nest_level = 0;
  for (const auto& instruction_and_logical_time : *instruction_schedule_) {
    // To avoid double counting, don't include the elapsed time of while and
    // conditional HLOs.
    const HloInstruction* instruction = instruction_and_logical_time.first;
    int64_t logical_time = instruction_and_logical_time.second;
    if (logical_time >= instructions_elapsed_time.size()) {
      instructions_elapsed_time.resize(logical_time + 1, 0.0);
      while_nest_level_.resize(logical_time + 1, 0);
    }
    int while_nest_level = cost_analysis_.CalculateComputationNestLevel(
        instruction_and_logical_time.first, /*while_only=*/true);
    while_nest_level_[logical_time] = while_nest_level;
    max_while_nest_level = std::max(max_while_nest_level, while_nest_level);
    int computation_nest_level = cost_analysis_.CalculateComputationNestLevel(
        instruction_and_logical_time.first, /*while_only=*/false);
    computation_nest_level_[logical_time] = computation_nest_level;
    if (instruction->opcode() == HloOpcode::kWhile ||
        instruction->opcode() == HloOpcode::kConditional) {
      continue;
    }
    float elapsed_time = cost_analysis_.GetInstructionElapsed(
        *instruction_and_logical_time.first);
    instructions_elapsed_time[logical_time] =
        elapsed_time * cost_analysis_.GetWhileNestMultiplier(while_nest_level);
  }
  // As an optimization, create a cumulative sum vector of elapsed time.
  float cumsum = 0.0;
  elapsed_time_cumsum_.reserve(instructions_elapsed_time.size());
  for (float elapsed_time : instructions_elapsed_time) {
    cumsum += elapsed_time;
    elapsed_time_cumsum_.push_back(cumsum);
  }
  // To be able to accurately determine the minimum nest level between a start
  // time and an end time efficiently, populate a data structure that stores the
  // closest 'smaller' nest level change index.
  const int64_t size = instructions_elapsed_time.size();
  CHECK_EQ(size, while_nest_level_.size());
  std::vector<int> most_recent_by_level(while_nest_level_.size(), -1);
  int prev_nest_level = 0;
  int change_idx = -1;
  while_nest_level_change_.reserve(size);
  for (int i = 0; i < size; ++i) {
    int nest_level = while_nest_level_[i];
    if (nest_level != prev_nest_level) {
      prev_nest_level = nest_level;
      // Compute last change index by choosing the most recent instruction index
      // with smaller nesting level. Note that it may happen that even though
      // there were few different regions with other nest levels before, all of
      // then are same or bigger than this one, in which case we'll end up with
      // -1, e.g. if you got nest level 0 no need checking anything else.
      change_idx = -1;
      for (int smaller_level = 0; smaller_level < nest_level; smaller_level++) {
        change_idx = std::max(change_idx, most_recent_by_level[smaller_level]);
      }
    }
    most_recent_by_level[nest_level] = i;
    while_nest_level_change_.push_back(change_idx);
  }
  for (int i = 0; i <= max_while_nest_level; ++i) {
    while_execution_counts_.push_back(cost_analysis_.GetWhileNestMultiplier(i));
    // The binary searches below (Begin, LatestPrefetchStartTime,
    // PreferredPrefetchStartTime) rely on GetLogicalIntervalElapsed being
    // monotone in its start time, which holds iff the per level multiplier
    // is non-decreasing (the while execution count flag is >= 1, its default
    // is 5). A decreasing multiplier would silently change picked times
    // versus the old linear scans.
    DCHECK(i == 0 ||
           while_execution_counts_[i] >= while_execution_counts_[i - 1])
        << "while nest multiplier must be non-decreasing per level for the "
           "prefetch picker's binary searches";
  }
  // Index the logical times by computation nest level, so searches restricted
  // to "times at the same nest level as the prefetch end time" can binary
  // search instead of scanning every time in a range.
  int max_computation_nest_level = 0;
  for (int level : computation_nest_level_) {
    max_computation_nest_level = std::max(max_computation_nest_level, level);
  }
  positions_by_computation_nest_level_.resize(max_computation_nest_level + 1);
  for (int64_t time = 0;
       time < static_cast<int64_t>(computation_nest_level_.size()); ++time) {
    positions_by_computation_nest_level_[computation_nest_level_[time]]
        .push_back(time);
  }
}

float CostAnalysisPrefetchIntervalPicker::GetMaxElapsedInAlternateMemory(
    float async_copy_elapsed) const {
  return max_async_copy_elapsed_;
}

bool CostAnalysisPrefetchIntervalPicker::CanAllocateInAlternateMemoryNoCopy(
    int64_t shape_size, int64_t start_time, int64_t end_time) const {
  // Even though this method returns if we allow the buffer in alternate memory
  // _without_ asynchronous copies, calculate how long it would have taken to
  // copy it and compare it to the elapsed time in the logical interval.
  int64_t effective_size = size_override_ ? *size_override_ : shape_size;
  float async_copy_elapsed = static_cast<float>(effective_size) /
                             cost_analysis_.DefaultMemBandwidthBytesPerSecond(
                                 /*use_scaling_factor=*/true);
  float logical_interval_elapsed =
      GetLogicalIntervalElapsed(start_time, end_time);
  return GetMaxElapsedInAlternateMemory(async_copy_elapsed) >
         logical_interval_elapsed;
}

int64_t CostAnalysisPrefetchIntervalPicker::PreferredEvictionEndTime(
    const Shape& shape, int64_t start_time, int64_t latest_end_time) const {
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(
      size_override_ ? *size_override_
                     : cost_analysis_.GetShapeSizeBytes(shape));
  int64_t end_time;
  for (end_time = start_time + 1; end_time <= latest_end_time; ++end_time) {
    float logical_interval_elapsed =
        GetLogicalIntervalElapsed(start_time, end_time);
    if (logical_interval_elapsed >=
        (1 + kEvictionRetryMultiplier * retry_number_) *
            preferred_overlap_to_async_copy_ratio_ * async_copy_elapsed) {
      break;
    }
  }
  return end_time;
}

int64_t CostAnalysisPrefetchIntervalPicker::LatestPrefetchStartTime(
    int64_t shape_size, int64_t start_time, int64_t end_time,
    const HloUse* use) const {
  // Find the earliest time that satisfies max_overlap_to_async_copy_ratio_.
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(
      size_override_ ? *size_override_ : shape_size);
  // If there is a use, estimate the time we would save by having this op in
  // alternate memory.
  float inst_elapsed_reduction = 0.0f;
  if (use) {
    float elapsed_time =
        cost_analysis_.GetInstructionElapsed(*use->instruction);
    float elapsed_time_in_alternate_mem =
        cost_analysis_.GetInstructionElapsedInAlternateMemory(
            *use->instruction,
            /*operands_in_alternate_mem=*/
            {std::make_pair(use->operand_number, use->operand_index)},
            /*outputs_in_alternate_mem=*/{});
    inst_elapsed_reduction = elapsed_time - elapsed_time_in_alternate_mem;
  }
  int end_nest_level = computation_nest_level_[end_time];

  // Find the latest time we're allowed to start prefetching.
  float min_interval = min_overlap_to_async_copy_ratio_ * async_copy_elapsed;
  // GetLogicalIntervalElapsed(t, end_time) is monotonically non increasing in t
  // (see Begin()), so "the interval is long enough: elapsed + reduction >=
  // min_interval" holds for every t at or below a single boundary. Binary
  // search for the largest such t instead of walking backward over every time
  // whose interval is still too short (an O(range) scan when a view stretches a
  // source's live range across a whole loop body), then step down to the first
  // matching nest level exactly as the linear scan would. Decision preserving:
  // times above the boundary all have elapsed + reduction < min_interval, so
  // the linear scan would skip them regardless of nest level.
  int64_t threshold_lo = start_time;
  int64_t threshold_hi = end_time - 1;
  int64_t latest_prefetch_time = start_time - 1;
  while (threshold_lo <= threshold_hi) {
    int64_t mid = threshold_lo + (threshold_hi - threshold_lo) / 2;
    if (GetLogicalIntervalElapsed(mid, end_time) + inst_elapsed_reduction >=
        min_interval) {
      latest_prefetch_time = mid;
      threshold_lo = mid + 1;
    } else {
      threshold_hi = mid - 1;
    }
  }
  while (latest_prefetch_time >= start_time &&
         computation_nest_level_[latest_prefetch_time] != end_nest_level) {
    --latest_prefetch_time;
  }

  return latest_prefetch_time;
}

int64_t CostAnalysisPrefetchIntervalPicker::PreferredPrefetchStartTime(
    const Shape& shape, int64_t earliest_prefetch_start_time,
    int64_t latest_prefetch_start_time, int64_t prefetch_end_time) const {
  // Between the earliest and latest prefetch interval, find the interval
  // closest to the preferred interval and start iterating from there.
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(
      size_override_ ? *size_override_
                     : cost_analysis_.GetShapeSizeBytes(shape));
  float preferred_interval =
      preferred_overlap_to_async_copy_ratio_ * async_copy_elapsed;
  int end_nest_level = computation_nest_level_[prefetch_end_time];

  // This replaces a linear scan that visited every time in
  // (earliest_prefetch_start_time, latest_prefetch_start_time], computed the
  // logical interval elapsed for each (an O(range) walk that dominates when a
  // buffer's live range spans a whole loop body), and kept the earliest time
  // (with earliest_prefetch_start_time as the initial candidate, regardless
  // of its nest level) whose deviation |preferred - interval(t)| strictly
  // improved on the best so far, considering only times at the end time's
  // nest level. Because GetLogicalIntervalElapsed(t, end) is monotonically
  // non increasing in t (see Begin()), the deviation over the matching nest
  // level times is V shaped: its minimum is attained at the last matching
  // time whose interval is >= preferred (A) or at the first matching time
  // whose interval is < preferred (B). On ties the scan kept the earliest
  // time, which on the A side is the first time of the plateau of equal
  // intervals ending at A (every earlier matching time has a strictly larger
  // interval, hence a strictly larger deviation). So the scan's answer is
  // always one of: the initial candidate, the first time of A's plateau, or
  // B. Binary search those out of the sorted list of times at
  // end_nest_level. Decision preserving: the same float quantities are
  // compared with the same strictness, and candidates are considered in
  // ascending time order so equal deviations keep the earlier time.
  int64_t best_time = earliest_prefetch_start_time;
  float best_interval = GetLogicalIntervalElapsed(earliest_prefetch_start_time,
                                                  prefetch_end_time);
  auto interval_at = [&](int64_t time) {
    return GetLogicalIntervalElapsed(time, prefetch_end_time);
  };
  auto try_improve = [&](int64_t time) {
    float interval = interval_at(time);
    if (std::abs(preferred_interval - interval) <
        std::abs(preferred_interval - best_interval)) {
      best_interval = interval;
      best_time = time;
    }
  };
  if (end_nest_level <
      static_cast<int>(positions_by_computation_nest_level_.size())) {
    const std::vector<int64_t>& level_positions =
        positions_by_computation_nest_level_[end_nest_level];
    auto lo_it =
        std::upper_bound(level_positions.begin(), level_positions.end(),
                         earliest_prefetch_start_time);
    auto hi_it = std::upper_bound(lo_it, level_positions.end(),
                                  latest_prefetch_start_time);
    if (lo_it != hi_it) {
      // Intervals are non increasing over these times, so "interval >=
      // preferred" holds on a prefix; partition_point finds B, the first
      // matching time whose interval is < preferred.
      auto b_it = std::partition_point(lo_it, hi_it, [&](int64_t time) {
        return interval_at(time) >= preferred_interval;
      });
      if (b_it != lo_it) {
        // A (= b_it - 1) exists; find the first time of A's plateau.
        float a_interval = interval_at(*(b_it - 1));
        auto plateau_first_it = std::partition_point(
            lo_it, b_it,
            [&](int64_t time) { return interval_at(time) > a_interval; });
        try_improve(*plateau_first_it);
      }
      if (b_it != hi_it) {
        try_improve(*b_it);
      }
    }
  }
  return best_time;
}

int64_t CostAnalysisPrefetchIntervalPicker::LatestPrefetchEndTime(
    int64_t original_prefetch_end_time,
    int64_t proposed_prefetch_end_time) const {
  // Iterate towards the beginning until we find a suitable end time that is the
  // same while nest level as the original prefetch end time.
  int64_t original_nest_level =
      computation_nest_level_[original_prefetch_end_time];
  int64_t new_prefetch_end_time;
  for (new_prefetch_end_time = proposed_prefetch_end_time;
       computation_nest_level_[new_prefetch_end_time] != original_nest_level;
       --new_prefetch_end_time) {
  }
  return new_prefetch_end_time;
}

int64_t CostAnalysisPrefetchIntervalPicker::EstimatedPrefetchEndTime(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(
      size_override_ ? *size_override_
                     : cost_analysis_.GetShapeSizeBytes(shape));
  int64_t estimated_end_time;
  for (estimated_end_time = start_time + 1; estimated_end_time < end_time;
       ++estimated_end_time) {
    float interval = GetLogicalIntervalElapsed(start_time, estimated_end_time);
    if (interval >= async_copy_elapsed) {
      break;
    }
  }
  return estimated_end_time;
}

void CostAnalysisPrefetchIntervalPicker::Begin(
    const HloUse& use, int64_t start_time, int64_t end_time,
    std::optional<int64_t> preferred_time) {
  const Shape& shape = ShapeUtil::GetSubshape(
      use.instruction->operand(use.operand_number)->shape(), use.operand_index);
  int64_t shape_size = size_override_ ? *size_override_
                                      : cost_analysis_.GetShapeSizeBytes(shape);
  // Find the earliest time that satisfies max_overlap_to_async_copy_ratio_.
  async_copy_elapsed_ = cost_analysis_.GetAsyncCopyElapsed(shape_size);
  // Estimate the time we would save by having this op in alternate memory.
  float elapsed_time = cost_analysis_.GetInstructionElapsed(*use.instruction);
  float elapsed_time_in_alternate_mem =
      cost_analysis_.GetInstructionElapsedInAlternateMemory(
          *use.instruction, /*operands_in_alternate_mem=*/
          {std::make_pair(use.operand_number, use.operand_index)},
          /*outputs_in_alternate_mem=*/{});
  inst_elapsed_reduction_ = elapsed_time - elapsed_time_in_alternate_mem;
  end_logical_time_ = end_time;
  int end_nest_level = computation_nest_level_[end_logical_time_];

  // Find the latest time we're allowed to start prefetching.
  float min_interval = min_overlap_to_async_copy_ratio_ * async_copy_elapsed_;
  latest_prefetch_time_ =
      LatestPrefetchStartTime(shape_size, start_time, end_time, &use);

  // Find the earliest time we're allowed to start prefetching.
  float max_interval = GetMaxElapsedInAlternateMemory(async_copy_elapsed_);
  // GetLogicalIntervalElapsed(t, end_logical_time_) is monotonically non
  // increasing in t: its numerator (a prefix sum difference) shrinks as t grows
  // and its divisor (the while execution count of the interval's min nest
  // level) does not shrink, so the predicate "the async copy fits, elapsed <=
  // max_interval" flips from false to true exactly once as t increases. Binary
  // search for that first fitting time instead of walking every earlier time
  // (an O(range) scan that dominates when a view stretches a source's live
  // range across a whole loop body). This returns the IDENTICAL
  // earliest_prefetch_time_ the linear scan below would: for any t before the
  // fit, elapsed > max_interval so the scan would advance regardless of nest
  // level; from the first fitting time the only remaining reason to advance is
  // a nest level mismatch, which the trailing loop reproduces exactly.
  {
    int64_t lo = start_time;
    int64_t hi = latest_prefetch_time_;
    while (lo < hi) {
      int64_t mid = lo + (hi - lo) / 2;
      if (GetLogicalIntervalElapsed(mid, end_logical_time_) <= max_interval) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    earliest_prefetch_time_ = lo;
    while (earliest_prefetch_time_ < latest_prefetch_time_ &&
           computation_nest_level_[earliest_prefetch_time_] != end_nest_level) {
      ++earliest_prefetch_time_;
    }
  }
  if (earliest_prefetch_time_ > latest_prefetch_time_) {
    // There is no available prefetch interval for the given start and end
    // times. Set the iterators accordingly to ensure Done() returns true.
    increasing_prefetch_time_iterator_ = earliest_prefetch_time_;
    decreasing_prefetch_time_iterator_ = latest_prefetch_time_;
    CHECK(Done());
    return;
  }

  int64_t starting_prefetch_time;
  if (preferred_time && *preferred_time <= latest_prefetch_time_) {
    starting_prefetch_time = *preferred_time;
  } else {
    starting_prefetch_time =
        PreferredPrefetchStartTime(shape, earliest_prefetch_time_,
                                   latest_prefetch_time_, end_logical_time_);
  }
  float preferred_interval =
      preferred_overlap_to_async_copy_ratio_ * async_copy_elapsed_;
  VLOG(4) << "Interval min/max/preferred = " << min_interval << " "
          << max_interval << " " << preferred_interval
          << " prefetch time earliest/latest/starting = "
          << earliest_prefetch_time_ << " " << latest_prefetch_time_ << " "
          << starting_prefetch_time;

  increasing_prefetch_time_iterator_ = starting_prefetch_time;
  decreasing_prefetch_time_iterator_ = starting_prefetch_time;
  using_increasing_prefetch_time_iterator_ = true;
  // Since both iterators start at the same position, call Next() once to
  // advance one of the iterators.
  Next();
}

int64_t CostAnalysisPrefetchIntervalPicker::Next() {
  CHECK(!Done()) << "Prefetch interval picker's Next() is called even though "
                    "Done() is false";
  if (using_increasing_prefetch_time_iterator_) {
    int64_t prefetch_time = increasing_prefetch_time_iterator_++;
    while (increasing_prefetch_time_iterator_ <= latest_prefetch_time_ &&
           computation_nest_level_[increasing_prefetch_time_iterator_] !=
               computation_nest_level_[end_logical_time_]) {
      ++increasing_prefetch_time_iterator_;
    }
    if (decreasing_prefetch_time_iterator_ >= earliest_prefetch_time_) {
      using_increasing_prefetch_time_iterator_ = false;
    }
    return prefetch_time;
  } else {
    int64_t prefetch_time = decreasing_prefetch_time_iterator_--;
    // As a compilation time optimization, reduce the number of intervals that
    // this prefetch interval picker returns. When we run out of the increasing
    // prefetch time iterator, only explore up to
    // kNumExploredDecreasingIntervals intervals. To do that, calculate the
    // 1/kNumExploredDecreasingIntervals of the elapsed time between the
    // earliest prefetch time and the use, and decrement the iterator until the
    // prefetch elapsed time is at least as large as this target value. This
    // allows us to reduce the number of expensive heap fit and resource checks
    // when the graph consists of a large number of fast-executing HLOs.
    //
    // Shown pictorially, assuming kNumExploredDecreasingIntervals = 3 and the
    // numbers indicating the elapsed time of the HLOs, only the indicated
    // options for prefetch start time would be explored:
    //
    //    ---1---1---3---1---1---1---1---0---0---0---0---1---5---X
    //     ^           ^                                   ^     ^
    //  Option3     Option2                             Option1 Use
    // (Earliest)
    float next_target_interval_elapsed = 0;
    if (increasing_prefetch_time_iterator_ > latest_prefetch_time_) {
      next_target_interval_elapsed =
          GetLogicalIntervalElapsed(prefetch_time, end_logical_time_) +
          (GetLogicalIntervalElapsed(earliest_prefetch_time_,
                                     end_logical_time_) /
           kNumExploredDecreasingIntervals);
      VLOG(3) << "Next target interval elapsed: "
              << next_target_interval_elapsed;
    }
    while (decreasing_prefetch_time_iterator_ >= earliest_prefetch_time_ &&
           (computation_nest_level_[decreasing_prefetch_time_iterator_] !=
                computation_nest_level_[end_logical_time_] ||
            GetLogicalIntervalElapsed(decreasing_prefetch_time_iterator_,
                                      end_logical_time_) <
                next_target_interval_elapsed)) {
      --decreasing_prefetch_time_iterator_;
    }
    if (increasing_prefetch_time_iterator_ <= latest_prefetch_time_) {
      using_increasing_prefetch_time_iterator_ = true;
    }
    return prefetch_time;
  }
}

bool CostAnalysisPrefetchIntervalPicker::Done() const {
  return increasing_prefetch_time_iterator_ > latest_prefetch_time_ &&
         decreasing_prefetch_time_iterator_ < earliest_prefetch_time_;
}

int64_t CostAnalysisPrefetchIntervalPicker::latest_time() const {
  return latest_prefetch_time_;
}

void CostAnalysisPrefetchIntervalPicker::SetRetryNumber(int retry_number) {
  retry_number_ = retry_number;
}

int CostAnalysisPrefetchIntervalPicker::GetMinWhileNestLevel(
    int64_t start_time, int64_t end_time) const {
  int min_nest_level =
      std::min(while_nest_level_[start_time], while_nest_level_[end_time]);
  int change_idx = while_nest_level_change_[end_time];
  while (change_idx >= start_time) {
    min_nest_level = std::min(min_nest_level, while_nest_level_[change_idx]);
    change_idx = while_nest_level_change_[change_idx];
  }
  return min_nest_level;
}

float CostAnalysisPrefetchIntervalPicker::GetLogicalIntervalElapsed(
    int64_t start_time, int64_t end_time) const {
  CHECK_LE(start_time, end_time);
  if (start_time == end_time) {
    return 0.0;
  }
  if (start_time < 0) {
    start_time = 0;
  }
  // Since elapsed_time_cumsum_ is already weighed by the while loop nesting
  // level, normalize the elapsed time by dividing with the nesting factor of
  // the interval (start and end times).
  int interval_while_nest_level = GetMinWhileNestLevel(start_time, end_time);
  return (elapsed_time_cumsum_[end_time - 1] -
          elapsed_time_cumsum_[start_time]) /
         while_execution_counts_[interval_while_nest_level];
}

std::string CostAnalysisPrefetchIntervalPicker::ToDebugString() const {
  int current_logical_prefetch_time = using_increasing_prefetch_time_iterator_
                                          ? increasing_prefetch_time_iterator_
                                          : decreasing_prefetch_time_iterator_;
  float logical_interval_elapsed = GetLogicalIntervalElapsed(
      current_logical_prefetch_time, end_logical_time_);
  return absl::StrCat(
      "Async copy elapsed (s) = ", async_copy_elapsed_,
      ", inst elapsed reduction (s) = ", inst_elapsed_reduction_,
      ", logical interval elapsed (s) = ", logical_interval_elapsed,
      ", interval = (", current_logical_prefetch_time, ", ", end_logical_time_,
      ")");
}

std::string CostAnalysisPrefetchIntervalPicker::ToNoCopyDebugString(
    const Shape& shape, int64_t start_time, int64_t end_time) const {
  float async_copy_elapsed = cost_analysis_.GetAsyncCopyElapsed(
      size_override_ ? *size_override_
                     : cost_analysis_.GetShapeSizeBytes(shape));
  float logical_interval_elapsed =
      GetLogicalIntervalElapsed(start_time, end_time);
  return absl::StrCat(
      "Async copy elapsed (s) = ", async_copy_elapsed,
      ", logical interval elapsed (s) = ", logical_interval_elapsed);
}

std::optional<float>
CostAnalysisPrefetchIntervalPicker::BufferIntervalAlternateMemoryBenefit(
    const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval)
    const {
  return cost_analysis_.GetMemoryBoundedness(interval);
}

}  // namespace memory_space_assignment
}  // namespace xla
