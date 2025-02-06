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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_PREFETCH_INTERVAL_PICKER_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_PREFETCH_INTERVAL_PICKER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/shape.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

// Abstract base class that memory space assignment uses to pick prefetch
// intervals.
class PrefetchIntervalPicker {
 public:
  PrefetchIntervalPicker() = default;
  virtual ~PrefetchIntervalPicker() = default;

  // Returns true if the buffer can be allocated in alternate memory space
  // without any copies (prefetches).
  virtual bool CanAllocateInAlternateMemoryNoCopy(const Shape& shape,
                                                  int64_t start_time,
                                                  int64_t end_time) const = 0;

  // Returns the preferred end time for an eviction that starts at a given time
  // and must end by the given end time.
  virtual int64_t PreferredEvictionEndTime(const Shape& shape,
                                           int64_t start_time,
                                           int64_t latest_end_time) const = 0;

  // Returns the latest time that a prefetch can start.
  virtual int64_t LatestPrefetchStartTime(const Shape& shape,
                                          int64_t start_time, int64_t end_time,
                                          const HloUse* use) const = 0;

  // Returns the preferred time that a prefetch can start.
  virtual int64_t PreferredPrefetchStartTime(
      const Shape& shape, int64_t earliest_prefetch_start_time,
      int64_t latest_prefetch_start_time, int64_t prefetch_end_time) const = 0;

  // Returns the latest time that a prefetch can end that is less than or equal
  // to proposed_prefetch_end_time.
  virtual int64_t LatestPrefetchEndTime(
      int64_t original_prefetch_end_time,
      int64_t proposed_prefetch_end_time) const {
    return proposed_prefetch_end_time;
  }

  // Returns the estimated end time of a prefetch that starts at the given time.
  virtual int64_t EstimatedPrefetchEndTime(const Shape& shape,
                                           int64_t start_time,
                                           int64_t end_time) const = 0;

  // Returns the elapsed time in seconds between the logical interval that
  // corresponds to the instruction schedule.
  virtual float GetLogicalIntervalElapsed(int64_t start_time,
                                          int64_t end_time) const = 0;

  // Begins the iterator for the first start time of the prefetch.
  virtual void Begin(const HloUse& use, int64_t start_time, int64_t end_time,
                     std::optional<int64_t> preferred_time) = 0;

  // Advances the start time of the prefetch and returns that value.
  virtual int64_t Next() = 0;

  // Returns true if the available prefetch intervals have been exhausted.
  virtual bool Done() const = 0;

  // Returns the latest time the prefetch interval picker will have pick.
  virtual int64_t latest_time() const = 0;

  // The retry number can be used to modify the interval picking policies. The
  // first attempt will have a retry_number of 0, then 1, etc.
  virtual void SetRetryNumber(int retry_number) {
    retry_number_ = retry_number;
  }
  int retry_number() const { return retry_number_; }

  // Returns a debug string for the current state of the prefetch interval
  // picker.
  virtual std::string ToDebugString() const = 0;

  // Returns a debug string for no-copy allocation.
  virtual std::string ToNoCopyDebugString(const Shape& shape,
                                          int64_t start_time,
                                          int64_t end_time) const = 0;

  // Prefetch interval pickers may return a value corresponding to the benefit
  // of placing the BufferInterval in the alternate memory. The larger value,
  // the more beneficial.
  virtual std::optional<float> BufferIntervalAlternateMemoryBenefit(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval)
      const {
    return std::nullopt;
  }

 protected:
  const absl::flat_hash_map<const HloInstruction*, int64_t>*
      instruction_schedule_ = nullptr;
  int retry_number_ = 0;
};

// Prefetch interval picker that uses instruction count to overlap asynchronous
// copies with independent computation. The min and max overlap counts describe
// the number of independent HLOs overlapped while a value is being prefetched
// into the alternate memory (between CopyStart and CopyDone HLO instructions).
// max_overlap_count attempts to prevent bringing tensors into the alternate
// memory too eagerly and hence occupying the space for other tensors which
// might use it.  min_overlap_count attempts to prevent cases where tensors are
// prefetched into the alternate memory without sufficient time for the copy to
// take place.  In those cases, it's just better to keep the tensor in the
// default memory instead of hurting the critical path with this copy that
// likely won't finish in time.
class InstructionCountPrefetchIntervalPicker : public PrefetchIntervalPicker {
 public:
  InstructionCountPrefetchIntervalPicker(int64_t min_overlap_count,
                                         int64_t max_overlap_count)
      : min_overlap_count_(min_overlap_count),
        max_overlap_count_(max_overlap_count) {}

  bool CanAllocateInAlternateMemoryNoCopy(const Shape& shape,
                                          int64_t start_time,
                                          int64_t end_time) const override;

  int64_t PreferredEvictionEndTime(const Shape& shape, int64_t start_time,
                                   int64_t latest_end_time) const override;

  int64_t LatestPrefetchStartTime(const Shape& shape, int64_t start_time,
                                  int64_t end_time,
                                  const HloUse* use) const override;

  int64_t PreferredPrefetchStartTime(const Shape& shape,
                                     int64_t earliest_prefetch_start_time,
                                     int64_t latest_prefetch_start_time,
                                     int64_t prefetch_end_time) const override;

  int64_t EstimatedPrefetchEndTime(const Shape& shape, int64_t start_time,
                                   int64_t end_time) const override;
  float GetLogicalIntervalElapsed(int64_t start_time,
                                  int64_t end_time) const override;

  void Begin(const HloUse& use, int64_t start_time, int64_t end_time,
             std::optional<int64_t> preferred_time) override;

  int64_t Next() override;
  bool Done() const override;

  int64_t latest_time() const override;

  std::string ToDebugString() const override;
  std::string ToNoCopyDebugString(const Shape& shape, int64_t start_time,
                                  int64_t end_time) const override;

 private:
  int64_t min_overlap_count_;
  int64_t max_overlap_count_;
  int64_t end_time_;
  int64_t current_prefetch_time_;
};

// Prefetch interval picker that uses cost analysis to overlap asynchronous
// copies with independent computation. It uses min (independent computation
// duration) / (asynchronous copy duration) ratio to guide whether the prefetch
// is within the lower bound. For the upper bound, it restricts the maximum
// duration that a buffer may occupy the alternate memory space as a multiple of
// the time it would take to copy a buffer that is the size of the alternate
// memory. It starts with the preferred ratio in Begin() and works its way for
// alternately earlier and later prefetches until hitting min and max ratios.
// The value for buffer size for max async copy is a mechanism to prevent
// copying small buffers between the two memories unnecessarily. For calculating
// the max time that the buffer can reside in alternate memory, we use the
// larger of this value and the actual size of the buffer. A shape override can
// also be provided which causes the interval picker to use that shape for async
// copy durations instead of the actual shape of the copy.
class CostAnalysisPrefetchIntervalPicker : public PrefetchIntervalPicker {
 public:
  CostAnalysisPrefetchIntervalPicker(
      const CostAnalysis& cost_analysis, float min_overlap_to_async_copy_ratio,
      float preferred_overlap_to_async_copy_ratio,
      float max_overlap_to_mem_size_async_copy_ratio, int64_t mem_size_bytes,
      const Shape* shape_override = nullptr);

  bool CanAllocateInAlternateMemoryNoCopy(const Shape& shape,
                                          int64_t start_time,
                                          int64_t end_time) const override;

  int64_t PreferredEvictionEndTime(const Shape& shape, int64_t start_time,
                                   int64_t latest_end_time) const override;

  int64_t LatestPrefetchEndTime(
      int64_t original_prefetch_end_time,
      int64_t proposed_prefetch_end_time) const override;

  int64_t LatestPrefetchStartTime(const Shape& shape, int64_t start_time,
                                  int64_t end_time,
                                  const HloUse* use) const override;

  int64_t PreferredPrefetchStartTime(const Shape& shape,
                                     int64_t earliest_prefetch_start_time,
                                     int64_t latest_prefetch_start_time,
                                     int64_t prefetch_end_time) const override;

  int64_t EstimatedPrefetchEndTime(const Shape& shape, int64_t start_time,
                                   int64_t end_time) const override;
  float GetLogicalIntervalElapsed(int64_t start_time,
                                  int64_t end_time) const override;

  void Begin(const HloUse& use, int64_t start_time, int64_t end_time,
             std::optional<int64_t> preferred_time) override;

  int64_t Next() override;
  bool Done() const override;

  int64_t latest_time() const override;

  void SetRetryNumber(int retry_number) override;

  std::string ToDebugString() const override;
  std::string ToNoCopyDebugString(const Shape& shape, int64_t start_time,
                                  int64_t end_time) const override;

  std::optional<float> BufferIntervalAlternateMemoryBenefit(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval)
      const override;

 private:
  // Finds the minimum nest level in the given interval.
  int GetMinWhileNestLevel(int64_t start_time, int64_t end_time) const;

  // Given the elapsed time to copy this buffer to the alternate memory, returns
  // the longest time that this buffer may reside in the alternate memory space.
  float GetMaxElapsedInAlternateMemory(float async_copy_elapsed) const;

  // For each instruction in the flattened schedule, maintain their elapsed time
  // (in cumulative sum) and while nesting level.
  std::vector<float> elapsed_time_cumsum_;
  std::vector<int> while_nest_level_;
  std::vector<int> computation_nest_level_;
  // Maintain the index of the most recent (before this instruction) nest level
  // change in order to efficiently determine the minimum nest level in an
  // interval.
  std::vector<int> while_nest_level_change_;

  const CostAnalysis& cost_analysis_;
  float min_overlap_to_async_copy_ratio_;
  float preferred_overlap_to_async_copy_ratio_;
  float max_async_copy_elapsed_;
  float async_copy_elapsed_;
  float inst_elapsed_reduction_;
  int64_t end_logical_time_;
  int64_t earliest_prefetch_time_;
  int64_t latest_prefetch_time_;
  bool using_increasing_prefetch_time_iterator_ = true;
  int64_t increasing_prefetch_time_iterator_;
  int64_t decreasing_prefetch_time_iterator_;

  std::vector<float> while_execution_counts_;
  // Shape override is used to override the shape of the shape of the async copy
  // to treat all async copies the same duration. Having an override forces
  // prefetches to be scheduled roughly in FIFO order.
  std::optional<Shape> shape_override_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_PREFETCH_INTERVAL_PICKER_H_
