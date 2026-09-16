/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
License for the specific language governing permissions and limitations under
the License.
==============================================================================*/
#ifndef XLA_HLO_UTILS_HLO_LIVE_RANGE_H_
#define XLA_HLO_UTILS_HLO_LIVE_RANGE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"

namespace xla {

// Class which computes live range of the output buffers of HLOs and their
// interference by flattening all computations. The live range is only available
// when all global computations (while, if, call, etc) have total order
// sequential orders.
class HloLiveRange {
 public:
  // Constructs a hlo live range object for the given module and computation
  // assuming the given HLO instruction ordering.
  static absl::StatusOr<std::unique_ptr<HloLiveRange>> Run(
      const HloSchedule& schedule, const HloAliasAnalysis& alias_analysis,
      const HloComputation* computation, bool module_scoped_analysis = true,
      absl::flat_hash_set<absl::string_view> execution_threads = {});

  // Returns all HloValues defined by this instruction.
  static std::vector<const HloValue*> GetValuesDefined(
      const HloInstruction* instruction, const HloDataflowAnalysis& dataflow);

  // Returns the distinct physical HloBuffers newly allocated/defined by this
  // instruction (excluding buffers forwarded or aliased from operands).
  static std::vector<const HloBuffer*> GetBuffersDefined(
      const HloInstruction* instruction,
      const HloAliasAnalysis& alias_analysis);

  // Returns the total bytes defined by this instruction according to size_fn.
  // Returns 0 for parameter instructions (parameters are attributed to
  // computation start).
  static int64_t GetBytesDefined(const HloInstruction* instruction,
                                 const HloAliasAnalysis& alias_analysis,
                                 const BufferValue::SizeFunction& size_fn);

  // Returns the distinct physical HloBuffers read by the operands of this
  // instruction.
  static std::vector<const HloBuffer*> GetBuffersUsed(
      const HloInstruction* instruction,
      const HloAliasAnalysis& alias_analysis);

  // Returns the total number of instruction reads across the computation for
  // all values contained in this buffer. If computation is null, counts across
  // all computations.
  static int32_t GetTotalUsers(const HloBuffer& buffer,
                               const HloComputation* computation = nullptr);

  // Returns the total parameter bytes allocated at the start of the computation
  // (matching HloLiveRange parameter attribution).
  static int64_t GetParameterBytesAtStart(
      const HloComputation& computation, const HloAliasAnalysis& alias_analysis,
      const BufferValue::SizeFunction& size_fn);

  // Returns true if any value in this buffer lives out of the computation.
  static bool BufferLivesOut(const HloBuffer& buffer,
                             const HloAliasAnalysis& alias_analysis,
                             const HloComputation* computation = nullptr);

  // LogicalTime represents the time in a virtual clock. Each instruction has
  // one monotonically increasing logical time assigned according to the
  // schedule.
  using LogicalTime = int64_t;

  struct LiveRangeBounds {
    LogicalTime start;
    LogicalTime end;

    // The buffer can hold multiple instructions during its life time (each
    // tenant exclusively owns the buffer at any given time). `end_position`
    // represents the last instruction that the buffer holds.
    HloPosition end_position;

    bool friend operator==(const LiveRangeBounds& a, const LiveRangeBounds& b) {
      return a.start == b.start && a.end == b.end;
    }
    bool friend operator!=(const LiveRangeBounds& a, const LiveRangeBounds& b) {
      return !(a == b);
    }
  };

  std::string ToString() const;

  const HloInstructionSequence& flattened_instruction_sequence() const {
    return flattened_instruction_sequence_;
  }

  // Returns the map from instruction to the end time of that instruction.
  const absl::flat_hash_map<const HloInstruction*, LogicalTime>&
  instruction_schedule() const {
    return instruction_schedule_;
  }

  // Returns the map from a hlo value to the definition time of that hlo value.
  const absl::flat_hash_map<const HloValue*, LiveRangeBounds>&
  buffer_live_ranges() const {
    return buffer_live_ranges_;
  }

  absl::flat_hash_map<const HloValue*, LiveRangeBounds>& buffer_live_ranges() {
    return buffer_live_ranges_;
  }

  // Returns the map from a computation and its time span in the schedule.
  const absl::flat_hash_map<const HloComputation*, LiveRangeBounds>&
  computation_span_times() const {
    return computation_span_times_;
  }

  // Returns the time stamp of the end of the program.
  LogicalTime schedule_end_time() const {
    return flattened_instruction_sequence_.size();
  }

  // Returns whether hlo live range is available on this entire module. Hlo live
  // range is not available if the module is partially ordered.
  bool total_order_scheduled() const { return total_order_scheduled_; }

 private:
  explicit HloLiveRange(
      const HloSchedule& schedule, const HloAliasAnalysis& alias_analysis,
      bool module_scoped_analysis,
      absl::flat_hash_set<absl::string_view> execution_threads = {})
      : schedule_(schedule),
        alias_analysis_(alias_analysis),
        module_scoped_analysis_(module_scoped_analysis),
        execution_threads_(std::move(execution_threads)) {}

  // FlattenSchedule walks through the instructions in `computation`, and
  // recurse into each called computations in module_scoped_analysis mode. As it
  // walks it also tracks down the ordinal number of each instruction in the
  // schedule and store it in the `instruction_schedule` and
  // 'flattened_instruction_sequence`. async_context contains the asynchronous
  // computation that this computation is in, if any. When this value is
  // non-null, it means that this computation is called by an async op or
  // another op in an asynchronous context.
  absl::Status FlattenSchedule(const HloComputation& computation,
                               const HloComputation* async_context = nullptr);

  // Computes the end of the live range of an HloValue. Returns the end time and
  // the position where the live range ends.
  std::pair<LogicalTime, HloPosition> ComputeValueLiveRangeEnd(
      const HloValue& value, LogicalTime defining_instruction_end_time) const;

  // Returns the time of the last use of a value.
  LogicalTime GetLastUsageTime(const HloValue& value) const;

  // Based on the flattened schedule, calculate the start and end of each
  // buffer.
  void CalculateBufferStartEndMap();

  // The aliased buffers could have overlapping live ranges.
  // NormalizeAliasedBuffers normalizes the buffer such that each alias buffer
  // has disjoint live range while keeping the live range union the same. This
  // avoid double counting aliased buffer sizes.
  //
  // Before(buffer1 and 2 are aliased):
  //
  //           +----+          live range of buffer1
  //   +------------------+    live range of buffer2
  //
  // After:
  //
  //           +----------+    live range of buffer1
  //   +-------+               live range of buffer2
  //
  // Before(buffer1 and 2 are aliased):
  //
  //           +----------+    live range of buffer1
  //   +------------+          live range of buffer2
  //
  // After:
  //
  //           +----------+    live range of buffer1
  //   +-------+               live range of buffer2
  //
  // Before(buffer1 and 2 are aliased):
  //
  //           +----------+    live range of buffer1
  //   +---+                   live range of buffer2
  //
  // After(unchanged):
  //
  //           +----------+    live range of buffer1
  //   +---+                   live range of buffer2
  //
  // As another example, imagine we have the following code sequence with live
  // ranges of each while-aliased buffers:
  //
  //                     a      p1    p2    e     b
  // a = ...             +
  //                     |
  // {                   |
  //   p1 = param        |       +
  //   ROOT true         |       |
  // }                   |       +
  // { // body           |
  //   p2 = param        +             +
  //   c = p2 + 1                      +
  //   d = c + 1
  //   ROOT e = d + 1                       +
  // }                                      |
  //                                        |
  // b = while (a)                          +     +
  //                                              |
  // f = b + 1                                    +
  //
  // After normalization it becomes:
  //
  //                     a      p1    p2    e     b
  // a = ...             +
  //                     |
  // {                   |
  //   p1 = param        +       +
  //   ROOT true                 |
  // }                           |
  // { // body                   |
  //   p2 = param                +     +
  //   c = p2 + 1                      +
  //   d = c + 1
  //   ROOT e = d + 1                       +
  // }                                      |
  //                                        |
  // b = while (a)                          +     +
  //                                              |
  // f = b + 1                                    +
  //
  // Note there is no overlap of live ranges after normalization.
  void NormalizeAliasedBuffers();

  LogicalTime ComputePeakMemoryMoment() const;

  const HloSchedule& schedule_;
  const HloAliasAnalysis& alias_analysis_;
  bool module_scoped_analysis_;
  bool total_order_scheduled_ = true;

  HloInstructionSequence flattened_instruction_sequence_;
  absl::flat_hash_map<const HloInstruction*, LogicalTime> instruction_schedule_;
  absl::flat_hash_map<const HloComputation*, LiveRangeBounds>
      computation_span_times_;
  absl::flat_hash_map<const HloValue*, LiveRangeBounds> buffer_live_ranges_;
  absl::flat_hash_map<const HloComputation*, const HloComputation*>
      computations_in_async_context_;
  absl::flat_hash_set<absl::string_view> execution_threads_;
};

// Returns the latest schedule time at which `view` (a value colored
// `view_color`, e.g. memory_space_assignment::Options::dus_view_color or
// BufferAssigner::Options::dus_view_color) still has its underlying storage
// read through it: the max schedule time over the transitive closure of the
// view's readers, following users that are themselves view colored. A view
// is an address into another buffer with no storage of its own, so that
// buffer must stay reserved until this time.
//
// REQUIRES: view->shape().IsTuple() == false.
int64_t ViewExtendedTransitiveUseTime(
    const HloInstruction* view, int64_t view_color,
    const absl::flat_hash_map<const HloInstruction*, int64_t>&
        instruction_schedule);

// Extends, in `hlo_live_range`, the live range end of every value used as
// the base (operand 0) of a view to the view's last transitive reader (see
// ViewExtendedTransitiveUseTime). The view is an address into the base's
// buffer with no storage of its own, so every consumer of liveness that can
// recycle or overlap storage (allocation reuse, heap simulation) must see
// the base held live until its last reader through the view. Values are
// visited in `dataflow_analysis.values()` order (the analysis `hlo_live_range`
// was built from).
void ExtendViewBaseLiveRanges(HloLiveRange* hlo_live_range,
                              const HloDataflowAnalysis& dataflow_analysis,
                              int64_t view_color);

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_LIVE_RANGE_H_
