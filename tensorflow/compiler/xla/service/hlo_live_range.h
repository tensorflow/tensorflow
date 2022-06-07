/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LIVE_RANGE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LIVE_RANGE_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// Class which computes live range of the output buffers of HLOs and their
// interference by flattening all computations. The live range is only available
// when all global computations (while, if, call, etc) have total order
// sequential orders.
class HloLiveRange {
 public:
  // Constructs a hlo live range object for the given module and computation
  // assuming the given HLO instruction ordering.
  static StatusOr<std::unique_ptr<HloLiveRange>> Run(
      const HloSchedule& schedule, const HloAliasAnalysis& alias_analysis,
      const HloComputation* computation, bool module_scoped_analysis = true);

  // LogicalTime represents the time in a virtual clock. Each instruction has
  // one monotonically increasing logical time assigned according to the
  // schedule.
  using LogicalTime = int64_t;

  struct TimeBound {
    LogicalTime start;
    LogicalTime end;
    // The buffer can hold multiple instructions during its life time (each
    // tenant exclusively owns the buffer at any given time). `end_instruction`
    // represents the last instruction that the buffer holds.
    HloPosition end_position;

    bool friend operator==(const TimeBound& a, const TimeBound& b) {
      return a.start == b.start && a.end == b.end;
    }
    bool friend operator!=(const TimeBound& a, const TimeBound& b) {
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
  const absl::flat_hash_map<const HloValue*, TimeBound>& buffer_live_ranges()
      const {
    return buffer_live_ranges_;
  }

  absl::flat_hash_map<const HloValue*, TimeBound>& buffer_live_ranges() {
    return buffer_live_ranges_;
  }

  // Returns the map from a computation and its time span in the schedule.
  const absl::flat_hash_map<const HloComputation*, TimeBound>&
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
  explicit HloLiveRange(const HloSchedule& schedule,
                        const HloAliasAnalysis& alias_analysis,
                        bool module_scoped_analysis)
      : schedule_(schedule),
        alias_analysis_(alias_analysis),
        module_scoped_analysis_(module_scoped_analysis) {}

  // FlattenSchedule walks through the instructions in `computation`, and
  // recurse into each called computations in module_scoped_analysis mode. As it
  // walks it also tracks down the ordinal number of each instruction in the
  // schedule and store it in the `instruction_schedule` and
  // 'flattened_instruction_sequence`.
  void FlattenSchedule(const HloComputation& computation);

  // Returns the last position of a value.
  TimeBound GetLastPosition(const HloValue& value,
                            LogicalTime definition_end_time) const;

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
  absl::flat_hash_map<const HloComputation*, TimeBound> computation_span_times_;
  absl::flat_hash_map<const HloValue*, TimeBound> buffer_live_ranges_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LIVE_RANGE_H_
