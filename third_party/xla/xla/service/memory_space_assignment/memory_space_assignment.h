/* Copyright 2019 The OpenXLA Authors.

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

/*
Quick reference

This section is meant as to be a quick reference for getting the gist of
commonly used terminology in the code and logging. Please see the code for more
details.

General concepts

  - Time: In MSA, time typically refers to an index into the flattened
    instruction schedule.

  - Cross-program prefetch: Cross-program prefetched tensors are copied from
    memory to alternate the first time a program executes, like usual
    prefetches. MSA keeps these buffers alive in alternate memory at the end of
    the program, such that if the same program is executed again, these tensors
    would not need to be prefetched again.

Classes

  - HloPosition (Hlo dataflow analysis concept): Identifies a tensor referenced
    in an instruction's output. Defined by <instruction, shape index>.

  - HloValue (Hlo dataflow analysis concept): The value of a tensor. Each
    HloValue is represented by a collection of HloPositions. Exactly 1 of those
    positions is the HloValue's defining position, i.e., the point in code where
    the value is created/written. The rest of the positions pertain to read-only
    uses of the value.
    * Example: A tensor that is inserted in a Tuple has 2 HloPositions, one for
      the instruction that creates the tensor, and one indexing into the Tuple
      instruction result.
    * The read-only positions of an HloValue should not be confused with
      HloUses. Read-only positions are references to the HloValue in the output
      of an instruction. Uses are references to an HloValue in the input of an
      instruction.
    * Dataflow analysis assigns HloValues for the instructions in computations
      pertaining to while loops, conditionals, and call ops. However, it does
      not assign HloValues to the computations pertaining to instructions with
      "call" semantics (e.g., fusions, reduce, and custom-call) because those
      computations are treated as black boxes.
    * If a while loop does not modify an input tensor, that tensor will be
      assigned 1 HloValue that lasts from its creation point through the while
      loop.
    * If a while loop modifies one of its input tensors, that tensor will
      receive at least the following HloValues:
      - An HloValue for the tensor's creation, with a use at the operand of the
        while instruction.
      - An HloValue with its defining position at the while body's parameter.
      - An HloValue whose defining position is an instruction in the while body
        that feeds the new tensor value to the body's ROOT instruction.
      - An HloValue with its defining position at the while instruction's
        result.

  - HloBuffer (Hlo alias analysis concept): A memory container that holds one
    or more HloValues that must alias. Typically, each HloValue corresponds to
    1 HloBuffer; however, many exceptions exist. For example, tensors that are
    modified by a while loop have their HloValues share an HloBuffer, for the
    HloValues that come immediately before, during, and immediately after the
    loop. HloBuffers are shared between HloValues wherever there is aliasing,
    whether implicit by the nature of the instruction (e.g.,
    dynamic-update-slice) or explicit (e.g., fusion input-output aliasing).

  - MsaBufferInterval (HeapSimulator concept): A MsaBufferInterval is defined by
    a buffer of a given size, with a defined lifetime. In MSA, the buffer
    corresponds to an HloValue.

  - AllocationValue: An AllocationValue is defined by an HloValue, and *one* of
    its HloPositions. Note that a given HloValue may be associated with multiple
    AllocationValues in this way.
    * We do not create AllocationValues for trivial HloPositions, e.g., ones
      defined by Tuple, GetTupleElement, and Bitcast instructions.
    * The HloPosition used to define the AllocationValue is referred to as the
      AllocationValue's defining position.
      * Typically, this is also the defining position of the HloValue. However,
        it may not be. For example, we would create an AllocationValue with an
        HloPosition of a read-only while loop parameter, but the HloValue
        corresponding to that HloPosition would have a different defining
        position.
    * The uses of an AllocationValue are limited to the direct uses of the
      AllocationValue's defining position.
    * An AllocationValue is associated with an AllocationSequence, describing
      what to do with the underlying tensor, in memory, over the lifetime of the
      AllocationValue.

  - (Use) Segment: Each AllocationValue and its uses are separated into periods
    of time called use segments. The first use segment is from the (inclusive)
    time of the AllocationValue's defining position to its first use
    (inclusive). The second use segment is from the first use (inclusive) to
    the second use (inclusive), etc.

  - AllocationRequest: A request to determine what to do with an
    AllocationValue, in memory, during a use segment. It also contains
    restrictions and preferences on what to do.
    * A request results in updates to the AllocationValue's AllocationSequence.
      It may add Allocations, or modify existing Allocations in the sequence.

  - Allocation: A description of what to do with an AllocationValue in memory,
    over a period of time.
    * Pure virtual base class of all Allocations.

  - AllocationSequence: A sequential list of Allocations, explaining what to do
    with an AllocationValue over its lifetime. Allocations in the sequence may
    overlap.

  - Pinned Allocation: Represents producing a tensor in a particular memory
    space, or keeping a tensor in a memory space in which it already exists.

  - Copy Allocation: Instructions to copy an AllocationValue from one memory
    space to another. Used for prefetching (default mem -> alt mem), and
    eviction (alt mem -> default mem).
    * A copy Allocation contains a copy_done_schedule_before_time. The buffer is
      available for use at that schedule time, through the Allocation's
      end_time.

  - Sliced Copy Allocation: Similar to a Copy Allocation, except the memory is
    copied in slices, in an effort to delay allocating memory in the destination
    memory space, for as long as possible.

  - Mirrored Allocation and Parent Allocation: R/W tensors passed to while loops
    typically have at least 3 AllocationValues, 1 for the producer of the tensor
    before the while loop, 1 for the while loop's body parameter, and 1 for the
    result of the while loop. There are situations heading into a while loop, in
    which the while loop input is both in alternate memory and default memory.
    (For example, this could happen beause we want the buffer in alternate
    memory for the while loop and default memory after the while loop, but we
    don't have resources to evict the buffer after the while loop.) In those
    cases, we use a mirrored allocation for the AllocationValue inside the
    while loop, to mirror the allocation in default memory. We use a parent
    allocation for the AllocationValue resulting from the while loop result.

Useful logging and error messages

  - Live range too long: The live range of a use segment is too long for a
    no-copy allocation in alternate memory; i.e., it is longer than we want to
    keep a buffer in alternate memory without being used.
    * If the CostAnalysisPrefetchIntervalPicker is used, which is the default,
      live range too long is governed by the picker's
      max_overlap_to_mem_size_async_copy_ratio argument.

  - Live range too short: The live range of a use segment is too short to
    prefetch a buffer to alternate memory, according to some heuristic and not
    based on limited copy resource.
    * If the CostAnalysisPrefetchIntervalPicker is used, which is the default,
      live range too long is governed by the picker's
      min_overlap_to_async_copy_ratio argument.

  - "Finding allocation for": Magical logging phrase indicating the point in
    time where we are are trying to determine how to update an AllocationValue's
    AllocationSequence, for a particular use segment.

  - To log the alternate memory allocations that MSA made at a given schedule
    time:
    * Find the time point of interest. For example, to find the time for an
      instruction fusion.1:
      - Set vlogging to 2 for algorithm.cc.
      - Find logging lines that look like:
        Initial resource[100] = 1.0 (fusion.1)
      - That tells us that the fusion.1 has schedule time 100.
    * Uncomment the line in memory_space_assignment.cc labeled
      DEBUG_LOG_ALLOCATIONS_AT, and use time 100.
*/

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_SPACE_ASSIGNMENT_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_SPACE_ASSIGNMENT_H_

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
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

// This class contains pre-set assignments determined by memory space
// assignment. It contains two data structures: (1) a chunks vector that maps a
// defining HloPosition to a Chunk (offset and size), and (2) an assignment_info
// vector that maps the memory space to information like its allocated size and
// heap memory trace. If there is only one alternate memory space like there is
// currently, there will be one entry in assignment_info.
class PresetAssignments {
 public:
  // Contains per-memory-space information like the allocated size and heap
  // simulator trace.
  struct AssignmentInformation {
    int64_t size;
    HeapSimulatorTrace heap_simulator_trace;
  };

  PresetAssignments() = default;

  void add_chunk(const HloPosition& position,
                 const HeapSimulator::Chunk& chunk) {
    chunks_.emplace_back(position, chunk);
  }

  void add_scoped_allocation_chunk(HloInstruction* instruction,
                                   const HeapSimulator::Chunk& chunk) {
    scoped_allocation_chunks_.emplace_back(instruction, chunk);
  }

  AssignmentInformation* assignment_information_for_space(
      int64_t memory_space) {
    for (auto& space_and_info : assignment_info_) {
      if (space_and_info.first == memory_space) {
        return &space_and_info.second;
      }
    }
    assignment_info_.emplace_back(memory_space, AssignmentInformation());
    return &assignment_info_.back().second;
  }

  absl::Span<const std::pair<HloPosition, HeapSimulator::Chunk>> chunks()
      const {
    return chunks_;
  }

  absl::Span<const std::pair<HloInstruction*, HeapSimulator::Chunk>>
  scoped_allocation_chunks() const {
    return scoped_allocation_chunks_;
  }

  absl::Span<const std::pair<int64_t, AssignmentInformation>>
  assignment_informations() const {
    return assignment_info_;
  }

  // Get debugging information.
  std::string buffer_info_str() const { return buffer_info_str_; }
  std::string allocation_info_str() const { return allocation_info_str_; }
  std::string instruction_schedule_str() const {
    return instruction_schedule_str_;
  }

 private:
  std::vector<std::pair<HloPosition, HeapSimulator::Chunk>> chunks_;
  std::vector<std::pair<HloInstruction*, HeapSimulator::Chunk>>
      scoped_allocation_chunks_;
  std::vector<std::pair<int64_t, AssignmentInformation>> assignment_info_;
  std::string buffer_info_str_;
  std::string allocation_info_str_;
  std::string instruction_schedule_str_;
};

// MemorySpaceAssignment assigns memory spaces (default or alternate) to each
// instruction in the module. It will greedily try placing as as many values in
// the alternate memory space as possible. It uses the heap simulator to
// determine the actual allocation offsets of values in the alternate memory
// space to account for fragmentation. The default memory space is assumed to be
// large enough to hold the values that could not be placed in the alternate
// memory space.
class MemorySpaceAssignment {
 public:
  // Statistics of asynchronous copies.
  struct AsyncCopyStats {
    // Includes both async copies and async sliced copies.
    int64_t max_outstanding_async_copies = 0;
    // Includes both async copies and async sliced copies.
    int64_t num_prefetches = 0;
    int64_t num_sliced_prefetches = 0;
    int64_t num_sliced_prefetch_slices = 0;
    int64_t prefetch_bytes = 0;
    int64_t num_evictions = 0;
    int64_t eviction_bytes = 0;
  };

  virtual ~MemorySpaceAssignment() = default;

  // Runs the MemorySpaceAssignment pass.
  static absl::StatusOr<std::unique_ptr<PresetAssignments>> Run(
      HloModule* module, const HloLiveRange& hlo_live_range,
      const HloAliasAnalysis& alias_analysis, const Options& options);

  // Calculates asynchronous copy statistics.
  absl::StatusOr<AsyncCopyStats> CalculateAsyncCopyStats(
      const HloDataflowAnalysis& dataflow_analysis) const;

  // Verify that allocations_ are free of overlapping Allocations in time and
  // space. This is a post-processing step called after all allocations have
  // been finalized, before the async copies get scheduled.
  absl::Status VerifyAllocations() const;

  // Verify that the memory space assignment is free of overlapping buffers and
  // export heap simulator trace to be used by buffer_assignment.
  //
  // If alt_mem_bytes_occupied is not null, it will be populated with the number
  // of bytes occupied in the alternate memory space at each instruction time.
  absl::Status VerifyAndExportHeapSimulatorTrace(
      const HloAliasAnalysis& alias_analysis,
      std::vector<int64_t>* alt_mem_bytes_occupied = nullptr);

  static constexpr absl::string_view kName = "memory-space-assignment";

 protected:
  // Main driver of the memory space assignment pass.
  virtual absl::StatusOr<std::unique_ptr<PresetAssignments>>
  RunMemorySpaceAssignment(const HloLiveRange& hlo_live_range,
                           const HloAliasAnalysis& alias_analysis);

  // Finds an AllocationSequence for placing buffers in alternate memory using
  // the MsaAlgorithm algorithm. Must be set before Process() is called.
  virtual absl::Status FindAllocationSequence(
      const HloLiveRange& hlo_live_range,
      const HloAliasAnalysis& alias_analysis);

  const Options& options() const { return options_; }

  MemorySpaceAssignment(HloModule* module, const Options& options,
                        const HloLiveRange& hlo_live_range)
      : module_(module),
        options_(options),
        flattened_instructions_(hlo_live_range.flattened_instruction_sequence()
                                    .instructions()
                                    .begin(),
                                hlo_live_range.flattened_instruction_sequence()
                                    .instructions()
                                    .end()),
        computations_in_schedule_(),
        preset_assignments_(std::make_unique<PresetAssignments>()) {
    for (const auto& computation_and_bound :
         hlo_live_range.computation_span_times()) {
      computations_in_schedule_.insert(computation_and_bound.first);
    }
  }

  AllocationSequence allocations_;

  HloModule* module() { return module_; }

 private:
  // Process calls Process methods of the allocations after the allocations have
  // been finalized.
  absl::Status Process(const HloLiveRange& hlo_live_range);

  // Process() might have altered the computation graph by inserting kTuple and
  // kGetTupleElement instructions. SimplifyGraph performs a simple DCE and
  // tuple simplification operation (e.g., given GetTupleElement(Tuple(a, b),
  // 1), simply forwards b). Runs to fixed point.
  absl::Status SimplifyGraph();

  // FixSchedule inserts asynchronous copies in the schedule.
  absl::Status FixSchedule();

  // Export the alternate memory assignments to the PresetAssignments and color
  // the HLO graph with the determined memory spaces.
  absl::Status ExportAndColorBuffers(const HloAliasAnalysis& alias_analysis);

  // Schedules asynchronous copies and ensures that the CopyStarts and their
  // corresponding CopyDones follow the same order.
  void ScheduleAsynchronousCopies();

  // Remove the positions and chunks associated with the instruction from
  // alternate_memory_assignments_.
  void RemoveAssignmentForInstruction(const HloInstruction* instruction);

  HloModule* module_;
  const Options& options_;
  std::vector<HloInstruction*> flattened_instructions_;
  absl::flat_hash_set<const HloComputation*> computations_in_schedule_;
  std::unique_ptr<PresetAssignments> preset_assignments_;
  std::vector<std::pair<HloPosition, HeapSimulator::Chunk>>
      alternate_memory_assignments_;
  std::vector<std::pair<HloInstruction*, HeapSimulator::Chunk>>
      scoped_memory_assignments_;
  int64_t alternate_memory_size_ = 0;

  // These maps hold vectors of new instructions that need to be scheduled after
  // (or before) the instruction index in the key. FixSchedule uses these maps
  // to modify and fix the schedule.
  absl::flat_hash_map<int64_t, std::vector<HloInstruction*>> schedule_after_;
  absl::flat_hash_map<int64_t, std::vector<HloInstruction*>> schedule_before_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_MEMORY_SPACE_ASSIGNMENT_H_
