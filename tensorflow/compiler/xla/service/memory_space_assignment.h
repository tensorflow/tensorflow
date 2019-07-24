/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_H_

#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// MemorySpaceAssignment assigns memory spaces (default or alternate) to each
// instruction in the module. It will greedily try placing as as many values in
// the alternate memory space as possible. It uses the heap simulator to
// determine the actual allocation offsets of values in the alternate memory
// space to account for fragmentation. The default memory space is assumed to be
// large enough to hold the values that could not be placed in the alternate
// memory space.
class MemorySpaceAssignment {
 public:
  using Chunk = HeapSimulator::Chunk;

  // MemorySpaceAssignment uses a notion of a slow and large default memory
  // space and a fast and small alternate memory space.
  enum class MemorySpace { kDefault, kAlternate };

  // This class represents an allocation that might either be in the default or
  // alternate memory. An HloValue might live in multiple different allocations
  // over its lifetime. The lifetimes of the allocations are defined using
  // start_time and end_time, which corresponds to the instruction indexes in
  // the flattened schedule. Each of these allocations might partially overlap
  // with each other. CopyAllocation defined below represents asynchronous
  // copies between Allocations.
  //
  // Consider an instruction Foo, and its users Bar and Baz, and the times given
  // in terms of the flattened schedule of the entire module:
  //
  //      Foo:10
  //       /   \
  //    Bar:14  \
  //           Baz:25
  //
  // A valid memory space assignment could be like the following:
  //
  //  Time:         10 ... 14        ...      25
  //                Foo    Bar                Baz
  //  Alternate     +-------+           +-----+
  //  Default           +---------------------+
  //                    ^   ^           ^     ^
  //                    |   |           |     |
  //                evict   evict  prefetch  prefetch
  //                start    end    start      end
  //
  // This would be represented with:
  //   - Allocation(memory_space=kAlternate, start_time=10, end_time=14)
  //   - CopyAllocation(memory_space=kDefault, start_time=12, end_time=25)
  //   - CopyAllocation(memory_space=kAlternate, start_time=22, end_time=25)
  class Allocation {
   public:
    Allocation(HloInstruction* defining_instruction, MemorySpace memory_space,
               Chunk chunk, int64 start_time, int64 end_time)
        : defining_instruction_(defining_instruction),
          memory_space_(memory_space),
          chunk_(chunk),
          start_time_(start_time),
          end_time_(end_time) {}
    virtual ~Allocation() = default;

    // Adds a use to this allocation.
    void AddUse(HloUse use) { uses_.push_back(use); }

    // Extends the end time of this allocation.
    void Extend(int64 end_time) { end_time_ = end_time; }

    // After all of the time ranges for the allocations have been assigned,
    // Process morphs the instructions affected to assign the memory spaces and
    // insert asynchronous copy instructions if necessary.
    virtual Status Process(MemorySpaceAssignment* memory_space_assignment);

    // Returns the defining instruction for this allocation.
    virtual HloInstruction* defining_instruction() const {
      return defining_instruction_;
    }

    const std::vector<HloUse>& uses() const { return uses_; }
    MemorySpace memory_space() const { return memory_space_; }
    Chunk chunk() const { return chunk_; }
    int64 start_time() const { return start_time_; }
    int64 end_time() const { return end_time_; }

   protected:
    HloInstruction* defining_instruction_;
    std::vector<HloUse> uses_;
    MemorySpace memory_space_;
    Chunk chunk_;
    int64 start_time_;
    int64 end_time_;
  };

  // This class represents an allocation as a result of an asynchronous copy.
  class CopyAllocation : public Allocation {
   public:
    CopyAllocation(const Allocation& prev_allocation, MemorySpace memory_space,
                   Chunk chunk, int64 start_time, int64 end_time,
                   HloInstruction* copy_start_schedule_after,
                   HloInstruction* copy_done_schedule_before)
        : Allocation(/*defining_instruction=*/nullptr, memory_space, chunk,
                     start_time, end_time),
          prev_allocation_(prev_allocation),
          copy_start_schedule_after_(copy_start_schedule_after),
          copy_done_schedule_before_(copy_done_schedule_before) {}

    Status Process(MemorySpaceAssignment* memory_space_assignment) override;

    HloInstruction* defining_instruction() const override {
      // Unless explicitly set, the defining instruction of a copy allocation in
      // retrieved from the previous allocation.
      if (defining_instruction_ != nullptr) {
        return defining_instruction_;
      } else {
        return prev_allocation_.defining_instruction();
      }
    }

   private:
    const Allocation& prev_allocation_;
    // These variables define the scheduling boundaries where CopyStart and
    // CopyDone can be scheduled. The earliest CopyStart can be scheduled is
    // after copy_start_schedule_after_ and the latest CopyDone can be scheduled
    // is before copy_done_schedule_before_.
    HloInstruction* copy_start_schedule_after_;
    HloInstruction* copy_done_schedule_before_;
  };

  using AllocationSequence = std::list<std::unique_ptr<Allocation>>;
  using AllocationMap =
      absl::flat_hash_map<const HloBuffer*, AllocationSequence>;

  // Runs the MemorySpaceAssignment pass. alternate_memory_space is the
  // architecture-specific integer value that describes the alternate memory.
  // max_size_in_bytes is the maximum size of the alternate memory.
  // min/max_prefetch_interval define min/max number of independent instructions
  // that can be overlapped while prefetching to decide how early can prefetch
  // begin. alternate_memory_space_alignment_in_bytes is the alignment required
  // in the alternate memory space, size_fn is the size function for buffer
  // values, and is_allowed_in_alternate_mem can be used to prevent certain
  // HloValues (e.g., based on the opcode) to be placed on the alternate memory.
  // TODO(berkin): Use the cost model instead of using number of instructions to
  // decide how early to prefetch.
  static StatusOr<bool> Run(
      HloModule* module, int64 alternate_memory_space, int64 max_size_in_bytes,
      int64 min_prefetch_interval, int64 max_prefetch_interval,
      int64 alternate_memory_space_alignment_in_bytes,
      BufferValue::SizeFunction size_fn,
      std::function<bool(const HloValue&)> is_allowed_in_alternate_mem);

 private:
  MemorySpaceAssignment(HloModule* module, int64 alternate_memory_space)
      : module_(module), alternate_memory_space_(alternate_memory_space) {}

  // Process calls Process methods of the allocations after the allocations have
  // been finalized.
  Status Process();

  // FixSchedule inserts asynchronous copies in the schedule.
  Status FixSchedule();

  // Schedules a pair of asynchronous copy instructions (copy_start and
  // copy_done) where copy_start will be scheduled after the instruction in
  // copy_start_schedule_after and copy_done will be scheduled before the
  // instruction in copy_done_schedule_before.
  void ScheduleAsynchronousCopy(HloInstruction* copy_start,
                                HloInstruction* copy_start_schedule_after,
                                HloInstruction* copy_done,
                                HloInstruction* copy_done_schedule_before);

  HloModule* module_;
  int64 alternate_memory_space_;
  AllocationMap allocation_map_;

  // These maps hold vectors of new instructions that need to be scheduled after
  // (or before) the instruction in the key. FixSchedule uses these maps to
  // modify and fix the schedule.
  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      schedule_after_;
  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      schedule_before_;
};

// This class inherits from GlobalDecreasingSizeBestFitHeap with a notion of
// maximum size.
class AlternateMemoryBestFitHeap : public GlobalDecreasingSizeBestFitHeap {
 public:
  using IsAllowedInAlternateMemoryFunction =
      std::function<bool(const HloValue&)>;
  using MemorySpace = MemorySpaceAssignment::MemorySpace;

  AlternateMemoryBestFitHeap(
      MemorySpaceAssignment::AllocationMap* allocation_map,
      int64 max_size_in_bytes, int64 min_prefetch_interval,
      int64 max_prefetch_interval, const HloAliasAnalysis& alias_analysis,
      int64 alignment, GlobalDecreasingSizeBestFitHeap::Type type,
      IsAllowedInAlternateMemoryFunction is_allowed_in_alternate_mem)
      : GlobalDecreasingSizeBestFitHeap(alignment, type),
        allocation_map_(allocation_map),
        max_size_in_bytes_(max_size_in_bytes),
        min_prefetch_interval_(min_prefetch_interval),
        max_prefetch_interval_(max_prefetch_interval),
        alias_analysis_(alias_analysis),
        is_allowed_in_alternate_mem_(is_allowed_in_alternate_mem) {}

  HeapSimulator::Result Finish() override;

 private:
  // Finds an allocation for the given interval. Internally, it will attempt to
  // find a suitable chunk candidate within the heap size and prefetch interval
  // limits, and append the new allocation(s) to allocations. The new
  // allocations can be in default or alternate memory spaces, or can be
  // prefetches or evictions.
  void FindAllocation(int64 start_time, int64 end_time, HloUse use,
                      const BufferInterval& interval,
                      MemorySpaceAssignment::AllocationSequence* allocations);

  // Returns the instruction at a particular time in the flattened instruction
  // schedule.
  HloInstruction* GetInstructionAt(int64 time) const;

  // Given a buffer interval, returns the colocated intervals. Unlike the
  // similar GlobalDecreasingSizeBestFitHeap::GetTransitiveColocations, it
  // returns the colocated intervals sorted by scheduled time.
  std::vector<const BufferInterval*> GetSortedColocatedIntervals(
      const BufferInterval& interval) const;

  // Since the allocations are recorded to the AllocationMap, we don't maintain
  // result_ in GlobalDecreasingSizeBestFitHeap. Override AddToChunkMap to avoid
  // unnecessarily adding the chunk to the chunk map.
  void AddToChunkMap(const HloValue* buffer, Chunk chunk) override {}

  MemorySpaceAssignment::AllocationMap* allocation_map_;
  int64 max_size_in_bytes_;
  // The min and max prefetch intervals decribe the number of independent HLOs
  // overlapped while a value is being prefetched into the alternate memory
  // (between CopyStart and CopyDone HLO instructions). max_prefetch_interval
  // attempts to prevent bringing tensors into the alternate memory too eagerly
  // and hence occupying the space for other tensors which might use it.
  // min_prefetch_interval attempts to prevent cases where tensors are
  // prefetched into the alternate memory without sufficient time for the copy
  // to take place. In those cases, it's just better to keep the tensor in the
  // default memory instead of hurting the critical path with this copy that
  // likely won't finish in time.
  // TODO(berkin): Explore heuristics that take into account the cost of copying
  // tensors between alternate and default memories.
  int64 min_prefetch_interval_;
  int64 max_prefetch_interval_;
  const HloAliasAnalysis& alias_analysis_;
  IsAllowedInAlternateMemoryFunction is_allowed_in_alternate_mem_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_H_
