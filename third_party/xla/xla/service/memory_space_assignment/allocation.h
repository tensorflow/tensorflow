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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALLOCATION_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALLOCATION_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/slice.h"
#include "xla/shape.h"
#include "xla/status.h"

namespace xla::memory_space_assignment {

// MemorySpaceAssignment uses a notion of a slow and large default memory
// space and a fast and small alternate memory space.
enum class MemorySpace : std::uint8_t { kDefault, kAlternate };

// An interface describing what to do with a value in memory over its lifetime.
// An allocation might either be placed in the default or alternate memory. An
// HloValue might live in multiple different allocations over its lifetime. The
// lifetimes of the allocations are defined using start_time and end_time, which
// corresponds to the instruction indexes in the flattened schedule. Each of
// these allocations might partially overlap with each other.
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
//   - PinnedAllocation(memory_space=kAlternate, start_time=10, end_time=14)
//   - CopyAllocation(memory_space=kDefault, start_time=12, end_time=25)
//   - CopyAllocation(memory_space=kAlternate, start_time=22, end_time=25)
class Allocation {
 public:
  virtual ~Allocation() = default;

  // Allocation source methods
  // --------------------------------------------------------------------------
  // Returns the defining position for this allocation.
  virtual HloPosition defining_position() const = 0;
  // Returns the cross-program prefetch index for this allocation.
  std::optional<int64_t> cross_program_prefetch_index() const;

  // Allocation timing methods
  // --------------------------------------------------------------------------
  // TODO(cl/604356742): update all timing methods to explicitly state that
  // they're representing inclusive intervals.
  int64_t start_time() const { return start_time_; }
  int64_t end_time() const { return end_time_; }
  // Returns the time the buffer is first available to be used
  virtual int64_t earliest_available_time() const = 0;
  void set_start_time(int64_t start_time) { start_time_ = start_time; }
  void set_end_time(int64_t end_time) { end_time_ = end_time; }
  // Extends the end time of this allocation.
  void Extend(int64_t end_time) { end_time_ = std::max(end_time_, end_time); }

  // Allocation space methods
  // --------------------------------------------------------------------------
  MemorySpace memory_space() const { return memory_space_; }
  // Returns the associated chunk that may be a nullopt if the allocation is
  // in the default memory space.
  std::optional<HeapSimulator::Chunk> maybe_chunk() const { return chunk_; }
  // Returns the associated chunk. The caller should ensure that the chunk is
  // defined (the allocation should be in the alternate memory space).
  HeapSimulator::Chunk chunk() const;
  HeapSimulator::Chunk* mutable_chunk() { return &*chunk_; }
  void set_offset(int64_t offset);
  bool is_scoped_allocation() const { return is_scoped_allocation_; }
  // Returns true if the allocation is in the alternate memory space.
  bool is_in_alternate_mem() const;
  // Returns true if the allocation is in the default memory space.
  bool is_in_default_mem() const;

  // Use methods
  // --------------------------------------------------------------------------
  const std::vector<HloUse>& uses() const { return uses_; }
  void clear_uses() { uses_.clear(); }
  bool has_no_uses() const { return uses_.empty(); }
  // Adds a use to this allocation.
  void AddUse(HloUse use);
  // Replaces all uses of the allocation with the copy_complete instruction.
  absl::Status UpdateUses(HloComputation* computation,
                          HloInstruction* producing_instruction);

  // Allocation type methods
  // --------------------------------------------------------------------------
  virtual bool is_copy_allocation() const = 0;
  virtual bool is_sliced_copy_allocation() const = 0;
  // True if the allocation is for a copy or a sliced-copy.
  bool is_copy_like_allocation() const;

  // Processing methods
  // --------------------------------------------------------------------------
  // Recursively create kGetTupleElement instructions if the defining position
  // shape is not an array. Returns the new instruction that has array shape.
  HloInstruction* AddGetTupleElements() const;
  // After all of the time ranges for the allocations have been assigned,
  // Process morphs the instructions affected to assign the memory spaces and
  // insert asynchronous copy instructions if necessary.
  virtual absl::Status Process() = 0;
  // An optional post-process step that will be called after all allocations
  // have been processed.
  virtual absl::Status PostProcess() = 0;
  // Marks (adds this allocation to needed_allocations) if this allocation is
  // needed. PinnedAllocation and CopyAllocations are always needed and
  // ParentAllocations are needed if they have any uses or if other
  // CopyAllocation or ParentAllocations depend on them.
  virtual void MarkIfNeeded(
      absl::flat_hash_set<const Allocation*>& needed_allocations) const = 0;
  // Marks this allocation as needed.
  virtual void MarkNeeded(
      absl::flat_hash_set<const Allocation*>& needed_allocations) const = 0;

  // Utility methods
  // --------------------------------------------------------------------------
  virtual std::string ToString() const = 0;
  virtual bool operator==(const Allocation& other) const = 0;

 protected:
  // Protected constructor to encourage use of the final subclasses (e.g.,
  // PinnedAllocation, CopyAllocation, etc.).
  Allocation(HloPosition defining_position, MemorySpace memory_space,
             std::optional<HeapSimulator::Chunk> chunk, int64_t start_time,
             int64_t end_time, bool is_scoped_allocation,
             std::optional<int64_t> cross_program_prefetch_index);

  // Returns the original defining position of this allocation.
  HloPosition original_defining_position() const;
  // Sets the original defining position of this allocation.
  void set_original_defining_position(HloPosition defining_position);
  bool base_is_equal(const Allocation& other) const;

 private:
  HloPosition original_defining_position_;
  MemorySpace memory_space_;
  std::optional<HeapSimulator::Chunk> chunk_;
  int64_t start_time_;
  int64_t end_time_;
  const bool is_scoped_allocation_;
  std::vector<HloUse> uses_;
  std::optional<int64_t> cross_program_prefetch_index_;
};

using AllocationSequence = std::vector<std::unique_ptr<Allocation>>;
std::tuple<int64_t, bool, int64_t> GetAllocationSortTuple(
    const std::unique_ptr<Allocation>& allocation);
void SortAllocationSequence(AllocationSequence& allocations);
std::string AllocationSequenceToString(AllocationSequence& allocations,
                                       bool sort_allocations = false);
std::vector<Allocation*> GetAllocationSequenceInRawPointers(
    AllocationSequence& allocations);

// This class represents an allocation that pins a tensor to
// a specific memory space.
class PinnedAllocation final : public Allocation {
 public:
  PinnedAllocation(HloPosition defining_position, MemorySpace memory_space,
                   std::optional<HeapSimulator::Chunk> chunk,
                   int64_t start_time, int64_t end_time,
                   bool is_scoped_allocation);

  // Overridden methods
  //
  // Returns the original defining position.
  HloPosition defining_position() const override;
  int64_t earliest_available_time() const override { return start_time(); }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  absl::Status Process() override;
  absl::Status PostProcess() override { return OkStatus(); }
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  bool operator==(const PinnedAllocation& other) const;
};

// This class represents an allocation as a result of an asynchronous copy.
// Note: CopyStart instructions are inserted after
// `copy_start_schedule_after`, while CopyDone instructions are inserted
// before `copy_done_schedule_before_time`.
class CopyAllocation final : public Allocation {
 public:
  // TODO(b/307342076): Reorder scheduling times to be
  // copy_start_schedule_after_time, copy_done_schedule_before_time, end_time
  CopyAllocation(
      Allocation& prev_allocation, MemorySpace memory_space,
      std::optional<HeapSimulator::Chunk> chunk,
      int64_t copy_start_schedule_after_time,
      int64_t copy_done_schedule_before_time, int64_t end_time,
      std::optional<int64_t> cross_program_prefetch_index = std::nullopt);

  // Overridden methods
  //
  HloPosition defining_position() const override;
  // Returns the time the buffer is first available to be used. For
  // CopyAllocation, this is when the copy ends, which is
  // copy_done_schedule_before.
  int64_t earliest_available_time() const override;
  bool is_copy_allocation() const override { return true; }
  bool is_sliced_copy_allocation() const override { return false; }
  absl::Status Process() override;
  absl::Status PostProcess() override { return OkStatus(); }
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  bool operator==(const CopyAllocation& other) const;

  const Allocation& prev_allocation() { return prev_allocation_; }
  Allocation& mutable_prev_allocation() { return prev_allocation_; }

  HloInstruction* copy_start() const { return copy_start_; }
  HloInstruction* copy_done() const { return copy_done_; }

  void set_copy_start_schedule_after(int64_t copy_start_schedule_after);
  void set_copy_done_schedule_before(int64_t copy_done_schedule_before);
  int64_t copy_start_schedule_after() const;
  int64_t copy_done_schedule_before() const;

 private:
  Allocation& prev_allocation_;
  // These variables define the scheduling boundaries where CopyStart and
  // CopyDone can be scheduled. The earliest CopyStart can be scheduled is
  // after copy_start_schedule_after_ and the latest CopyDone can be scheduled
  // is before copy_done_schedule_before_.
  int64_t copy_start_schedule_after_;
  int64_t copy_done_schedule_before_;
  HloInstruction* copy_start_ = nullptr;
  HloInstruction* copy_done_ = nullptr;
};

// This class represents an allocation resulting from asynchronous sliced
// copies.
//
// Let the sliced allocation be represented as follows, and imagine that t3
// is the time when the entire buffer [p0, p3) is available for use
//
//   space
//    ^
// p3 |       +-----------+
//    |       |           |
// p2 |   +---+           |
//    |   |               |
// p1 |   +-------+       |
//    |           |       |
// p0 |           +-------+
//    +---|---|---|---|---|----> time
//        t0  t1  t2  t3  t4
//
// The PinnedAllocation underlying the SlicedCopyAllocation will use the
// following dimensions:
// - chunk = [p0, p3)
// - start time = t2
// - earliest_available_time = t3
// - end_time = t4
class SlicedCopyAllocation final : public Allocation {
 public:
  // Full details about a slice in the sliced allocation.
  struct SliceDetail {
    std::string ToString() const;
    std::tuple<const SliceDecision&, int64_t, int64_t, const HloInstruction*,
               const HloInstruction*>
    ToTuple() const;
    bool operator==(const SliceDetail& other) const;

    // Create the instructions to copy the slice. This method updates
    // copy_start and copy_done.
    absl::Status CreateAsyncSlice(const Shape& original_shape,
                                  HloInstruction& producer,
                                  HloComputation& parent);

    SliceDecision slice_decision;
    int64_t copy_start_after_time = -1;
    int64_t copy_done_before_time = -1;
    HloInstruction* copy_start = nullptr;
    HloInstruction* copy_done = nullptr;
  };

  // REQUIRES:
  // - slice_decisions_sorted_by_exclusive_start_time.size() >= 2, otherwise,
  //   CopyAllocation should be used.
  SlicedCopyAllocation(
      const Allocation& prev_allocation, MemorySpace memory_space,
      std::vector<SliceDecision> slice_decisions_sorted_by_exclusive_start_time,
      int64_t copy_done_schedule_before_time, int64_t end_time,
      const SlicedPrefetchOptions& sliced_prefetch_options,
      absl::FunctionRef<Shape(const Shape&)> get_equivalent_s8_shape_fn);

  // Overridden methods
  //
  HloPosition defining_position() const override;
  // Returns the time the buffer is first available to be used. For
  // SlicedCopyAllocation, this is when all copies have ended.
  int64_t earliest_available_time() const override;
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return true; }
  // MemorySpaceAssignment::Process() calls Process() to create asynchronous
  // slice copies, and a bitcast-concat call to glue the slices back together.
  absl::Status Process() override;
  absl::Status PostProcess() override { return OkStatus(); }
  // Marks the allocation as needed.
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  bool operator==(const SlicedCopyAllocation& other) const;

  std::vector<int64_t> SliceOffsetsSortedByStartTime() const;
  void AddDiffToAllSliceOffsets(int64_t diff);
  // Used to update offsets and start times after repacking.
  void ImportRepackedSliceData(const SlicedAllocationData& data);
  const std::vector<SliceDetail>& slice_details_sorted_by_start_time() const;
  std::vector<SliceDetail>& mutable_slice_details_sorted_by_start_time();
  HloInstruction* concat() const { return concat_; }

 private:
  SlicedCopyAllocation() = delete;

  // Create an instruction to concatenate the slices. Populates concat_.
  absl::Status CreateBitcastConcat(const Shape& shape,
                                   absl::Span<HloInstruction* const> slices);

  Shape original_shape_to_slice_;
  const Allocation& prev_allocation_;
  // REQUIRES:
  // - sorted_segments_[i].copy_start_after_time <=
  //   sorted_segments_[i+j].copy.start_after_time
  // - sorted_segments_[i].copy_done_before_time <=
  //   sorted_segments_[i+j].copy.start_before_time
  std::vector<SliceDetail> slice_details_sorted_by_exclusive_start_time_;
  HloInstruction* concat_ = nullptr;
  const SlicedPrefetchOptions& sliced_prefetch_options_;
  absl::FunctionRef<Shape(const Shape&)> get_equivalent_s8_shape_fn_;
};

// An allocation in the default memory space that mirrors another Allocation
// object. This is useful to model an eviction that happens before a while op
// so that we don't need to redundantly evict the buffer after the while op as
// well.
class MirroredAllocation final : public Allocation {
 public:
  MirroredAllocation(const Allocation& original_allocation, int64_t time);

  // Overridden methods
  //
  // Returns the original defining position.
  HloPosition defining_position() const override;
  int64_t earliest_available_time() const override { return start_time(); }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  absl::Status Process() override;
  absl::Status PostProcess() override { return OkStatus(); }
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  bool operator==(const MirroredAllocation& other) const;

 private:
  const Allocation& original_allocation_;
};

// An allocation in default memory space that is defined in the parent
// computation. If a value has a copy in the default memory space in the
// parent computation, we don't need to evict this buffer in a while loop.
class ParentAllocation final : public Allocation {
 public:
  ParentAllocation(const Allocation& original_allocation,
                   HloInstruction* calling_instruction, HloPosition position,
                   int64_t time);

  // Overridden methods
  //
  // Returns the original defining position.
  HloPosition defining_position() const override;
  int64_t earliest_available_time() const override { return start_time(); }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  absl::Status Process() override;
  absl::Status PostProcess() override;
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  bool operator==(const ParentAllocation& other) const;

 private:
  const Allocation& original_allocation_;
  HloInstruction* calling_instruction_;
};

}  // namespace xla::memory_space_assignment

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALLOCATION_H_
