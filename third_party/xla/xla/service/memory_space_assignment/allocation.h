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

#include <stdbool.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/slice.h"
#include "xla/shape.h"

namespace xla::memory_space_assignment {

// MemorySpaceAssignment uses a notion of a slow and large default memory
// space and a fast and small alternate memory space.
enum class MemorySpace : std::uint8_t { kDefault, kAlternate };
std::string MemorySpaceToString(MemorySpace memory_space);

using BitcastSplitFn = std::function<absl::StatusOr<int64_t>(
    const HloInstruction* instruction, int64_t split_dim)>;

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

  void set_split_shape(const std::optional<Shape>& split_shape) {
    split_shape_ = split_shape;
  }
  const std::optional<Shape>& split_shape() const { return split_shape_; }
  std::optional<Shape>& mutable_split_shape() { return split_shape_; }

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
  void RemoveUse(HloUse use);
  // Replaces all uses of the allocation with the copy_complete instruction.
  absl::Status UpdateUses(HloComputation* computation,
                          HloInstruction* producing_instruction,
                          const BitcastSplitFn& bitcast_split_fn);

  // Allocation type methods
  // --------------------------------------------------------------------------
  virtual bool is_pinned_allocation() const = 0;
  virtual bool is_copy_allocation() const = 0;
  virtual bool is_sliced_copy_allocation() const = 0;
  virtual bool is_window_prefetched_allocation() const = 0;
  virtual bool is_scoped_allocation() const = 0;
  virtual bool is_reserved_allocation() const = 0;
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
  virtual absl::Status Process(const BitcastSplitFn& bitcast_split_fn) = 0;
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
             int64_t end_time,
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
  std::vector<HloUse> uses_;
  std::optional<int64_t> cross_program_prefetch_index_;
  // If present, indicates the newly split shape.
  std::optional<Shape> split_shape_;
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
                   int64_t start_time, int64_t end_time);

  // Overridden methods
  //
  // Returns the original defining position.
  HloPosition defining_position() const override;
  int64_t earliest_available_time() const override { return start_time(); }
  bool is_pinned_allocation() const override { return true; }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  bool is_window_prefetched_allocation() const override { return false; }
  bool is_scoped_allocation() const override { return false; }
  bool is_reserved_allocation() const override { return false; }
  absl::Status Process(const BitcastSplitFn& bitcast_split_fn) override;
  absl::Status PostProcess() override { return absl::OkStatus(); }
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  bool operator==(const PinnedAllocation& other) const;
};

// This class represents an allocation that is used to reserve a chunk of
// memory. If an HloPosition or an HloUse is colored in alternate memory, to
// make sure we are able to satisfy the coloring requirements, we reserve a
// chunk in the alternate memory before we start processing the buffers in
// sorted order. The reserved chunk serves as a fallback in case we are not able
// to satisfy the coloring requirements using the buffers in sorted order.
class ReservedAllocation final : public Allocation {
 public:
  ReservedAllocation(HloPosition defining_position, HeapSimulator::Chunk chunk,
                     int64_t start_time, int64_t end_time);

  // Overridden methods
  //
  // Returns the original defining position.
  HloPosition defining_position() const override;
  int64_t earliest_available_time() const override { return start_time(); }
  bool is_pinned_allocation() const override { return false; }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  bool is_window_prefetched_allocation() const override { return false; }
  bool is_scoped_allocation() const override { return false; }
  bool is_reserved_allocation() const override { return true; }
  absl::Status Process(const BitcastSplitFn& bitcast_split_fn) override;
  absl::Status PostProcess() override { return absl::OkStatus(); }
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  bool operator==(const ReservedAllocation& other) const;

  bool is_chunk_reserved_in_interval_tree() const { return reserved_; }
  void chunk_freed_in_interval_tree() { reserved_ = false; }

 private:
  // Indicates whether the chunk is still reserved in the interval_tree_.
  bool reserved_;
};

// This class represents an allocation as a result of an asynchronous copy.
// Note: CopyStart instructions are inserted after
// `copy_start_schedule_after`, while CopyDone instructions are inserted
// before `copy_done_schedule_before_time`.
class CopyAllocation final : public Allocation {
 public:
  CopyAllocation(
      Allocation& prev_allocation, MemorySpace memory_space,
      std::optional<HeapSimulator::Chunk> chunk,
      int64_t copy_start_schedule_after_time,
      int64_t copy_done_schedule_before_time, int64_t end_time,
      std::optional<int64_t> cross_program_prefetch_index = std::nullopt,
      HloInstruction* sync_mem_op = nullptr);

  // Overridden methods
  //
  HloPosition defining_position() const override;
  // Returns the time the buffer is first available to be used. For
  // CopyAllocation, this is when the copy ends, which is
  // copy_done_schedule_before.
  int64_t earliest_available_time() const override;
  bool is_pinned_allocation() const override { return false; }
  bool is_copy_allocation() const override { return true; }
  bool is_sliced_copy_allocation() const override { return false; }
  bool is_window_prefetched_allocation() const override { return false; }
  bool is_scoped_allocation() const override { return false; }
  bool is_reserved_allocation() const override { return false; }
  absl::Status Process(const BitcastSplitFn& bitcast_split_fn) override;
  absl::Status PostProcess() override { return absl::OkStatus(); }
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  const HloInstruction* sync_mem_op() const { return sync_mem_op_; }
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
  // The sync data movement instruction that this copy is associated with.
  HloInstruction* sync_mem_op_ = nullptr;
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
      absl::FunctionRef<Shape(const Shape&)> get_equivalent_s8_shape_fn,
      HloInstruction* sync_mem_op = nullptr);

  // Overridden methods
  //
  HloPosition defining_position() const override;
  // Returns the time the buffer is first available to be used. For
  // SlicedCopyAllocation, this is when all copies have ended.
  int64_t earliest_available_time() const override;
  bool is_pinned_allocation() const override { return false; }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return true; }
  bool is_window_prefetched_allocation() const override { return false; }
  bool is_scoped_allocation() const override { return false; }
  bool is_reserved_allocation() const override { return false; }
  // MemorySpaceAssignment::Process() calls Process(const BitcastSplitFn&
  // bitcast_split_fn) to create asynchronous slice copies, and a bitcast-concat
  // call to glue the slices back together.
  absl::Status Process(const BitcastSplitFn& bitcast_split_fn) override;
  absl::Status PostProcess() override { return absl::OkStatus(); }
  // Marks the allocation as needed.
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  const HloInstruction* sync_mem_op() const { return sync_mem_op_; }
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
  // The sync data movement instruction that this copy is associated with.
  HloInstruction* sync_mem_op_ = nullptr;
};

// This class represents an allocation resulting from asynchronously prefetching
// a window buffer. When a tensor is placed in the default memory, we can
// prefetch the window buffer of the tensor to the alternate memory space. This
// is called window prefetching.
class WindowPrefetchedAllocation final : public Allocation {
 public:
  struct Options {
    int64_t bytes = 0;
    int64_t uid = 0;
    int64_t alternate_memory_space = 0;
    std::function<void(HloInstruction*, int64_t, int64_t)>
        notify_operand_appended_fn =
            [](const HloInstruction*, int64_t, int64_t) {};
  };

  WindowPrefetchedAllocation(Allocation& prev_allocation, HloUse use,
                             const HeapSimulator::Chunk& chunk,
                             int64_t prefetch_start_schedule_after_time,
                             int64_t prefetch_done_schedule_before_time,
                             const Options& options);

  // Overridden methods
  //
  HloPosition defining_position() const override;
  int64_t earliest_available_time() const override;
  bool is_pinned_allocation() const override { return false; }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  bool is_window_prefetched_allocation() const override { return true; }
  bool is_scoped_allocation() const override { return false; }
  bool is_reserved_allocation() const override { return false; }
  // MemorySpaceAssignment::Process() calls Process(const BitcastSplitFn&
  // bitcast_split_fn) to create asynchronous window prefetches.
  absl::Status Process(const BitcastSplitFn& bitcast_split_fn) override;
  absl::Status PostProcess() override { return absl::OkStatus(); }
  // Marks the allocation as needed.
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const WindowPrefetchedAllocation& other) const;
  bool operator==(const Allocation& other) const override;
  int64_t bytes() const { return bytes_; }
  int64_t prefetch_start_schedule_after() const {
    return prefetch_start_schedule_after_;
  }
  int64_t prefetch_done_schedule_before() const {
    return prefetch_done_schedule_before_;
  }
  HloInstruction* prefetch() const { return prefetch_instruction_; }

 private:
  // This method is called by Process(const BitcastSplitFn& bitcast_split_fn) to
  // create window prefetch instructions. These instructions include a pair of
  // async WindowPrefetch which is passed to the fusion.
  absl::Status InsertWindowPrefetchInstruction(
      HloInstruction* producing_instruction, HloInstruction* use_instruction,
      HloComputation* computation);

  Options options_;
  HloInstruction* prefetch_instruction_ = nullptr;
  Allocation& prev_allocation_;
  HloUse use_;
  int64_t prefetch_start_schedule_after_;
  int64_t prefetch_done_schedule_before_;
  int64_t bytes_;
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
  bool is_pinned_allocation() const override { return false; }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  bool is_window_prefetched_allocation() const override { return false; }
  bool is_scoped_allocation() const override { return false; }
  bool is_reserved_allocation() const override { return false; }
  absl::Status Process(const BitcastSplitFn& bitcast_split_fn) override;
  absl::Status PostProcess() override { return absl::OkStatus(); }
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
  bool is_pinned_allocation() const override { return false; }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  bool is_window_prefetched_allocation() const override { return false; }
  bool is_scoped_allocation() const override { return false; }
  bool is_reserved_allocation() const override { return false; }
  absl::Status Process(const BitcastSplitFn& bitcast_split_fn) override;
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

// An allocation representing scoped alternate memory.
class ScopedAllocation final : public Allocation {
 public:
  // is_post_module is true if the allocation is for a scoped allocation that
  // is used after the module.
  ScopedAllocation(HeapSimulator::Chunk chunk, int64_t allocation_time,
                   HloInstruction* defining_instruction, bool is_post_module);

  // Overridden methods
  HloPosition defining_position() const override;
  int64_t earliest_available_time() const override { return start_time(); }
  bool is_pinned_allocation() const override { return false; }
  bool is_copy_allocation() const override { return false; }
  bool is_sliced_copy_allocation() const override { return false; }
  bool is_window_prefetched_allocation() const override { return false; }
  bool is_scoped_allocation() const override { return true; }
  bool is_reserved_allocation() const override { return false; }
  absl::Status Process(const BitcastSplitFn& bitcast_split_fn) override;
  absl::Status PostProcess() override { return absl::OkStatus(); }
  void MarkIfNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  void MarkNeeded(absl::flat_hash_set<const Allocation*>& needed_allocations)
      const override;
  std::string ToString() const override;
  bool operator==(const Allocation& other) const override;

  // New non-virtual methods
  bool operator==(const ScopedAllocation& other) const;
  bool is_post_module() const { return is_post_module_; }

 private:
  bool is_post_module_;
};

// A class with some utility functions that are useful in debugging.
struct AllocationSequenceDebugging {
  // Developers can call this method to log all the allocations in alternate
  // memory, at a given instruction time.
  //
  // REQUIRED:
  // - This method is intended to be called before MSA modifies the HloModule.
  static void LogAltMemAllocationsAt(const AllocationSequence& allocations,
                                     int64_t time);
};

}  // namespace xla::memory_space_assignment

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALLOCATION_H_
