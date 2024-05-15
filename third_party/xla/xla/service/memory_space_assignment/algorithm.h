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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALGORITHM_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALGORITHM_H_

#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// TODO(b/210891274): Use btree_map after build issue in Windows is resolved.
#if defined(__GNUC__) || defined(__clang__)
#include "absl/container/btree_map.h"
#endif
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/buffer_value.h"
#include "xla/service/call_graph.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/buffer_interval_comparator.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/service/memory_space_assignment/slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

// AllocationValue is used to break up HloValues for each non-trivial position
// (trivial positions are considered Tuple, GetTupleElement, and Bitcast). An
// HloValue may include positions and uses that alias with each other across
// multiple computations. We use this class to break these HloValues such that
// every AllocationValue has one defining position (that may alias with other
// AllocationValues). The uses field of the AllocationValue contains only the
// direct uses of the AllocationValue's defining position.
//
// For example, consider the following HLO snippet:
//
// Body {
//   body_param = (f32[4,3]{1,0}, f32[]) parameter(0)
//   get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element(body_param),
//   index=0
//   ...
//   ROOT tuple = (f32[4,3]{1,0}, f32[]) tuple(get-tuple-element.3, ...)
// }
//
// Cond {
//   cond_param = (f32[4,3]{1,0}, f32[]) parameter(0)
//   ...
// }
//
// add.4 = f32[4,3]{1,0} add(...)
// tuple.1 = (f32[4,3]{1,0}, f32[]) tuple(add.4, ...)
// while = (f32[4,3]{1,0}, f32[]) while(tuple.1), body=Body, condition=Cond
// get-tuple-element.5 = f32[4,3]{1,0} get-tuple-element(while), index=0
// add.5 = f32[4,3]{1,0} add(get-tuple-element.5, ...)
//
// This contains an HloValue that looks like the following:
// positions:
//  add.4
//  body_param {0}
//  get-tuple-element.3
//  tuple {0}
//  cond_param {0}
//  tuple.1 {0}
//  while {0}
//  get-tuple-element.5
// uses:
//  add.1, operand 0
//  tuple, operand 0
//  while, operand 0 {0}
//  add.5, operand 0
//
// We break this HloValue up into the following AllocationValues for each
// non-trivial position:
// AllocationValue1: computation = Entry
//  position:
//   add.4
//  uses:
//   while, operand 0 {0}
// AllocationValue2: computation = Cond
//  position:
//   cond_param {0}
//  uses:
// AllocationValue3: computation = Body
//  position:
//   body_param {0}
//  uses:
//   add.1, operand 0
//   tuple, operand 0
// AllocationValue4: computation = Entry
//  position:
//   while {0}
//  uses:
//   add.5, operand 0
class AllocationValue {
 public:
  // This data structure wraps an HloUse and adds additional metadata that are
  // useful for allocation.
  struct Use {
    // The wrapped HloUse object.
    HloUse hlo_use;
    // The logical time this use is scheduled.
    int64_t time;
    // All the positions where this use aliases with. The aliased positions
    // must get the same allocation.
    std::vector<HloPosition> aliases;

    bool operator==(const Use& other) const {
      return hlo_use == other.hlo_use && time == other.time &&
             aliases == other.aliases;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Use& s) {
      return H::combine(std::move(h), s.hlo_use, s.time, s.aliases);
    }
  };

  AllocationValue(const HloValue* value, const HloPosition& position,
                  int64_t size)
      : value_(value),
        defining_position_(position),
        size_(size),
        requires_contiguous_allocation_(false) {}

  const HloPosition& defining_position() const { return defining_position_; }
  const HloInstruction* defining_instruction() const {
    return defining_position().instruction;
  }
  int64_t size() const { return size_; }
  const std::vector<Use>& uses() const { return uses_; }
  std::vector<Use>& uses() { return uses_; }
  const HloValue* value() const { return value_; }
  const HloComputation* computation() const {
    return defining_instruction()->parent();
  }
  AllocationSequence* mutable_allocation_sequence() {
    return &allocation_sequence_;
  }
  const AllocationSequence* allocation_sequence() const {
    return &allocation_sequence_;
  }

  // Sets/gets whether this AllocationValue requires allocating it
  // contiguously throughout its live range (without any copies).
  bool requires_contiguous_allocation() const {
    return requires_contiguous_allocation_;
  }
  void set_requires_contiguous_allocation(bool requires_contiguous_allocation) {
    requires_contiguous_allocation_ = requires_contiguous_allocation;
  }

  void AddUse(const HloUse& use, int64_t use_time) {
    uses_.push_back({use, use_time, {}});
  }

  std::string ToString() const;
  std::string ToShortString() const;

 private:
  const HloValue* value_;
  HloPosition defining_position_;
  int64_t size_;
  // If true, there must be a contiguous allocation for this buffer without
  // any copies.
  bool requires_contiguous_allocation_;
  std::vector<Use> uses_;
  AllocationSequence allocation_sequence_;
};

// A struct representing an asynchronous copy with its logical start and end
// time (time that copy done is scheduled), the resource this copy would use,
// its destination memory space, and a unique ID.
struct AsynchronousCopy {
  int64_t exclusive_start_time;
  int64_t end_time;
  float resource;
  MemorySpace destination;
  int64_t id;

  std::tuple<int64_t, int64_t, float, MemorySpace, int64_t> AsTuple() const {
    return std::make_tuple(exclusive_start_time, end_time, resource,
                           destination, id);
  }
};

// Compare asynchronous copies such that an earlier start time has the same or
// earlier end time and an earlier end time has the same or earlier start time.
bool operator<(const AsynchronousCopy& a, const AsynchronousCopy& b);

bool operator==(const AsynchronousCopy& a, const AsynchronousCopy& b);
bool operator!=(const AsynchronousCopy& a, const AsynchronousCopy& b);

// Helper class to enforce asynchronous copy ordering. If the appropriate option
// is enabled, we only allow asynchronous copies that are pipelined: if an
// asynchronous copy ends earlier than another asynchronous copy, it must start
// the same time or earlier than the other asynchronous copy; and if an
// asynchronous copy starts earlier than another asynchronous copy, it must end
// the same time or earlier than the other asynchronous copy.
class AsynchronousCopyOrdering {
 public:
  AsynchronousCopyOrdering() = default;

  // Adds an asynchronous copy.
  void AddCopy(const AsynchronousCopy& copy);

  // Removes an asynchronous copy. CHECKs that it is removed.
  void RemoveCopy(const AsynchronousCopy& copy);

  // Returns true if the addition of an asynchronous copy in the given time
  // interval would violate the asynchronous copy ordering. E.g., consider the
  // following scenario:
  //                                  CS          CD
  //  already committed async copy:   +-----------+
  //                new async copy:     +--------+
  //
  // The new asynchronous copy would violate the ordering guarantee because the
  // copy start is after an already committed asynchronous copy while its copy
  // done is before the committed copy.
  bool ViolatesOrdering(int64_t exclusive_start_time, int64_t end_time) const;

 private:
  // We use this data structure for keys into the map that has a custom
  // comparator for the ordering guarantees.
  struct Interval {
    int64_t exclusive_start_time;
    int64_t end_time;

    // We allow multiple prefetches that have one or both of the same start and
    // end times. std::map considers two values as equal if neither are less
    // than the other.  Using this comparator, we can ensure that the only
    // intervals that evaluate to be equal are those with the same start and end
    // times or those with intervals that violate the FIFO order.
    bool operator<(const Interval& other) const {
      return (exclusive_start_time < other.exclusive_start_time &&
              end_time <= other.end_time) ||
             (exclusive_start_time <= other.exclusive_start_time &&
              end_time < other.end_time);
    }
  };
  // Stores asynchronous copies in a tree set respecting the pipelining order.
  std::map<Interval, std::set<AsynchronousCopy>> ranges_;
};

// Helper class to enforce asynchronous copy resources by keeping track of
// available copy bandwidth and elapsed times of overlapped operations. It
// maintains a list of initial resources that correspond to the elapsed times of
// overlapped operations. As asynchronous copies are added, the available
// resource is subtracted to keep track of the current state.
class AsynchronousCopyResource {
 public:
  // A specification of needed asynchronous copy resources.
  struct ResourceSpec {
    int64_t exclusive_start_time;
    int64_t end_time;
    float resource;
  };

  AsynchronousCopyResource() = default;

  // The constructor needs the initial resources.
  explicit AsynchronousCopyResource(absl::Span<const float> initial_resources)
      : initial_resources_(initial_resources.begin(), initial_resources.end()),
        delay_(initial_resources.size(), 0) {}

  // Adds the given asynchronous copy and updates the current resources. CHECK
  // fails if there aren't enough resources to satisfy this copy (the caller
  // should use HasEnoughResource first to ensure there is enough resource).
  void AddCopy(const AsynchronousCopy& copy);

  // Removes the given copy and frees the resource.
  void RemoveCopy(const AsynchronousCopy& copy);

  // Returns true if a copy with the given start and end times and resource can
  // be satisfied.
  bool HasEnoughResource(int64_t exclusive_start_time, int64_t end_time,
                         float resource);

  // Returns true if a set of copy specifications can be satisfied in the
  // order specified.
  bool HasEnoughResourceMultiCheck(const std::vector<ResourceSpec>& specs);

  // This is only used for debugging and testing purposes, it returns the
  // currently available resource at each logical time.
  std::vector<float> GetCurrentResources() const {
    std::vector<float> current_resources(initial_resources_.begin(),
                                         initial_resources_.end());
    for (int i = 0; i < current_resources.size(); ++i) {
      current_resources[i] -= std::min(current_resources[i], delay_[i]);
    }
    return current_resources;
  }

  // A useful debugging tool for printing several pieces of information about
  // AsynchronousCopyResource.
  std::string Dump(int64_t start_time, int64_t end_time,
                   MemorySpace memory_space_filter) const;

 private:
  // Internal helper method to implement adding/removing/checking resources.
  // ConsumeResource() may modify delay_. If delay_change_map is not null,
  // for any change to delay_[i], {i, delay_[i]} will be added to
  // delay_change_map, allowing callers to undo any modifications.
  bool ConsumeResource(
      int64_t exclusive_start_time, int64_t end_time, float resource,
      absl::flat_hash_map<int64_t, float>* delay_change_map = nullptr,
      float resource_to_free = 0.0);

  // Same as the public RemoveCopy except it works on the async_copies_
  // iterator. Assumes copy_it points to the last copy for its start time;
  // otherwise the public RemoveCopy method is supposed to temporarily remove
  // these later copies that share the same start time before removing the
  // requested copy.
  void RemoveCopy(std::list<AsynchronousCopy>::iterator& copy_it);

  // We maintain a linked list of asynchronous copies sorted by the start times.
  // This allows us to efficiently find the copy that starts right after another
  // one because adding a copy might push a copy further into the future.
  std::list<AsynchronousCopy> async_copies_;
// To make the lookups into async_copies_ more efficient, we also maintain a
// binary tree that is indexed by the start time, containing iterators into
// async_copies_.
// TODO(b/210891274): Use btree_map after build issue in Windows is resolved.
#if defined(__GNUC__) || defined(__clang__)
  absl::btree_map<int64_t, std::list<AsynchronousCopy>::iterator>
      async_copy_time_map_;
#else
  std::map<int64_t, std::list<AsynchronousCopy>::iterator> async_copy_time_map_;
#endif
  std::vector<float> initial_resources_;
  std::vector<float> delay_;
};

// This class inherits from GlobalDecreasingSizeBestFitHeap with a notion of
// maximum size.
class MsaAlgorithm : public GlobalDecreasingSizeBestFitHeap<HloValue> {
 public:
  using HloPositionOrUse = std::variant<HloPosition, HloUse>;

  MsaAlgorithm(AllocationSequence* allocations, const Options& options,
               const HloAliasAnalysis& alias_analysis,
               const HloLiveRange& hlo_live_range);

  // Allocates a buffer in preferred memory with whole program lifetime and
  // enables prefetching prefetch_candidate from default memory across program
  // boundaries.
  void AllocateCrossProgramPrefetchBuffer(
      HloModule* module, const MsaBufferInterval& prefetch_candidate);

  absl::StatusOr<HeapSimulator::Result<HloValue>> Finish() override;

 protected:
  // Given a buffer interval, returns the colocated intervals. Unlike the
  // similar GlobalDecreasingSizeBestFitHeap::GetTransitiveColocations, it
  // returns the colocated intervals sorted by scheduled time.
  std::vector<const MsaBufferInterval*> GetSortedColocatedIntervals(
      const MsaBufferInterval& interval) const;

  // Given a MsaBufferInterval, creates AllocationValue objects and
  // corresponding AllocationSequences and appends them into
  // allocation_sequence_list_.
  void CreateAllocationValues(
      const MsaBufferInterval& buffer_interval,
      std::vector<AllocationValue>& allocation_values) const;

  // Given colocated intervals, populates allocation_values with the
  // corresponding AllocationValue objects.
  virtual void CreateAllocationValuesFromColocatedIntervals(
      absl::Span<const MsaBufferInterval* const> colocated_intervals,
      std::vector<AllocationValue>& allocation_values);

  // Go through all the uses in the AllocationValues and find the aliasing
  // positions.
  void FindAliases(std::vector<AllocationValue>* allocation_values) const;

  AllocationSequence* allocations() { return allocations_; }
  const Options& options() const { return options_; }
  const HloAliasAnalysis& alias_analysis() { return alias_analysis_; }
  const HloLiveRange& hlo_live_range() { return hlo_live_range_; }

 private:
  // We inherit AllocationBlock struct to attach the Allocation information to
  // make importing repacked offsets easier.
  struct RepackAllocationBlock : AllocationBlock {
    Allocation* allocation;
  };

  // A data structure we use to associate Allocation objects that are aliased
  // and must get the same offset.
  struct AliasedOffset {
    int64_t offset;
    absl::flat_hash_set<const Allocation*> allocations;
  };

  // An allocation request for a use segment. A use segment is the time segment
  // between the definition and the first use, and the time segment between the
  // uses of a buffer. For example, the time between the definition and Use1, is
  // the first segment, and the time between Use1 and Use2 is the second segment
  // and so on:
  //
  //        +------+----------+-------+
  //       /        \          \       \
  //      /          v          v       v
  //    Def         Use1       Use2    Use3
  //     <----------> <--------> <----->
  //        Segment    Segment   Segment
  //
  // start_time and end_time are the start and end logical times of the segment.
  // use_times is a sorted sequence of the times of all uses.
  // latest_prefetch_time is the latest time we can schedule the CopyDone for a
  // prefetch.
  // If allow_no_copy_alternate_mem_allocation is false, an eviction is forced.
  // If earliest_prefetch_time is set, prefetches cannot start before this
  // value.
  struct AllocationRequest {
    int64_t inclusive_start_time;
    int64_t end_time;
    int64_t latest_prefetch_time;
    int64_t size;
    bool prefer_no_copy_alternate_mem_allocation;
    bool allow_no_copy_alternate_mem_allocation;
    bool require_no_copy_alternate_mem_allocation;
    bool allow_prefetch;
    std::optional<int64_t> earliest_prefetch_time;
    std::optional<int64_t> preferred_prefetch_time;
    AliasedOffset* preferred_offset;
    const AllocationValue::Use* use;
    AllocationValue* allocation_value;
    absl::Span<const int64_t> all_use_times;
  };

  // This struct contains mandatory memory assignments at a given time. E.g., an
  // input's required memory assignment time would correspond to the definition
  // time of the parameter instruction, and an output's time would correspond to
  // the time of last use.
  struct RequiredMemoryAssignment {
    MemorySpace memory_space;
    int64_t time;
    AliasedOffset* offset;

    bool equals_ignoring_time(const RequiredMemoryAssignment& other) const {
      return memory_space == other.memory_space && offset == other.offset;
    }

    bool operator==(const RequiredMemoryAssignment& other) const {
      return memory_space == other.memory_space && time == other.time &&
             offset == other.offset;
    }

    bool operator!=(const RequiredMemoryAssignment& other) const {
      return !(*this == other);
    }
  };

  // A struct that contains a pointer to loop-optimized allocation along with
  // essential data about the loop itself.
  struct LoopOptimizedAllocationInfo {
    // The use_idx is the instruction index of the use within the loop.
    int64_t use_index;
    // The number of instructions in one iteration of the loop. We use use_index
    // and loop_size to calculate when exactly to schedule a prefetch
    // instruction.
    int64_t loop_size;
    // A pointer into an Allocation in loop_optimized_allocations_.
    const Allocation* loop_optimized_allocation;
  };

  // A context object that is used to share state amongst the methods that
  // implement Prefetch(). Prefetch tries to find both a sliced solution and an
  // unsliced solution at the same time. We store both in this structure.
  struct PrefetchContext {
    // Prefetching is designed to operate on a SlicedBufferInterval that is
    // backed by a standard MsaBufferInterval, even if the number of slices
    // == 1. WorkingIntervals is used to store a SlicedBufferInterval and its
    // backing MsaBufferInterval.
    struct WorkingIntervals {
      MsaBufferInterval full;
      // sliced is a unique_ptr because it won't necessarily be initialized
      // when the WorkingBufferIntervals are created, and there is no way to
      // create an empty SlicedBufferInterval.
      std::unique_ptr<SlicedBufferInterval> sliced;
    };

    struct SlicedSolution {
      // When we talk about a slice, we think of spatial slices, where each
      // slice is allocated at different times. The following example shows
      // 3 slices that are used to form a contiguous buffer from [p0, p3]
      //
      //   space
      //    ^
      // p3 |       +-----------+
      //    |       |    s2     |
      // p2 |   +---+-----------+
      //    |   |      s1       |
      // p1 |   +-------+-------+
      //    |           |  s0   |
      // p0 |           +-------+
      //    +---|---|---|---|---|----> time
      //        t0  t1  t2  t3  t4
      std::vector<SliceDecision> slice_decisions_sorted_by_start_time;

      // In order to support colocated buffer calculations, we need to add a
      // MsaBufferInterval-Chunk pair to pending_chunks_, such that:
      // - The duration of the MsaBufferInterval is non-zero.
      // - All slices have been allocated by the start of the MsaBufferInterval.
      // - The MsaBufferInterval ends at the end time for all slices.
      // - The Chunk covers the space allocated for all slices.
      //
      // In order to meet that requirement,
      // we create MsaBufferInterval-Chunk pairs from
      // slice_decisions_sorted_by_start_time that meet those requirement but do
      // not cause any memory to be allocated in more than one Chunk at a time.
      // The result is stored in slices_for_pending_chunks.
      //
      // The illustration below demonstrates how we would construct such
      // MsaBufferInterval-Chunk pairs from the
      // slice_decisions_sorted_by_start_time example above.
      //
      //   space
      //    ^
      // p3 |       +---+---+---+
      //    |       |c2 |       |
      // p2 |   +---+---+       |
      //    |   |  c0   |   c2  |
      // p1 |   +-------+       |
      //    |           |       |
      // p0 |           +-------+
      //    +---|---|---|---|---|----> time
      //        t0  t1  t2  t3  t4
      std::vector<std::pair<MsaBufferInterval, Chunk>>
          slices_for_pending_chunks;

      // The prefetch_picker_debug_string will only be set with the appropriate
      // VLOG level.
      std::string prefetch_picker_debug_string;
    };

    struct UnslicedSolution {
      Chunk chunk_candidate;    // The chunk chosen for the solution.
      float prefetch_resource;  // The amount of required prefetch resource.
      // The prefetch_picker_debug_string will only be set with the appropriate
      // VLOG level.
      std::string prefetch_picker_debug_string;
    };

    WorkingIntervals& GetMutableWorkingIntervals(bool for_sliced_solution) {
      if (for_sliced_solution) {
        return sliced_solution_intervals;
      }
      return unsliced_solution_intervals;
    }

    const WorkingIntervals& GetWorkingIntervals(
        bool for_sliced_solution) const {
      if (for_sliced_solution) {
        return sliced_solution_intervals;
      }
      return unsliced_solution_intervals;
    }

    // Parameters to Prefetch().
    const AllocationRequest* request;
    Allocation* prev_allocation_in_default_mem;

    // Intermediate calculations common to both the sliced and unsliced
    // solutions.
    int64_t exclusive_prefetch_start_time = -1;
    int64_t prefetch_end_time = -1;
    const Shape* full_shape;
    int64_t extra_async_copy_limit = 0;
    // As a compilation time optimization, store the prefetch start time where
    // we have first seen out of memory. There is no point of exploring prefetch
    // start times earlier than this point.
    std::optional<int64_t> exclusive_out_of_mem_start = std::nullopt;

    // Data structures used to compute and store the sliced solution.
    std::optional<SliceProposalCollection> slice_proposal_collection =
        std::nullopt;
    WorkingIntervals sliced_solution_intervals;
    std::optional<SlicedSolution> sliced_solution;

    // Data structures used to compute and store the unsliced solution.
    WorkingIntervals unsliced_solution_intervals;
    std::optional<UnslicedSolution> unsliced_solution;
  };

  // Result of an allocation, prefetch, eviction etc. request.  The result is
  // either kSuccess or a bitwise OR of one or more failures. The values are
  // unique powers of two. To check if a result contains a particular failure,
  // use the result_is method. To add a new failure to a result, use the
  // result_mark method.
  enum class Result {
    // Successful allocation.
    kSuccess = 0,
    // Allocation failed because we ran out of alternate memory.
    kFailOutOfMemory = 1,
    // A no-copy allocation couldn't be performed because the previous
    // allocation wasn't in the alternate memory space.
    kFailPrevAllocationNotInAlternateMem = 2,
    // A no-copy allocation couldn't be performed because the live range was too
    // long.
    kFailLiveRangeTooLong = 4,
    // A prefetching couldn't be performed because the live range was too short.
    kFailLiveRangeTooShort = 8,
    // Ran out of outstanding asynchronous copy limit either during prefetching
    // or eviction.
    kFailOutOfAsyncCopies = 16,
    // A prefetching couldn't be performed because the asynchronous copy
    // resource was violated.
    kFailViolatesAsyncCopyResource = 32,
    // An allocation failure happened that requires uncommitting all the pending
    // allocations. Usually this is due to a situation requiring an eviction but
    // the eviction couldn't be performed.
    kFailRequiresUncommit = 64,
    // For prefetching, indicates that all slices have the same start time, in
    // which case, we fallback to an unsliced solution.
    kAllSlicesHaveTheSameStartTime = 128,
    // There were conflicting preferred offsets.
    kFailConflictingPreferredOffsets = 256
  };

  // Return true if the result belongs to a failure.
  static bool result_is(Result result, Result failure) {
    return static_cast<int>(result) & static_cast<int>(failure);
  }

  // Mark (bitwise OR) a failure to the result.
  static Result result_mark(Result failure, Result& result) {
    result = static_cast<Result>(static_cast<int>(result) |
                                 static_cast<int>(failure));
    return result;
  }

  // Return true if the result is a failure that requires us to uncommit pending
  // chunks.
  static bool result_requires_uncommit(Result result) {
    return result_is(result, Result::kFailRequiresUncommit);
  }

  // Return true if the result is a failure either due to running out of
  // outstanding asynchronous copies or due to violating asynchronous copy
  // ordering.
  static bool result_failed_because_of_async_copy(Result result) {
    return result_is(result, Result::kFailOutOfAsyncCopies) ||
           result_is(result, Result::kFailViolatesAsyncCopyResource);
  }

  // For the given loop with the start and end index and loop size, run the
  // MemoryBoundLoopOptimizer and record its outputs into
  // optimized_allocations_map_.
  absl::Status OptimizeMemoryBoundLoop(int loop_start_idx, int loop_end_idx,
                                       int loop_size);

  // Identify memory-bound loops in the graph and call OptimizeMemoryBoundLoop
  // for the found loops.
  void IdentifyAndOptimizeMemoryBoundLoops();

  // Allocates buffers for instructions that need reserved scoped allocations in
  // the alternate memory space.
  void AllocateReservedScopedAllocations();

  // Returns the AliasedOffset object associated with the allocation.
  AliasedOffset* GetAliasedOffset(const Allocation& allocation);

  // If aliased_offset is non-null, this method adds the allocation to
  // aliased_offset. Otherwise, it creates a new AliasedOffset object and adds
  // the allocation to this new AliasedOffset.
  void CreateOrAddToAliasedOffset(const Allocation& allocation,
                                  AliasedOffset* aliased_offset);

  // Given an allocation sequence, returns the live allocation at time with a
  // preference towards allocations in alternate memory. Returns nullptr if no
  // allocation is alive at that time.
  static Allocation* GetLiveAllocationAt(const AllocationSequence& allocations,
                                         int64_t time);

  // Returns true if the use is allowed in the alternate memory.
  bool IsUseAllowedInAlternateMemory(const AllocationValue& value,
                                     const HloUse& use) const;

  // Adjusts the preferred memory offset for a given use, taking aliasing
  // constraints into account. If the use already has a preferred offset in the
  // alternate memory space (e.g., due to prior allocations), the offset derived
  // from aliasing considerations must match the existing preferred offset.
  AliasedOffset* UpdatePreferredOffsetForUse(
      const AllocationValue::Use& use, AliasedOffset* preferred_offset) const;

  // Propagate the allocation at the use time to any aliases that this use might
  // have had.
  void UpdateAllocationRequirementForUseAliases(
      const AllocationValue& allocation_value, const AllocationValue::Use& use,
      int64_t use_time);

  // For while uses that are allocated in the alternate memory space, if
  // they also have an allocation in the default memory space in their
  // allocation sequence, create a "parent" allocation that mirrors this
  // default memory space allocation. When we process the parent
  // allocation, we add an additional parameter to the while that is a
  // reference to the buffer in the default memory space. With parent
  // allocations, we don't need to unnecessarily evict buffers since they
  // already have a copy in the default memory space. We search backwards
  // (latest to earliest in execution time) for a suitable allocation in
  // order to find the most recent one.
  void MaybeCreateMirroredParentAllocationForWhileUse(
      const AllocationValue& allocation_value, const AllocationValue::Use& use,
      int64_t use_time, absl::Span<AllocationValue> allocation_values,
      absl::flat_hash_map<const HloComputation*, AliasedOffset*>&
          preferred_offset_for_computation);

  // Creates a detailed memory allocation request for a given use of an
  // allocation value. Analyzes the usage pattern of the use to determine if it
  // can be placed in alternate memory, considering the restrictions for loops
  // and conditionals. Also calculates the timing for prefetching, taking into
  // account instruction schedules, operation type (e.g., sequential vs.
  // non-sequential calls), and prior usage patterns.
  AllocationRequest CreateAllocationRequest(
      AllocationValue& allocation_value, const AllocationValue::Use& use,
      const AllocationValue::Use* previous_use, AliasedOffset* preferred_offset,
      int64_t definition_time, bool require_no_copy_alternate_mem_allocation,
      const std::vector<int64_t>& all_use_times);

  // Finds allocations for allocation values generated from colocated intervals.
  // All of the allocation values have a must-alias relationship with each
  // other. Returns either kSuccess if all of the sites could be placed in the
  // alternate memory or a bitwise OR of failure reasons why they couldn't
  absl::StatusOr<Result> AllocateAllocationValues(
      absl::Span<AllocationValue> allocation_values);

  // Finds an allocation for an allocation request for a segment (see the
  // documentation for AllocationRequest above how a segment is defined).
  //
  // It performs three things in the following order:
  //  1- Allocate the allocation request entirely in the alternate memory, if
  //     there is enough space and if the prefetch interval picker allows.
  //  2- If (1) was unsuccessful, and the only allocation for
  //     this buffer was in the alternate memory, we try to perform a prefetch.
  //  3- If (1) was unsuccessful, prefetch the buffer into the alternate memory,
  //     if there is enough space and if the prefetch interval picker allows.
  //
  // If an eviction (2) was requested and was unsuccessful, this method returns
  // Result::kFailRequiresUncommit. This means we could not find a suitable
  // allocation, so all previous allocations for this buffer must be removed and
  // allocated in the default memory. Otherwise, this method may return
  // Result::kSuccess if the buffer could be placed in alternate memory or some
  // other Result with an OR of reasons why the buffer couldn't be placed in
  // alternate memory.
  Result AllocateSegment(const AllocationRequest& request);

  // Try allocating in alternate memory without any copies.
  Result AllocateInAlternateMemoryNoCopy(const AllocationRequest& request);

  // Try evicting to default memory space.
  Result Evict(const AllocationRequest& request);

  // Returns the time a copy done of a prefetch should be scheduled.
  int64_t FindPrefetchEndTime(const AllocationRequest& request,
                              int64_t earliest_prefetch_time) const;

  // Try prefetching to alternate memory space.
  Result Prefetch(const AllocationRequest& request,
                  Allocation& prev_allocation_in_default_mem);

  // Helper methods used to implement Prefetch().
  //
  // Generates a SliceProposal in context, if options dictate and one can be
  // constructed.
  void GenerateSliceProposal(PrefetchContext& context) const;
  // Calls GenerateSliceProposal to potentially create a SliceProposal, and
  // sets up WorkingIntervals for a sliced and unsliced solution. Updates
  // context.
  void SetupPrefetchWorkingIntervalsAndSliceProposal(
      PrefetchContext& context) const;
  // Initializes the PrefetchIntervalPicker and associated data structures in
  // context.
  Result InitializePrefetchIntervalPicker(PrefetchContext& context);
  // As a compile time optimization, try a prefetch allocation that is as late
  // as possible. If this is not able to find a solution, none of the
  // earlier tries will succeed either.
  Result EnsureSomeSpatialPrefetchFitExists(PrefetchContext& context) const;
  // Check if for the specified type of solution, using the parameters in
  // context. If we find a solution, it will be stored in context.
  Result CheckPrefetchFit(bool for_sliced_solution, PrefetchContext& context);
  // Creates a debugging string describing the timing of the prefetch solution
  // we are currently attempting (as dictated by for_sliced_solution and
  // context).
  std::string AlternateMemoryAllocationAttemptToString(
      bool for_sliced_solution, const PrefetchContext& context) const;

  // Find the best possible chunk candidate, where it has the longest possible
  // availability if no preferred offset is given, or at the preferred_offset if
  // it is given.
  std::optional<Chunk> FindBestChunkCandidate(
      const AllocationRequest& request, const AliasedOffset* preferred_offset,
      MsaBufferInterval* alternate_mem_interval) const;
  // The same as FindBestChunkCandidate() but allocates the request in slices.
  // The ith returned chunk should be allocated at slice time i.
  std::vector<Chunk> FindBestChunkCandidates(
      const AllocationRequest& request, const AliasedOffset* preferred_offset,
      SlicedBufferInterval* alternate_mem_interval) const;

  // Returns the corrected schedule time of an HloUse. The corrected time is
  // equivalent to the actual time of the use instructions for all instructions
  // except for while and conditional instructions. For while instructions, the
  // corrected time is the time of the body parameter, and for conditional, the
  // corrected time is the time of the parameter of the earliest-scheduled
  // called computation.
  int64_t GetCorrectedUseTime(const HloUse& use) const;

  // Returns the required assignment at a particular time, if available.
  std::optional<RequiredMemoryAssignment> RequiredMemoryAssignmentAt(
      const HloValue* buffer, int64_t time) const;

  // Searches for aliases in the use for a required assignment, and returns it
  // if found.
  std::optional<RequiredMemoryAssignment> AliasedRequiredAssignmentForUse(
      const AllocationValue::Use& use) const;

  // Goes through the colocated intervals and adds any required assignment.
  void AddRequiredAssignmentsForColocatedIntervals(
      absl::Span<const MsaBufferInterval* const> colocated_intervals);

  // Propagates aliased required assignment for a given position.
  void AddAliasedRequiredAssignment(const HloInstruction* instruction,
                                    ShapeIndex index,
                                    const Allocation* aliased_allocation);

  // This sets a required assignment. CHECK fails if there is a conflicting
  // required assignment at the same time.
  void AddRequiredAssignment(const HloValue* value,
                             const HloInstruction* instruction,
                             MemorySpace memory_space, int64_t time,
                             AliasedOffset* offset = nullptr,
                             bool add_to_pending = true);
  void AddRequiredAssignment(const HloInstruction* instruction,
                             ShapeIndex index, MemorySpace memory_space,
                             AliasedOffset* offset = nullptr,
                             bool add_to_pending = true);
  void AddRequiredAssignment(const HloPosition& position,
                             MemorySpace memory_space,
                             AliasedOffset* offset = nullptr,
                             bool add_to_pending = true);
  void AddRequiredAssignment(const HloUse& use, MemorySpace memory_space,
                             AliasedOffset* offset = nullptr,
                             bool add_to_pending = true);

  // Adds input and outputs as required assignments.
  void AddInputAndOutputRequiredAssignments();

  // Returns a list of "linked" allocations in the alternate memory. Linked
  // allocations all share a common allocation site (a use or position) with
  // each other. This can be used to determine if a group of linked allocations
  // are considered efficient or not.
  std::vector<std::vector<const Allocation*>>
  GetLinkedAllocationsInAlternateMemory(
      absl::Span<const AllocationValue> allocation_values) const;

  // Returns allocation sites (use or position) that are allocated in the
  // alternate memory, but is considered inefficient.  These arise in the
  // context of in-place operation like dynamic-update-slice.  We will typically
  // have an allocation that has the DUS as a use, and another allocation that
  // has the DUS as a defining position. These two allocation will be part of
  // the same linked allocation group.
  //
  // One reason why an allocation site could be inefficient is because the
  // amount of data that is asynchronously copied (prefetch and eviction) is
  // much larger than the amount of data that is used by the HLOs. If we find
  // inefficient allocation sites, we can require these sites default memory
  // allocations and allocate them again.
  std::vector<HloPositionOrUse> GetInefficientAllocationSites(
      absl::Span<const AllocationValue> allocation_values) const;

  // Returns true if the colocated intervals in the argument are in a parameter
  // or root instruction of the entry computation and are reserved by the user
  // to be in the alternate memory space.
  bool AreIntervalsReservedInAlternateMemory(
      absl::Span<const MsaBufferInterval* const> colocated_intervals) const;

  // Since the allocations are recorded to the AllocationSequence, we don't
  // maintain result_ in GlobalDecreasingSizeBestFitHeap. Override AddToChunkMap
  // to avoid unnecessarily adding the chunk to the chunk map.
  //
  // Sliced prefetching requires that we override this method because we
  // associate more than one chunk with a buffer (i.e., 1 chunk per slice),
  // which would cause the original implementation of this method to CHECK fail.
  void AddToChunkMap(const HloValue* buffer, Chunk chunk) override {}

  // Returns true if the addition of num_additional_copies asynchronous copies
  // in the given time interval would violate the maximum number of asynchronous
  // copies. An extra  async copy limit can be provided to increase the limit of
  // asynchronous copies for this instance.
  bool ViolatesMaximumOutstandingAsyncCopies(
      int64_t inclusive_start_time, int64_t end_time, bool is_prefetch,
      int64_t extra_async_copy_limit = 0,
      int64_t num_additional_copies = 1) const;

  // Exports the allocations for repacking and puts them into the vector in the
  // parameter.
  void ExportAllocationsForRepacking(
      std::vector<AllocationBlock*>& allocations);

  // Update reserved scoped allocation size for instructions when their
  // operand/output has been allocated in alternate memory by invoking
  // reserved_scoped_memory_fn
  void UpdateReservedScopedAllocationSize();

  // Imports repacked allocations and updates the internal data structures
  // consistent with the new packing.
  void ImportRepackedAllocations();
  // Helper functions to implement ImportRepackedAllocations.
  void ImportRepackedNonSlicedAllocation(RepackAllocationBlock& block);
  void ImportRepackedSlicedAllocation(RepackAllocationBlock& block);
  absl::Status AreRepackedSlicesValid(const RepackAllocationBlock& block);

  // Adds an asynchronous copy to allocations.
  void AddAsyncCopy(
      Allocation& prev_allocation, MemorySpace memory_space,
      std::optional<Chunk> chunk, int64_t exclusive_start_time,
      int64_t end_time, int64_t copy_done_schedule_before_time,
      AllocationSequence* allocations, AliasedOffset* aliased_offset,
      float resource,
      std::optional<int> cross_program_prefetch_index = std::nullopt);

  // For prefetching, adds a SlicedCopyAllocation to allocations. Also updates
  // asynchronous copy data structures, prefetch_interval_tree_, and aliasing
  // data structures
  void AddAsyncSlicesForPrefetch(
      const Allocation& prev_allocation, AllocationSequence* allocations,
      AliasedOffset* aliased_offset,
      const std::vector<SliceDecision>& slice_decisions_sorted_by_start_time,
      int64_t prefetch_end_time, int64_t allocation_end_time);

  // This method is used for committing the chunk candidate but adding it to
  // pending_chunks_ so that we can "uncommit" them in case we need to roll back
  // this allocation sequence.
  void AddToPendingChunks(const MsaBufferInterval& buffer_interval,
                          const Chunk& chunk);
  // If we need to remove the allocations for this allocation sequence, this
  // removes pending chunks and asynchronous copies in the respective pending
  // buffers from the interval trees. If an allocation request returns
  // kFailRequiresUncommit, this method must be called.
  void UncommitPendingChunks(absl::Span<AllocationValue> allocation_values);

  // Finalizes the allocations where they can no longer be uncommitted.
  void FinalizeAllocations(absl::Span<AllocationValue> allocation_values);

  // Clears all pending chunks and asynchronous copies.
  void ClearPendingChunks();

  // Append buffer and allocation infos for debugging and dump it into a file,
  // if enabled.
  void AppendBufferInfoDebugString(const MsaBufferInterval& interval,
                                   std::string* debug_str) const;
  void AppendScopedAllocationBufferInfoDebugString(
      const HloInstruction* instruction, int64_t time, int64_t size,
      std::string& debug_str) const;
  void AppendAllocationInfoDebugString(const Allocation& allocation,
                                       std::string& debug_str) const;
  void DumpDebugStringsIfEnabled() const;

  // Returns the available heap size in the alternate memory.
  int64_t available_heap_size() const {
    return options_.max_size_in_bytes - reserved_in_bytes_;
  }

  // Returns the earliest time in the (exclusive_start_time, end_time) range
  // that a new allocation with the given size would fit in the alternate
  // memory. If it doesn't fit, it returns nullopt.
  std::optional<int> FindEarliestExclusiveTimeToSatisfyPeakMemory(
      int exclusive_start_time, int end_time, int64_t size) const;

  // Creates and returns a RepackAllocationBlock.
  static RepackAllocationBlock MakeRepackAllocationBlock(
      int64_t start_time, int64_t end_time, int64_t size,
      int64_t initial_offset, int64_t id, Allocation* allocation) {
    RepackAllocationBlock allocation_block;
    allocation_block.inclusive_start_time = start_time;
    allocation_block.end_time = end_time;
    allocation_block.size = size;
    allocation_block.offset = -1;
    allocation_block.initial_offset = initial_offset;
    allocation_block.id = id;
    allocation_block.next_colocated = nullptr;
    allocation_block.allocation = allocation;
    return allocation_block;
  }

  // Returns a vector of instructions that have the same fingerprint as this
  // instruction.
  const std::vector<const HloInstruction*>* GetRepeatedInstructionList(
      const HloInstruction* instruction) const;

  // Returns true if the interval is pinned in the alternate memory. Buffers are
  // pinned when their layout has the alternate memory space before MSA runs.
  bool IsIntervalPinnedToAlternateMemory(
      const MsaBufferInterval& interval) const;

  AllocationSequence* allocations_;
  const Options& options_;
  const HloAliasAnalysis& alias_analysis_;
  const HloLiveRange& hlo_live_range_;
  std::unique_ptr<CallGraph> call_graph_;
  // We use a interval tree to keep track of the number of outstanding
  // prefetches and evictions.
  BufferIntervalTree prefetch_interval_tree_;
  BufferIntervalTree eviction_interval_tree_;
  AsynchronousCopyOrdering async_copy_ordering_;
  AsynchronousCopyResource prefetch_async_copy_resource_;
  AsynchronousCopyResource eviction_async_copy_resource_;
  // A list of RepackAllocationBlock objects that mirrors allocation sequences,
  // used for repacking. We use a list here because we need pointer stability
  // for aliased allocations.
  std::list<RepackAllocationBlock> repack_allocation_blocks_;
  int64_t num_repacks_ = 0;
  int64_t num_repacks_successful_ = 0;
  std::vector<std::pair<MsaBufferInterval, Chunk>> pending_chunks_;
  std::vector<AsynchronousCopy> pending_async_copies_;
  std::vector<std::pair<const HloValue*, RequiredMemoryAssignment>>
      pending_required_assignments_;
  // A cache to keep the peak memory usage at each point in the graph. We use
  // this to see if the proposed allocation in the alternate memory would fit
  // ignoring fragmentation, and if not, we can skip the more expensive lookup
  // in the BufferIntervalTree, which also considers fragmentation.
  std::vector<int64_t> peak_memory_usage_;
  // The data structure that contains AliasedOffset objects and Allocation to
  // AliasedOffset map for efficient lookup.
  std::list<AliasedOffset> aliased_offsets_;
  absl::flat_hash_map<const Allocation*, AliasedOffset*> aliased_offset_map_;
  // This map contains required memory assignments for HloValues (e.g., input
  // and outputs).
  absl::flat_hash_map<const HloValue*, std::vector<RequiredMemoryAssignment>>
      required_assignments_;
  // Number of bytes reserved in alternate memory space.
  int64_t reserved_in_bytes_ = 0;
  // A rough measure of the memory pressure of the model, in bytes. Note that
  // this is pressure for memory capacity (and not accessed bytes), and for
  // alternate memory (not default memory).
  int64_t memory_pressure_ = 0;
  int64_t next_async_copy_id_ = 0;
  // Fingerprint cache.
  absl::flat_hash_map<const HloInstruction*, std::string> fingerprint_map_;
  // Vector of repeated instructions (that have the same fingerprint) indexed by
  // fingerprint.
  absl::flat_hash_map<std::string, std::vector<const HloInstruction*>>
      repeated_inst_map_;

  // Loop-optimized allocations found by MemoryBoundLoopOptimizer. These
  // allocation objects describe the allocations for one iteration of the loop,
  // so we translate them into the program-level Allocation objects in
  // allocations_.
  std::vector<AllocationSequence> loop_optimized_allocations_;
  // A map to look up the loop-optimized allocation info by use.
  absl::flat_hash_map<HloUse, LoopOptimizedAllocationInfo>
      loop_optimized_allocations_map_;
  // A map to look the operands of each instruction that are assigned in
  // alternate memory.
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<std::pair<int, ShapeIndex>>>
      operands_in_alternate_memory_map_;
  // A map to look the outputs of each instruction that are assigned in
  // alternate memory.
  absl::flat_hash_map<const HloInstruction*, absl::flat_hash_set<ShapeIndex>>
      outputs_in_alternate_memory_map_;
  // Debug strings.
  std::string buffer_info_str_;
  std::string allocation_info_str_;
  std::string instruction_schedule_str_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALGORITHM_H_
