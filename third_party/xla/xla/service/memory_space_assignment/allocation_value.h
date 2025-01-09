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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALLOCATION_VALUE_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALLOCATION_VALUE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"

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
    // A synchronous memory operation that feeds this use.
    // TODO(mehrdadk): extend this to support multiple sync data movement
    // operands.
    HloInstruction* sync_mem_op_operand = nullptr;

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
//
// In case we are trying to replace synchronous copies, and for example Use2
// is a replaceable sync copy candidate, we now skip Use2 and segments will be
// between Def, Use1, Use2.1, Use2.2, Use3:
//        +------+----------+-------------------+
//       /        \          \                   \
  //      /          v          v                   v
//    Def         Use1       Use2(Sync Copy)     Use3
//    |            |           \         \        |
//    |            |            v         v       |
//    |            |           Use2.1    Use2.2   |
//    |<---------->|<---------->|<------->|<----->|
//    |  Segment   |   Segment  | Segment |Segment|

struct AllocationRequest {
  int64_t inclusive_start_time;
  int64_t end_time;
  int64_t latest_prefetch_time;
  // See the comment for require_copy_allocation
  int64_t required_copy_allocation_latest_time;
  int64_t size;
  bool prefer_no_copy_alternate_mem_allocation;
  bool allow_no_copy_alternate_mem_allocation;
  bool require_no_copy_alternate_mem_allocation;
  // If true, indicates we are requiring a copy allocation between def and
  // use, that finishes by required_copy_allocation_latest_time.
  // required_copy_allocation_for is a synchronous copy instruction that will
  // be removed, if we are successful in adding the copy allocation.
  bool require_copy_allocation;
  bool allow_prefetch;
  std::optional<int64_t> earliest_prefetch_time;
  std::optional<int64_t> preferred_prefetch_time;
  AliasedOffset* preferred_offset;
  const AllocationValue::Use* use;
  AllocationValue* allocation_value;
  absl::Span<const int64_t> all_use_times;
  // See the comment for require_copy_allocation
  HloInstruction* required_copy_allocation_for;
  // If the required copy in require_copy_allocation is only for a slice of
  // the allocation_value
  bool required_copy_for_slice;
  // The resulting Allocation will be added to the AllocationSequence of
  // allocation_value_to_update. We only expect allocation_value_to_update to
  // be different from allocation_value in the case of a synchronous memory
  // operation conversion to asynchronous, otherwise, they should be the same.
  AllocationValue* allocation_value_to_update;
  // No new Allocation is needed to be created and we will only extend an
  // existing one.
  bool only_extend_existing_allocation;
  // Data structure that contains the options for making window prefetched
  // allocations.
  const WindowPrefetchedAllocation::Options* window_prefetch_options = nullptr;
};

// Result of an allocation, prefetch, eviction etc. request.  The result is
// either kSuccess or a bitwise OR of one or more failures. The values are
// unique powers of two. To check if a result contains a particular failure,
// use the result_is method. To add a new failure to a result, use the
// result_mark method.
enum class AllocationResult {
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
  kFailConflictingPreferredOffsets = 256,
  // Could not replace the synchronous data movement instruction (e.g., kCopy,
  // kSlice) with an asynchronous one
  kFailSyncDataMoveReplacement = 512
};

}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_ALLOCATION_VALUE_H_
