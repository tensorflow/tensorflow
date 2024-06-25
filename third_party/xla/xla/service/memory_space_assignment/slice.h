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

// This file contains definitions for MSA slicing. Slicing is an allocation
// technique in which we allocate a buffer in slices that can start at different
// times, but once allocated, form a contiguous buffer. When copying buffers, we
// may want to allocate a buffer in slices, so that we delay allocating memory
// that would otherwise not be in use, due to copy bandwidth constraints.
//
// The following illustrates a buffer that is fully allocated at time t2, via
// slices.
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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SLICE_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SLICE_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/shape.h"

namespace xla::memory_space_assignment {

// The target of a custom call that slicing uses to concatenate slices
// that are already contiguous in memory, into a larger buffer.
inline constexpr char kConcatBitcastCustomCall[] = "ConcatBitcast";

// The parameters for slicing a single dimension of a tensor.
struct SliceParam {
  std::string ToString() const;
  bool operator==(const SliceParam& other) const;

  int64_t start_inclusive;
  int64_t end_exclusive;
};

// A proposed way to slice a buffer.
struct SliceProposal {
  std::string ToString() const;
  friend std::ostream& operator<<(std::ostream& os,
                                  const SliceProposal& proposal);
  std::tuple<const Shape&, const std::vector<SliceParam>&, int64_t> ToTuple()
      const;
  bool operator==(const SliceProposal& other) const;

  // Shape resulting from the slice.
  Shape slice_shape;

  // slice_params map to the parameters that would be passed to a slice
  // instruction. Thus:
  // * There should be a slice parameter for every dimension in the shape of
  //   the tensor being sliced.
  // * The ith slice_param applies to the ith logical dimension in the shape
  //   being sliced.
  // * If a dimension is not being sliced, it should have a SliceParam of
  //   {0, dim size}.
  std::vector<SliceParam> slice_params;

  // The size to be allocated for the slice. Note, this may be > the size of
  // the slice shape, due to additional padding that may occur when the slices
  // are concatenated back together.
  int64_t slice_size;
};

// A SliceProposalCollection proposes a way to to slice an AllocationRequest.
// A SliceProposalCollection is generated from a SliceProposalFunction and is
// used when we want to slice a prefetch.
using SliceProposalCollection = std::vector<SliceProposal>;
using SliceProposalFunction =
    std::function<absl::StatusOr<SliceProposalCollection>(
        const Shape& shape, const SlicedPrefetchOptions& options)>;

// A SliceDecision is a SliceProposal that we've determined where and when to
// allocate.
struct SliceDecision {
  std::string ToString() const;
  bool operator==(const SliceDecision& other) const;

  HeapSimulator::Chunk chunk;
  int64_t exclusive_start_time;
  SliceProposal sizing;
  float copy_resource_consumed;
};

// Returns true if the options indicates that there is a preferred slice
// size.
bool IsUniformSliceSizingEnabled(const SlicedPrefetchOptions& options);

// A class for turning a copy start time and end time into slice start times.
class SlicedPrefetchStartTimePicker {
 public:
  // Returns the amount of time elapsed in the instruction schedule between
  // (exclusive_start_time, exclusive_end_time).
  using ElapsedTimeFn = std::add_pointer<float(
      int64_t exclusive_start_time, int64_t exclusive_end_time) const>::type;

  // Returns true if the instructions at lhs_time and rhs_time are in the same
  // computation.
  using SameComputationParentFn =
      std::add_pointer<bool(int64_t lhs_time, int64_t rhs_time) const>::type;

  // Picks slice start times, given the num_slices, prefetch_start_time, and
  // prefetch_end_time. The returned times are exclusive.
  //
  // REQUIRES:
  // - The instructions following each start time are guaranateed to be in the
  //   same computation.
  // - The returned times sorted.
  // - The first returned time is equal to prefetch_start_time.
  static std::vector<int64_t> Pick(
      int64_t num_slices, int64_t exclusive_prefetch_start_time,
      int64_t prefetch_end_time, absl::AnyInvocable<ElapsedTimeFn> elapsed_fn,
      absl::AnyInvocable<SameComputationParentFn> has_same_parent_fn);
};

}  // namespace xla::memory_space_assignment

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SLICE_H_
