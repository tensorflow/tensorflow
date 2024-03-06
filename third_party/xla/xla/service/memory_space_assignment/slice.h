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
#include <vector>

#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

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
using SliceProposalFunction = std::function<StatusOr<SliceProposalCollection>(
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

}  // namespace xla::memory_space_assignment

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SLICE_H_
