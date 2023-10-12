/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/statusor.h"
#include "xla/types.h"

namespace xla {
namespace memory_space_assignment {

// An interface to define allocation repacking algorithms.
class MemorySpaceAssignmentRepacker {
 public:
  MemorySpaceAssignmentRepacker(int64_t max_size, int64_t alignment)
      : max_size_(max_size), alignment_(alignment) {}
  virtual ~MemorySpaceAssignmentRepacker() = default;

  // Data about a slice in a sliced allocation.
  struct Slice {
    int64_t size;
    int64_t offset;
    int64_t start_time;

    std::string ToString() const {
      return absl::StrCat("{ size: ", size, ", offset: ", offset,
                          ", start_time: ", start_time, " }");
    }

    std::tuple<int64_t, int64_t, int64_t> ToTuple() const {
      return std::make_tuple(size, offset, start_time);
    }

    bool operator==(const Slice& rhs) const {
      return ToTuple() == rhs.ToTuple();
    }
  };

  // Slice data about a sliced allocation.
  struct SlicedAllocationData {
    std::vector<Slice> slices_sorted_by_offset;

    std::vector<int64_t> SizesSortedByOffset() const {
      std::vector<int64_t> sizes_sorted_by_offset;
      sizes_sorted_by_offset.reserve(slices_sorted_by_offset.size());
      absl::c_for_each(slices_sorted_by_offset,
                       [&sizes_sorted_by_offset](const Slice& slice) {
                         sizes_sorted_by_offset.push_back(slice.size);
                       });
      return sizes_sorted_by_offset;
    }

    std::vector<int64_t> SortedStartTimes() const {
      std::vector<int64_t> sorted_start_times;
      sorted_start_times.reserve(slices_sorted_by_offset.size());
      absl::c_for_each(slices_sorted_by_offset,
                       [&sorted_start_times](const Slice& slice) {
                         sorted_start_times.push_back(slice.start_time);
                       });
      absl::c_sort(sorted_start_times);
      return sorted_start_times;
    }

    std::string ToString() const {
      return absl::StrCat(
          "{ slices_sorted_by_offset: [ ",
          absl::StrJoin(slices_sorted_by_offset, ", ",
                        [](std::string* out, const Slice& slice) {
                          absl::StrAppend(out, slice.ToString());
                        }),
          " ] }");
    }

    bool operator==(const SlicedAllocationData& rhs) const {
      return slices_sorted_by_offset == rhs.slices_sorted_by_offset;
    }
  };

  // A contiguous block of allocation consisting of start and end (logical)
  // times, size, and the initial offset. After repacking, if the repacking was
  // successful and the allocations were modified, the offset field holds the
  // new offset. To support aliased allocations, AllocationBlock also includes a
  // vector of AllocationBlock pointers, called colocations. All AllocationBlock
  // objects within the colocations must get the same offset. The id should be
  // unique and is used to ensure determinism for comparison tie-breaker.
  //
  // Each AllocationBlock can be treated as an allocation that requires size
  // space from start_time to end_time. However, some allocations are really
  // composed of slices. In such cases, the repacker can utilize
  // the information in the original_slice_data field to achieve an even more
  // efficient repacking.
  struct AllocationBlock {
    int64_t start_time;
    int64_t end_time;
    int64_t size;
    int64_t offset;
    int64_t initial_offset;
    int64_t id;
    std::vector<AllocationBlock*> colocations;

    // Optional data structures that are used to improve repacking, when an
    // allocation is sliced, e.g., from a sliced prefetch.
    std::optional<SlicedAllocationData> original_slice_data;
    std::optional<SlicedAllocationData> repacked_slice_data;

    std::string ToString() const {
      std::string original_slicing_str;
      if (original_slice_data.has_value()) {
        original_slicing_str = absl::StrCat("; original_slice_data: ",
                                            original_slice_data->ToString());
      }
      std::string repacked_slicing_str;
      if (repacked_slice_data.has_value()) {
        repacked_slicing_str = absl::StrCat("; repacked_slice_data: ",
                                            repacked_slice_data->ToString());
      }
      return absl::StrCat("[", start_time, ", ", end_time, "]; size: ", size,
                          "; offset: ", offset,
                          "; initial offset: ", initial_offset,
                          "; # colocations: ", colocations.size(),
                          original_slicing_str, repacked_slicing_str);
    }

    // This is required by BufferIntervalCompare as a tie breaker. Use a unique
    // and deterministic id.
    bool operator<(const AllocationBlock& other) const { return id < other.id; }
  };

  // Repack the AllocationBlocks provided in the parameter. Returns true if
  // allocations have been modified and false if not. Returns a non-ok status if
  // there was an error.
  virtual StatusOr<bool> Repack(absl::Span<AllocationBlock*> allocations) = 0;

 protected:
  int64_t max_size_;
  int64_t alignment_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_
