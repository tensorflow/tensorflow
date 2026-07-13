/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/util/buffer_slice_merge.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/util/sorted_range.h"

namespace xla {

std::vector<BufferAllocation::Slice> MergeOverlappingSlices(
    absl::Span<const BufferAllocation::Slice> slices) {
  if (slices.empty()) {
    return {};
  }

  std::vector<BufferAllocation::Slice> merged_slices;
  auto compare_by_offset = [](const BufferAllocation::Slice& a,
                              const BufferAllocation::Slice& b) {
    return a.offset() < b.offset();
  };

  auto sorted_range = tsl::SortedRange(slices, compare_by_offset);
  auto it = sorted_range.begin();

  BufferAllocation::Slice current_slice = *it;
  while (++it != sorted_range.end()) {
    const BufferAllocation::Slice& next_slice = *it;
    CHECK_EQ(current_slice.allocation(), next_slice.allocation())
        << "Slices must belong to the same allocation.";

    if (current_slice.OverlapsWith(next_slice)) {
      int64_t new_end = std::max(current_slice.offset() + current_slice.size(),
                                 next_slice.offset() + next_slice.size());
      current_slice = BufferAllocation::Slice(current_slice.allocation(),
                                              current_slice.offset(),
                                              new_end - current_slice.offset());
    } else {
      // Disjoint interval.
      merged_slices.push_back(current_slice);
      current_slice = next_slice;
    }
  }
  merged_slices.push_back(current_slice);
  return merged_slices;
}

}  // namespace xla
