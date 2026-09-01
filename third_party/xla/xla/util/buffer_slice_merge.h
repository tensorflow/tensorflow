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

#ifndef XLA_UTIL_BUFFER_SLICE_MERGE_H_
#define XLA_UTIL_BUFFER_SLICE_MERGE_H_

#include <vector>

#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"

namespace xla {

// Merges overlapping buffer slices belonging to the same BufferAllocation.
// The returned vector contains non-overlapping slices sorted by offset.
std::vector<BufferAllocation::Slice> MergeOverlappingSlices(
    absl::Span<const BufferAllocation::Slice> slices);

}  // namespace xla

#endif  // XLA_UTIL_BUFFER_SLICE_MERGE_H_
