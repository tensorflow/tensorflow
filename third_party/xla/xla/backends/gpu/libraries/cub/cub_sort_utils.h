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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_UTILS_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_UTILS_H_

#include <cstdint>

namespace xla::gpu {

// For CUB segmented sorts (batch_size > 1), XLA appends the segment offsets to
// the end of the scratch buffer. This function updates the given `scratch_size`
// to include the space needed for these offsets.
inline int64_t AddSegmentedSortOffsetsToScratchSize(int64_t scratch_size,
                                                    int64_t batch_size) {
  if (batch_size > 1) {
    // TODO(b/502873525): This adds 4 bytes even if already aligned.
    // Fix to proper alignment in a follow-up.
    scratch_size += sizeof(int32_t) - scratch_size % sizeof(int32_t);
    scratch_size += (batch_size + 1) * sizeof(int32_t);
  }
  return scratch_size;
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SORT_UTILS_H_
