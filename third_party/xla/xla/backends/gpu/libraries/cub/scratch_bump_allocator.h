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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_SCRATCH_BUMP_ALLOCATOR_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_SCRATCH_BUMP_ALLOCATOR_H_

#include <cstdint>

#include "absl/status/statusor.h"

namespace xla::gpu {

// Byte range [offset, offset + size) inside a scratch buffer.
struct ScratchRange {
  int64_t offset = 0;
  int64_t size = 0;

  friend bool operator==(const ScratchRange&, const ScratchRange&) = default;
};

// Bump allocator placing independent scratch sub-buffers on aligned boundaries
// with signed 64-bit integer overflow checking.
class ScratchBumpAllocator {
 public:
  explicit ScratchBumpAllocator(int64_t alignment_bytes);

  // Reserves an aligned sub-buffer of `size_bytes` bytes. A zero-byte request
  // succeeds without advancing the cursor and returns `{0, 0}`. Returns
  // InvalidArgumentError if `alignment_bytes <= 0`, `size_bytes < 0`, or
  // offset arithmetic overflows int64_t.
  absl::StatusOr<ScratchRange> Allocate(int64_t size_bytes);

  // Returns the total scratch size rounded up to `alignment_bytes`. Returns
  // InvalidArgumentError if `alignment_bytes <= 0` or rounding overflows
  // int64_t.
  absl::StatusOr<int64_t> TotalBytes() const;

 private:
  int64_t alignment_bytes_ = 0;
  int64_t current_offset_ = 0;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_SCRATCH_BUMP_ALLOCATOR_H_
