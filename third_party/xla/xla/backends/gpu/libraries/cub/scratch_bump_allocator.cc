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

#include "xla/backends/gpu/libraries/cub/scratch_bump_allocator.h"

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/overflow_util.h"

namespace xla::gpu {

ScratchBumpAllocator::ScratchBumpAllocator(int64_t alignment_bytes)
    : alignment_bytes_(alignment_bytes) {}

absl::StatusOr<ScratchRange> ScratchBumpAllocator::Allocate(
    int64_t size_bytes) {
  if (alignment_bytes_ <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Scratch alignment must be positive, got ", alignment_bytes_));
  }
  if (size_bytes < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Scratch allocation size must be non-negative, got ", size_bytes));
  }
  if (size_bytes == 0) {
    return ScratchRange{};
  }
  std::optional<int64_t> offset =
      OverflowSafeRoundUpTo(current_offset_, alignment_bytes_);
  if (!offset.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Integer overflow aligning scratch offset ",
                     current_offset_, " to ", alignment_bytes_, " bytes"));
  }
  std::optional<int64_t> next_offset = OverflowSafeAdd(*offset, size_bytes);
  if (!next_offset.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Integer overflow adding scratch allocation ", *offset,
                     " + ", size_bytes));
  }
  current_offset_ = *next_offset;
  return ScratchRange{*offset, size_bytes};
}

absl::StatusOr<int64_t> ScratchBumpAllocator::TotalBytes() const {
  if (alignment_bytes_ <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Scratch alignment must be positive, got ", alignment_bytes_));
  }
  std::optional<int64_t> total =
      OverflowSafeRoundUpTo(current_offset_, alignment_bytes_);
  if (!total.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Integer overflow rounding total scratch bytes ",
                     current_offset_, " to ", alignment_bytes_, " bytes"));
  }
  return *total;
}

}  // namespace xla::gpu
