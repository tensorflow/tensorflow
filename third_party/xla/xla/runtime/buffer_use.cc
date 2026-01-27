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

#include "xla/runtime/buffer_use.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"

namespace xla {

BufferUse::ReadWriteSet::ReadWriteSet() = default;

void BufferUse::ReadWriteSet::Add(BufferUse use) {
  switch (use.access()) {
    case BufferUse::MemoryAccess::kRead:
      AddRead(use);
      break;
    case BufferUse::MemoryAccess::kWrite:
      AddWrite(use);
      break;
  }
}

void BufferUse::ReadWriteSet::AddRead(const BufferUse& use) {
  read_.push_back(use);
}

void BufferUse::ReadWriteSet::AddWrite(const BufferUse& use) {
  write_.push_back(use);
}

void BufferUse::ReadWriteSet::AddAll(absl::Span<const BufferUse> uses) {
  for (const auto& use : uses) {
    Add(use);
  }
}

bool BufferUse::ReadWriteSet::HasConflicts(const BufferUse& use) const {
  // Returns true if `use` overlaps with any of the slices in set.
  auto overlaps = [](const std::vector<BufferUse>& set, const BufferUse& use) {
    return absl::c_any_of(set, [&](const BufferUse& other) {
      return other.slice_.OverlapsWith(use.slice()) ||
             other.slice_ == use.slice_;
    });
  };

  return use.access() == MemoryAccess::kWrite
             ? overlaps(write_, use) || overlaps(read_, use)
             : overlaps(write_, use);
}

bool BufferUse::ReadWriteSet::HasConflicts(const ReadWriteSet& other) {
  return absl::c_any_of(
             other.read_,
             [&](const BufferUse& other) { return HasConflicts(other); }) ||
         absl::c_any_of(other.write_, [&](const BufferUse& other) {
           return HasConflicts(other);
         });
}

}  // namespace xla
