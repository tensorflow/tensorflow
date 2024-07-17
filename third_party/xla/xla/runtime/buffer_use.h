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

#ifndef XLA_RUNTIME_BUFFER_USE_H_
#define XLA_RUNTIME_BUFFER_USE_H_

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"

namespace xla {

// BufferUse tracks memory access type for a buffer slice, so that XLA can
// correctly insert synchronization primitives at run time to avoid read/write
// conflicts. Synchronization primitives are specific to the target backend.
class BufferUse {
 public:
  enum class MemoryAccess { kRead, kWrite };

  static constexpr MemoryAccess kRead = MemoryAccess::kRead;
  static constexpr MemoryAccess kWrite = MemoryAccess::kWrite;

  BufferUse(BufferAllocation::Slice slice, MemoryAccess access)
      : slice_(slice), access_(access) {}

  static BufferUse Read(BufferAllocation::Slice slice) {
    return BufferUse(slice, MemoryAccess::kRead);
  }

  static BufferUse Write(BufferAllocation::Slice slice) {
    return BufferUse(slice, MemoryAccess::kWrite);
  }

  // ReadWriteSet tracks a set of read and write buffer slices.
  class ReadWriteSet {
   public:
    ReadWriteSet();

    void Add(BufferUse use);
    void AddRead(BufferAllocation::Slice slice);
    void AddWrite(BufferAllocation::Slice slice);

    void AddAll(absl::Span<const BufferUse> uses);

    // Returns true if any of the buffer use(s) has a conflict with tracked
    // buffer slice reads or writes.
    bool HasConflicts(const BufferUse& use) const;
    bool HasConflicts(const ReadWriteSet& other);

   private:
    absl::flat_hash_set<BufferAllocation::Slice> read_;
    absl::flat_hash_set<BufferAllocation::Slice> write_;
  };

  bool operator==(const BufferUse& other) const {
    return slice_ == other.slice_ && access_ == other.access_;
  }

  bool operator!=(const BufferUse& other) const { return !(*this == other); }

  const BufferAllocation::Slice& slice() const { return slice_; }
  MemoryAccess access() const { return access_; }

  template <typename H>
  friend H AbslHashValue(H h, const BufferUse& use) {
    return H::combine(std::move(h), use.slice_, use.access_);
  }

 private:
  BufferAllocation::Slice slice_;
  MemoryAccess access_;
};

}  // namespace xla

#endif  // XLA_RUNTIME_BUFFER_USE_H_
