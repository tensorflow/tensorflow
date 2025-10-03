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

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"

namespace xla {

// BufferUse tracks memory access type for a buffer slice. This is used to
// let XLA:
// - Correctly insert synchronization primitives at run time to avoid read/write
//   conflicts. Synchronization primitives are specific to the target backend.
// - Determine whether a buffer has defined contents before/after we execute a
//   thunk. This is used to detect non-deterministic behavior via checksumming.
class BufferUse {
 public:
  enum class MemoryAccess : uint32_t {
    kRead = 1 << 0,
    // Written to, with meaningful contents.
    kWrite = 1 << 1,
    // Written to, but contents are not meaningful.
    kScratch = 1 << 2,
    kReadWrite = kRead | kWrite,
    // Read from, but left undefined after use.
    kConsume = kRead | kScratch,
  };

  static constexpr MemoryAccess kRead = MemoryAccess::kRead;
  static constexpr MemoryAccess kWrite = MemoryAccess::kWrite;
  static constexpr MemoryAccess kScratch = MemoryAccess::kScratch;
  static constexpr MemoryAccess kReadWrite = MemoryAccess::kReadWrite;
  static constexpr MemoryAccess kConsume = MemoryAccess::kConsume;

  BufferUse(BufferAllocation::Slice slice, MemoryAccess access)
      : slice_(slice), access_(access) {}

  static BufferUse Read(BufferAllocation::Slice slice) {
    return BufferUse(slice, MemoryAccess::kRead);
  }

  static BufferUse Write(BufferAllocation::Slice slice) {
    return BufferUse(slice, MemoryAccess::kWrite);
  }

  static BufferUse ReadWrite(BufferAllocation::Slice slice) {
    return BufferUse(slice, MemoryAccess::kReadWrite);
  }

  static BufferUse Scratch(BufferAllocation::Slice slice) {
    return BufferUse(slice, MemoryAccess::kScratch);
  }

  static BufferUse Consume(BufferAllocation::Slice slice) {
    return BufferUse(slice, MemoryAccess::kConsume);
  }

  bool HasReadAccess() const {
    return static_cast<uint32_t>(access_) & static_cast<uint32_t>(kRead);
  }

  bool HasWriteAccess() const {
    return static_cast<uint32_t>(access_) &
           (static_cast<uint32_t>(kWrite) | static_cast<uint32_t>(kScratch));
  }

  // Returns true if the buffer contains initialized data when thunk starts
  // execution.
  bool HasDefinedContentsOnInput() const {
    return static_cast<uint32_t>(access_) & static_cast<uint32_t>(kRead);
  }

  // Returns true if the buffer contains initialized data when thunk finishes
  // execution.
  bool HasDefinedContentsOnOutput() const {
    return !(static_cast<uint32_t>(access_) & static_cast<uint32_t>(kScratch));
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

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const BufferUse& use) {
    absl::Format(
        &sink, "slice: %v, access: %s%s%s", use.slice_,
        (static_cast<uint32_t>(use.access()) & static_cast<uint32_t>(kRead))
            ? "R"
            : "",
        (static_cast<uint32_t>(use.access()) & static_cast<uint32_t>(kWrite))
            ? "W"
            : "",
        (static_cast<uint32_t>(use.access()) & static_cast<uint32_t>(kScratch))
            ? "S"
            : "");
  }

 private:
  BufferAllocation::Slice slice_;
  MemoryAccess access_;
};

}  // namespace xla

#endif  // XLA_RUNTIME_BUFFER_USE_H_
