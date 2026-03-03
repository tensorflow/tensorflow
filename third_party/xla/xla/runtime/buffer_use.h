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
#include <tuple>
#include <vector>

#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla {

// BufferUse tracks memory access type for a buffer slice. This is used to
// let XLA:
// - Correctly insert synchronization primitives at run time to avoid read/write
//   conflicts. Synchronization primitives are specific to the target backend.
// - Determine whether a buffer has defined contents before/after we execute a
//   thunk. This is used to detect non-deterministic behavior via checksumming.
// - We also use shape to know how the bytes in the slice are reinterpreted by
//   thunks. Shape can be used by rewriters in ThunkPassPipeline.
class BufferUse {
 public:
  enum class MemoryAccess {
    // The buffer is only read.
    kRead,
    // The buffer is read and written to.
    kWrite,
  };

  // Flags that indicate whether the contents of a buffer are defined before and
  // after execution of a thunk.
  enum class ContentValidity : uint32_t {
    // The thunk uses the buffer as a scratch space. There are no guarantees
    // about the buffer's contents outside of the thunk's execution.
    kUndefined = 0,
    // The buffer is initialized when thunk starts execution. This is the case
    // for parameters.
    kDefinedOnInput = 1 << 0,
    // The buffer is initialized when thunk finishes execution. This is the case
    // for outputs and parameters the thunk does not modify.
    kDefinedOnOutput = 1 << 1,

    kDefinedOnInputAndOutput = kDefinedOnInput | kDefinedOnOutput,
  };

  BufferUse(BufferAllocation::Slice slice, MemoryAccess access, Shape shape)
      : BufferUse(slice, access,
                  access == MemoryAccess::kRead
                      ? ContentValidity::kDefinedOnInputAndOutput
                      : ContentValidity::kDefinedOnOutput,
                  shape) {}

  BufferUse(BufferAllocation::Slice slice, MemoryAccess access,
            ContentValidity content_validity, Shape shape)
      : slice_(slice),
        shape_(shape),
        access_(access),
        content_validity_(content_validity) {}

  static BufferUse Read(BufferAllocation::Slice slice, Shape shape) {
    return BufferUse(slice, MemoryAccess::kRead,
                     ContentValidity::kDefinedOnInputAndOutput, shape);
  }

  static BufferUse Write(BufferAllocation::Slice slice, Shape shape) {
    return BufferUse(slice, MemoryAccess::kWrite,
                     ContentValidity::kDefinedOnOutput, shape);
  }

  static BufferUse Scratch(BufferAllocation::Slice slice, Shape shape) {
    return BufferUse(slice, MemoryAccess::kWrite, ContentValidity::kUndefined,
                     shape);
  }

  static BufferUse Consume(BufferAllocation::Slice slice, Shape shape) {
    return BufferUse(slice, MemoryAccess::kWrite,
                     ContentValidity::kDefinedOnInput, shape);
  }

  // Returns true if the buffer contains initialized data when thunk starts
  // execution.
  bool HasDefinedContentsOnInput() const {
    return static_cast<uint32_t>(content_validity_) &
           static_cast<uint32_t>(ContentValidity::kDefinedOnInput);
  }

  // Returns true if the buffer contains initialized data when thunk finishes
  // execution.
  bool HasDefinedContentsOnOutput() const {
    return static_cast<uint32_t>(content_validity_) &
           static_cast<uint32_t>(ContentValidity::kDefinedOnOutput);
  }

  // ReadWriteSet tracks a set of read and write buffer slices.
  class ReadWriteSet {
   public:
    ReadWriteSet();

    void Add(BufferUse use);
    void AddRead(const BufferUse& use);
    void AddWrite(const BufferUse& use);

    void AddAll(absl::Span<const BufferUse> uses);

    // Returns true if any of the buffer use(s) has a conflict with tracked
    // buffer slice reads or writes.
    bool HasConflicts(const BufferUse& use) const;
    bool HasConflicts(const ReadWriteSet& other);

   private:
    std::vector<BufferUse> read_;
    std::vector<BufferUse> write_;
  };

  bool operator==(const BufferUse& other) const {
    return std::tie(slice_, access_, content_validity_) ==
           std::tie(other.slice_, other.access_, other.content_validity_);
  }

  bool operator!=(const BufferUse& other) const { return !(*this == other); }

  const BufferAllocation::Slice& slice() const { return slice_; }
  MemoryAccess access() const { return access_; }
  ContentValidity content_validity() const { return content_validity_; }

  template <typename H>
  friend H AbslHashValue(H h, const BufferUse& use) {
    return H::combine(std::move(h), use.slice_, use.access_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const BufferUse& use) {
    absl::Format(&sink, "{slice: %v, access: %s, content_validity: %s%s}",
                 use.slice_, use.access() == MemoryAccess::kRead ? "R" : "W",
                 use.HasDefinedContentsOnInput() ? "I" : "",
                 use.HasDefinedContentsOnOutput() ? "O" : "");
  }

  const Shape& shape() const { return shape_; }

 private:
  BufferAllocation::Slice slice_;
  Shape shape_;
  MemoryAccess access_;
  ContentValidity content_validity_;
};

}  // namespace xla

#endif  // XLA_RUNTIME_BUFFER_USE_H_
