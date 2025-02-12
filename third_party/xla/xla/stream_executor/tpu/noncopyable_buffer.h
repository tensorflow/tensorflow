/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_
#define XLA_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/mem.h"

namespace tensorflow {
namespace tpu {

using BufferDeallocator = std::function<void(void*)>;
using OwnedDataPtr = std::unique_ptr<uint8_t[], BufferDeallocator>;
using BufferAllocator = absl::FunctionRef<OwnedDataPtr(size_t)>;

inline OwnedDataPtr DefaultAllocator(size_t size) {
  return {static_cast<uint8_t*>(malloc(size)), free};
}

// Uncopyable buffer type with optional ownership of the underlying data. If
// data is not owned then ensuring lifetime of the data exceeds the lifetime of
// the buffer is the responsibility of the user.
class NoncopyableBuffer {
 public:
  NoncopyableBuffer() = default;

  // Allocate an owning buffer without initializing the data. Useful when it
  // will be filled by a subsequent function and want to avoid initialization
  // cost. Size is specified in number of bytes.
  explicit NoncopyableBuffer(size_t size,
                             BufferAllocator allocator = DefaultAllocator)
      : data_(allocator(size)), buf_(data_.get()), size_(size) {}

  // Allocates an owning buffer and initializes it with the specified data. Size
  // is specified in number of uint32_t's.
  NoncopyableBuffer(size_t size_in_u32s, std::optional<uint32_t> value,
                    BufferAllocator allocator = DefaultAllocator)
      : NoncopyableBuffer(size_in_u32s * sizeof(uint32_t), allocator) {
#ifndef MEMORY_SANITIZER
    if (!value.has_value()) {
      return;
    }
#endif
    uint32_t* data_u32 = reinterpret_cast<uint32_t*>(data_.get());
    uint32_t v = value.value_or(uint32_t{0});
    std::fill_n(data_u32, size_in_u32s, v);
  }

  // Directly use buf pointer without copying it to owning data_. This delays
  // the memcpy until mutable access is requested. "buf" is not owned by this
  // data structure, so it is the user's duty to ensure the live range of "buf"
  // is longer than this data structure.
  NoncopyableBuffer(const uint8_t* buf, size_t size)  // Size is in uint8's.
      : buf_(buf), size_(size) {}
  NoncopyableBuffer(const uint32_t* buf,
                    size_t size_in_u32s)  // Size is in uint32_t's.
      : buf_(buf), size_(size_in_u32s * sizeof(uint32_t)) {}

  NoncopyableBuffer(const NoncopyableBuffer&) = delete;
  NoncopyableBuffer(NoncopyableBuffer&&) = default;

  NoncopyableBuffer& operator=(const NoncopyableBuffer&) = delete;
  NoncopyableBuffer& operator=(NoncopyableBuffer&&) = default;

  // Ensure that the buffer owns the data and returns a mutable view into the
  // owned data for modification.
  template <typename T>
  absl::Span<T> mutable_data() {
    static_assert(std::is_arithmetic<T>::value, "Must be arithmetic type.");
    EnsureDataOwned();
    DCHECK_EQ(size_ % sizeof(T), 0);
    return absl::Span<T>(reinterpret_cast<T*>(data_.get()), size_ / sizeof(T));
  }

  template <typename T>
  absl::Span<const T> const_data() const {
    static_assert(std::is_arithmetic<T>::value, "Must be arithmetic type.");
    DCHECK_EQ(size_ % sizeof(T), 0);
    return absl::Span<const T>(static_cast<const T*>(buf_), size_ / sizeof(T));
  }
  // Clone the content to a given buffer.
  void CloneTo(void* buf) { memcpy(buf, buf_, size_); }

  // Return true if data is owned by this buffer (have been copied to `data_`).
  bool owns_data() const { return data_ != nullptr; }

  // Returns a copy of the object that owns its buffer.
  NoncopyableBuffer Clone(size_t alignment = 1) const {
    auto clone = alignment <= 1
                     ? NoncopyableBuffer(size_)
                     : NoncopyableBuffer(AlignedAlloc(size_, alignment), size_);
    memcpy(clone.data_.get(), buf_, size_);
    return clone;
  }
  // Returns a copy of the object that owns its buffer. It uses `allocator` to
  // allocate the new buffer, which can have custom properties like special
  // alignment.
  NoncopyableBuffer Clone(BufferAllocator allocator) const {
    NoncopyableBuffer clone(size_, allocator);
    memcpy(clone.data_.get(), buf_, size_);
    return clone;
  }

  // Ensure that the buffer owns the data.
  void EnsureDataOwned(BufferAllocator allocator = DefaultAllocator) {
    if (data_ == nullptr) {
      data_ = allocator(size_);
      memcpy(data_.get(), buf_, size_);
      buf_ = data_.get();
    }
  }

  static OwnedDataPtr AlignedAlloc(size_t size, size_t alignment) {
    return OwnedDataPtr(
        static_cast<uint8_t*>(tsl::port::AlignedMalloc(size, alignment)),
        tsl::port::AlignedFree);
  }

 private:
  NoncopyableBuffer(OwnedDataPtr data, size_t size)
      : data_(std::move(data)), buf_(data_.get()), size_(size) {}

  // If data_ != nullptr then buf_ == data_.get()
  OwnedDataPtr data_{nullptr, free};  // Owning data pointer.
  const void* buf_;                   // Non-owning data pointer.
  size_t size_;                       // Size in number of bytes.
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_
