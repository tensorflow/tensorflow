/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_

#include <memory>

#include "absl/base/casts.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tpu {

// Uncopyable buffer type with optional ownership of the underlying data. If
// data is not owned then ensuring lifetime of the data exceeds the lifetime of
// the buffer is the responsibility of the user.
class NoncopyableBuffer {
 public:
  NoncopyableBuffer() = default;

  // Allocate an owning buffer without initializing the data. Useful when it
  // will be filled by a subsequent function and want to avoid initialization
  // cost. Size is specified in number of uint32's.
  explicit NoncopyableBuffer(size_t size)
      : data_(new uint32[size]), buf_(data_.get()), size_(size) {}

  // Allocates an owning buffer and initializes it with the specified data. Size
  // is specified in number of uint32's.
  NoncopyableBuffer(size_t size, absl::optional<uint32> value)
      : NoncopyableBuffer(size) {
#ifndef MEMORY_SANITIZER
    if (!value.has_value()) {
      return;
    }
#endif
    uint32 v = value.value_or(0);
    for (int64 i = 0; i < size; ++i) {
      data_[i] = v;
    }
  }

  // Directly use buf pointer without copying it to owning data_. This delays
  // the memcpy until mutable access is requested. "buf" is not owned by this
  // data structure, so it is the user's duty to ensure the live range of "buf"
  // is longer than this data structure.
  NoncopyableBuffer(const uint8* buf, uint64 size)  // Size is in uint8's.
      : buf_(buf), size_(size / sizeof(uint32)) {
    CHECK_EQ(size % sizeof(uint32), 0);
  }
  NoncopyableBuffer(const uint32* buf, uint64 size)  // Size is in uint32's.
      : buf_(buf), size_(size) {}

  NoncopyableBuffer(const NoncopyableBuffer&) = delete;
  NoncopyableBuffer(NoncopyableBuffer&&) = default;

  NoncopyableBuffer& operator=(const NoncopyableBuffer&) = delete;
  NoncopyableBuffer& operator=(NoncopyableBuffer&&) = default;

  // Ensure that the buffer owns the data and returns a mutable view into the
  // owned data for modification.
  absl::Span<uint32> mutable_data() {
    if (data_ == nullptr) {
      data_.reset(new uint32[size_]);
      memcpy(data_.get(), buf_, size_ * sizeof(uint32));
      buf_ = data_.get();
    }
    return absl::Span<uint32>(data_.get(), size_);
  }

  absl::Span<const uint32> const_data() const {
    return absl::Span<const uint32>(absl::bit_cast<uint32*>(buf_), size_);
  }
  // Clone the content to a given buffer.
  void CloneTo(void* buf) { memcpy(buf, buf_, size_ * sizeof(uint32)); }

  // Return true if data is owned by this buffer (have been copied to `data_`).
  bool owns_data() const { return data_ != nullptr; }

  // Returns a copy of the object that owns its buffer.
  NoncopyableBuffer Clone() const {
    NoncopyableBuffer clone(size_);
    memcpy(clone.data_.get(), buf_, size_ * sizeof(uint32));
    return clone;
  }

 private:
  // If data_ != nullptr then buf_ == data_.get()
  std::unique_ptr<uint32[]> data_;  // Owning data pointer.
  const void* buf_;                 // Non-owning data pointer.
  uint64 size_;                     // Size in number of uint32's.
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_NONCOPYABLE_BUFFER_H_
