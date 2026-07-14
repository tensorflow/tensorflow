/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_BUFFER_H_
#define XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_BUFFER_H_

#include <cstddef>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

namespace xla {

// HostOffloadingBuffer is a non-owning view into an opaque region of memory in
// the host memory space that holds data for host offloading executable
// parameters or a destination for results. Underlying storage can be either a
// TpuHostBuffer, or a host offloading buffer allocated by
// HostOffloadingAllocator. It is a caller responsibility to pass buffers of
// correct size, and to guarantee they will stay alive untile the host
// offloading execution is completed.
class HostOffloadingBuffer {
 public:
  HostOffloadingBuffer() : base_(nullptr), size_in_bytes_(0) {}

  HostOffloadingBuffer(void* base, size_t size_in_bytes)
      : base_(base), size_in_bytes_(size_in_bytes) {}

  template <typename T>
  explicit HostOffloadingBuffer(absl::Span<T> data)
      : base_(reinterpret_cast<void*>(data.data())),
        size_in_bytes_(data.size() * sizeof(T)) {}

  // Returns an opaque pointer pointing to the base of the buffer.
  void* opaque_base() const { return base_; }

  // Returns the size of the buffer in bytes.
  size_t size_in_bytes() const { return size_in_bytes_; }

  template <typename T>
  absl::Span<T> data() const {
    DCHECK_EQ(size_in_bytes_ % sizeof(T), 0);
    return absl::MakeSpan(reinterpret_cast<T*>(base_),
                          size_in_bytes_ / sizeof(T));
  }

  bool operator==(const HostOffloadingBuffer& other) const {
    return base_ == other.base_ && size_in_bytes_ == other.size_in_bytes_;
  }

  std::string ToString() const {
    return absl::StrFormat("Buffer{base=%p, size_in_bytes=%d}", base_,
                           size_in_bytes_);
  }

 private:
  void* base_;
  size_t size_in_bytes_;
};

}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_BUFFER_H_
