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

#ifndef XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_ALLOCATOR_H_
#define XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

namespace xla {

// Host offloading allocator is responsible for allocating host memory for
// running host offloading executables.
//
// TODO(ezhulenev): Consider adding explicit NUMA node argument to allocation
// functions, instead of relying on the NUMA node of the caller thread, as it
// might be different from the NUMA node of the device.
class HostOffloadingAllocator {
 public:
  // A base class for HostOffloadingAllocator buffers. Host offloading allocator
  // might use its own arena-based allocator, or it might use builtin operators
  // new and delete. Allocation details are implementation and backend specific.
  //
  // A host offloading allocator buffer is must be an RAII container that owns
  // the underlying memory and frees it when it gets destructed.
  class Buffer {
   public:
    Buffer() = default;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    virtual ~Buffer() = default;
    virtual absl::Span<uint8_t> data() const = 0;

    void* untyped_data() const { return data().data(); }
    size_t size_bytes() const { return data().size(); }

    template <typename T>
    absl::Span<T> data() const {
      DCHECK_EQ(size_bytes() % sizeof(T), 0);
      return absl::MakeSpan(tsl::safe_reinterpret_cast<T*>(data().data()),
                            size_bytes() / sizeof(T));
    }
  };

  HostOffloadingAllocator() = default;
  virtual ~HostOffloadingAllocator() = default;

  // ALlocates a transfer buffer that can be used to transfer data between
  // device and the host: it has sufficient alignment and size to be used
  // with DMA transfers, and might be pre-mapped with the TPU system.
  virtual absl::StatusOr<std::unique_ptr<Buffer>> AllocateTransferBuffer(
      size_t num_bytes) = 0;

  // Allocates a staging buffer that can be used as a parameter or result buffer
  // for host offloading executable: it has sufficient alignment, required by
  // XLA:CPU, but it's not intended to be used for DMA transfers.
  virtual absl::StatusOr<std::unique_ptr<Buffer>> AllocateStagingBuffer(
      size_t num_bytes) = 0;
};

}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_ALLOCATOR_H_
