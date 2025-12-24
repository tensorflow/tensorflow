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

#ifndef XLA_STREAM_EXECUTOR_MEMORY_ALLOCATION_H_
#define XLA_STREAM_EXECUTOR_MEMORY_ALLOCATION_H_

#include <cstdint>

#include "absl/base/macros.h"
#include "xla/stream_executor/device_address.h"

namespace stream_executor {

// A MemoryAllocation is a block of physical memory allocated on the
// StreamExecutor device.
//
// MemoryAllocation is not necessarily a physical memory on the physical device
// (i.e. GPU), it can be a memory on the host pre-mapped for the host to device
// communication. It can be pinned host memory, unified memory, device memory,
// etc. depending on what kinds of memories are supported by underlying device.
//
// MemoryAllocation can be mapped to a DeviceAddress, which can be used to
// access the memory from device or host. Multiple device address ranges can be
// mapped to the same MemoryAllocation.
class MemoryAllocation {
 public:
  MemoryAllocation() = default;
  virtual ~MemoryAllocation() = default;

  MemoryAllocation(MemoryAllocation&&) = delete;
  MemoryAllocation& operator=(MemoryAllocation&&) = delete;

  // A device address which gives access to the memory allocation. Can be
  // nullptr if memory allocation is not adressable, i.e. physical allocation
  // might not be mapped to any virtual address by default.
  virtual DeviceAddressBase address() const = 0;

  ABSL_DEPRECATE_AND_INLINE()
  void* opaque() const { return address().opaque(); }

  ABSL_DEPRECATE_AND_INLINE()
  uint64_t size() const { return address().size(); }
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MEMORY_ALLOCATION_H_
