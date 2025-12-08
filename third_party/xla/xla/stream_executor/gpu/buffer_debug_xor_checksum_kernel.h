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

#ifndef XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_XOR_CHECKSUM_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_XOR_CHECKSUM_KERNEL_H_

#include <cstdint>

#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {

// Trait for a kernel that computes the checksum of given input buffer and
// appends it to the buffer debug log.
//
// This kernel MUST execute on a single thread block.
struct BufferDebugXorChecksumKernel {
  using KernelType =
      TypedKernel<xla::gpu::BufferDebugLogEntryId, DeviceAddress<uint8_t>,
                  uint64_t, DeviceAddress<xla::gpu::BufferDebugLogHeader>,
                  DeviceAddress<xla::gpu::BufferDebugLogEntry>>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_XOR_CHECKSUM_KERNEL_H_
