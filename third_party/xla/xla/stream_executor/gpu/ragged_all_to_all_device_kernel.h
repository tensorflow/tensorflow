/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_H_

#include <cstdint>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"

namespace xla {
class SymmetricMemory;
namespace gpu {
class GpuDeviceCommunicator;
}  // namespace gpu
}  // namespace xla

namespace stream_executor::gpu {

template <int64_t kVectorSize>
struct RaggedAllToAllDeviceKernel {
  using KernelType = stream_executor::TypedKernel<
      xla::gpu::GpuDeviceCommunicator*,    // dev_comm
      xla::SymmetricMemory*,               // send_win (input buffer)
      xla::SymmetricMemory*,               // recv_win (output buffer)
      stream_executor::DeviceAddressBase,  // input_offsets
      stream_executor::DeviceAddressBase,  // send_sizes
      stream_executor::DeviceAddressBase,  // output_offsets
      int64_t,                             // num_updates_per_replica
      int64_t,                             // num_row_elements
      int64_t,                             // input_buffer_offset_bytes
      int64_t>;                            // output_buffer_offset_bytes
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_DEVICE_KERNEL_H_
