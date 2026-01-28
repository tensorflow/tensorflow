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

#ifndef XLA_STREAM_EXECUTOR_GPU_MULTI_GPU_BARRIER_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_MULTI_GPU_BARRIER_KERNEL_H_

#include <array>
#include <cstdint>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {

// Kernel signature:
// 1. rank (int64_t)
// 2. num_ranks (int64_t)
// 3. signal_buffers (array of pointers)
// 4. sync_counter: DeviceMemory<uint32_t>: Device ptr to a monotonic counter
//    that increments after every barrier execution. Stores last signal_value.
struct MultiGpuBarrierKernel {
  // Maximum number of peers supported by the barrier.
  // Can be extended to support larger GPU clusters in the future.
  static constexpr int64_t kMaxPeers = 32;

  using KernelType =
      stream_executor::TypedKernel<int64_t, int64_t,
                                   std::array<void*, kMaxPeers>,
                                   stream_executor::DeviceAddress<uint32_t>>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_MULTI_GPU_BARRIER_KERNEL_H_
