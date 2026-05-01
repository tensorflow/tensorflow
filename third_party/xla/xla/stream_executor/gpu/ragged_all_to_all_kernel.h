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

#ifndef XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_KERNEL_H_

#include <sys/types.h>

#include <array>
#include <cstdint>

#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {
inline constexpr int64_t kMaxNumRaggedAllToAllOutputPtrs = 8;

using RaggedAllToAllOutputPtrs =
    std::array<void*, kMaxNumRaggedAllToAllOutputPtrs>;

// Defines a trait for the RaggedAllToAll kernel that can be used to register
// and look up the kernel in the GPU kernel registry.
template <typename PtrStorage, int64_t kVectorSize>
struct RaggedAllToAllKernel {
  using KernelType = stream_executor::TypedKernel<
      /*input_ptr=*/stream_executor::DeviceAddressBase,
      /*output_ptrs=*/PtrStorage,
      /*input_offsets_ptr=*/stream_executor::DeviceAddressBase,
      /*send_sizes_ptr=*/stream_executor::DeviceAddressBase,
      /*output_offsets_ptr=*/stream_executor::DeviceAddressBase,
      /*num_updates_per_replica=*/int64_t,
      /*num_row_elements=*/int64_t>;
};

template <int64_t kVectorSize>
struct RaggedAllToAllWithSymmetricMemoryKernel {
  using KernelType = stream_executor::TypedKernel<
      /*input_ptr=*/stream_executor::DeviceAddressBase,
      /*output_ptrs=*/xla::SymmetricMemory*,
      /*input_offsets_ptr=*/stream_executor::DeviceAddressBase,
      /*send_sizes_ptr=*/stream_executor::DeviceAddressBase,
      /*output_offsets_ptr=*/stream_executor::DeviceAddressBase,
      /*num_updates_per_replica=*/int64_t,
      /*num_row_elements=*/int64_t>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_KERNEL_H_
