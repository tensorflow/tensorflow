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

#ifndef XLA_STREAM_EXECUTOR_GPU_PREFIX_SUM_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_PREFIX_SUM_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/types.h"

namespace stream_executor::gpu {
struct PrefixSumBF16Kernel {
  using KernelType = TypedKernel<const DeviceMemory<xla::bfloat16>,
                                 DeviceMemory<xla::bfloat16>, size_t>;
};
struct PrefixSumF16Kernel {
  using KernelType = TypedKernel<const DeviceMemory<xla::half>,
                                 DeviceMemory<xla::half>, size_t>;
};
struct PrefixSumF32Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<float>, DeviceMemory<float>, size_t>;
};
struct PrefixSumF64Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<double>, DeviceMemory<double>, size_t>;
};
struct PrefixSumS8Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<int8_t>, DeviceMemory<int8_t>, size_t>;
};
struct PrefixSumS16Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<int16_t>, DeviceMemory<int16_t>, size_t>;
};
struct PrefixSumS32Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<int32_t>, DeviceMemory<int32_t>, size_t>;
};
struct PrefixSumS64Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<int64_t>, DeviceMemory<int64_t>, size_t>;
};
struct PrefixSumU8Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<uint8_t>, DeviceMemory<uint8_t>, size_t>;
};
struct PrefixSumU16Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<uint16_t>, DeviceMemory<uint16_t>, size_t>;
};
struct PrefixSumU32Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<uint32_t>, DeviceMemory<uint32_t>, size_t>;
};
struct PrefixSumU64Kernel {
  using KernelType =
      TypedKernel<const DeviceMemory<uint64_t>, DeviceMemory<uint64_t>, size_t>;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_PREFIX_SUM_KERNEL_H_
