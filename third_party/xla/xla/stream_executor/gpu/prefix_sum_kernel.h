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

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/types.h"

namespace stream_executor::gpu {
struct PrefixSumBF16Kernel {
  using KernelType = TypedKernel<const DeviceAddress<xla::bfloat16>,
                                 DeviceAddress<xla::bfloat16>, size_t>;
};
struct PrefixSumF16Kernel {
  using KernelType = TypedKernel<const DeviceAddress<xla::half>,
                                 DeviceAddress<xla::half>, size_t>;
};
struct PrefixSumF32Kernel {
  using KernelType =
      TypedKernel<const DeviceAddress<float>, DeviceAddress<float>, size_t>;
};
struct PrefixSumF64Kernel {
  using KernelType =
      TypedKernel<const DeviceAddress<double>, DeviceAddress<double>, size_t>;
};
struct PrefixSumS8Kernel {
  using KernelType =
      TypedKernel<const DeviceAddress<int8_t>, DeviceAddress<int8_t>, size_t>;
};
struct PrefixSumS16Kernel {
  using KernelType =
      TypedKernel<const DeviceAddress<int16_t>, DeviceAddress<int16_t>, size_t>;
};
struct PrefixSumS32Kernel {
  using KernelType =
      TypedKernel<const DeviceAddress<int32_t>, DeviceAddress<int32_t>, size_t>;
};
struct PrefixSumS64Kernel {
  using KernelType =
      TypedKernel<const DeviceAddress<int64_t>, DeviceAddress<int64_t>, size_t>;
};
struct PrefixSumU8Kernel {
  using KernelType =
      TypedKernel<const DeviceAddress<uint8_t>, DeviceAddress<uint8_t>, size_t>;
};
struct PrefixSumU16Kernel {
  using KernelType = TypedKernel<const DeviceAddress<uint16_t>,
                                 DeviceAddress<uint16_t>, size_t>;
};
struct PrefixSumU32Kernel {
  using KernelType = TypedKernel<const DeviceAddress<uint32_t>,
                                 DeviceAddress<uint32_t>, size_t>;
};
struct PrefixSumU64Kernel {
  using KernelType = TypedKernel<const DeviceAddress<uint64_t>,
                                 DeviceAddress<uint64_t>, size_t>;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_PREFIX_SUM_KERNEL_H_
