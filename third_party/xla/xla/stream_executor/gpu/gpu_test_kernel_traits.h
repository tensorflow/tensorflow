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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNEL_TRAITS_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNEL_TRAITS_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {
template <typename T>
struct Ptrs3 {
  T* a;
  T* b;
  T* c;
};

namespace internal {

struct AddI32Kernel {
  using KernelType = TypedKernel<DeviceAddress<int32_t>, DeviceAddress<int32_t>,
                                 DeviceAddress<int32_t>>;
};

struct IncrementBy5I32KernelWithCustomArgsPacking {
  using KernelType = TypedKernel<DeviceAddress<int32_t>>;
};

struct MulI32Kernel {
  using KernelType = TypedKernel<DeviceAddress<int32_t>, DeviceAddress<int32_t>,
                                 DeviceAddress<int32_t>>;
};

struct IncAndCmpKernel {
  using KernelType = TypedKernel<DeviceAddress<int32_t>, DeviceAddress<bool>,
                                 DeviceAddress<int32_t>>;
};

struct AddI32Ptrs3Kernel {
  using KernelType = TypedKernel<Ptrs3<int32_t>>;
};

struct CopyKernel {
  using KernelType =
      TypedKernel<DeviceAddress<std::byte>, std::array<std::byte, 16>>;
};

}  // namespace internal
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_TEST_KERNEL_TRAITS_H_
