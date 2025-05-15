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

#ifndef XLA_STREAM_EXECUTOR_GPU_TOPK_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_TOPK_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
namespace stream_executor::gpu {

// We perform 2 32-way reductions, which means the largest number of threads per
// block we support is 1024.
static constexpr size_t kTopKMaxThreadsPerBlock = 1024;

// Defines a trait for the TopK kernel that can be used to register
// and look up the kernel in the GPU kernel registry.
template <size_t K, typename KT, typename VT>
struct TopKKernel {
  using KernelType =
      stream_executor::TypedKernel<stream_executor::DeviceMemory<KT>, size_t,
                                   stream_executor::DeviceMemory<KT>,
                                   stream_executor::DeviceMemory<uint32_t>,
                                   size_t>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_TOPK_KERNEL_H_
