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

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "absl/base/casts.h"
#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"

namespace stream_executor::gpu {
#if defined(__GFX9__)
__device__ static constexpr uint64_t kWarpSize = 64;
#else
__device__ static constexpr uint64_t kWarpSize = 32;
#endif
}  // namespace stream_executor::gpu

#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel_lib.cu.h.inc"

namespace stream_executor::gpu {

template <>
__device__ constexpr hip_bfloat16 kInfinity<hip_bfloat16> =
    __builtin_bit_cast(hip_bfloat16, uint16_t{0x7F80});

template <>
__device__ constexpr _Float16 kInfinity<_Float16> =
    __builtin_bit_cast(_Float16, uint16_t{0x7C00});

template <>
__device__ inline bool IsNan(hip_bfloat16 v) {
  return __isnan(v);
}
template <>
__device__ inline bool IsInf(hip_bfloat16 v) {
  return __isinf(v);
}
template <>
__device__ inline bool IsZero(hip_bfloat16 v) {
  return v == hip_bfloat16(0.0f);
}
template <>
__device__ inline bool IsNan(_Float16 v) {
  return isnan(static_cast<float>(v));
}
template <>
__device__ inline bool IsInf(_Float16 v) {
  return isinf(static_cast<float>(v));
}
template <>
__device__ inline bool IsZero(_Float16 v) {
  return v == 0.0f;
}

template <>
__device__ _Float16 WarpShuffleDown(_Float16 value, unsigned int offset) {
  return __builtin_bit_cast(
      _Float16, static_cast<uint16_t>(
                    __shfl_down(__builtin_bit_cast(uint16_t, value), offset)));
}

template <typename T>
__device__ inline T WarpShuffleDown(T value, unsigned int offset) {
  return __shfl_down(value, offset);
}

__device__ inline uint32_t AtomicIncSystem(uint32_t* write_idx) {
  return atomicAdd_system(write_idx, 1);
}

}  // namespace stream_executor::gpu

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckF32Kernel,
    stream_executor::gpu::BufferDebugFloatCheckF32Kernel,
    stream_executor::rocm::kROCmPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::FloatCheck<float>),
          "BufferDebugFloatCheckF32Kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckBf16Kernel,
    stream_executor::gpu::BufferDebugFloatCheckBf16Kernel,
    stream_executor::rocm::kROCmPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(
              &stream_executor::gpu::FloatCheck<hip_bfloat16>),
          "BufferDebugFloatCheckBf16Kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckF16Kernel,
    stream_executor::gpu::BufferDebugFloatCheckF16Kernel,
    stream_executor::rocm::kROCmPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::FloatCheck<_Float16>),
          "BufferDebugFloatCheckF16Kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckF64Kernel,
    stream_executor::gpu::BufferDebugFloatCheckF64Kernel,
    stream_executor::rocm::kROCmPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::FloatCheck<double>),
          "BufferDebugFloatCheckF64Kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugReduceFloatCheckResultsKernel,
    stream_executor::gpu::BufferDebugAppendReducedFloatCheckResultsKernel,
    stream_executor::rocm::kROCmPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::ReduceFloatCheckResults),
          "BufferDebugReduceFloatCheckResultsKernel", arity);
    }));
