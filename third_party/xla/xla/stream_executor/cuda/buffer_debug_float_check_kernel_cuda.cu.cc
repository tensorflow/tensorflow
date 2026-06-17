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

#include <cassert>
#include <cstdint>

#include "absl/base/casts.h"
#include "Eigen/Core"
#include "third_party/gpus/cuda/include/cuda/atomic"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::gpu {
__device__ static constexpr uint64_t kWarpSize = 32;
}  // namespace stream_executor::gpu

#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel_lib.cu.h"

namespace stream_executor::gpu {

// __nv_bfloat16 is a distinct type different from Eigen::bfloat16.
// CUDA doesn't provide std::numeric_limits<__nv_bfloat16>::infinity(), which
// makes it default to returning 0. This is suboptimal and results in a lot of
// fun when debugging.
//
// Neither the constructor of __nv_bfloat16 nor the CUDART_INF_BF16 constants
// are constexpr, so we can't use them directly. However:
// - Eigen::bfloat16 *does* have a valid numeric_limits::infinity(),
// - absl::bit_cast is constexpr,
// - __nv_bfloat16 and Eigen::bfloat16 are bitwise identical
// So we can get a constant this way.
template <>
__device__ constexpr __nv_bfloat16 kInfinity<__nv_bfloat16> =
    absl::bit_cast<__nv_bfloat16>(kInfinity<Eigen::bfloat16>);
// - __half lacks std::numeric_limits specialization, and Eigen::half is not a
// literal type (non-constexpr constructors), so we construct infinity from raw
// bits.
template <>
__device__ constexpr __half kInfinity<__half> =
    absl::bit_cast<__half>(uint16_t{0x7C00});

template <>
__device__ inline bool IsNan(__nv_bfloat16 v) {
  return __isnan(v);
}
template <>
__device__ inline bool IsInf(__nv_bfloat16 v) {
  return __isinf(v);
}
template <>
__device__ inline bool IsZero(__nv_bfloat16 v) {
  return v == __nv_bfloat16(0.0f);
}
template <>
__device__ inline bool IsNan(__half v) {
  return __isnan(v);
}
template <>
__device__ inline bool IsInf(__half v) {
  return __isinf(v);
}
template <>
__device__ inline bool IsZero(__half v) {
  return v == __half(0.0f);
}

template <typename T>
__device__ inline T WarpShuffleDown(T value, unsigned int offset) {
  static constexpr uint32_t kFullMask = ~0u;
  return __shfl_down_sync(kFullMask, value, offset);
}

__device__ inline uint32_t AtomicIncSystem(uint32_t* write_idx) {
#if __CUDA_ARCH__ >= 600
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> log_write_idx(
      *write_idx);
  return log_write_idx.fetch_add(1);
#else
  // Our toolchains generate a fetch_add PTX instructions with system scope,
  // which is not supported on pre-Pascal architectures.
  assert(false);
  return ~0u;
#endif
}

}  // namespace stream_executor::gpu

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckF32Kernel,
    stream_executor::gpu::BufferDebugFloatCheckF32Kernel,
    stream_executor::cuda::kCudaPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::FloatCheck<float>),
          "BufferDebugFloatCheckF32Kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckBf16Kernel,
    stream_executor::gpu::BufferDebugFloatCheckBf16Kernel,
    stream_executor::cuda::kCudaPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(
              &stream_executor::gpu::FloatCheck<__nv_bfloat16>),
          "BufferDebugFloatCheckBf16Kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckF64Kernel,
    stream_executor::gpu::BufferDebugFloatCheckF64Kernel,
    stream_executor::cuda::kCudaPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::FloatCheck<double>),
          "BufferDebugFloatCheckF64Kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckF16Kernel,
    stream_executor::gpu::BufferDebugFloatCheckF16Kernel,
    stream_executor::cuda::kCudaPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::FloatCheck<__half>),
          "BufferDebugFloatCheckF16Kernel", arity);
    }));

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugReduceFloatCheckResultsKernel,
    stream_executor::gpu::BufferDebugAppendReducedFloatCheckResultsKernel,
    stream_executor::cuda::kCudaPlatformId, ([](int arity) {
      return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
          absl::bit_cast<void*>(&stream_executor::gpu::ReduceFloatCheckResults),
          "BufferDebugReduceFloatCheckResultsKernel", arity);
    }));
