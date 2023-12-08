/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_GPU_KERNEL_HELPER_H_
#define XLA_SERVICE_GPU_RUNTIME_GPU_KERNEL_HELPER_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <type_traits>

#include "tsl/lib/math/math_util.h"

namespace xla {
namespace gpu {

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#else
#include "rocm/include/hip/hip_runtime.h"
#endif

#if GOOGLE_CUDA
#define WAVEFRONT_SIZE 32
#define FORCEINLINE __forceinline__
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
using gpuEvent_t = cudaEvent_t;
#define gpuSuccess cudaSuccess
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventDestroy cudaEventDestroy
#define gpuEventCreate cudaEventCreate
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuLaunchKernel cudaLaunchKernel
#define gpuMemcpy cudaMemcpy
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamSynchronize cudaStreamSynchronize

#elif TENSORFLOW_USE_ROCM
using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
using gpuEvent_t = hipEvent_t;
#define gpuSuccess hipSuccess
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuEventRecord hipEventRecord
#define gpuEventDestroy hipEventDestroy
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuLaunchKernel hipLaunchKernel
#define gpuMemcpy hipMemcpy
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuStreamCreate hipStreamCreate
#define gpuStreamSynchronize hipStreamSynchronize

#ifdef __AMDGCN_WAVEFRONT_SIZE
#define WAVEFRONT_SIZE __AMDGCN_WAVEFRONT_SIZE
#else
#define WAVEFRONT_SIZE 64
#endif
#define FORCEINLINE __forceinline__
#endif

// macro wrapper to declare dynamic shared memory
#if GOOGLE_CUDA

#define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
  extern __shared__ __align__(ALIGN)                   \
  TYPE NAME[]

#elif TENSORFLOW_USE_ROCM

#define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
  HIP_DYNAMIC_SHARED(TYPE, NAME)

#endif

enum class ShflType { Sync, Up, Down, Xor };

template <ShflType Type, class NT>
__device__ FORCEINLINE NT GpuShuffle(NT val, uint32_t idx,
                                     uint32_t allmsk = 0xffffffffu) {
  constexpr uint32_t SZ =
      tsl::MathUtil::CeilOfRatio(sizeof(NT), sizeof(uint32_t));
  union S {
    NT v;
    uint32_t d[SZ];
  };
  S in{val}, res{};

#pragma unroll
  for (uint32_t i = 0; i < SZ; i++) {
#if GOOGLE_CUDA
    if constexpr (Type == ShflType::Sync)
      res.d[i] = __shfl_sync(allmsk, in.d[i], idx);
    else if constexpr (Type == ShflType::Up)
      res.d[i] = __shfl_up_sync(allmsk, in.d[i], idx);
    else if constexpr (Type == ShflType::Down)
      res.d[i] = __shfl_down_sync(allmsk, in.d[i], idx);
    else if constexpr (Type == ShflType::Xor)
      res.d[i] = __shfl_xor_sync(allmsk, in.d[i], idx);
#elif TENSORFLOW_USE_ROCM  // ROcm does not support sync shuffle intrinsics
    if constexpr (Type == ShflType::Sync)
      res.d[i] = __shfl(in.d[i], idx);
    else if constexpr (Type == ShflType::Up)
      res.d[i] = __shfl_up(in.d[i], idx);
    else if constexpr (Type == ShflType::Down)
      res.d[i] = __shfl_down(in.d[i], idx);
    else if constexpr (Type == ShflType::Xor)
      res.d[i] = __shfl_xor(in.d[i], idx);
#endif
  }
  return res.v;
}

}  // namespace gpu
}  // namespace xla

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // XLA_SERVICE_GPU_RUNTIME_GPU_KERNEL_HELPER_H_
