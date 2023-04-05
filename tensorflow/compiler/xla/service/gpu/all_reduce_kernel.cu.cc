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

#include "tensorflow/compiler/xla/service/gpu/all_reduce_kernel.h"

#include <cassert>
#include <cstdint>

namespace {

using xla::gpu::kLaunchBounds;
using xla::gpu::kMaxBuffers;
using xla::gpu::kMaxNumGpus;
using xla::gpu::SyncFlag;

// Like std::array<T, kMaxNumGpus>, without the need for `relaxed-constexpr`.
template <typename T>
struct Array {
  __device__ constexpr const T& operator[](int i) const { return data[i]; }

 private:
  T data[kMaxNumGpus];
};

struct float2 {
  __device__ explicit float2(__nv_bfloat162 value)
      : x(__bfloat162float(value.x)), y(__bfloat162float(value.y)) {}
  __device__ operator __nv_bfloat162() const {
    __nv_bfloat162 result;
    result.x = __float2bfloat16_rn(x);
    result.y = __float2bfloat16_rn(y);
    return result;
  }
  __device__ float2& operator+=(const float2& rhs) {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }

 private:
  float x, y;
};

template <typename T>
struct MathType {
  using type = T;
};
template <>
struct MathType<__nv_bfloat16> {
  using type = float;
};
template <>
struct MathType<__nv_bfloat162> {
  using type = float2;
};
}  // namespace

static __device__ uint32_t atomic_inc_release_system(uint32_t* ptr,
                                                     uint32_t value) {
#if __CUDA_ARCH__ >= 700
  uint32_t result = 0;
  asm volatile("atom.inc.release.sys.u32 %0, [%1], %2;"
               : "=r"(result)
               : "l"(ptr), "r"(value)
               : "memory");
  return result;
#elif __CUDA_ARCH__ >= 600
  return atomicInc_system(ptr, value);
#else
  return __trap(), 0;  // Unsupported.
#endif
}

static __device__ uint32_t atomic_load_acquire_system(uint32_t* ptr) {
  uint32_t result = 0;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.b32 %0, [%1];"
               : "=r"(result)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.volatile.b32 %0, [%1];"
               : "=r"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

static __global__ void SyncKernel(uint32_t* counter) {
  atomic_inc_release_system(counter, kMaxNumGpus);
  while (atomic_load_acquire_system(counter) != 0) {
  }
}

template <typename T>
static __global__ void __launch_bounds__(kLaunchBounds)
    AllReduceKernel(int num_gpus, Array<const T* __restrict> send_buffers,
                    Array<T* __restrict> recv_buffers, int64_t num_elements,
                    uint32_t* counter, SyncFlag sync_flag) {
  if (sync_flag & SyncFlag::SYNC_START) {
    if (threadIdx.x == 0) {
      while (atomic_load_acquire_system(counter) != num_gpus - 1) {
      }
    }
    __syncthreads();
  }

  T vals[kMaxNumGpus];
  for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_elements;
       tid += blockDim.x * gridDim.x) {
    // Static loop bounds is required to store 'vals' in registers.
    for (int i = 0; i < kMaxNumGpus; ++i) {
      if (i >= num_gpus) break;
      vals[i] = send_buffers[i][tid];
    }
    using MathType = typename MathType<T>::type;
    MathType result = static_cast<MathType>(vals[0]);
    for (int i = 1; i < kMaxNumGpus; ++i) {
      if (i >= num_gpus) break;
      result += static_cast<MathType>(vals[i]);
    }
    for (int i = 0; i < kMaxNumGpus; ++i) {
      if (i >= num_gpus) break;
      recv_buffers[i][tid] = result;
    }
  }

  if (sync_flag & SyncFlag::SYNC_END) {
    __syncthreads();
    if (threadIdx.x == 0) {
      atomic_inc_release_system(counter, num_gpus + gridDim.x - 2);
    }
  }
}

// bfloat16x2 kernel for sm80+ that requires num_elements to be multiple of 32.
static __global__ void __launch_bounds__(kLaunchBounds)
    AllReduceKernelAsync(int num_gpus,
                         Array<const __nv_bfloat162* __restrict> send_buffers,
                         Array<__nv_bfloat162* __restrict> recv_buffers,
                         int64_t num_elements, uint32_t* counter,
                         SyncFlag sync_flag) {
  assert(num_elements % 32 == 0);

  if (sync_flag & SyncFlag::SYNC_START) {
    if (threadIdx.x == 0) {
      while (atomic_load_acquire_system(counter) != num_gpus - 1) {
      }
    }
    __syncthreads();
  }

#if __CUDA_ARCH__ >= 800
  __shared__ __nv_bfloat162 data[kMaxNumGpus][kLaunchBounds];

  // Groups of 4 consecutive threads load 4*bfloat16x2 (16B) each from 4
  // different GPUs at a time. That is, thread 4*k+i loads
  // elements [4*k, 4*k+1, 4*k+2, 4*k+3] from GPUs [i, i+4, i+8, i+12].
  int start_gpu = threadIdx.x & 0x3;
  int start_offset = threadIdx.x & ~0x3;
  uint32_t start_shared =
      __cvta_generic_to_shared(data[start_gpu] + start_offset);

  for (int offset = blockDim.x * blockIdx.x + start_offset;
       offset < num_elements; offset += blockDim.x * gridDim.x) {
    uint32_t shared = start_shared;
    for (int i = start_gpu; i < kMaxNumGpus; i += 4) {
      if (i >= num_gpus) break;
      asm volatile(
          "cp.async.ca.shared.global [%0], [%1], 16, 16;" ::"r"(shared),
          "l"(send_buffers[i] + offset)
          : "memory");
      shared += 4 * kLaunchBounds * sizeof(__nv_bfloat162);
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncwarp();

    const __nv_bfloat162* ptr = data[0] + threadIdx.x;
    auto f32x2 = __bfloat1622float2(*ptr);
    for (int i = 1; i < kMaxNumGpus; ++i) {
      if (i >= num_gpus) break;
      ptr += kLaunchBounds;
      auto tmp = __bfloat1622float2(*ptr);
      f32x2.x += tmp.x;
      f32x2.y += tmp.y;
    }
    __nv_bfloat162 bf16x2 = __floats2bfloat162_rn(f32x2.x, f32x2.y);
    unsigned result = reinterpret_cast<const unsigned&>(bf16x2);
    uint4 results = {
        __shfl_sync(~0u, result, 0, 4),  // x
        __shfl_sync(~0u, result, 1, 4),  // y
        __shfl_sync(~0u, result, 2, 4),  // z
        __shfl_sync(~0u, result, 3, 4)   // w
    };

    for (int i = start_gpu; i < kMaxNumGpus; i += 4) {
      if (i >= num_gpus) break;
      *reinterpret_cast<uint4* __restrict>(recv_buffers[i] + offset) = results;
    }
  }
#else
  __trap();  // Unsupported.
#endif

  if (sync_flag & SyncFlag::SYNC_END) {
    __syncthreads();
    if (threadIdx.x == 0) {
      atomic_inc_release_system(counter, num_gpus + gridDim.x - 2);
    }
  }
}

const void* xla::gpu::GetSyncKernel() {
  return reinterpret_cast<const void*>(&SyncKernel);
}

const void* xla::gpu::GetAllReduceKernel(ncclDataType_t dtype,
                                         int64_t* num_elements, int cc_major) {
  // Clang crashes if not wrapped in a IFEE.
  return [&]() -> const void* {
    switch (dtype) {
      case ncclBfloat16:
        if (cc_major >= 8 && *num_elements % 64 == 0) {
          *num_elements /= 2;
          return reinterpret_cast<const void*>(&AllReduceKernelAsync);
        }
        if (*num_elements % 2 == 0) {
          *num_elements /= 2;
          return reinterpret_cast<const void*>(
              &AllReduceKernel<__nv_bfloat162>);
        }
        return reinterpret_cast<const void*>(&AllReduceKernel<__nv_bfloat16>);
      case ncclFloat32:
        return reinterpret_cast<const void*>(&AllReduceKernel<float>);
      case ncclInt32:
        return reinterpret_cast<const void*>(&AllReduceKernel<int32_t>);
      default:
        return nullptr;
    }
  }();
}
