/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_
#define TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_

#if GOOGLE_CUDA

#include <algorithm>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

struct CudaLaunchConfig {
  // Logical number of thread that works on the elements. If each logical
  // thread works on exactly a single element, this is the same as the working
  // element count.
  int virtual_thread_count = -1;
  // Number of threads per block.
  int thread_per_block = -1;
  // Number of blocks for Cuda kernel launch.
  int block_count = -1;
};

// Calculate the Cuda launch config we should use for a kernel launch.
// This is assuming the kernel is quite simple and will largely be
// memory-limited.
inline CudaLaunchConfig GetCudaLaunchConfig(int work_element_count,
                                            const GPUDevice& d) {
  const int virtual_thread_count = work_element_count;
  const int physical_thread_count = std::min(
      d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor(),
      virtual_thread_count);
  const int thread_per_block = std::min(1024, d.maxCudaThreadsPerBlock());
  const int block_count = std::min(
      (physical_thread_count + thread_per_block - 1) / thread_per_block,
      d.getNumCudaMultiProcessors());

  CudaLaunchConfig config;
  config.virtual_thread_count = virtual_thread_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}

template <typename T>
__device__ __host__ inline T ldg(const T* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}

// CUDA provides atomic ops, but not for all types.  We provide wrappers
// for some ops and provide implementation for all reasonable types.
#define CUDA_ATOMIC_WRAPPER(op, T) \
  __device__ __forceinline__ T CudaAtomic##op(T* address, T val)

// Reason of guarding: NVCC cannot compile the "::" in "cuda_builtin::atomicOp".
#ifdef __GCUDACC__
#define USE_CUDA_ATOMIC(op, T) \
  CUDA_ATOMIC_WRAPPER(op, T) { return cuda_builtin::atomic##op(address, val); }
#else
#define USE_CUDA_ATOMIC(op, T) \
  CUDA_ATOMIC_WRAPPER(op, T) { return atomic##op(address, val); }
#endif

// For atomicAdd.
USE_CUDA_ATOMIC(Add, int32);
USE_CUDA_ATOMIC(Add, uint32);
USE_CUDA_ATOMIC(Add, uint64);
USE_CUDA_ATOMIC(Add, float);

// For atomicMax.
USE_CUDA_ATOMIC(Max, int32);
USE_CUDA_ATOMIC(Max, uint32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
USE_CUDA_ATOMIC(Max, uint64);
#else
// The uint64 overload of atomicMax() is only available for __CUDA_ARCH__ >=
// 350.  If not satisfied, we provide a custom implementation using atomicCAS().
CUDA_ATOMIC_WRAPPER(Max, uint64) {
  uint64* address_as_ull = (uint64*)address;
  uint64 old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, max(val, assumed));
  } while (assumed != old);

  return old;
}
#endif

// Custom implementation of atomicAdd for double.
// This implementation is copied from CUDA manual.
CUDA_ATOMIC_WRAPPER(Add, double) {
  uint64* address_as_ull = (uint64*)address;
  uint64 old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return __longlong_as_double(old);
}

// Helper functions for CudaAtomicAdd(half*, half), below.
//
// Note that if __CUDA_ARCH__ >= 530, we could probably use __hadd2()
// for a more efficient implementation, assuming that adding -0.0
// will never harm the neighboring value. In this version, we take special
// care to guarantee the bits of the untouched value are unchanged.
inline __device__ uint32 add_to_low_half(uint32 val, float x) {
  Eigen::half low_half;
  low_half.x = static_cast<uint16>(val & 0xffffu);
  low_half = static_cast<Eigen::half>(static_cast<float>(low_half) + x);
  return (val & 0xffff0000u) | low_half.x;
}

inline __device__ uint32 add_to_high_half(uint32 val, float x) {
  Eigen::half high_half;
  high_half.x = static_cast<uint16>(val >> 16);
  high_half = static_cast<Eigen::half>(static_cast<float>(high_half) + x);
  return (val & 0xffffu) | (high_half.x << 16);
}

// Custom implementation of atomicAdd for half. Note that we don't have
// atomicCAS() for anything less than 32 bits, so we need to include the
// other 16 bits in the operation.
//
// Unlike the other atomic adds, this version is going to be very slow
// under high concurrency, since most threads will be spinning on failing
// their compare-and-swap tests. (The fact that we get false sharing on the
// neighboring fp16 makes this even worse.) If you are doing a large reduction,
// you are much better off with doing the intermediate steps in fp32 and then
// switching to fp16 as late as you can in the calculations.
//
// Note: Assumes little endian.
CUDA_ATOMIC_WRAPPER(Add, Eigen::half) {
  float val_as_float(val);
  intptr_t address_int = reinterpret_cast<intptr_t>(address);
  if ((address_int & 0x2) == 0) {
    // The half is in the first part of the uint32 (lower 16 bits).
    uint32* address_as_uint32 = reinterpret_cast<uint32*>(address);
    assert(((intptr_t)address_as_uint32 & 0x3) == 0);
    uint32 old = *address_as_uint32, assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_uint32, assumed,
                      add_to_low_half(assumed, val_as_float));

      // Note: uses integer comparison to avoid hang in case of NaN
    } while (assumed != old);

    Eigen::half ret;
    ret.x = old & 0xffffu;
    return ret;
  } else {
    // The half is in the second part of the uint32 (upper 16 bits).
    uint32* address_as_uint32 = reinterpret_cast<uint32*>(address_int - 2);
    assert(((intptr_t)address_as_uint32 & 0x3) == 0);
    uint32 old = *address_as_uint32, assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_uint32, assumed,
                      add_to_high_half(assumed, val_as_float));

      // Note: uses integer comparison to avoid hang in case of NaN
    } while (assumed != old);

    Eigen::half ret;
    ret.x = old >> 16;
    return ret;
  }
}

template <typename T>
__global__ void SetZero(const int nthreads, T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) { *(bottom_diff + index) = T(0); }
}

// For atomicSub.

// Custom implementation for sub by just negating the value.
#define WRAPPED_ATOMIC_SUB(T) \
  CUDA_ATOMIC_WRAPPER(Sub, T) { return CudaAtomicAdd(address, -val); }

WRAPPED_ATOMIC_SUB(uint64);
WRAPPED_ATOMIC_SUB(int32);
WRAPPED_ATOMIC_SUB(uint32);
WRAPPED_ATOMIC_SUB(float);
WRAPPED_ATOMIC_SUB(double);

#undef WRAPPED_ATOMIC_SUB

#undef USE_CUDA_ATOMIC
#undef CUDA_ATOMIC_WRAPPER

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T tf_min(const T& x, const T& y) {
  return x > y ? y : x;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T tf_max(const T& x, const T& y) {
  return x < y ? y : x;
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_
