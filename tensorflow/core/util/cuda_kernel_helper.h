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

#include "tensorflow/core/util/cuda_device_functions.h"
#include "tensorflow/core/util/cuda_launch_config.h"

// Deprecated, use 'for(int i : CudaGridRangeX(n))' instead.
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::CudaGridRangeX<int>(n))
// Deprecated, use 'for(int i : CudaGridRange?(n))' instead.
#define CUDA_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::tensorflow::CudaGridRange##axis<int>(n))

namespace tensorflow {
__host__ __device__ inline tensorflow::bfloat16 CudaLdg(
    const tensorflow::bfloat16* address) {
  tensorflow::bfloat16 return_value;
  return_value.value = CudaLdg(reinterpret_cast<const uint16_t*>(address));
  return return_value;
}

template <typename T>
__host__ __device__ inline T ldg(const T* ptr) {
  return CudaLdg(ptr);
}

template <typename T>
__host__ __device__ inline const T& tf_min(const T& x, const T& y) {
  return x < y ? x : y;
}

template <typename T>
__host__ __device__ inline const T& tf_max(const T& x, const T& y) {
  return x < y ? y : x;
}

// Overloads of the above functions for float and double.
__host__ __device__ inline float tf_min(float x, float y) {
  return fminf(x, y);
}
__host__ __device__ inline double tf_min(double x, double y) {
  return fmin(x, y);
}
__host__ __device__ inline float tf_max(float x, float y) {
  return fmaxf(x, y);
}
__host__ __device__ inline double tf_max(double x, double y) {
  return fmax(x, y);
}

__device__ inline Eigen::half CudaShuffleSync(unsigned mask, Eigen::half value,
                                              int src_lane,
                                              int width = warpSize) {
  return Eigen::half(
      CudaShuffleSync(mask, static_cast<uint16>(value), src_lane, width));
}

__device__ EIGEN_ALWAYS_INLINE Eigen::half CudaShuffleUpSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
  return Eigen::half(
      CudaShuffleUpSync(mask, static_cast<uint16>(value), delta, width));
}

__device__ EIGEN_ALWAYS_INLINE Eigen::half CudaShuffleDownSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
  return Eigen::half(
      CudaShuffleDownSync(mask, static_cast<uint16>(value), delta, width));
}

__device__ EIGEN_ALWAYS_INLINE Eigen::half CudaShuffleXorSync(
    unsigned mask, Eigen::half value, int lane_mask, int width = warpSize) {
  return Eigen::half(
      CudaShuffleXorSync(mask, static_cast<uint16>(value), lane_mask, width));
}

namespace detail {
// Overload of above function for half. Note that we don't have
// atomicCAS() for anything less than 32 bits, so we need to include the
// other 16 bits in the operation.
//
// This version is going to be very slow
// under high concurrency, since most threads will be spinning on failing
// their compare-and-swap tests. (The fact that we get false sharing on the
// neighboring fp16 makes this even worse.) If you are doing a large reduction,
// you are much better off with doing the intermediate steps in fp32 and then
// switching to fp16 as late as you can in the calculations.
//
// Note: Assumes little endian.
template <typename F>
__device__ Eigen::half CudaAtomicCasHelper(Eigen::half* ptr, F accumulate) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Not little endian");
#endif
  namespace half_impl = Eigen::half_impl;
  intptr_t intptr = reinterpret_cast<intptr_t>(ptr);
  assert(!(intptr & 0x1));  // should be 2-aligned.
  if (intptr & 0x2) {
    // The half is in the second part of the uint32 (upper 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr - 2);
    uint32 result = CudaAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short high = static_cast<unsigned short>(arg >> 16);
      Eigen::half acc = accumulate(half_impl::raw_uint16_to_half(high));
      return (static_cast<uint32>(acc.x) << 16) | (arg & 0xffff);
    });
    return half_impl::raw_uint16_to_half(static_cast<uint16>(result >> 16));
  } else {
    // The half is in the first part of the uint32 (lower 16 bits).
    uint32* address = reinterpret_cast<uint32*>(intptr);
    uint32 result = CudaAtomicCasHelper(address, [accumulate](uint32 arg) {
      unsigned short low = static_cast<unsigned short>(arg & 0xffff);
      Eigen::half acc = accumulate(half_impl::raw_uint16_to_half(low));
      return (arg & 0xffff0000) | static_cast<uint32>(acc.x);
    });
    return half_impl::raw_uint16_to_half(static_cast<uint16>(result & 0xffff));
  }
}
}  // namespace detail

__device__ inline Eigen::half CudaAtomicAdd(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a + value; });
}
__device__ inline Eigen::half CudaAtomicSub(Eigen::half* ptr,
                                            Eigen::half value) {
  return detail::CudaAtomicCasHelper(
      ptr, [value](Eigen::half a) { return a - value; });
}

namespace cuda_helper {
template <typename IntType>
__device__ IntType upper_bound(IntType* first, IntType count, IntType val) {
  IntType* orig = first;
  IntType* it = nullptr;
  IntType step = 0;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (!(val < *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return first - orig;
}
}  // namespace cuda_helper
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_
