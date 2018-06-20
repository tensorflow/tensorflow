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

#ifndef TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
#define TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#if GPU_VERSION >= 7050
#include "cuda/include/cuda_fp16.h"
#define TF_HAS_GPU_FP16
#endif

#if GOOGLE_CUDA
#define TF_RED_WARPSIZE 32
#elif TENSORFLOW_USE_ROCM
#define TF_RED_WARPSIZE 64
#endif

#if GOOGLE_CUDA
#define GPU_LAUNCH_KERNEL(kernel, block_count, threads_per_block, \
                          shared_mem, stream, ...) \
  kernel<<<block_count, threads_per_block, shared_mem, stream>>>(__VA_ARGS__);
#elif TENSORFLOW_USE_ROCM
#define GPU_LAUNCH_KERNEL(kernel, block_count, threads_per_block, \
                          shared_mem, stream, ...) \
  hipLaunchKernelGGL(kernel, \
    block_count, threads_per_block, shared_mem, stream, \
    __VA_ARGS__);
#endif

// Deprecated, use 'for(int i : GpuGridRangeX(n))' instead.
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))
// Deprecated, use 'for(int i : GpuGridRange?(n))' instead.
#define GPU_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::tensorflow::GpuGridRange##axis<int>(n))

#if GOOGLE_CUDA
#define gpuSuccess cudaSuccess
#define GPUGETERRORSTRING(error) cudaGetErrorString(error)
using gpuStream_t = cudaStream_t;
#define GPUGETLASTERROR() cudaGetLastError()
using gpuError_t = cudaError_t;
#define GPUSUCCESSS cudaSuccess
#elif TENSORFLOW_USE_ROCM
#define gpuSuccess hipSuccess
#define GPUGETERRORSTRING(error) hipGetErrorString(error)
using gpuStream_t = hipStream_t;
#define GPUGETLASTERROR() hipGetLastError()
using gpuError_t = hipError_t;
#define GPUSUCCESSS hipSuccess
#endif

#if GOOGLE_CUDA
#define GetGPUStream(context) context->eigen_gpu_device().stream()
#elif TENSORFLOW_USE_ROCM
#define GetGPUStream(context) context->eigen_gpu_device().stream()
#endif

namespace tensorflow {
__host__ __device__ inline tensorflow::bfloat16 GpuLdg(
    const tensorflow::bfloat16* address) {
  tensorflow::bfloat16 return_value;
  return_value.value = GpuLdg(reinterpret_cast<const uint16_t*>(address));
  return return_value;
}

template <typename T>
__host__ __device__ inline T ldg(const T* ptr) {
  return GpuLdg(ptr);
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

// ROCM TODO re-enable them after adding fp16 support logic
#if GOOGLE_CUDA
__device__ inline Eigen::half GpuShuffleSync(unsigned mask, Eigen::half value,
                                              int src_lane,
                                              int width = warpSize) {
  return Eigen::half(
      GpuShuffleSync(mask, static_cast<uint16>(value), src_lane, width));
}

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleUpSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
  return Eigen::half(
      GpuShuffleUpSync(mask, static_cast<uint16>(value), delta, width));
}

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleDownSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
  return Eigen::half(
      GpuShuffleDownSync(mask, static_cast<uint16>(value), delta, width));
}

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleXorSync(
    unsigned mask, Eigen::half value, int lane_mask, int width = warpSize) {
  return Eigen::half(
      GpuShuffleXorSync(mask, static_cast<uint16>(value), lane_mask, width));
}
#endif

namespace gpu_helper {
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
}  // namespace gpu_helper
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
