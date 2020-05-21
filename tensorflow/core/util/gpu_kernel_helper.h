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

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#endif
#include "tensorflow/core/util/gpu_cuda_alias.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#if GOOGLE_CUDA
#define TF_RED_WARPSIZE 32
#elif TENSORFLOW_USE_ROCM
#define TF_RED_WARPSIZE 64
#endif

// Deprecated, use 'for(int i : GpuGridRangeX(n))' instead.
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))

// Deprecated, use 'for(int i : GpuGridRange?(n))' instead.
#define GPU_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::tensorflow::GpuGridRange##axis<int>(n))
#define CUDA_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::tensorflow::GpuGridRange##axis<int>(n))

#if GOOGLE_CUDA
#define gpuSuccess cudaSuccess
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
#elif TENSORFLOW_USE_ROCM
#define gpuSuccess hipSuccess
using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
#endif

// macro wrapper to declare dynamic shared memory
#if GOOGLE_CUDA

#if defined(__APPLE__)
    // Cannot use alignment on macOS due to #14174.
   #define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
      extern __shared__ TYPE NAME[]

      //The following is semantically correct on macOS but still cannot be used due to #14174.
      //extern __shared__  __attribute__((aligned(ALIGN))) TYPE NAME[]
#else
   #define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
      extern __shared__ __align__(ALIGN) TYPE NAME[]
#endif

#elif TENSORFLOW_USE_ROCM

#define GPU_DYNAMIC_SHARED_MEM_DECL(ALIGN, TYPE, NAME) \
  HIP_DYNAMIC_SHARED(TYPE, NAME)

#endif

namespace tensorflow {

#if GOOGLE_CUDA
// cudaGetErrorString is available to both host and device
__host__ __device__ inline const char* GpuGetErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}
#elif TENSORFLOW_USE_ROCM
// hipGetErrorString is available on host side only
inline const char* GpuGetErrorString(hipError_t error) {
  return hipGetErrorString(error);
}
#endif

// Returns a raw reference to the current cuda stream. Required by a
// number of kernel calls (for which StreamInterface* does not work),
// i.e. CUB and certain cublas primitives.
inline const gpuStream_t& GetGpuStream(OpKernelContext* context) {
  const gpuStream_t* ptr = CHECK_NOTNULL(
      reinterpret_cast<const gpuStream_t*>(context->op_device_context()
                                               ->stream()
                                               ->implementation()
                                               ->GpuStreamMemberHack()));
  return *ptr;
}

// Launches a GPU kernel through cudaLaunchKernel in CUDA environment, or
// hipLaunchKernel in ROCm environment with the given arguments.
//
// The kernel parameters 'Ts' must be constructible from the arguments 'Args'.
template <typename... Ts, typename... Args>
Status GpuLaunchKernel(void (*function)(Ts...), dim3 grid_dim, dim3 block_dim,
                       size_t shared_memory_size_bytes, gpuStream_t stream,
                       Args... arguments) {
  static_assert(detail::NoneIsReference<Ts...>(),
                "Kernels with reference arguments have undefined behaviour.");
#if GOOGLE_CUDA
  auto func_ptr = absl::bit_cast<const void*>(function);
  // Cast arguments and forward them as an array of pointers.
  auto args_tuple = std::tuple<Ts...>(arguments...);
  auto arg_ptrs = detail::GetArrayOfElementPointers(&args_tuple);
  auto result = cudaLaunchKernel(func_ptr, grid_dim, block_dim, arg_ptrs.data(),
                                 shared_memory_size_bytes, stream);
  if (result != cudaSuccess) {
    return errors::Internal(cudaGetErrorString(result));
  }
#elif TENSORFLOW_USE_ROCM
  hipLaunchKernelGGL(function, grid_dim, block_dim, shared_memory_size_bytes,
                     stream, std::forward<Args>(arguments)...);
#endif
  return Status::OK();
}

// Perfect forwarding to make CudaLaunchKernel available to both ROCm and CUDA
// builds
template <typename... Args>
auto CudaLaunchKernel(Args&&... args)
    -> decltype(GpuLaunchKernel(std::forward<Args>(args)...)) {
  return GpuLaunchKernel(std::forward<Args>(args)...);
}

__host__ __device__ inline tensorflow::bfloat16 GpuLdg(
    const tensorflow::bfloat16* address) {
  tensorflow::bfloat16 return_value;
  return_value.value = GpuLdg(reinterpret_cast<const uint16_t*>(address));
  return return_value;
}
// Already aliased in gpu_device_functions.h

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
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleUpSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
  return Eigen::half(
      GpuShuffleUpSync(mask, static_cast<uint16>(value), delta, width));
}
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleDownSync(
    unsigned mask, Eigen::half value, int delta, int width = warpSize) {
  return Eigen::half(
      GpuShuffleDownSync(mask, static_cast<uint16>(value), delta, width));
}
// Aliased in gpu_device_functions.h

__device__ EIGEN_ALWAYS_INLINE Eigen::half GpuShuffleXorSync(
    unsigned mask, Eigen::half value, int lane_mask, int width = warpSize) {
  return Eigen::half(
      GpuShuffleXorSync(mask, static_cast<uint16>(value), lane_mask, width));
}
// Aliased in gpu_device_functions.h
#endif

namespace gpu_helper {
template <typename T, typename OutType = int32>
__device__ OutType upper_bound(const T* first, OutType count, T val) {
  const T* orig = first;
  const T* it = nullptr;
  OutType step = 0;
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

template <typename T, typename OutType = int32>
__device__ OutType lower_bound(const T* first, OutType count, T val) {
  const T* orig = first;
  const T* it = nullptr;
  OutType step = 0;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (*it < val) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }

  return first - orig;
}

}  // namespace gpu_helper

#ifndef TENSORFLOW_USE_ROCM
namespace cuda_helper = gpu_helper;
#endif

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_UTIL_GPU_KERNEL_HELPER_H_
