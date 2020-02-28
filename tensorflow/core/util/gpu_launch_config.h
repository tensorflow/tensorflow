/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_GPU_LAUNCH_CONFIG_H_
#define TENSORFLOW_CORE_UTIL_GPU_LAUNCH_CONFIG_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <algorithm>

#include "absl/base/casts.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_cuda_alias.h"

// Usage of GetGpuLaunchConfig, GetGpu2DLaunchConfig, and
// GetGpu3DLaunchConfig:
//
// There are two versions of GetGpuLaunchConfig and GetGpu2DLaunchConfig, one
// version uses heuristics without any knowledge of the device kernel, the other
// version uses cudaOccupancyMaxPotentialBlockSize to determine the theoretical
// launch parameters that maximize occupancy. Currently, only the maximum
// occupancy version of GetGpu3DLaunchConfig is available.
//
// For large number of work elements, the convention is that each kernel would
// iterate through its assigned range. The return value of GetGpuLaunchConfig
// is struct GpuLaunchConfig, which contains all the information needed for the
// kernel launch, including: virtual number of threads, the number of threads
// per block and number of threads per block used inside <<< >>> of a kernel
// launch. GetGpu2DLaunchConfig and GetGpu3DLaunchConfig does the same thing
// as GpuLaunchConfig. The only difference is the dimension. The macros
// GPU_1D_KERNEL_LOOP and GPU_AXIS_KERNEL_LOOP might be used to do inner loop.
//
/* Sample code:

__global__ void MyKernel1D(GpuLaunchConfig config, other_args...) {
  GPU_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    do_your_job_here;
  }
}

__global__ void MyKernel2D(Gpu2DLaunchConfig config, other_args...) {
  GPU_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    GPU_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      do_your_job_here;
    }
  }
}

__global__ void MyKernel3D(Gpu3DLaunchConfig config, other_args...) {
  GPU_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    GPU_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      GPU_AXIS_KERNEL_LOOP(z, config.virtual_thread_count, z) {
        do_your_job_here;
      }
    }
  }
}

void MyDriverFunc(const Eigen::GpuDevice &d) {
  // use heuristics
  GpuLaunchConfig cfg1 = GetGpuLaunchConfig(10240, d);
  MyKernel1D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg1, other_args...);
  Gpu2DLaunchConfig cfg2 = GetGpu2DLaunchConfig(10240, 10240, d);
  MyKernel2D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg2, other_args...);
  Gpu3DLaunchConfig cfg3 = GetGpu3DLaunchConfig(4096, 4096, 100, d);
  MyKernel3D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg3, other_args...);

  // maximize occupancy
  GpuLaunchConfig cfg4 = GetGpuLaunchConfig(10240, d, MyKernel1D, 0, 0 );
  MyKernel1D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg4, other_args...);
  Gpu2DLaunchConfig cfg5 = GetGpu2DLaunchConfig(10240, 10240, d,
                                                  MyKernel1D, 0, 0);
  MyKernel2D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg5, other_args...);
  Gpu3DLaunchConfig cfg6 = GetGpu3DLaunchConfig(4096, 4096, 100, d,
                                                  MyKernel1D, 0, 0);
  MyKernel3D <<<config.block_count,
                config.thread_per_block, 0, d.stream()>>> (cfg6, other_args...);
}

// See the test for this for more example:
//
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/gpu_kernel_helper_test.cu.cc

*/

namespace tensorflow {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

struct GpuLaunchConfig {
  // Logical number of thread that works on the elements. If each logical
  // thread works on exactly a single element, this is the same as the working
  // element count.
  int virtual_thread_count = -1;
  // Number of threads per block.
  int thread_per_block = -1;
  // Number of blocks for GPU kernel launch.
  int block_count = -1;
};
CREATE_CUDA_TYPE_ALIAS(GpuLaunchConfig, CudaLaunchConfig);

// Calculate the GPU launch config we should use for a kernel launch.
// This is assuming the kernel is quite simple and will largely be
// memory-limited.
// REQUIRES: work_element_count > 0.
inline GpuLaunchConfig GetGpuLaunchConfig(int work_element_count,
                                          const Eigen::GpuDevice& d) {
  CHECK_GT(work_element_count, 0);
  GpuLaunchConfig config;
  const int virtual_thread_count = work_element_count;
  const int physical_thread_count = std::min(
      d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor(),
      virtual_thread_count);
  const int thread_per_block = std::min(1024, d.maxGpuThreadsPerBlock());
  const int block_count =
      std::min(DivUp(physical_thread_count, thread_per_block),
               d.getNumGpuMultiProcessors());

  config.virtual_thread_count = virtual_thread_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}
#ifndef TENSORFLOW_USE_ROCM
inline CudaLaunchConfig GetCudaLaunchConfig(int work_element_count,
                                            const Eigen::GpuDevice& d) {
  return GetGpuLaunchConfig(work_element_count, d);
}
#endif

// Calculate the GPU launch config we should use for a kernel launch. This
// variant takes the resource limits of func into account to maximize occupancy.
// REQUIRES: work_element_count > 0.
template <typename DeviceFunc>
GpuLaunchConfig GetGpuLaunchConfig(int work_element_count,
                                   const Eigen::GpuDevice& d, DeviceFunc func,
                                   size_t dynamic_shared_memory_size,
                                   int block_size_limit) {
  CHECK_GT(work_element_count, 0);
  GpuLaunchConfig config;
  int block_count = 0;
  int thread_per_block = 0;

#if GOOGLE_CUDA
  cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
      &block_count, &thread_per_block, func, dynamic_shared_memory_size,
      block_size_limit);
  CHECK_EQ(err, cudaSuccess);
#elif TENSORFLOW_USE_ROCM
  // Earlier versions of this HIP routine incorrectly returned void.
  // TODO re-enable hipError_t error checking when HIP is fixed.
  // ROCm interface uses unsigned int, convert after checking
  uint32_t block_count_uint = 0;
  uint32_t thread_per_block_uint = 0;
  CHECK_GE(block_size_limit, 0);
  uint32_t block_size_limit_uint = static_cast<uint32_t>(block_size_limit);
  hipOccupancyMaxPotentialBlockSize(&block_count_uint, &thread_per_block_uint,
                                    func, dynamic_shared_memory_size,
                                    block_size_limit_uint);
  block_count = static_cast<int>(block_count_uint);
  thread_per_block = static_cast<int>(thread_per_block_uint);
#endif

  block_count =
      std::min(block_count, DivUp(work_element_count, thread_per_block));

  config.virtual_thread_count = work_element_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}
CREATE_CUDA_HOST_FUNCTION_ALIAS(GetGpuLaunchConfig, GetCudaLaunchConfig);

// Calculate the GPU launch config we should use for a kernel launch. This
// variant takes the resource limits of func into account to maximize occupancy.
// The returned launch config has thread_per_block set to fixed_block_size.
// REQUIRES: work_element_count > 0.
template <typename DeviceFunc>
GpuLaunchConfig GetGpuLaunchConfigFixedBlockSize(
    int work_element_count, const Eigen::GpuDevice& d, DeviceFunc func,
    size_t dynamic_shared_memory_size, int fixed_block_size) {
  CHECK_GT(work_element_count, 0);
  GpuLaunchConfig config;
  int block_count = 0;

#if GOOGLE_CUDA
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &block_count, func, fixed_block_size, dynamic_shared_memory_size);
  CHECK_EQ(err, cudaSuccess);
  block_count = std::min(block_count * d.getNumGpuMultiProcessors(),
                         DivUp(work_element_count, fixed_block_size));
#elif TENSORFLOW_USE_ROCM
  // ROCM TODO re-enable this after hipOccupancyMaxActiveBlocksPerMultiprocessor
  // is implemented
  // hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
  //    &block_count, &thread_per_block, func, dynamic_shared_memory_size,
  //    block_size_limit);
  // CHECK_EQ(err, hipSuccess);

  // Apply the heuristic in GetGpuLaunchConfig(int, const Eigen::GpuDevice&)
  // that the kernel is quite simple and will largely be memory-limited.
  const int physical_thread_count = std::min(
      d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor(),
      work_element_count);
  // Assume the kernel be simple enough that it is okay to use 1024 threads
  // per workgroup.
  int thread_per_block = std::min(1024, d.maxGpuThreadsPerBlock());
  block_count = std::min(DivUp(physical_thread_count, thread_per_block),
                         d.getNumGpuMultiProcessors());
#endif

  config.virtual_thread_count = work_element_count;
  config.thread_per_block = fixed_block_size;
  config.block_count = block_count;
  return config;
}
CREATE_CUDA_HOST_FUNCTION_ALIAS(GetGpuLaunchConfigFixedBlockSize,
                                GetCudaLaunchConfigFixedBlockSize);

struct Gpu2DLaunchConfig {
  dim3 virtual_thread_count = dim3(0, 0, 0);
  dim3 thread_per_block = dim3(0, 0, 0);
  dim3 block_count = dim3(0, 0, 0);
};
CREATE_CUDA_TYPE_ALIAS(Gpu2DLaunchConfig, Cuda2DLaunchConfig);

inline Gpu2DLaunchConfig GetGpu2DLaunchConfig(int xdim, int ydim,
                                              const Eigen::GpuDevice& d) {
  Gpu2DLaunchConfig config;

  if (xdim <= 0 || ydim <= 0) {
    return config;
  }

  const int kThreadsPerBlock = 256;
  int block_cols = std::min(xdim, kThreadsPerBlock);
  // ok to round down here and just do more loops in the kernel
  int block_rows = std::max(kThreadsPerBlock / block_cols, 1);

  const int physical_thread_count =
      d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor();

  const int max_blocks = std::max(physical_thread_count / kThreadsPerBlock, 1);

  config.virtual_thread_count = dim3(xdim, ydim, 1);
  config.thread_per_block = dim3(block_cols, block_rows, 1);

  int grid_x = std::min(DivUp(xdim, block_cols), max_blocks);

  config.block_count = dim3(
      grid_x, std::min(max_blocks / grid_x, std::max(ydim / block_rows, 1)), 1);
  return config;
}
#ifndef TENSORFLOW_USE_ROCM
inline Cuda2DLaunchConfig GetCuda2DLaunchConfig(int xdim, int ydim,
                                                const Eigen::GpuDevice& d) {
  return GetGpu2DLaunchConfig(xdim, ydim, d);
}
#endif

// Calculate the GPU 2D and 3D launch config we should use for a kernel launch.
// This variant takes the resource limits of func into account to maximize
// occupancy.
using Gpu3DLaunchConfig = Gpu2DLaunchConfig;
CREATE_CUDA_TYPE_ALIAS(Gpu3DLaunchConfig, Cuda3DLaunchConfig);

template <typename DeviceFunc>
Gpu3DLaunchConfig GetGpu3DLaunchConfig(int xdim, int ydim, int zdim,
                                       const Eigen::GpuDevice& d,
                                       DeviceFunc func,
                                       size_t dynamic_shared_memory_size,
                                       int block_size_limit) {
  Gpu3DLaunchConfig config;

  if (xdim <= 0 || ydim <= 0 || zdim <= 0) {
    return config;
  }

  int dev;
#if GOOGLE_CUDA
  cudaGetDevice(&dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
#elif TENSORFLOW_USE_ROCM
  hipGetDevice(&dev);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, dev);
#endif
  int xthreadlimit = deviceProp.maxThreadsDim[0];
  int ythreadlimit = deviceProp.maxThreadsDim[1];
  int zthreadlimit = deviceProp.maxThreadsDim[2];
  int xgridlimit = deviceProp.maxGridSize[0];
  int ygridlimit = deviceProp.maxGridSize[1];
  int zgridlimit = deviceProp.maxGridSize[2];

  int block_count = 0;
  int thread_per_block = 0;

#if GOOGLE_CUDA
  cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
      &block_count, &thread_per_block, func, dynamic_shared_memory_size,
      block_size_limit);
  CHECK_EQ(err, cudaSuccess);
#elif TENSORFLOW_USE_ROCM
  // ROCM TODO re-enable this after hipOccupancyMaxPotentialBlockSize is
  // implemented
  // hipError_t err = hipOccupancyMaxPotentialBlockSize(
  //    &block_count, &thread_per_block, func, dynamic_shared_memory_size,
  //    block_size_limit);
  // CHECK_EQ(err, hipSuccess);

  const int physical_thread_count =
      d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor();
  thread_per_block = std::min(1024, d.maxGpuThreadsPerBlock());
  block_count = std::min(DivUp(physical_thread_count, thread_per_block),
                         d.getNumGpuMultiProcessors());
#endif

  int threadsx = std::min({xdim, thread_per_block, xthreadlimit});
  int threadsy =
      std::min({ydim, std::max(thread_per_block / threadsx, 1), ythreadlimit});
  int threadsz =
      std::min({zdim, std::max(thread_per_block / (threadsx * threadsy), 1),
                zthreadlimit});

  int blocksx = std::min({block_count, DivUp(xdim, threadsx), xgridlimit});
  int blocksy = std::min(
      {DivUp(block_count, blocksx), DivUp(ydim, threadsy), ygridlimit});
  int blocksz = std::min({DivUp(block_count, (blocksx * blocksy)),
                          DivUp(zdim, threadsz), zgridlimit});

  config.virtual_thread_count = dim3(xdim, ydim, zdim);
  config.thread_per_block = dim3(threadsx, threadsy, threadsz);
  config.block_count = dim3(blocksx, blocksy, blocksz);
  return config;
}
CREATE_CUDA_HOST_FUNCTION_ALIAS(GetGpu3DLaunchConfig, GetCuda3DLaunchConfig);

template <typename DeviceFunc>
Gpu2DLaunchConfig GetGpu2DLaunchConfig(int xdim, int ydim,
                                       const Eigen::GpuDevice& d,
                                       DeviceFunc func,
                                       size_t dynamic_shared_memory_size,
                                       int block_size_limit) {
  return GetGpu3DLaunchConfig(xdim, ydim, 1, d, func,
                              dynamic_shared_memory_size, block_size_limit);
}
CREATE_CUDA_HOST_FUNCTION_ALIAS(GetGpu2DLaunchConfig, GetCuda2DLaunchConfig);

#if GOOGLE_CUDA
template <typename DeviceFunc>
Cuda2DLaunchConfig GetCuda2DLaunchConfig(int xdim, int ydim,
                                         const Eigen::GpuDevice& d,
                                         DeviceFunc func,
                                         size_t dynamic_shared_memory_size,
                                         int block_size_limit) {
  return GetGpu2DLaunchConfig(xdim, ydim, d, func, dynamic_shared_memory_size,
                              block_size_limit);
}
#endif  // GOOGLE_CUDA

namespace detail {
template <typename... Ts, size_t... Is>
std::array<void*, sizeof...(Ts)> GetArrayOfElementPointersImpl(
    std::tuple<Ts...>* tuple, absl::index_sequence<Is...>) {
  return {{&std::get<Is>(*tuple)...}};
}
// Returns an array of void pointers to the elements of the given tuple.
template <typename... Ts>
std::array<void*, sizeof...(Ts)> GetArrayOfElementPointers(
    std::tuple<Ts...>* tuple) {
  return GetArrayOfElementPointersImpl(tuple,
                                       absl::index_sequence_for<Ts...>{});
}

template <bool...>
struct BoolPack;
template <bool... Bs>
using NoneTrue = std::is_same<BoolPack<Bs..., false>, BoolPack<false, Bs...>>;
// Returns whether none of the types in Ts is a reference.
template <typename... Ts>
constexpr bool NoneIsReference() {
  return NoneTrue<(std::is_reference<Ts>::value)...>::value;
}
}  // namespace detail
}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_UTIL_GPU_LAUNCH_CONFIG_H_
