/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_

#if GOOGLE_CUDA

#include <algorithm>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

struct CudaLaunchConfig {
  // Logical number of thread that works on the elements. If each logic thread
  // works on exactly a single element, this is the same as the working element
  // count.
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

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // THIRD_PARTY_TENSORFLOW_CORE_UTIL_CUDA_KERNEL_HELPER_H_
