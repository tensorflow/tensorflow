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

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#define EIGEN_USE_GPU
#endif

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#define EIGEN_USE_GPU
#endif

#include "xla/tsl/platform/logging.h"
#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"

namespace tensorflow {

void* GpuManagedAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* ptr = nullptr;
#if GOOGLE_CUDA
  CUdeviceptr result = 0;
  CHECK_EQ(cuMemAllocManaged(&result, num_bytes, CU_MEM_ATTACH_GLOBAL),
           CUDA_SUCCESS);
  ptr = reinterpret_cast<void*>(result);
#elif TENSORFLOW_USE_ROCM
  void** result = 0;
  CHECK_EQ(hipHostMalloc(&result, num_bytes, 0), 0);
  ptr = reinterpret_cast<void*>(result);
#endif
  CHECK(!(reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)));
  return ptr;
}

void GpuManagedAllocator::DeallocateRaw(void* ptr) {
#if GOOGLE_CUDA
  CHECK_EQ(cudaFree(ptr), cudaSuccess);
#elif TENSORFLOW_USE_ROCM
  CHECK_EQ(hipFree(ptr), hipSuccess);
#endif
}

}  // namespace tensorflow
