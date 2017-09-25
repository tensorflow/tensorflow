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

#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"

#ifdef GOOGLE_CUDA
#include "cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

void* GpuManagedAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
#ifdef GOOGLE_CUDA
  CUdeviceptr ptr = 0;
  CHECK_EQ(cuMemAllocManaged(&ptr, num_bytes, CU_MEM_ATTACH_GLOBAL),
           CUDA_SUCCESS);
  CHECK(!(ptr & (alignment - 1)));
  return reinterpret_cast<void*>(ptr);
#else
  return nullptr;
#endif
}

void GpuManagedAllocator::DeallocateRaw(void* ptr) {
#ifdef GOOGLE_CUDA
  CHECK_EQ(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)), CUDA_SUCCESS);
#endif
}

}  // namespace tensorflow
