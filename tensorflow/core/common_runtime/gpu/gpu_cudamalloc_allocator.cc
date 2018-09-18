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

#ifdef GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

GPUcudaMallocAllocator::GPUcudaMallocAllocator(VisitableAllocator* allocator,
                                               CudaGpuId cuda_gpu_id)
    : base_allocator_(allocator) {
  stream_exec_ = GpuIdUtil::ExecutorForCudaGpuId(cuda_gpu_id).ValueOrDie();
}

GPUcudaMallocAllocator::~GPUcudaMallocAllocator() { delete base_allocator_; }

void* GPUcudaMallocAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
#ifdef GOOGLE_CUDA
  // allocate with cudaMalloc
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  CUdeviceptr rv = 0;
  CUresult res = cuMemAlloc(&rv, num_bytes);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "cuMemAlloc failed to allocate " << num_bytes;
    return nullptr;
  }
  return reinterpret_cast<void*>(rv);
#else
  return nullptr;
#endif  // GOOGLE_CUDA
}
void GPUcudaMallocAllocator::DeallocateRaw(void* ptr) {
#ifdef GOOGLE_CUDA
  // free with cudaFree
  CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "cuMemFree failed to free " << ptr;
  }
#endif  // GOOGLE_CUDA
}

void GPUcudaMallocAllocator::AddAllocVisitor(Visitor visitor) {
  return base_allocator_->AddAllocVisitor(visitor);
}

void GPUcudaMallocAllocator::AddFreeVisitor(Visitor visitor) {
  return base_allocator_->AddFreeVisitor(visitor);
}

bool GPUcudaMallocAllocator::TracksAllocationSizes() { return false; }

}  // namespace tensorflow
