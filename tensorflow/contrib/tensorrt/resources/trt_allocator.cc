/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"

#include "tensorflow/core/platform/logging.h"
#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#if NV_TENSORRT_MAJOR > 2
#include "cuda/include/cuda_runtime_api.h"

namespace tensorflow {
namespace tensorrt {
void* TRTCudaAllocator::allocate(uint64_t size, uint64_t alignment,
                                 uint32_t flags) {
  assert((alignment & (alignment - 1)) == 0);  // zero or a power of 2.
  void* memory;
  cudaMalloc(&memory, size);
  return memory;
}

void TRTCudaAllocator::free(void* memory) { cudaFree(memory); }

void* TRTDeviceAllocator::allocate(uint64_t size, uint64_t alignment,
                                   uint32_t flags) {
  assert((alignment & (alignment - 1)) == 0);  // zero or a power of 2.
  void* mem = allocator_->AllocateRaw(alignment, size);
  VLOG(2) << "Allocated " << size << " bytes with alignment " << alignment
          << " @ " << mem;
  return mem;
}

TRTDeviceAllocator::TRTDeviceAllocator(tensorflow::Allocator* allocator)
    : allocator_(allocator) {
  VLOG(1) << "Using " << allocator->Name() << " allocator from TensorFlow";
}

void TRTDeviceAllocator::free(void* memory) {
  VLOG(2) << "Deallocating " << memory;
  allocator_->DeallocateRaw(memory);
}

}  // namespace tensorrt
}  // namespace tensorflow
#endif
#endif
#endif
