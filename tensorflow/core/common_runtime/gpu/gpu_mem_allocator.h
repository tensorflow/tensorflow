/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_MEM_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_MEM_ALLOCATOR_H_

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// Suballocator for GPU memory.
class GPUMemAllocator : public SubAllocator {
 public:
  // 'platform_gpu_id' refers to the ID of the GPU device within
  // the process and must reference a valid ID in the process.
  // Note: stream_exec cannot be null.
  explicit GPUMemAllocator(se::StreamExecutor* stream_exec,
                           PlatformGpuId gpu_id, bool use_unified_memory,
                           const std::vector<Visitor>& alloc_visitors,
                           const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors),
        stream_exec_(stream_exec),
        gpu_id_(gpu_id),
        use_unified_memory_(use_unified_memory) {
    CHECK(stream_exec_ != nullptr);
  }
  ~GPUMemAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    void* ptr = nullptr;
    if (num_bytes > 0) {
      if (use_unified_memory_) {
        ptr = stream_exec_->UnifiedMemoryAllocate(num_bytes);
      } else {
        ptr = stream_exec_->AllocateArray<char>(num_bytes).opaque();
      }
      VisitAlloc(ptr, gpu_id_.value(), num_bytes);
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      VisitFree(ptr, gpu_id_.value(), num_bytes);
      if (use_unified_memory_) {
        stream_exec_->UnifiedMemoryDeallocate(ptr);
      } else {
        se::DeviceMemoryBase gpu_ptr(ptr);
        stream_exec_->Deallocate(&gpu_ptr);
      }
    }
  }

 private:
  se::StreamExecutor* stream_exec_;  // not owned, non-null
  const PlatformGpuId gpu_id_;
  const bool use_unified_memory_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(GPUMemAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_MEM_ALLOCATOR_H_
