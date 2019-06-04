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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_HOST_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_HOST_ALLOCATOR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
// Allocator for pinned CPU RAM that is made known to GPU for the
// purpose of efficient DMA with a GPU.
class GpuHostAllocator : public SubAllocator {
 public:
  // Note: stream_exec cannot be null.
  explicit GpuHostAllocator(se::StreamExecutor* stream_exec, int numa_node,
                            const std::vector<Visitor>& alloc_visitors,
                            const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors),
        stream_exec_(stream_exec),
        numa_node_(numa_node) {
    CHECK(stream_exec_ != nullptr);
  }
  ~GpuHostAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    void* ptr = nullptr;
    if (num_bytes > 0) {
      ptr = stream_exec_->HostMemoryAllocate(num_bytes);
      if (ptr == nullptr) {
        LOG(WARNING) << "could not allocate pinned host memory of size: "
                     << num_bytes;
        return ptr;
      }
      VisitAlloc(ptr, numa_node_, num_bytes);
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      VisitFree(ptr, numa_node_, num_bytes);
      stream_exec_->HostMemoryDeallocate(ptr);
    }
  }

 private:
  se::StreamExecutor* stream_exec_;  // not owned, non-null
  const int numa_node_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuHostAllocator);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_HOST_ALLOCATOR_H_
