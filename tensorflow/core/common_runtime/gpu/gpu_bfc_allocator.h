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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.
class GPUBFCAllocator : public BFCAllocator {
 public:
  // 'cuda_gpu_id' refers to the ID of the GPU device within
  // the process and must reference a valid ID in the process.
  GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                  const string& name,
                  const std::vector<CudaGpuId>& valid_cuda_gpu_ids);
  GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                  const GPUOptions& gpu_options, const string& name,
                  const std::vector<CudaGpuId>& valid_cuda_gpu_ids);
  virtual ~GPUBFCAllocator() {}

  TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);
};

// Suballocator for GPU memory.
class GPUMemAllocator : public SubAllocator {
 public:
  // Note: stream_exec cannot be null.
  explicit GPUMemAllocator(se::StreamExecutor* stream_exec,
                           bool use_unified_memory,
                           const std::vector<CudaGpuId> valid_cuda_gpu_ids)
      : stream_exec_(stream_exec),
        use_unified_memory_(use_unified_memory),
        valid_cuda_gpu_ids_(valid_cuda_gpu_ids) {
    CHECK(stream_exec_ != nullptr);
  }
  ~GPUMemAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    char* ptr = nullptr;
    if (num_bytes > 0) {
      if (use_unified_memory_) {
        ptr = static_cast<char*>(stream_exec_->UnifiedMemoryAllocate(num_bytes));

        // TODO(fo40225): advise only on linux and cc_major >= 6

        size_t blockSize = num_bytes / valid_cuda_gpu_ids_.size();
        for (size_t i = 0; i < valid_cuda_gpu_ids_.size(); ++i) {

          // CU_MEM_ADVISE_SET_PREFERRED_LOCATION
          stream_exec_->UnifiedMemoryAdvise(ptr + i * blockSize, blockSize, 3,
                                              valid_cuda_gpu_ids_[i].value());
          for (size_t j = 0; j < valid_cuda_gpu_ids_.size(); ++j) {

            // CU_MEM_ADVISE_SET_ACCESSED_BY
            stream_exec_->UnifiedMemoryAdvise(ptr + i * blockSize, blockSize, 5,
                                              valid_cuda_gpu_ids_[j].value());
          }
        }

        if (num_bytes % valid_cuda_gpu_ids_.size()) {
          size_t advised_size = blockSize * valid_cuda_gpu_ids_.size();
          size_t remain_size = num_bytes - advised_size;

          // CU_MEM_ADVISE_SET_PREFERRED_LOCATION
          stream_exec_->UnifiedMemoryAdvise(ptr + advised_size, remain_size, 3);
          for (size_t k = 0; k < valid_cuda_gpu_ids_.size(); ++k) {

            // CU_MEM_ADVISE_SET_ACCESSED_BY
            stream_exec_->UnifiedMemoryAdvise(ptr + advised_size, remain_size,
                                              5, valid_cuda_gpu_ids_[k].value());
          }
        }
      } else {
        ptr = static_cast<char*>(stream_exec_->AllocateArray<char>(num_bytes).opaque());
      }
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
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
  const bool use_unified_memory_ = false;
  const std::vector<CudaGpuId> valid_cuda_gpu_ids_;
  TF_DISALLOW_COPY_AND_ASSIGN(GPUMemAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
