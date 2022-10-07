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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_MEM_ALLOCATOR_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_MEM_ALLOCATOR_H_

#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/framework/allocator.h"
#include "tensorflow/tsl/framework/device_id.h"

namespace stream_executor {

// Suballocator for StreamExecutor-based device memory.
class DeviceMemAllocator : public tsl::SubAllocator {
 public:
  // 'platform_device_id' refers to the ID of the device within
  // the process and must reference a valid ID in the process.
  // Note: stream_exec cannot be null.
  explicit DeviceMemAllocator(StreamExecutor* stream_exec,
                              tsl::PlatformDeviceId device_id,
                              bool use_unified_memory,
                              const std::vector<Visitor>& alloc_visitors,
                              const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors),
        stream_exec_(stream_exec),
        device_id_(device_id),
        use_unified_memory_(use_unified_memory) {
    CHECK(stream_exec_ != nullptr);
  }
  ~DeviceMemAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    void* ptr = nullptr;
    *bytes_received = num_bytes;
    if (num_bytes > 0) {
      if (use_unified_memory_) {
        ptr = stream_exec_->UnifiedMemoryAllocate(num_bytes);
      } else {
        ptr = stream_exec_->AllocateArray<char>(num_bytes).opaque();
      }
      VisitAlloc(ptr, device_id_.value(), num_bytes);
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      VisitFree(ptr, device_id_.value(), num_bytes);
      if (use_unified_memory_) {
        stream_exec_->UnifiedMemoryDeallocate(ptr);
      } else {
        DeviceMemoryBase device_ptr(ptr);
        stream_exec_->Deallocate(&device_ptr);
      }
    }
  }

  bool SupportsCoalescing() const override { return false; }

  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kDevice;
  }

 private:
  StreamExecutor* stream_exec_;  // not owned, non-null
  const tsl::PlatformDeviceId device_id_;
  const bool use_unified_memory_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceMemAllocator);
};

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_MEM_ALLOCATOR_H_
