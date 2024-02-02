/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_MEM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_MEM_ALLOCATOR_H_

#include <vector>

#include "xla/stream_executor/stream_executor.h"
#include "tsl/framework/allocator.h"
#include "tsl/framework/device_id.h"
#include "tsl/profiler/lib/traceme.h"

namespace stream_executor {

// The type of memory that the allocator will use.
enum class MemoryType { kDevice = 0, kUnified, kCollective };

// Suballocator for StreamExecutor-based device memory.
class DeviceMemAllocator : public tsl::SubAllocator {
 public:
  // 'platform_device_id' refers to the ID of the device within
  // the process and must reference a valid ID in the process.
  // Note: stream_exec cannot be null.
  explicit DeviceMemAllocator(StreamExecutor* stream_exec,
                              tsl::PlatformDeviceId device_id,
                              MemoryType memory_type,
                              const std::vector<Visitor>& alloc_visitors,
                              const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors),
        stream_exec_(stream_exec),
        device_id_(device_id),
        memory_type_(memory_type) {
    CHECK(stream_exec_ != nullptr);
  }

  ~DeviceMemAllocator() override = default;

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    tsl::profiler::TraceMe traceme("DeviceMemAllocator::Alloc");

    void* ptr = nullptr;
    *bytes_received = num_bytes;
    if (num_bytes > 0) {
      if (memory_type_ == MemoryType::kUnified) {
        ptr = stream_exec_->UnifiedMemoryAllocate(num_bytes);
      } else if (memory_type_ == MemoryType::kCollective) {
        auto status_or = stream_exec_->CollectiveMemoryAllocate(num_bytes);
        CHECK(status_or.ok()) << status_or.status().message();
        ptr = status_or.value();
      } else {
        ptr = stream_exec_->AllocateArray<char>(num_bytes).opaque();
      }
      VisitAlloc(ptr, device_id_.value(), num_bytes);
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    tsl::profiler::TraceMe traceme("DeviceMemAllocator::Free");

    if (ptr != nullptr) {
      VisitFree(ptr, device_id_.value(), num_bytes);
      if (memory_type_ == MemoryType::kUnified) {
        stream_exec_->UnifiedMemoryDeallocate(ptr);
      } else if (memory_type_ == MemoryType::kCollective) {
        auto status = stream_exec_->CollectiveMemoryDeallocate(ptr);
        CHECK(status.ok()) << status.message();
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
  const MemoryType memory_type_ = MemoryType::kDevice;

  DeviceMemAllocator(const DeviceMemAllocator&) = delete;
  void operator=(const DeviceMemAllocator&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_MEM_ALLOCATOR_H_
