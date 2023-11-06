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

#ifndef XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_HOST_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_HOST_ALLOCATOR_H_

#include <vector>

#include "xla/stream_executor/stream_executor.h"
#include "tsl/framework/allocator.h"
#include "tsl/profiler/lib/traceme.h"

namespace stream_executor {
// Allocator for pinned CPU RAM that is made known to a StreamExecutor-based
// device for the purpose of efficient DMA with the device.
class DeviceHostAllocator : public tsl::SubAllocator {
 public:
  // Note: stream_exec cannot be null.
  explicit DeviceHostAllocator(StreamExecutor* stream_exec, int numa_node,
                               const std::vector<Visitor>& alloc_visitors,
                               const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors),
        stream_exec_(stream_exec),
        numa_node_(numa_node) {
    CHECK(stream_exec_ != nullptr);
  }

  ~DeviceHostAllocator() override = default;

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    tsl::profiler::TraceMe traceme("DeviceHostAllocator::Alloc");

    void* ptr = nullptr;
    *bytes_received = num_bytes;
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
    tsl::profiler::TraceMe traceme("DeviceHostAllocator::Free");

    if (ptr != nullptr) {
      VisitFree(ptr, numa_node_, num_bytes);
      stream_exec_->HostMemoryDeallocate(ptr);
    }
  }

  bool SupportsCoalescing() const override { return false; }

  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kHostPinned;
  }

 private:
  StreamExecutor* stream_exec_;  // not owned, non-null
  const int numa_node_;

  DeviceHostAllocator(const DeviceHostAllocator&) = delete;
  void operator=(const DeviceHostAllocator&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_INTEGRATIONS_DEVICE_HOST_ALLOCATOR_H_
