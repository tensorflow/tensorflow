/* Copyright 2018 The OpenXLA Authors.

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
#include "xla/stream_executor/integrations/stream_executor_allocator.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "tsl/profiler/lib/traceme.h"

namespace stream_executor {

StreamExecutorAllocator::StreamExecutorAllocator(
    std::unique_ptr<MemoryAllocator> memory_allocator, MemoryType memory_type,
    int index, const std::vector<Visitor>& alloc_visitors,
    const std::vector<Visitor>& free_visitors)
    : tsl::SubAllocator(alloc_visitors, free_visitors),
      memory_allocator_(std::move(memory_allocator)),
      memory_type_(memory_type),
      index_(index) {}

// Maps MemoryType to human-readable strings for allocation error messages
const auto& kMemoryTypeStrings =
    *new absl::flat_hash_map<MemoryType, std::string>{
        {MemoryType::kDevice, "device"},
        {MemoryType::kUnified, "unified"},
        {MemoryType::kHost, "pinned host"},
        {MemoryType::kCollective, "collective"}};

void* StreamExecutorAllocator::Alloc(size_t alignment, size_t num_bytes,
                                     size_t* bytes_received) {
  tsl::profiler::TraceMe traceme("StreamExecutorAllocator::Alloc");

  void* ptr = nullptr;

  if (num_bytes > 0) {
    auto allocation = memory_allocator_->Allocate(num_bytes);
    const auto memory_type_iter = kMemoryTypeStrings.find(memory_type_);
    if (!allocation.ok()) {
      LOG(WARNING) << "could not allocate "
                   << (memory_type_iter == kMemoryTypeStrings.end()
                           ? "unknown"
                           : memory_type_iter->second)
                   << " of size: " << num_bytes;
      *bytes_received = 0;
      return nullptr;
    }

    ptr = (*allocation)->opaque();
    VisitAlloc(ptr, index_, num_bytes);

    absl::MutexLock lock(&mutex_);
    allocations_[ptr] = std::move(*allocation);
  }

  *bytes_received = num_bytes;

  return ptr;
}

void StreamExecutorAllocator::Free(void* ptr, size_t num_bytes) {
  tsl::profiler::TraceMe traceme("StreamExecutorAllocator::Free");

  if (ptr != nullptr) {
    VisitFree(ptr, index_, num_bytes);
    absl::MutexLock lock(&mutex_);
    allocations_.erase(ptr);
  }
}

bool StreamExecutorAllocator::SupportsCoalescing() const { return false; }

tsl::AllocatorMemoryType StreamExecutorAllocator::GetMemoryType() const {
  if (memory_type_ == MemoryType::kHost) {
    return tsl::AllocatorMemoryType::kHostPinned;
  } else {
    return tsl::AllocatorMemoryType::kDevice;
  }
}

}  // namespace stream_executor
