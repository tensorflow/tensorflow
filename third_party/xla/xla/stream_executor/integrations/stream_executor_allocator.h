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

#ifndef XLA_STREAM_EXECUTOR_INTEGRATIONS_STREAM_EXECUTOR_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_INTEGRATIONS_STREAM_EXECUTOR_ALLOCATOR_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"

namespace stream_executor {

// Implements a tsl::SubAllocator interface for StreamExecutor-based devices.
class StreamExecutorAllocator : public tsl::SubAllocator {
 public:
  StreamExecutorAllocator(std::unique_ptr<MemoryAllocator> memory_allocator,
                          MemoryType memory_type, int index,
                          const std::vector<Visitor>& alloc_visitors = {},
                          const std::vector<Visitor>& free_visitors = {});

  ~StreamExecutorAllocator() override = default;
  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override;
  void Free(void* ptr, size_t num_bytes) override;
  bool SupportsCoalescing() const override;
  tsl::AllocatorMemoryType GetMemoryType() const override;

 private:
  std::unique_ptr<MemoryAllocator> memory_allocator_;
  MemoryType memory_type_;
  int index_;

  StreamExecutorAllocator(const StreamExecutorAllocator&) = delete;
  void operator=(const StreamExecutorAllocator&) = delete;

  absl::Mutex mutex_;
  absl::flat_hash_map<void*, std::unique_ptr<MemoryAllocation>> allocations_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_INTEGRATIONS_STREAM_EXECUTOR_ALLOCATOR_H_
