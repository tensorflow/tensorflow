/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/host_memory_pool.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::unique_ptr<HostMemoryPool>> HostMemoryPool::Create(
    se::StreamExecutor* executor, PrimitiveType type) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> allocation,
                      executor->HostMemoryAllocate(
                          kNumElems * primitive_util::ByteWidth(type)));
  return absl::WrapUnique(new HostMemoryPool(std::move(allocation), type));
}

absl::StatusOr<HostMemoryPool::Handle> HostMemoryPool::Acquire() {
  absl::MutexLock lock(&mutex_);
  if (free_list_.empty()) {
    return absl::ResourceExhaustedError(
        absl::StrCat("All ", kNumElems,
                     " elements in the host memory pool are in use. This is "
                     "likely because there are more than ",
                     kNumElems, " concurrent calls to an XLA executable."));
  }
  void* ptr = free_list_.front();
  free_list_.pop();
  return Handle(this, ptr);
}

HostMemoryPool::Handle::Handle(Handle&& other)
    : pool_(other.pool_), ptr_(other.ptr_) {
  other.ptr_ = nullptr;
}
HostMemoryPool::Handle& HostMemoryPool::Handle::operator=(Handle&& other) {
  if (this != &other) {
    if (ptr_) {
      pool_->Release(ptr_);
    }
    pool_ = other.pool_;
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }
  return *this;
}

HostMemoryPool::Handle::~Handle() {
  if (ptr_) {
    pool_->Release(ptr_);
  }
}
HostMemoryPool::HostMemoryPool(std::unique_ptr<se::MemoryAllocation> allocation,
                               PrimitiveType type)
    : allocation_(std::move(allocation)), type_(type) {
  for (int i = 0; i < kNumElems; ++i) {
    free_list_.push(static_cast<char*>(allocation_->opaque()) +
                    i * primitive_util::ByteWidth(type_));
  }
}
}  // namespace gpu
}  // namespace xla
