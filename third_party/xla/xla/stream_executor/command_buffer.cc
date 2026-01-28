/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/command_buffer.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"

namespace stream_executor {

CommandBuffer::ResourceTypeId CommandBuffer::GetNextResourceTypeId() {
  absl::NoDestructor<std::atomic<int64_t>> counter(1);
  return ResourceTypeId(counter->fetch_add(1));
}

CommandBuffer::Resource* CommandBuffer::GetOrNullResource(
    ResourceTypeId type_id) {
  absl::MutexLock lock(resource_mutex_);
  auto it = resources_.find(type_id);
  return (it != resources_.end()) ? it->second.get() : nullptr;
}

CommandBuffer::Resource* CommandBuffer::GetOrCreateResource(
    ResourceTypeId type_id,
    absl::FunctionRef<std::unique_ptr<Resource>()> create) {
  // First, try to find the resource under lock
  {
    absl::MutexLock lock(resource_mutex_);
    auto it = resources_.find(type_id);
    if (ABSL_PREDICT_TRUE(it != resources_.end())) {
      return it->second.get();
    }
  }

  // Resource not found, create it outside the lock
  auto resource = create();
  Resource* ptr = resource.get();

  // Acquire lock again to insert the new resource
  {
    absl::MutexLock lock(resource_mutex_);
    auto it = resources_.find(type_id);
    if (ABSL_PREDICT_TRUE(it == resources_.end())) {
      // We won the race — insert our resource
      resources_.emplace(type_id, std::move(resource));
    } else {
      // Another thread inserted it in the meantime
      ptr = it->second.get();
    }
  }

  return ptr;
}

}  // namespace stream_executor
