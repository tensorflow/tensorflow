/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/stream_executor.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"

namespace stream_executor {

// A structure to hold resources for a specific executor instance.
struct StreamExecutor::ResourceStorage {
  absl::Mutex mu;
  absl::flat_hash_map<StreamExecutor::ResourceTypeId,
                      std::unique_ptr<StreamExecutor::Resource>>
      map;
};

StreamExecutor::StreamExecutor()
    : resources_(std::make_unique<ResourceStorage>()) {}

StreamExecutor::~StreamExecutor() = default;

StreamExecutor::ResourceTypeId StreamExecutor::GetNextResourceTypeId() {
  static std::atomic<int64_t> counter(1);
  return ResourceTypeId(counter.fetch_add(1));
}

StreamExecutor::Resource* StreamExecutor::GetOrNullResource(
    ResourceTypeId type_id) {
  if (!resources_) {
    return nullptr;
  }
  absl::MutexLock lock(resources_->mu);
  auto it = resources_->map.find(type_id);
  return (it != resources_->map.end()) ? it->second.get() : nullptr;
}

StreamExecutor::Resource* StreamExecutor::GetOrCreateResource(
    ResourceTypeId type_id,
    absl::FunctionRef<std::unique_ptr<Resource>()> create) {
  // 1. Fast path: try to find the resource under lock
  {
    absl::MutexLock lock(resources_->mu);
    auto it = resources_->map.find(type_id);
    if (ABSL_PREDICT_TRUE(it != resources_->map.end())) {
      return it->second.get();
    }
  }

  // 2. Resource not found, create it outside the lock
  auto new_resource = create();
  Resource* ptr = new_resource.get();

  // 3. Insertion with double-check
  {
    absl::MutexLock lock(resources_->mu);
    auto [it, inserted] =
        resources_->map.try_emplace(type_id, std::move(new_resource));
    if (!inserted) {
      ptr = it->second.get();
    }
  }
  return ptr;
}

}  // namespace stream_executor
