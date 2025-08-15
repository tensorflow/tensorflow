
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

#include "xla/stream_executor/stream.h"

#include <atomic>
#include <cstdint>
#include <memory>

#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"

namespace stream_executor {

Stream::ResourceTypeId Stream::GetNextResourceTypeId() {
  absl::NoDestructor<std::atomic<int64_t>> counter(1);
  return ResourceTypeId(counter->fetch_add(1));
}

Stream::Resource* Stream::GetOrNullResource(ResourceTypeId type_id) {
  absl::MutexLock lock(&resource_mutex_);
  if (auto it = resources_.find(type_id); it != resources_.end()) {
    return it->second.get();
  }
  return nullptr;
}

Stream::Resource* Stream::GetOrCreateResource(
    ResourceTypeId type_id,
    absl::FunctionRef<std::unique_ptr<Resource>()> create) {
  absl::MutexLock lock(&resource_mutex_);
  auto [it, inserted] = resources_.try_emplace(type_id, create());
  if (ABSL_PREDICT_FALSE(inserted)) {
    it->second = create();
  }
  return it->second.get();
}

}  // namespace stream_executor
