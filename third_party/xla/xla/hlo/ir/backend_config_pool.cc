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

#include "xla/hlo/ir/backend_config_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "re2/re2.h"

namespace xla {

namespace {
std::string NormalizeRawBackendConfigString(std::string s) {
  static constexpr LazyRE2 kReWaitOnOperationQueues = {
      R"("wait_on_operation_queues"\s*:\s*\[\s*\]\s*,)"};
  RE2::GlobalReplace(&s, *kReWaitOnOperationQueues, "");
  return s;
}
}  // namespace
BackendConfigPool* BackendConfigPool::Get() {
  static absl::NoDestructor<BackendConfigPool> pool;
  return &*pool;
}

std::shared_ptr<const std::string> BackendConfigPool::Intern(
    absl::string_view json) {
  std::string normalized_json =
      NormalizeRawBackendConfigString(std::string(json));
  uint64_t hash = absl::HashOf(normalized_json);

  absl::MutexLock lock(mutex_);

  // Periodically garbage collect if the map grows large.
  static int call_count = 0;
  static constexpr size_t kGcThreshold = 10000;
  if (++call_count % 1000 == 0 && registry_.size() > kGcThreshold) {
    GarbageCollectLocked();
  }

  auto& bucket = registry_[hash];

  // Check existing entries and do lazy GC.
  for (auto it = bucket.begin(); it != bucket.end();) {
    if (auto shared = it->lock()) {
      if (*shared == normalized_json) {
        return shared;  // Hit
      }
      ++it;
    } else {
      it = bucket.erase(it);  // Lazy GC
    }
  }

  // Miss — create new entry.
  // TODO(b/510802287): Consider moving allocation outside the lock to reduce
  // contention, using double-checked locking with ReaderMutexLock.
  auto shared = std::make_shared<const std::string>(std::move(normalized_json));
  bucket.push_back(shared);
  return shared;
}

size_t BackendConfigPool::GarbageCollect() {
  absl::MutexLock lock(mutex_);
  return GarbageCollectLocked();
}

size_t BackendConfigPool::GarbageCollectLocked() {
  size_t total_erased = 0;
  // NOLINTNEXTLINE
  for (auto it = registry_.begin(); it != registry_.end();) {
    auto& bucket = it->second;
    auto remove_it =
        std::remove_if(bucket.begin(), bucket.end(),
                       [](const auto& ptr) { return ptr.expired(); });
    total_erased += std::distance(remove_it, bucket.end());
    bucket.erase(remove_it, bucket.end());
    if (bucket.empty()) {
      registry_.erase(it++);
    } else {
      ++it;
    }
  }

  VLOG(3) << "Garbage collected " << total_erased << " entries";
  return total_erased;
}

void BackendConfigPool::ResetForTesting() {
  absl::MutexLock lock(mutex_);
  registry_.clear();
}

}  // namespace xla
