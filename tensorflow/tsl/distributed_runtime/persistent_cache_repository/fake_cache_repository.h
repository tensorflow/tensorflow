/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// In-memory implementation of the persistent cache repository interface. Use
// for testing caching solutions.

#ifndef TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_FAKE_CACHE_REPOSITORY_H_
#define TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_FAKE_CACHE_REPOSITORY_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/tsl/distributed_runtime/persistent_cache_repository/persistent_cache_repository.h"

namespace tsl {

// In-memory implementation of the persistent cache repository for testing
// caching solutions. This implementation does not persist any entries;
// entries survive only as long as the instance is alive. The value of
// Options.reset_ttl_on_access has no effect.
class FakeCacheRepository : public PersistentCacheRepository {
 public:
  explicit FakeCacheRepository(Options options)
      : PersistentCacheRepository(std::move(options)) {}
  ~FakeCacheRepository() override = default;

  absl::Status Put(const std::string& key,
                   const std::string& serialized_entry) override {
    absl::MutexLock lock(&mu_);
    const auto it = repository_.find(key);
    if (it == repository_.end()) {
      repository_[key] = serialized_entry;
      return absl::OkStatus();
    }

    if (it->second != serialized_entry) {
      return absl::AlreadyExistsError("Put: operation not idempotent");
    }

    return absl::OkStatus();
  }

  absl::StatusOr<std::string> Get(const std::string& key) override {
    absl::MutexLock lock(&mu_);
    const auto it = repository_.find(key);
    if (it == repository_.end()) {
      return absl::NotFoundError("Get: key not found");
    } else {
      return it->second;
    }
  }

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::string> repository_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_FAKE_CACHE_REPOSITORY_H_
