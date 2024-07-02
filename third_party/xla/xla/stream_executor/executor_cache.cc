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

#include "xla/stream_executor/executor_cache.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"

namespace stream_executor {

ExecutorCache::ExecutorCache() = default;
ExecutorCache::~ExecutorCache() { DestroyAllExecutors(); }

absl::StatusOr<StreamExecutor*> ExecutorCache::GetOrCreate(
    const StreamExecutorConfig& config, const ExecutorFactory& factory) {
  // In the fast path case, the cache already has an entry and we can just
  // return after Get() which only takes a shared lock and not a unique lock.
  // If we need to create, we take a unique lock on cache_.
  if (auto fast_result = Get(config); fast_result.ok()) {
    return fast_result;
  }

  Entry* entry = nullptr;
  {
    absl::MutexLock lock{&mutex_};
    entry = &cache_[{config.ordinal, config.stream_id}];
    // Release the map lock; the address of 'entry' is stable because
    // absl::node_hash_map guarantees reference stability.
  }

  // Acquire the per-Entry mutex without holding the map mutex. Initializing
  // an Executor may be expensive, so we want to allow concurrent
  // initialization of different entries.
  absl::MutexLock lock{&entry->configurations_mutex};
  for (const auto& iter : entry->configurations) {
    VLOG(2) << "hit in cache";
    return iter.second.get();
  }

  VLOG(2) << "building executor";
  absl::StatusOr<std::unique_ptr<StreamExecutor>> result = factory();
  if (!result.ok()) {
    VLOG(2) << "failed to get build executor: " << result.status();
    // If construction failed, leave the cache Entry around, but with a null
    // executor.
    return result.status();
  }
  entry->configurations.emplace_back(config, std::move(result.value()));
  return entry->configurations.back().second.get();
}

absl::StatusOr<StreamExecutor*> ExecutorCache::Get(
    const StreamExecutorConfig& config) {
  Entry* entry = nullptr;
  {
    absl::ReaderMutexLock lock{&mutex_};

    // If gpu stream is not nullptr we have to find StreamExecutor that owns it,
    // and return NOT_FOUND error if we can't find it.
    if (config.gpu_stream) {
      for (auto& [ordinal, e] : cache_) {
        absl::ReaderMutexLock l{&e.configurations_mutex};
        for (auto& [c, executor] : e.configurations) {
          if (executor->FindAllocatedStream(config.gpu_stream)) {
            return executor.get();
          }
        }
      }
      return absl::NotFoundError(
          absl::StrFormat("No executors own stream %p", config.gpu_stream));
    }

    if (auto it = cache_.find({config.ordinal, config.stream_id});
        it != cache_.end()) {
      entry = &it->second;
    } else {
      return absl::NotFoundError(absl::StrFormat(
          "No executors registered for ordinal %d, stream group %d",
          config.ordinal, config.stream_id));
    }
  }

  absl::ReaderMutexLock lock{&entry->configurations_mutex};
  if (entry->configurations.empty()) {
    return absl::NotFoundError(absl::StrFormat(
        "No executors registered for ordinal %d, stream group %d",
        config.ordinal, config.stream_id));
  }

  for (auto& [entry_config, entry_executor] : entry->configurations) {
    return entry_executor.get();
  }

  return absl::NotFoundError("No executor found with a matching config.");
}

void ExecutorCache::DestroyAllExecutors() {
  absl::MutexLock lock{&mutex_};
  cache_.clear();
}

ExecutorCache::Entry::~Entry() {
  absl::MutexLock lock{&configurations_mutex};
  configurations.clear();
}

}  // namespace stream_executor
