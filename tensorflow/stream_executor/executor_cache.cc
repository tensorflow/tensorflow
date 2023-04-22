/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/executor_cache.h"

#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"

namespace stream_executor {

port::StatusOr<StreamExecutor*> ExecutorCache::GetOrCreate(
    const StreamExecutorConfig& config,
    const std::function<ExecutorFactory>& factory) {
  // In the fast path case, the cache already has an entry and we can just
  // return after Get() which only takes a shared lock and not a unique lock.
  // If we need to create, we take a unique lock on cache_.
  auto fast_result = Get(config);
  if (fast_result.ok()) {
    return fast_result;
  }

  Entry* entry = nullptr;
  {
    absl::MutexLock lock{&mutex_};
    entry = &cache_[config.ordinal];
    // Release the map lock; the address of 'entry' is stable because
    // std::map guarantees reference stability.
  }

  // Acquire the per-Entry mutex without holding the map mutex. Initializing
  // an Executor may be expensive, so we want to allow concurrent
  // initialization of different entries.
  absl::MutexLock lock{&entry->configurations_mutex};
  for (const auto& iter : entry->configurations) {
    if (iter.first.plugin_config == config.plugin_config &&
        iter.first.device_options == config.device_options) {
      VLOG(2) << "hit in cache";
      return iter.second.get();
    }
  }

  VLOG(2) << "building executor";
  port::StatusOr<std::unique_ptr<StreamExecutor>> result = factory();
  if (!result.ok()) {
    VLOG(2) << "failed to get build executor: " << result.status();
    // If construction failed, leave the cache Entry around, but with a null
    // executor.
    return result.status();
  }
  entry->configurations.emplace_back(config, std::move(result.ValueOrDie()));
  return entry->configurations.back().second.get();
}

port::StatusOr<StreamExecutor*> ExecutorCache::Get(
    const StreamExecutorConfig& config) {
  Entry* entry = nullptr;
  {
    absl::ReaderMutexLock lock{&mutex_};
    auto it = cache_.find(config.ordinal);
    if (it != cache_.end()) {
      entry = &it->second;
    } else {
      return port::Status(
          port::error::NOT_FOUND,
          absl::StrFormat("No executors registered for ordinal %d",
                          config.ordinal));
    }
  }
  absl::ReaderMutexLock lock{&entry->configurations_mutex};
  if (entry->configurations.empty()) {
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrFormat("No executors registered for ordinal %d",
                        config.ordinal));
  }
  for (const auto& iter : entry->configurations) {
    if (iter.first.plugin_config == config.plugin_config &&
        iter.first.device_options == config.device_options) {
      VLOG(2) << "hit in cache for device ordinal " << config.ordinal;
      return iter.second.get();
    }
  }
  return port::Status(port::error::NOT_FOUND,
                      "No executor found with a matching config.");
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
