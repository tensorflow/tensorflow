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

#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {

port::StatusOr<StreamExecutor*> ExecutorCache::GetOrCreate(
    const StreamExecutorConfig& config,
    const std::function<ExecutorFactory>& factory) {
  Entry* entry = nullptr;
  {
    mutex_lock lock{mutex_};
    entry = &cache_[config.ordinal];
    // Release the map lock; the address of 'entry' is stable because
    // std::map guarantees reference stability.
  }

  // Acquire the per-Entry mutex without holding the map mutex. Initializing
  // an Executor may be expensive, so we want to allow concurrent
  // initialization of different entries.
  mutex_lock lock{entry->configurations_mutex};
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
    mutex_lock lock{mutex_};
    entry = &cache_[config.ordinal];
    // Release the map lock; the address of 'entry' is stable because
    // std::map guarantees reference stability.
  }
  mutex_lock lock{entry->configurations_mutex};
  if (entry->configurations.empty()) {
    return port::Status(
        port::error::NOT_FOUND,
        port::Printf("No executors registered for ordinal %d", config.ordinal));
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
  mutex_lock lock{mutex_};
  cache_.clear();
}

ExecutorCache::Entry::~Entry() {
  mutex_lock lock{configurations_mutex};
  configurations.clear();
}

}  // namespace gputools
}  // namespace perftools
