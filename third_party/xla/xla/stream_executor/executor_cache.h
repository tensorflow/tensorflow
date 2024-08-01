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

#ifndef XLA_STREAM_EXECUTOR_EXECUTOR_CACHE_H_
#define XLA_STREAM_EXECUTOR_EXECUTOR_CACHE_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor {

// Forward declare.
class StreamExecutor;

// Utility class to allow Platform objects to manage cached StreamExecutors.
// Thread-safe.
class ExecutorCache {
 public:
  using ExecutorFactory =
      std::function<absl::StatusOr<std::unique_ptr<StreamExecutor>>()>;

  ExecutorCache();
  ~ExecutorCache();

  // Looks up 'config' in the cache. Returns a pointer to the existing executor,
  // if already present, or creates it using 'factory', if it does not.
  // Factories may be executed concurrently for different device ordinals.
  absl::StatusOr<StreamExecutor*> GetOrCreate(
      const StreamExecutorConfig& config, const ExecutorFactory& factory);

  // Returns a pointer to the described executor (if one with a matching config
  // has been created), or a NOT_FOUND status.
  absl::StatusOr<StreamExecutor*> Get(const StreamExecutorConfig& config);

  // Destroys all Executors and clears the cache.
  // Performs no synchronization with the executors - undefined behavior may
  // occur if any executors are active!
  void DestroyAllExecutors();

 private:
  // Each Entry contains zero or more cached executors for a device ordinal.
  struct Entry {
    ~Entry();

    // Mutex that guards the contents of each entry. The 'mutex_' of the
    // ExecutorCache class protects both the 'cache_' and the existence of each
    // Entry, but not the Entry's contents. 'configurations_mutex' protects the
    // contents of the entry after 'mutex_' has been dropped.
    absl::Mutex configurations_mutex;

    // Vector of cached {config, executor} pairs.
    std::vector<
        std::pair<StreamExecutorConfig, std::unique_ptr<StreamExecutor>>>
        configurations ABSL_GUARDED_BY(configurations_mutex);
  };

  // Maps ordinal and stream_id to a list of cached executors for that ordinal
  // and stream_id. We key off of the ordinal-stream pair (instead of just
  // looking up all fields in the StreamExecutorConfig) for a slight improvement
  // in lookup time.
  absl::Mutex mutex_;
  absl::node_hash_map<std::pair<int, int>, Entry> cache_
      ABSL_GUARDED_BY(mutex_);

  ExecutorCache(const ExecutorCache&) = delete;
  void operator=(const ExecutorCache&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_EXECUTOR_CACHE_H_
