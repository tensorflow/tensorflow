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

#ifndef TENSORFLOW_STREAM_EXECUTOR_EXECUTOR_CACHE_H_
#define TENSORFLOW_STREAM_EXECUTOR_EXECUTOR_CACHE_H_

#include <functional>
#include <map>

#include "absl/synchronization/mutex.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {

// Utility class to allow Platform objects to manage cached StreamExecutors.
// Thread-safe.
class ExecutorCache {
 public:
  ExecutorCache() {}

  // Looks up 'config' in the cache. Returns a pointer to the existing executor,
  // if already present, or creates it using 'factory', if it does not.
  // Factories may be executed concurrently for different device ordinals.
  typedef port::StatusOr<std::unique_ptr<StreamExecutor>> ExecutorFactory();
  port::StatusOr<StreamExecutor*> GetOrCreate(
      const StreamExecutorConfig& config,
      const std::function<ExecutorFactory>& factory);

  // Returns a pointer to the described executor (if one with a matching config
  // has been created), or a NOT_FOUND status.
  port::StatusOr<StreamExecutor*> Get(const StreamExecutorConfig& config);

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
        configurations GUARDED_BY(configurations_mutex);
  };

  // Maps ordinal number to a list of cached executors for that ordinal.
  // We key off of ordinal (instead of just looking up all fields in the
  // StreamExecutorConfig) for a slight improvement in lookup time.
  absl::Mutex mutex_;
  std::map<int, Entry> cache_ GUARDED_BY(mutex_);

  SE_DISALLOW_COPY_AND_ASSIGN(ExecutorCache);
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_EXECUTOR_CACHE_H_
