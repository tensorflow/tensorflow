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

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Utility class to allow Platform objects to manage cached StreamExecutors.
// Thread-safe.
class ExecutorCache {
 public:
  using ExecutorFactory =
      std::function<absl::StatusOr<std::unique_ptr<StreamExecutor>>()>;

  ExecutorCache();
  ~ExecutorCache();

  // Looks up 'ordinal' in the cache. Returns a pointer to the existing
  // executor, if already present, or creates it using 'factory', if it does
  // not. Factories may be executed concurrently for different device ordinals.
  absl::StatusOr<StreamExecutor*> GetOrCreate(int ordinal,
                                              const ExecutorFactory& factory);

  // Returns a pointer to the described executor (if one with a matching ordinal
  // has been created), or a NOT_FOUND status.
  absl::StatusOr<StreamExecutor*> Get(int ordinal);

 private:
  // Protects cache_.
  absl::Mutex mutex_;

  // Maps ordinal number to a cached executor for that ordinal.
  absl::flat_hash_map<int, std::unique_ptr<StreamExecutor>> cache_
      ABSL_GUARDED_BY(mutex_);

  ExecutorCache(const ExecutorCache&) = delete;
  void operator=(const ExecutorCache&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_EXECUTOR_CACHE_H_
