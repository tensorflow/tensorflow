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
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

ExecutorCache::ExecutorCache() = default;
ExecutorCache::~ExecutorCache() = default;

absl::StatusOr<StreamExecutor*> ExecutorCache::GetOrCreate(
    int ordinal, const ExecutorFactory& factory) {
  // In the fast path case, the cache already has an entry and we can just
  // return after Get() which only takes a shared lock and not a unique lock.
  // If we need to create, we take a unique lock on cache_.
  if (auto fast_result = Get(ordinal); fast_result.ok()) {
    return fast_result;
  }

  VLOG(2) << "building executor";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<StreamExecutor> result, factory());
  auto returned_executor = result.get();
  absl::MutexLock lock(&mutex_);
  cache_.emplace(ordinal, std::move(result));
  return returned_executor;
}

absl::StatusOr<StreamExecutor*> ExecutorCache::Get(int ordinal) {
  absl::ReaderMutexLock lock{&mutex_};

  if (auto it = cache_.find(ordinal); it != cache_.end()) {
    return it->second.get();
  }

  return absl::NotFoundError(
      absl::StrFormat("No executors registered for ordinal %d", ordinal));
}

}  // namespace stream_executor
