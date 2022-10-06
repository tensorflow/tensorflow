/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/compilation_cache.h"

#include <utility>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/strcat.h"

namespace xla {

namespace {

int64_t GetUniqueId() {
  static absl::Mutex mu(absl::kConstInit);
  static int64_t counter = 0;
  absl::MutexLock loc(&mu);
  const int64_t id = counter++;
  return id;
}

}  // namespace

ExecutionHandle CompilationCache::Insert(
    std::unique_ptr<Executable> executable) {
  absl::MutexLock lock(&mutex_);

  CacheKey key = GetUniqueId();
  VLOG(2) << "inserting cache key: " << key;
  CHECK_EQ(cache_.count(key), 0);
  cache_.emplace(key, std::move(executable));

  ExecutionHandle handle;
  handle.set_handle(key);
  return handle;
}

StatusOr<std::shared_ptr<Executable>> CompilationCache::LookUp(
    const ExecutionHandle& handle) const {
  absl::MutexLock lock(&mutex_);

  CacheKey key = handle.handle();
  VLOG(2) << "looking up cache key: " << key;
  if (cache_.count(key) == 0) {
    VLOG(2) << "cache key not found: " << key;
    return InvalidArgumentStrCat("can not find executable with handle ", key);
  } else {
    auto& result = cache_.at(key);
    VLOG(2) << "hit executable: " << result->module().name();
    return result;
  }
}

}  // namespace xla
