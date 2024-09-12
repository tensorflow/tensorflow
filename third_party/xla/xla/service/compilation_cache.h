/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COMPILATION_CACHE_H_
#define XLA_SERVICE_COMPILATION_CACHE_H_

#include <map>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/types.h"

namespace xla {

// A cache which stores Executables indexed by computation handle and version.
//
// TODO(b/119042872): Provide mechanism for removing computations from the
// compilation cache.
class CompilationCache {
 public:
  CompilationCache() {}

  ExecutionHandle Insert(std::unique_ptr<Executable> executable);

  // Lookup the Executable for the specified handle in the cache. Return a
  // shared_ptr to the Executable if it exists in the cache.
  absl::StatusOr<std::shared_ptr<Executable>> LookUp(
      const ExecutionHandle& handle) const;

 protected:
  mutable absl::Mutex mutex_;

  using CacheKey = int64_t;

  absl::flat_hash_map<CacheKey, std::shared_ptr<Executable>> cache_
      ABSL_GUARDED_BY(mutex_);

 private:
  CompilationCache(const CompilationCache&) = delete;
  CompilationCache& operator=(const CompilationCache&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_COMPILATION_CACHE_H_
