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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_CACHE_H_

#include <map>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

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
  StatusOr<std::shared_ptr<Executable>> LookUp(
      const ExecutionHandle& handle) const;

 protected:
  mutable tensorflow::mutex mutex_;

  using CacheKey = int64;

  absl::flat_hash_map<CacheKey, std::shared_ptr<Executable>> cache_
      GUARDED_BY(mutex_);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CompilationCache);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_CACHE_H_
