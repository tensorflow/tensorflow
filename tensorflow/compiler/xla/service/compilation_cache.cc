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
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

std::shared_ptr<Executable> CompilationCache::Insert(
    std::unique_ptr<Executable> executable,
    const HloModuleConfig& module_config) {
  tensorflow::mutex_lock lock(mutex_);

  CacheKey key =
      BuildKey(executable->entry_computation_handle(), module_config);
  VLOG(2) << "inserting cache key: " << key;
  if (cache_.count(key) == 0) {
    cache_.emplace(key, std::move(executable));
  } else {
    // Executable already exists in the cache. This can happen if two Execute
    // calls for a new computation are received simultaneously by the
    // service. In this case, we discard the Executable given as a parameter and
    // return what is in the cache. This is necessary because the service relies
    // on the cache to keep ownership of the Executable. We only want to store
    // one Executable for a given computation version and we can't discard the
    // executable which is in the cache because it may be in use.
    executable.reset();
  }
  return cache_.at(key);
}

std::shared_ptr<Executable> CompilationCache::LookUp(
    const VersionedComputationHandle& versioned_handle,
    const HloModuleConfig& module_config) const {
  tensorflow::mutex_lock lock(mutex_);

  CacheKey key = BuildKey(versioned_handle, module_config);
  VLOG(2) << "looking up cache key: " << key;
  if (cache_.count(key) == 0) {
    VLOG(2) << "cache key not found: " << key;
    return nullptr;
  } else {
    std::shared_ptr<Executable> result = cache_.at(key);
    VLOG(2) << "hit executable with module config: "
            << result->module_config().compilation_cache_key();
    return result;
  }
}

CompilationCache::CacheKey CompilationCache::BuildKey(
    const VersionedComputationHandle& versioned_handle,
    const HloModuleConfig& module_config) const {
  // The computation shape is represented entirely by its ProgramShape member,
  // so just serialize the proto as part of the key.
  return tensorflow::strings::StrCat(versioned_handle.handle.handle(), "::",
                                     versioned_handle.version, "::",
                                     module_config.compilation_cache_key());
}

}  // namespace xla
