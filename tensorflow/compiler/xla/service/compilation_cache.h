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

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {

// A cache which stores Executables indexed by computation handle and version.
class CompilationCache {
 public:
  CompilationCache() {}

  // Insert the given Executable into the cache. Return a bare Executable
  // pointer for the caller to use. Note: the returned pointer will *not* be the
  // same as the given unique pointer if the computation already exists in the
  // cache. See comments in the .cc implementation for details of this case.
  //
  // module_config is provided by the caller, instead of being taken from the
  // executable, so that we can insert keys into the compilation cache that are
  // devoid of layout (where XLA gets to choose what layout to compile).
  //
  // A shared_ptr is returned so the caller can keep the Executable from being
  // destructed in the event that the Executable is evicted from the
  // computation cache (and the cache's shared_ptr to the Executable is
  // destructed).
  std::shared_ptr<Executable> Insert(std::unique_ptr<Executable> executable,
                                     const HloModuleConfig& module_config);

  // Lookup the Executable for the specified versioned computation in the cache.
  // Return a shared_ptr to the Executable if it exists in the cache. Return
  // nullptr otherwise.
  std::shared_ptr<Executable> LookUp(
      const VersionedComputationHandle& versioned_handle,
      const HloModuleConfig& module_config) const;

 protected:
  mutable tensorflow::mutex mutex_;

  // Map from versioned handle with program layout to Executable built
  // for that computation version and program layout.
  using CacheKey = string;

  CacheKey BuildKey(const VersionedComputationHandle& versioned_handle,
                    const HloModuleConfig& module_config) const;
  std::map<CacheKey, std::shared_ptr<Executable>> cache_ GUARDED_BY(mutex_);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CompilationCache);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_CACHE_H_
