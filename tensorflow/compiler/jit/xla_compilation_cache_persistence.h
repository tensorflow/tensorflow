/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_PERSISTENCE_H_
#define TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_PERSISTENCE_H_

#include <functional>
#include <memory>
#include <string>

#include "tensorflow/compiler/jit/xla_compilation_cache.pb.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace tensorflow {

class XlaCompilationCacheSaver;
class XlaCompilationCacheLoader;

using XlaCompilationCacheSaverCreatorFn =
    std::function<std::unique_ptr<XlaCompilationCacheSaver>()>;

using XlaCompilationCacheLoaderCreatorFn =
    std::function<std::unique_ptr<XlaCompilationCacheLoader>()>;

// Registers a function to create a XLA compilation cache saver object. The
// creator function will be called when a new XLA compliation cache resource in
// allocated.
Status RegisterXlaCompilationCacheSaver(
    XlaCompilationCacheSaverCreatorFn&& creator);

// Registers a function to create a XLA compilation cache loader object. The
// creator function will be called when a new XLA compliation cache resource in
// allocated.
Status RegisterXlaCompilationCacheLoader(
    XlaCompilationCacheLoaderCreatorFn&& creator);

// Unregisteres the creator function previously registered with
// `RegisterXlaCompilationCacheSaver`
void UnregisterXlaCompilationCacheSaver();

// Unregisters the creator function previously registered with
// `RegisterXlaCompilationCacheLoader`
void UnregisterXlaCompilationCacheLoader();

// Calls the creator function registered with
// `RegisterXlaCompilationCacheSaver`. If no function is registered a nullptr is
// returned.
std::unique_ptr<XlaCompilationCacheSaver> CreateXlaCompilationCacheSaver();

// Calls the creator function registered with
// `RegisterXlaCompilationCacheLoader`. If no function is registered a nullptr
// is returned.
std::unique_ptr<XlaCompilationCacheLoader> CreateXlaCompilationCacheLoader();

// Base class for XLA compliation cache savers.
class XlaCompilationCacheSaver {
 public:
  virtual ~XlaCompilationCacheSaver() = default;

  // Saves the cache entry.
  virtual Status Save(const XlaSerializedCacheEntry& entry) = 0;
};

// Base class for XLA compilation cache loaders.
class XlaCompilationCacheLoader {
 public:
  virtual ~XlaCompilationCacheLoader() = default;

  // Try to load a cache entry given a `key`.
  virtual StatusOr<absl::optional<XlaSerializedCacheEntry>> TryLoad(
      const XlaSerializedCacheKey& key) = 0;
};

// Saves XLA compilation cache entries to files.
class XlaCompilationCacheFileSaver : public XlaCompilationCacheSaver {
 public:
  enum Mode { kBINARY = 0, kTEXT = 1 };
  ~XlaCompilationCacheFileSaver() override = default;
  XlaCompilationCacheFileSaver(absl::string_view directory, Mode mode,
                               bool allow_overwrite)
      : directory_(directory), mode_(mode), allow_overwrite_(allow_overwrite) {}

  Status Save(const XlaSerializedCacheEntry& entry) override;

 private:
  std::string directory_;
  Mode mode_;
  bool allow_overwrite_;
};

// Loads XLA compilation cache entries from files.
class XlaCompilationCacheFileLoader : public XlaCompilationCacheLoader {
 public:
  ~XlaCompilationCacheFileLoader() override = default;
  explicit XlaCompilationCacheFileLoader(absl::string_view directory)
      : directory_(directory) {}

  StatusOr<absl::optional<XlaSerializedCacheEntry>> TryLoad(
      const XlaSerializedCacheKey& key) override;

 private:
  std::string directory_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_PERSISTENCE_H_
