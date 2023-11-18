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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_CACHE_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_CACHE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace device_compilation_cache_internal {
template <typename ExecutableType>
int64_t ExecutableSize(const ExecutableType* executable) {
  return 0;
}

template <>
inline int64_t ExecutableSize<xla::LocalExecutable>(
    const xla::LocalExecutable* executable) {
  if (executable != nullptr && executable->executable() != nullptr) {
    return executable->executable()->SizeOfGeneratedCodeInBytes();
  }

  return 0;
}

template <>
inline int64_t ExecutableSize<xla::PjRtLoadedExecutable>(
    const xla::PjRtLoadedExecutable* executable) {
  if (executable != nullptr) {
    return executable->SizeOfGeneratedCodeInBytes();
  }

  return 0;
}
}  // namespace device_compilation_cache_internal

// Cache to store compiled HLO, executables and related metadata keyed by
// `DeviceCompilationClusterSignature`. The cache owns the stored
// CompilationResults and Executables.
// Currently no cache eviction policy is implemented and the cache grows without
// bound.
template <typename ExecutableType>
class DeviceCompilationCache {
 public:
  DeviceCompilationCache() = default;
  ~DeviceCompilationCache() = default;

  using Key = DeviceCompilationClusterSignature;
  struct Value {
    DeviceCompileState compile_state = DeviceCompileState::kUncompiled;
    Status compilation_status;
    int64_t request_count = 0;
    const XlaCompiler::CompilationResult* compilation_result = nullptr;
    ExecutableType* executable = nullptr;
  };

  // Returns std::nullopt if value for the supplied key is not found. If a value
  // is found, `request_count` is incremented before returning the value.
  std::optional<Value> Lookup(const Key& key) const;

  // Inserts an empty value if value is not found and returns it. If a value is
  // found, `request_count` is incremented before returning the value.
  Value LookupOrCreate(const Key& key);

  // Caches `compile_state`, `compilation_status`, `compilation_result` and
  // `executable` and associates them with the provided `key`. Takes ownership
  // of `compilation_result` and `executable`. Does not increment the
  // corresponding `request_count`. Only arguments that are not std::nullopt are
  // updated in the cache.
  void Store(const Key& key, std::optional<DeviceCompileState> compile_state,
             std::optional<Status> compilation_status,
             std::optional<std::unique_ptr<XlaCompiler::CompilationResult>>
                 compilation_result,
             std::optional<std::unique_ptr<ExecutableType>> executable);

  std::string DebugString() const;

 private:
  // The value associated with a cache entry.
  struct Entry {
    mutable mutex mu;

    // The current compilation state for this entry.
    DeviceCompileState compile_state TF_GUARDED_BY(mu) =
        DeviceCompileState::kUncompiled;

    // The number of times a compilation with this signature has been requested.
    int64_t request_count TF_GUARDED_BY(mu) = 0;

    // Did compilation succeed?
    Status compilation_status TF_GUARDED_BY(mu);

    // Output of the XlaCompiler.
    std::unique_ptr<XlaCompiler::CompilationResult> compilation_result
        TF_GUARDED_BY(mu);

    // The XLA executable compiled from <computation>. May be null if no
    // executable has been built.
    std::unique_ptr<ExecutableType> executable TF_GUARDED_BY(mu);

    std::string DebugString() const {
      mutex_lock lock(mu);

      int64_t executable_size =
          device_compilation_cache_internal::ExecutableSize<ExecutableType>(
              executable.get());

      int64_t hlo_module_size = 0;
      if (compilation_result != nullptr &&
          compilation_result->computation != nullptr) {
        hlo_module_size =
            compilation_result->computation->proto().ByteSizeLong();
      }

      return absl::StrCat(
          "{compile_state: ", compile_state, ", request_count: ", request_count,
          ", compilation_status: ", compilation_status.ToString(),
          ", compilation_result?: ", compilation_result != nullptr,
          ", hlo_module_size: ", hlo_module_size, " bytes",
          ", executable?: ", executable != nullptr,
          ", executable_size: ", executable_size, " bytes}");
    }
  };

  mutable mutex compile_cache_mu_;
  absl::flat_hash_map<Key, std::unique_ptr<Entry>, Key::Hash> cache_
      TF_GUARDED_BY(compile_cache_mu_);

  DeviceCompilationCache(const DeviceCompilationCache&) = delete;
  void operator=(const DeviceCompilationCache&) = delete;
};

template <typename ExecutableType>
std::optional<typename DeviceCompilationCache<ExecutableType>::Value>
DeviceCompilationCache<ExecutableType>::Lookup(const Key& key) const {
  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    // Find cache entry.
    auto it = cache_.find(key);
    if (it == cache_.cend()) {
      return std::nullopt;
    }

    entry = it->second.get();
  }

  mutex_lock lock(entry->mu);
  Value value = {/*compile_state=*/entry->compile_state,
                 /*compilation_status=*/entry->compilation_status,
                 /*request_count=*/++entry->request_count,
                 /*compilation_result=*/entry->compilation_result.get(),
                 /*executable=*/entry->executable.get()};
  return value;
}

template <typename ExecutableType>
typename DeviceCompilationCache<ExecutableType>::Value
DeviceCompilationCache<ExecutableType>::LookupOrCreate(const Key& key) {
  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    // Emplace empty cache entry if not found.
    auto it = cache_.emplace(key, std::make_unique<Entry>()).first;
    entry = it->second.get();
  }

  mutex_lock lock(entry->mu);
  Value value = {/*compile_state=*/entry->compile_state,
                 /*compilation_status=*/entry->compilation_status,
                 /*request_count=*/++entry->request_count,
                 /*compilation_result=*/entry->compilation_result.get(),
                 /*executable=*/entry->executable.get()};
  return value;
}

template <typename ExecutableType>
void DeviceCompilationCache<ExecutableType>::Store(
    const Key& key, std::optional<DeviceCompileState> compile_state,
    std::optional<Status> compilation_status,
    std::optional<std::unique_ptr<XlaCompiler::CompilationResult>>
        compilation_result,
    std::optional<std::unique_ptr<ExecutableType>> executable) {
  Entry* entry;
  {
    mutex_lock lock(compile_cache_mu_);
    // Emplace empty cache entry if not found.
    auto it = cache_.emplace(key, std::make_unique<Entry>()).first;
    entry = it->second.get();
  }

  {
    mutex_lock lock(entry->mu);
    if (compile_state.has_value()) {
      entry->compile_state = *compile_state;
    }
    if (compilation_status.has_value()) {
      entry->compilation_status = *compilation_status;
    }
    if (compilation_result.has_value()) {
      entry->compilation_result = std::move(*compilation_result);
    }
    if (executable.has_value()) {
      entry->executable = std::move(*executable);
    }
  }

  VLOG(4) << "Added/updated cache entry: key=" << key.HumanString()
          << ", entry=" << entry->DebugString();
}

template <typename ExecutableType>
std::string DeviceCompilationCache<ExecutableType>::DebugString() const {
  std::string s = "DeviceCompilationCache<ExecutableType> {\n";
  {
    mutex_lock lock(compile_cache_mu_);
    for (const auto& [key, entry] : cache_) {
      absl::StrAppend(&s, key.HumanString(), " : ", entry->DebugString(),
                      ",\n");
    }
  }
  absl::StrAppend(&s, "}");

  return s;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_COMPILATION_CACHE_H_
