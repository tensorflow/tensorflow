/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_SUPPORT_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_SUPPORT_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "grpcpp/security/credentials.h"
#include "grpcpp/support/slice.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"

namespace tensorflow {
namespace tpu {

// A cache entry for remote TPU compilation.
struct CacheEntry {
  CacheEntry() : size(0), last_use(-1) {}
  virtual ~CacheEntry() {
    if (tpu_program_group != nullptr) {
      tpu_program_group->UnloadAndDestroyPrograms();
    }
  }
  std::unique_ptr<TpuProgramGroupInterface> tpu_program_group;
  std::string key;
  int64_t size;

  // An integer-based monotonically increasing counter used by the TPU
  // compilation cache to sort and evict the least recently used entry when the
  // cache size exceeded the maximum size limit. The value is initialized to
  // `-1` as an initial value.
  int64_t last_use;
};

// Implementation of `CompilationCacheEntryRef` that holds a shared_ptr to the
// local cache entry until the wrapper is destroyed.
class CacheWrapper : public CompilationCacheEntryRef {
 public:
  explicit CacheWrapper(std::shared_ptr<CacheEntry> entry)
      : cache_entry_(std::move(entry)) {}
  ~CacheWrapper() override = default;

  TpuCompilationCacheEntry get() override {
    if (cache_entry_->size == 0) {
      // Create an empty entry if the size is 0. This corresponds to
      // non-existing sharding/unsharding entries.
      return TpuCompilationCacheEntry();
    }
    return TpuCompilationCacheEntry(cache_entry_->tpu_program_group.get(),
                                    /*core_index=*/0);
  }

  Status ToSubEntryRef(CompilationCacheFetchTarget fetch_target) override {
    LOG(FATAL) << "Not implemented by designed.";
  }

 private:
  std::shared_ptr<CacheEntry> cache_entry_;
};

// Creates gRPC channel credentials for the current runtime env.
std::shared_ptr<::grpc::ChannelCredentials> CreateChannelCredentials();

// Fills an uinitialized `CacheEntry` from `GetTpuProgramResponse` proto. The
// `cache_entry` will be instantiated by the function.
template <typename ResponseType>
Status DeserializeRpcResponseToCacheEntry(
    const absl::string_view local_proto_key, ResponseType* response,
    std::shared_ptr<CacheEntry>* cache_entry);

// Serializes `TpuCompilationCacheEntry` to gRPC bufer slices.
xla::StatusOr<std::vector<::grpc::Slice>> SerializeCacheEntryToBufferSlices(
    const TpuCompilationCacheEntry& cache_entry);
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_SUPPORT_H_
