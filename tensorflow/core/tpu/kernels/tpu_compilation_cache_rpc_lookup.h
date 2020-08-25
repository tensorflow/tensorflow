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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_RPC_LOOKUP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_RPC_LOOKUP_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_support.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"

namespace tensorflow {
namespace tpu {

// Class for looking up and caching TPU program via RPC.
class TpuCompilationCacheRpcLookup : public TpuCompilationCacheLookup {
 public:
  using StubType = tpu::grpc::TpuCompilationCacheService::Stub;

  TpuCompilationCacheRpcLookup(const string& server_address,
                               int64 max_cache_size);
  ~TpuCompilationCacheRpcLookup() override = default;

  Status Lookup(const string& proto_key,
                std::unique_ptr<tpu::CompilationCacheEntryRef>* entry,
                tpu::CompilationCacheFetchTarget fetch_target) override;

  Status Lookup(int64 uid, int proto_index,
                std::unique_ptr<tpu::CompilationCacheEntryRef>* entry,
                tpu::CompilationCacheFetchTarget fetch_target) override;

  string DebugString() const override;

 private:
  // Helper method to make the RPC request to the central cache.
  Status RemoteLookupLocked(const string& local_proto_key,
                            const tpu::GetTpuProgramRequest& request,
                            std::shared_ptr<CacheEntry>* cache_entry)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper method to adjust datastructures after a cache lookup.
  // We use `removed_entries` so that actual CacheEntry destruction happens
  // outside the lock.
  void PostLookupLocked(
      std::shared_ptr<CacheEntry>* cache_entry,
      std::unique_ptr<tpu::CompilationCacheEntryRef>* entry,
      std::vector<std::shared_ptr<CacheEntry>>* removed_entries)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The maximum size of entries that are stored in the cache before entries are
  // evicted.
  const int64 max_cache_size_;

  std::unique_ptr<StubType> stub_;

  // Protect concurrent access to member variables below.
  mutable absl::Mutex mu_;

  // The total size of entries in the cache.
  int64 cache_size_ ABSL_GUARDED_BY(mu_) = 0;
  // The value to assign to the last_use field of the next entry that is looked
  // up.
  int64 use_counter_ ABSL_GUARDED_BY(mu_) = 0;
  // The entries that can be looked up in the cache. An entry is deleted from
  // the cache as soon as it is evicted, but the underlying shared_ptr won't be
  // freed until any wrappers holding it go out of scope.
  std::unordered_map<std::string, std::shared_ptr<CacheEntry>> cache_
      ABSL_GUARDED_BY(mu_);
  // Map from last_use to entry, used to evict entries in LRU order.
  std::map<int64, CacheEntry*> entries_by_last_use_ ABSL_GUARDED_BY(mu_);
};
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_RPC_LOOKUP_H_
