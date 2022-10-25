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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_UNLOADER_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_UNLOADER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"

namespace tensorflow {
namespace tpu {

class TpuCompilationCacheEntryUnloader : public ResourceBase {
 public:
  explicit TpuCompilationCacheEntryUnloader(TpuCompilationCacheInterface* cache)
      : cache_(cache) {
    // Hold a reference to the cache until the unloader is destroyed.
    cache_->Ref();
    VLOG(1) << "Will unload compilation cache entries when session closes.";
  }

  ~TpuCompilationCacheEntryUnloader() override {
    absl::MutexLock lock(&mu_);
    for (int64_t uid : cache_entry_uids_) {
      Status s = cache_->MarkEntryForEviction(uid);
      if (!s.ok()) {
        LOG(WARNING) << "MarkEntryForEviction in "
                        "~CompilationCacheEntryUnloader fails with error "
                     << s;
      }
    }
    // Release our reference to the cache.
    cache_->Unref();
  }

  // Add cache entry uid to be unloaded in destructor.
  void AddCacheEntryUid(int64_t uid) {
    absl::MutexLock lock(&mu_);
    cache_entry_uids_.insert(uid);
  }

  std::string DebugString() const override {
    return "CompilationCacheEntryUnloader";
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TpuCompilationCacheEntryUnloader);
  mutable absl::Mutex mu_;
  TpuCompilationCacheInterface* cache_;  // Not owned.
  absl::flat_hash_set<int64_t> cache_entry_uids_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_UNLOADER_H_
