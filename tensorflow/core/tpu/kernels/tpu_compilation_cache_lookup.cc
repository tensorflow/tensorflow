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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"

#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_external.h"

namespace tensorflow {
namespace tpu {

namespace {
class CompilationCacheFetchTargetUtility {
 public:
  CompilationCacheFetchTargetUtility()
      : names_({"Invalid", "Main", "Sharding", "Unsharding"}) {}

  std::string name(CompilationCacheFetchTarget target) const {
    return names_[static_cast<int>(target)];
  }

 private:
  const std::vector<std::string> names_;
};

std::string GetName(CompilationCacheFetchTarget target) {
  static const auto* util = new CompilationCacheFetchTargetUtility();
  return util->name(target);
}

}  // namespace

TpuCompilationCacheLocalLookup::TpuCompilationCacheLocalLookup(
    TpuCompilationCacheInterface* cache)
    : cache_(cache) {}

TpuCompilationCacheLocalLookup::~TpuCompilationCacheLocalLookup() {
  cache_->Unref();
}

Status TpuCompilationCacheLocalLookup::Lookup(
    const string& proto_key,
    std::unique_ptr<TpuCompilationCacheEntryRef>* entry,
    CompilationCacheFetchTarget fetch_target) {
  profiler::TraceMe proto_lookup_traceme("Local TPU proto cache lookup",
                                         /*level=*/2);
  Status s = cache_->Lookup<TpuCompilationCacheEntryRef, EntryRefImpl>(
      proto_key, entry);
  VLOG(1) << "Looked up key " << proto_key << " in local subgraph cache status "
          << s;
  if (!s.ok()) {
    return s;
  }
  s = (*entry)->ToSubEntryRef(fetch_target);

  VLOG(1) << "Fetched subentry: " << GetName(fetch_target) << " with status "
          << s;
  return s;
}

Status TpuCompilationCacheLocalLookup::Lookup(
    int64 uid, int proto_index,
    std::unique_ptr<TpuCompilationCacheEntryRef>* entry,
    CompilationCacheFetchTarget fetch_target) {
  profiler::TraceMe proto_lookup_traceme("Local TPU proto cache lookup by uid",
                                         /*level=*/2);
  Status s = cache_->Lookup<TpuCompilationCacheEntryRef, EntryRefImpl>(
      uid, proto_index, entry);
  VLOG(1) << "Looked up uid " << uid << ", index " << proto_index
          << " in local subgraph cache status " << s;
  if (!s.ok()) {
    return s;
  }
  s = (*entry)->ToSubEntryRef(fetch_target);
  VLOG(1) << "Fetched subentry: " << GetName(fetch_target) << " with status "
          << s;
  return s;
}

string TpuCompilationCacheLocalLookup::DebugString() const {
  return "TpuCompilationCacheLocalLookup";
}

}  // namespace tpu
}  // namespace tensorflow
