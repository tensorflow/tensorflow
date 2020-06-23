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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_IMPL_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_IMPL_H_

#include "tensorflow/core/tpu/kernels/compiled_subgraph.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"

namespace tensorflow {
namespace tpu {

// Wrapper for a cache entry that holds a reference to the entry until the
// wrapper is deleted. This wrapper is the concrete type of
// CompilationCacheEntryRef returned by Lookup.
template <typename CacheEntryType>
class CompilationCacheEntryRefImpl
    : public CompilationCacheEntryRef<CacheEntryType> {
 public:
  CompilationCacheEntryRefImpl(TpuCompilationCacheInterface* parent,
                               CompiledSubgraph* entry, int index);

  ~CompilationCacheEntryRefImpl() override;

  Status ToSubEntryRef(CompilationCacheFetchTarget fetch_target) override;

 protected:
  TpuCompilationCacheInterface* parent_;  // Not owned.
  // A reference to entry_ is acquired in the constructor and released via
  // parent->DiscardEntryRefs in the destructor.
  CompiledSubgraph* entry_;
  // The index of the program in entry_ that is returned by the get method.
  int index_;
};

template <typename CacheEntryType>
CompilationCacheEntryRefImpl<CacheEntryType>::CompilationCacheEntryRefImpl(
    TpuCompilationCacheInterface* parent, CompiledSubgraph* entry, int index)
    : parent_(parent), entry_(entry), index_(index) {
  if (entry_ == nullptr) {
    return;
  }
  if (entry_->main_entry == nullptr) {
    entry_->Ref();
  } else {
    // This is a sharding/unsharding entry nested in a main entry. Only
    // refcount the main entry.
    entry_->main_entry->Ref();
  }
}

template <typename CacheEntryType>
CompilationCacheEntryRefImpl<CacheEntryType>::~CompilationCacheEntryRefImpl() {
  if (entry_ == nullptr) {
    return;
  }
  if (entry_->main_entry == nullptr) {
    parent_->DiscardEntryRefs({entry_});
  } else {
    parent_->DiscardEntryRefs({entry_->main_entry});
  }
}

template <typename CacheEntryType>
Status CompilationCacheEntryRefImpl<CacheEntryType>::ToSubEntryRef(
    CompilationCacheFetchTarget fetch_target) {
  CompiledSubgraph* target = nullptr;
  switch (fetch_target) {
    case CompilationCacheFetchTarget::MAIN:
      target = entry_;
      break;
    case CompilationCacheFetchTarget::SHARDING:
      target = entry_->sharding_entry.get();
      break;
    case CompilationCacheFetchTarget::UNSHARDING:
      target = entry_->unsharding_entry.get();
      break;
    default:
      return xla::InvalidArgument("Invalid fetch target: %d", fetch_target);
  }

  if (target == nullptr) {
    // Cache entry does not have an unsharding subentry. Unref and replace
    // with nullptr.
    parent_->DiscardEntryRefs({entry_});
  }
  // Otherwise, since the refcount is always on the main entry, we don't
  // need ref/unref.
  entry_ = target;
  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_ENTRY_IMPL_H_
