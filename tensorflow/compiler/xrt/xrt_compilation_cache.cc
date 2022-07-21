/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xrt/xrt_compilation_cache.h"

#include <stdlib.h>

#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

int64_t get_uid() {
  uint64 unsigned_rand = random::New64() & INT64_MAX;
  return static_cast<int64_t>(unsigned_rand);
}

int64_t GetCompilationCacheSizeFromEnv() {
  const char* env = getenv("TF_XRT_COMPILATION_CACHE_SIZE");
  return env == nullptr ? 1024 : std::stol(env);
}

}  // namespace

const char* kXRTCompilationCacheResourceName = "xrt_compilation_cache";

XRTCompilationCache::EntryRefImpl::EntryRefImpl(XRTCompilationCache* parent,
                                                CompiledSubgraph* entry)
    : parent_(parent), entry_(entry) {
  entry_->Ref();
}

XRTCompilationCache::EntryRefImpl::~EntryRefImpl() {
  parent_->DiscardEntryRef(entry_);
}

XRTCompilationCacheEntry XRTCompilationCache::EntryRefImpl::get() {
  return XRTCompilationCacheEntry(entry_->program.get());
}

XRTCompilationCache::XRTCompilationCache(int max_number_of_entries)
    : max_cache_entries_(max_number_of_entries) {
  CHECK_GE(max_cache_entries_, 0);
  VLOG(1) << "Created compilation cache max " << max_cache_entries_
          << " entries.";
}

XRTCompilationCache::~XRTCompilationCache() {
  VLOG(1) << "XRTCompilationCache::~XRTCompilationCache()";
  // A buggy client may be holding onto a reference, or a client might have
  // crashed while holding onto a reference. In either case, discard all
  // outstanding client references to avoid leaking storage.
  for (const auto& entry : entries_by_uid_) {
    while (!entry.second->RefCountIsOne()) {
      entry.second->Unref();
    }
  }
  while (!entries_by_last_use_.empty()) {
    MarkOldestEntryForEviction();
  }
  CHECK_EQ(cache_.size(), 0);
  CHECK_EQ(entries_by_uid_.size(), 0);
  CHECK_EQ(cache_entries_, 0);
  CHECK_EQ(marked_for_eviction_entries_, 0);
}

Status XRTCompilationCache::Release(int64_t uid) {
  absl::MutexLock lock(&mu_);
  auto iter = entries_by_uid_.find(uid);

  if (iter == entries_by_uid_.end()) {
    return errors::NotFound("No cache entry found for uid ", uid);
  }

  DiscardEntryRefLocked(iter->second);

  VLOG(1) << "After releasing entry " << uid << " refs cache is "
          << cache_.size() << " entries ("
          << cache_entries_ + marked_for_eviction_entries_
          << "), marked for eviction "
          << (cache_.size() - entries_by_last_use_.size()) << " entries ("
          << marked_for_eviction_entries_ << ").";

  return OkStatus();
}

void XRTCompilationCache::DiscardEntryRef(CompiledSubgraph* entry) {
  absl::MutexLock lock(&mu_);
  DiscardEntryRefLocked(entry);
}

void XRTCompilationCache::DiscardEntryRefLocked(CompiledSubgraph* entry) {
  if (entry->RefCountIsOne()) {
    // The last reference to this entry is going away, so really delete it from
    // the cache in such a way that it can't be restored by being looked up
    // again.

    // Sanity-check that it has been marked for eviction.
    CHECK(entries_by_last_use_.find(entry->last_use) ==
          entries_by_last_use_.end());
    // Update the counter tracking how much space is taken up by entries that
    // are marked for eviction.
    --marked_for_eviction_entries_;

    // Remove the entry from the cache.
    auto erased = cache_.erase(entry->key);
    if (erased == 0) {
      LOG(FATAL) << "Tried to discard nonexistent cache entry";
    }
    erased = entries_by_uid_.erase(entry->uid);
    CHECK_EQ(erased, 1);
  }
  entry->Unref();
}

void XRTCompilationCache::MarkOldestEntryForEviction() {
  CompiledSubgraph* entry_to_mark = entries_by_last_use_.begin()->second;
  VLOG(1) << "Marking " << entry_to_mark->key << " for eviction";
  entries_by_last_use_.erase(entry_to_mark->last_use);
  --cache_entries_;
  ++marked_for_eviction_entries_;
  // Discard the cache's reference to entry. If steps are holding onto
  // references to entry it won't be deleted until the last step holding it
  // completes. It stays in the cache in the meantime and can be resurrected
  // by a call to CompileIfKeyAbsent if that occurs before the last reference
  // expires.
  DiscardEntryRefLocked(entry_to_mark);
}

void XRTCompilationCache::LookupEntryMarkedForEviction(
    CompiledSubgraph* entry) {
  // The entry was previously marked for eviction (or is newly created) so
  // unmark it. Add a reference (owned by the cache), update the cache size, and
  // mark something old for eviction if necessary.
  entry->Ref();
  --marked_for_eviction_entries_;
  ++cache_entries_;

  // Mark the least-recently-used non-marked entry for eviction. Never mark the
  // most-recently used entry (i.e., do nothing if entries_by_last_use_ == 1
  // which means there's only one entry not already marked for eviction), so
  // that an entry persists in the cache even if it is larger than the allocated
  // cache size.
  while (entries_by_last_use_.size() > 1 &&
         cache_entries_ > max_cache_entries_) {
    MarkOldestEntryForEviction();
  }
}

XRTCompilationCache::CompiledSubgraph* XRTCompilationCache::InitializeEntry(
    const string& key,
    const std::function<Status(std::unique_ptr<xla::LocalExecutable>*)>&
        initialize_program) {
  CompiledSubgraph* entry = new CompiledSubgraph();
  entry->parent = this;
  entry->key = key;
  entry->uid = get_uid();
  // Add the entry to the cache. Once the computation has been compiled,
  // UpdateEntryAfterCompilation will be called to potentially mark old entries
  // that don't fit any more for eviction.
  //
  // At this point there is one reference to entry, which is owned by the caller
  // who created the entry. A second reference, owned by the cache, will be
  // added below since we leave the entry in the 'marked for eviction' state
  // here.
  auto cache_inserted =
      cache_.insert(std::pair<string, CompiledSubgraph*>(key, entry));
  CHECK(cache_inserted.second);

  // Initialize the program outside the lock so that other cache operations
  // can proceed during the (potentially lengthy) initialization.
  Status s;
  std::unique_ptr<xla::LocalExecutable> program;
  {
    mu_.Unlock();
    { s = initialize_program(&program); }
    mu_.Lock();
  }

  // Add the entry to the uid index.
  auto uid_inserted = entries_by_uid_.insert(
      std::pair<int64_t, CompiledSubgraph*>(entry->uid, entry));
  CHECK(uid_inserted.second);

  entry->initialized = true;
  entry->initialization_status = s;
  if (s.ok()) {
    entry->program = std::move(program);
  }
  // Add the entry to marked_for_eviction_entries_ since it will be adjusted
  // down again when the newly-created entry gets unmarked.
  ++marked_for_eviction_entries_;
  return entry;
}

Status XRTCompilationCache::CompileIfKeyAbsent(
    const string& key, int64_t* uid,
    const std::function<Status(std::unique_ptr<xla::LocalExecutable>*)>&
        compile_function) {
  CompiledSubgraph* entry = nullptr;

  absl::MutexLock lock(&mu_);
  auto iter = cache_.find(key);

  if (iter == cache_.end()) {
    // The single ref on the newly-created entry is owned by the caller.
    VLOG(1) << "Before adding new entry for key " << key << " cache is "
            << cache_.size() << " entries ("
            << cache_entries_ + marked_for_eviction_entries_ << "), "
            << " marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_entries_ << ").";
    entry = InitializeEntry(key, compile_function);
  } else {
    VLOG(1) << "Before refreshing entry for key " << key << " cache is "
            << cache_.size() << " entries ("
            << cache_entries_ + marked_for_eviction_entries_ << "), "
            << " marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_entries_ << ").";
    entry = iter->second;
    // Make a new reference that is owned by the caller.
    entry->Ref();
    // Block if necessary until the subgraph has been initialized.
    mu_.Await(absl::Condition(
        +[](CompiledSubgraph* e) { return e->initialized; }, entry));
  }

  // Let the caller know the uid of the entry.
  *uid = entry->uid;

  // Remove the old LRU-table entry if it wasn't already marked for eviction.
  auto erased = entries_by_last_use_.erase(entry->last_use);
  // Update the LRU table indicating this entry is the most recently used.
  entry->last_use = use_counter_++;
  entries_by_last_use_[entry->last_use] = entry;
  if (erased == 0) {
    // The entry had been marked for eviction, or is newly created.
    LookupEntryMarkedForEviction(entry);
  }

  VLOG(1) << "After refreshing entry for key " << key << " cache is "
          << cache_.size() << " entries ("
          << cache_entries_ + marked_for_eviction_entries_ << "), "
          << " marked for eviction "
          << (cache_.size() - entries_by_last_use_.size()) << " entries ("
          << marked_for_eviction_entries_ << ").";

  return entry->initialization_status;
}

Status XRTCompilationCache::Lookup(
    int64_t uid, std::unique_ptr<XRTCompilationCacheEntryRef>* entry) {
  entry->reset();

  absl::MutexLock lock(&mu_);
  const auto iter = entries_by_uid_.find(uid);
  if (iter == entries_by_uid_.end()) {
    return errors::NotFound("No executable found for uid ", uid);
  }
  CompiledSubgraph* cache_entry = iter->second;
  *entry = std::unique_ptr<XRTCompilationCacheEntryRef>(
      new EntryRefImpl(this, cache_entry));
  return OkStatus();
}

string XRTCompilationCache::DebugString() const {
  return "XRTCompilationCache";
}

xla::StatusOr<RefPtr<XRTCompilationCache>> GetOrCreateCompilationCache(
    ResourceMgr* rm, int64_t max_number_of_entries) {
  if (max_number_of_entries == 0) {
    max_number_of_entries = GetCompilationCacheSizeFromEnv();
  }
  XRTCompilationCache* cache;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XRTCompilationCache>(
      rm->default_container(), kXRTCompilationCacheResourceName, &cache,
      [&](XRTCompilationCache** new_cache) {
        *new_cache = new XRTCompilationCache(max_number_of_entries);
        return OkStatus();
      }));
  return RefPtr<XRTCompilationCache>(cache);
}

}  // namespace tensorflow
