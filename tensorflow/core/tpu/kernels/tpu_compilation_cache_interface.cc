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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"

#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"

namespace tensorflow {
namespace tpu {

TpuCompilationCacheInterface::RefHolder::RefHolder(
    TpuCompilationCacheInterface* parent)
    : parent_(parent) {
  // Hold a reference to the parent until the holder is discarded.
  parent_->Ref();
}

TpuCompilationCacheInterface::RefHolder::~RefHolder() {
  parent_->DiscardEntryRefs(entries_);
  // Release our reference to the parent.
  parent_->Unref();
}

void TpuCompilationCacheInterface::RefHolder::AddRef(CompiledSubgraph* entry) {
  entries_.push_back(entry);
}

string TpuCompilationCacheInterface::RefHolder::DebugString() const {
  return "TpuCompilationCacheRefHolder";
}

TpuCompilationCacheInterface::TpuCompilationCacheInterface(int64 max_cache_size)
    : max_cache_size_(max_cache_size) {
  CHECK_GE(max_cache_size_, 0);
  VLOG(1) << "Created compilation cache size " << max_cache_size_ << " bytes.";
}

TpuCompilationCacheInterface::~TpuCompilationCacheInterface() {
  VLOG(1) << "TpuCompilationCacheInterface::~TpuCompilationCacheInterface()";
  // A buggy client may be holding onto a reference, or a client might have
  // crashed while holding onto a reference. In either case, discard all
  // outstanding client references to avoid leaking storage.
  for (const auto& entry : entries_by_uid_) {
    while (entry.second->external_references > 0) {
      Status s = Release(entry.first);
      CHECK(s.ok());
    }
  }
  while (!entries_by_last_use_.empty()) {
    UnloadAndDestroy(MarkOldestEntryForEviction());
  }
  // By the time the cache is deleted all reference holders should have already
  // been deleted, since they were holding references to the cache. So all
  // entries should be gone at this point.
  CHECK_EQ(cache_.size(), 0);
  CHECK_EQ(entries_by_uid_.size(), 0);
  CHECK_EQ(entries_by_proto_key_.size(), 0);
  CHECK_EQ(cache_size_, 0);
  CHECK_EQ(marked_for_eviction_size_, 0);
}

Status TpuCompilationCacheInterface::MarkEntryForEviction(int64 subgraph_uid) {
  profiler::TraceMe key_release_traceme(
      "TPU compilation cache possibly evict uid",
      /*level=*/2);
  CompiledSubgraph* deleted_entry = nullptr;
  {
    absl::MutexLock lock(&mu_);
    auto iter = entries_by_uid_.find(subgraph_uid);
    if (iter == entries_by_uid_.end()) {
      // If already evicted, return ok.
      return Status::OK();
    }

    // Mark entry for eviction.
    CompiledSubgraph* subgraph_to_evict = iter->second;
    // If there are external references, should not use this API.
    if (subgraph_to_evict->external_references != 0) {
      return errors::Internal("Subgraph ", subgraph_to_evict->subgraph_key,
                              " external_references greater than zero. Should "
                              "use TpuCompilationCacheInterface::Release.");
    }

    VLOG(1) << "Marking " << subgraph_to_evict->subgraph_key
            << " for eviction. Debug string: "
            << subgraph_to_evict->cache_entry_debug_string;
    entries_by_last_use_.erase(subgraph_to_evict->last_use);
    cache_size_ -= subgraph_to_evict->total_size;
    marked_for_eviction_size_ += subgraph_to_evict->total_size;

    // Evict if refcount exactly one, otherwise only discard cache's reference
    // to the entry while the actual eviction will happen when refholder's
    // references go away.
    deleted_entry = DiscardEntryRef(subgraph_to_evict);

    VLOG(1) << "After possibly evicting entry " << subgraph_uid
            << " refs cache is " << cache_.size() << " entries ("
            << cache_size_ + marked_for_eviction_size_
            << " bytes), marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_size_ << " bytes).";
  }

  // Unload from device cache if entry is evicted from host cache.
  UnloadAndDestroy(deleted_entry);
  return Status::OK();
}

Status TpuCompilationCacheInterface::Release(int64 subgraph_uid) {
  profiler::TraceMe key_release_traceme("TPU compilation cache release uid",
                                        /*level=*/2);

  CompiledSubgraph* deleted_entry = nullptr;
  {
    absl::MutexLock lock(&mu_);
    auto iter = entries_by_uid_.find(subgraph_uid);

    if (iter == entries_by_uid_.end()) {
      return errors::NotFound("No cache entry found for uid ", subgraph_uid);
    }

    CHECK_GT(iter->second->external_references, 0);
    --iter->second->external_references;

    deleted_entry = DiscardEntryRef(iter->second);

    VLOG(1) << "After releasing entry " << subgraph_uid << " refs cache is "
            << cache_.size() << " entries ("
            << cache_size_ + marked_for_eviction_size_
            << " bytes), marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_size_ << " bytes).";
  }
  UnloadAndDestroy(deleted_entry);
  return Status::OK();
}

void TpuCompilationCacheInterface::UnloadAndDestroy(CompiledSubgraph* entry) {
  if (!entry) return;

  CHECK(entry->RefCountIsOne());
  entry->tpu_program_group->UnloadAndDestroyPrograms();
  entry->Unref();
}

size_t TpuCompilationCacheInterface::RemoveEntry(const string& key) {
  auto erased = cache_.erase(key);
  tpu::TpuCompilationCacheMetrics::SetCacheEntryCount(cache_.size());

  auto parsed_key_or_status = ParseCompilationCacheKey(key);
  CHECK(parsed_key_or_status.status().ok());
  const TpuCompilationCacheKey parsed_key =
      parsed_key_or_status.ConsumeValueOrDie();
  if (!parsed_key.has_guaranteed_const) {
    return erased;
  }
  session_key_map_.erase(
      strings::StrCat(parsed_key.prefix, parsed_key.session_handle));
  fingerprint_key_map_.erase(strings::StrCat(
      parsed_key.prefix, parsed_key.guaranteed_const_fingerprint()));
  return erased;
}

CompiledSubgraph* TpuCompilationCacheInterface::DiscardEntryRef(
    CompiledSubgraph* entry) {
  if (entry->RefCountIsOne()) {
    // The last reference to this entry is going away, so really delete it from
    // the cache in such a way that it can't be restored by being looked up
    // again.

    // Sanity-check that it has been marked for eviction.
    CHECK(entries_by_last_use_.find(entry->last_use) ==
          entries_by_last_use_.end());
    // Update the counter tracking how much space is taken up by entries that
    // are marked for eviction.
    marked_for_eviction_size_ -= entry->total_size;

    // Remove the entry from the cache.
    auto erased = RemoveEntry(entry->subgraph_key);

    if (erased == 0) {
      LOG(FATAL) << "Tried to discard nonexistent cache entry";
    }
    erased = entries_by_uid_.erase(entry->uid);
    CHECK_EQ(erased, 1);
    for (const string& key : entry->proto_key) {
      erased = entries_by_proto_key_.erase(key);
      CHECK_EQ(erased, 1);
    }
    // The actual deletion will happen outside the lock in UnloadAndDestroy().
    return entry;
  }
  entry->Unref();
  return nullptr;
}

CompilationRefHolder* TpuCompilationCacheInterface::MakePerStepRefHolder() {
  return new RefHolder(this);
}

void TpuCompilationCacheInterface::DiscardEntryRefs(
    gtl::ArraySlice<CompiledSubgraph*> entries) {
  std::vector<CompiledSubgraph*> removed_entries;
  {
    absl::MutexLock lock(&mu_);

    for (auto entry : entries) {
      removed_entries.push_back(DiscardEntryRef(entry));
    }

    VLOG(1) << "After discarding entry refs cache is " << cache_.size()
            << " entries (" << cache_size_ + marked_for_eviction_size_
            << " bytes), marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_size_ << " bytes).";
  }
  for (auto removed_entry : removed_entries) {
    UnloadAndDestroy(removed_entry);
  }
}

CompiledSubgraph* TpuCompilationCacheInterface::MarkOldestEntryForEviction() {
  CompiledSubgraph* entry_to_mark = entries_by_last_use_.begin()->second;
  VLOG(1) << "Marking " << entry_to_mark->subgraph_key
          << " for eviction. Debug string: "
          << entry_to_mark->cache_entry_debug_string;
  entries_by_last_use_.erase(entry_to_mark->last_use);
  cache_size_ -= entry_to_mark->total_size;
  marked_for_eviction_size_ += entry_to_mark->total_size;
  // Discard the cache's reference to entry. If steps are holding onto
  // references to entry it won't be deleted until the last step holding it
  // completes. It stays in the cache in the meantime and can be resurrected
  // by a call to CompileIfKeyAbsent if that occurs before the last reference
  // expires.
  return DiscardEntryRef(entry_to_mark);
}

void TpuCompilationCacheInterface::LookupEntryMarkedForEviction(
    CompiledSubgraph* entry, std::vector<CompiledSubgraph*>* removed_entries) {
  // The entry was previously marked for eviction (or is newly created) so
  // unmark it. Add a reference (owned by the cache), update the cache size, and
  // mark something old for eviction if necessary.
  entry->Ref();
  marked_for_eviction_size_ -= entry->total_size;
  cache_size_ += entry->total_size;

  // Mark the least-recently-used non-marked entry for eviction. Never mark the
  // most-recently used entry (i.e., do nothing if entries_by_last_use_ == 1
  // which means there's only one entry not already marked for eviction), so
  // that an entry persists in the cache even if it is larger than the allocated
  // cache size.
  while (entries_by_last_use_.size() > 1 && cache_size_ > max_cache_size_) {
    if (auto entry_to_evict = MarkOldestEntryForEviction()) {
      removed_entries->push_back(entry_to_evict);
    }
  }
}

void TpuCompilationCacheInterface::InsertEntry(const string& key,
                                               CompiledSubgraph* entry) {
  auto cache_inserted =
      cache_.insert(std::pair<string, CompiledSubgraph*>(key, entry));
  CHECK(cache_inserted.second);
  tpu::TpuCompilationCacheMetrics::SetCacheEntryCount(cache_.size());

  auto parsed_key_or_status = ParseCompilationCacheKey(key);
  CHECK(parsed_key_or_status.status().ok());
  const TpuCompilationCacheKey parsed_key =
      parsed_key_or_status.ConsumeValueOrDie();
  if (!parsed_key.has_guaranteed_const) {
    return;
  }
  session_key_map_.insert(std::make_pair(
      strings::StrCat(parsed_key.prefix, parsed_key.session_handle), key));
  fingerprint_key_map_.insert(
      std::make_pair(strings::StrCat(parsed_key.prefix,
                                     parsed_key.guaranteed_const_fingerprint()),
                     key));
}

Status TpuCompilationCacheInterface::CompileIfKeyAbsent(
    const TpuCompilationCacheKey& subgraph_key,
    const SessionMetadata* session_metadata,
    CompilationRefHolder* per_step_ref_holder, int64* uid,
    std::vector<string>* proto_key, std::vector<bool>* may_modify_variables,
    absl::Span<const xla::HloProto* const>* hlo_metadatas,
    const std::function<Status(TpuProgramGroupInterface*)>& compile_function) {
  std::vector<CompiledSubgraph*> removed_entries;
  auto status = CompileIfKeyAbsentHelper(
      subgraph_key, session_metadata, per_step_ref_holder, uid, proto_key,
      may_modify_variables, &removed_entries, hlo_metadatas, compile_function);
  for (auto entry : removed_entries) {
    UnloadAndDestroy(entry);
  }
  return status;
}

string TpuCompilationCacheInterface::FindCacheKey(
    const TpuCompilationCacheKey& subgraph_key) {
  if (!subgraph_key.has_guaranteed_const) {
    return subgraph_key.prefix;
  }
  auto iter = session_key_map_.find(
      strings::StrCat(subgraph_key.prefix, subgraph_key.session_handle));
  if (iter != session_key_map_.end()) {
    return iter->second;
  }
  iter = fingerprint_key_map_.find(strings::StrCat(
      subgraph_key.prefix, subgraph_key.guaranteed_const_fingerprint()));
  if (iter != session_key_map_.end()) {
    return iter->second;
  }
  VLOG(1) << "No matching cache key found for key " << subgraph_key.ToString();
  return "";
}

Status TpuCompilationCacheInterface::CompileIfKeyAbsentHelper(
    const TpuCompilationCacheKey& subgraph_key,
    const SessionMetadata* session_metadata,
    CompilationRefHolder* per_step_ref_holder, int64* uid,
    std::vector<string>* proto_key, std::vector<bool>* may_modify_variables,
    std::vector<CompiledSubgraph*>* removed_entries,
    absl::Span<const xla::HloProto* const>* hlo_metadatas,
    const std::function<Status(TpuProgramGroupInterface*)>& compile_function) {
  CompiledSubgraph* entry = nullptr;

  profiler::TraceMe subgraph_lookup_traceme(
      "TPU compilation cache subgraph lookup",
      /*level=*/2);

  // NOTE: In spite of the fact that we use MutexLock, we do not hold the lock
  // for the lifetime of the object, see InitializeEntry() call below.
  absl::MutexLock lock(&mu_);

  string cache_key = FindCacheKey(subgraph_key);
  auto iter = cache_.find(cache_key);
  bool is_new_key = iter == cache_.end();

  const string session_name = tpu::SessionNameFromMetadata(session_metadata);

  if (is_new_key) {
    cache_key = subgraph_key.ToString();
    tpu::TpuCompilationCacheMetrics::IncrementCacheLookupCount(
        /*is_cache_hit=*/false, session_name);
    const string msg =
        strings::StrCat("TPU host compilation cache miss: cache_key(",
                        cache_key, "), session_name(", session_name, ")");
    TRACESTRING(msg);
    LOG(INFO) << msg;

    // Check if caller has disabled compilation. Set using
    // internal::ScopedTpuCompileDisabler.
    if (!IsTpuCompilationEnabled()) {
      const string error_msg = strings::StrCat(
          "[TpuCompilationDisabled]: Compilation cache miss, but compilation "
          "disabled, session_name(",
          session_name, ") Debug String: ", subgraph_key.debug_string);
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "Cache Missed. Current cache entries: ";
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
          VLOG(2) << "Cache Debug Info: ";
          VLOG(2) << it->second->cache_entry_debug_string;
        }
      }

      LOG_EVERY_N_SEC(WARNING, 30) << error_msg;
      return errors::NotFound(error_msg);
    }

    // The single ref on the newly-created entry is owned by the caller.
    VLOG(1) << "Before adding new entry for key " << cache_key
            << " with session_name( " << session_name << ");"
            << "; cache is " << cache_.size() << " entries ("
            << cache_size_ + marked_for_eviction_size_ << " bytes), "
            << " marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_size_ << " bytes).";
    // Note that InitializeEntry() will Release/Reacquire mu_.
    entry = InitializeEntry(cache_key, compile_function, subgraph_key);
    TRACELITERAL("TPU host compilation cache: compilation done.");
    LOG(INFO) << strings::StrCat(
        "TPU host compilation cache: compilation done for cache_key(",
        cache_key, "), session_name(", session_name, "), subgraph_key(",
        subgraph_key.debug_string, ")");
    // If session_name is present, log some additional stats related to HBM
    // here, so that they can be associated directly to the session.
    if (!session_name.empty()) {
      entry->tpu_program_group->LogProgramMemorySummary();
    }
  } else {
    tpu::TpuCompilationCacheMetrics::IncrementCacheLookupCount(
        /*is_cache_hit=*/true, session_name);
    const string msg =
        strings::StrCat("TPU host compilation cache hit: cache_key(", cache_key,
                        "), session_name(", session_name, ")");
    TRACESTRING(msg);
    VLOG(1) << msg;
    VLOG(1) << "Before refreshing entry for key " << cache_key
            << " with session_name( " << session_name << "); cache is "
            << cache_.size() << " entries ("
            << cache_size_ + marked_for_eviction_size_ << " bytes), "
            << " marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_size_ << " bytes).";
    entry = iter->second;
    // Make a new reference that is owned by the caller.
    entry->Ref();
    // Block if necessary until the subgraph has been initialized.
    mu_.Await(absl::Condition(
        +[](CompiledSubgraph* e) { return e->initialized; }, entry));
  }

  // Let the caller know the uid of the entry.
  *uid = entry->uid;
  // Let the caller know the keys for each of the cached protos.
  *proto_key = entry->proto_key;
  *may_modify_variables = entry->tpu_program_group->may_modify_variables();
  *hlo_metadatas = entry->tpu_program_group->hlo_metadatas();

  // If the caller didn't supply a per_step_ref_holder then the caller is going
  // to manually release the reference later via a call to Release().
  if (per_step_ref_holder == nullptr) {
    ++entry->external_references;
  } else {
    // The caller wants its reference to be handed off to a per-step holder that
    // will discard the reference when the step completes.
    RefHolder* cast_ref_holder =
        tensorflow::down_cast<RefHolder*>(per_step_ref_holder);
    CHECK_NE(cast_ref_holder, nullptr);
    cast_ref_holder->AddRef(entry);
  }

  // Remove the old LRU-table entry if it wasn't already marked for eviction.
  auto erased = entries_by_last_use_.erase(entry->last_use);
  // Update the LRU table indicating this entry is the most recently used.
  entry->last_use = use_counter_++;
  entries_by_last_use_[entry->last_use] = entry;
  if (erased == 0) {
    // The entry had been marked for eviction, or is newly created.
    LookupEntryMarkedForEviction(entry, removed_entries);
  }

  // Log a little more verbosely when a key is added.
  if (VLOG_IS_ON(1) || is_new_key) {
    LOG(INFO) << "After " << (is_new_key ? "adding" : "refreshing")
              << " entry for key " << cache_key << " with session_name "
              << session_name << " cache is " << cache_.size() << " entries ("
              << cache_size_ + marked_for_eviction_size_ << " bytes), "
              << " marked for eviction "
              << (cache_.size() - entries_by_last_use_.size()) << " entries ("
              << marked_for_eviction_size_ << " bytes).";
  }
  return entry->initialization_status;
}

Status TpuCompilationCacheInterface::GetKeysFromUid(int64 uid,
                                                    std::vector<string>* keys) {
  keys->clear();

  absl::MutexLock lock(&mu_);
  const auto iter = entries_by_uid_.find(uid);
  if (iter == entries_by_uid_.end()) {
    return errors::NotFound("No subgraph found for uid ", uid);
  }
  *keys = iter->second->proto_key;
  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
