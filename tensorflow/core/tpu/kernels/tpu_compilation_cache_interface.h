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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_INTERFACE_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_INTERFACE_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "xla/util.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tpu/kernels/compiled_subgraph.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_metrics.h"
#include "tensorflow/core/tpu/kernels/trace_util.h"

namespace tensorflow {
namespace tpu {

// Base class that holds references to compiled protos so that the protos are
// not garbage-collected before being used by execute ops. Use
// TpuCompilationCache::MakePerStepRefHolder to create an instance of a concrete
// ref holder object.
class CompilationRefHolder : public ResourceBase {
 public:
  ~CompilationRefHolder() override = default;
};

// Wrapper for a cache entry returned by all the TpuCompilationCacheInterface
// `Lookup` methods, and ensures the underlying proto is not garbage-collected
// until the client discards the ptr.
class CompilationCacheEntryRef {
 public:
  CompilationCacheEntryRef();
  CompilationCacheEntryRef(TpuCompilationCacheInterface* parent,
                           CompiledSubgraph* entry, int index);

  virtual ~CompilationCacheEntryRef();

  // Returns a TpuCompilationCacheEntry that should not be used beyond the
  // lifetime of the CompilationCacheEntryRef.
  virtual TpuCompilationCacheEntry get();

  // Mutates this ref to point to the entry's subentry (for
  // sharding/unsharding) or main entry (unchanged) as specified by
  // fetch_target. The refcount is kept unchanged, since we only track the
  // refcount of the main entry. The entry ref needs to point to the main
  // entry before this call.
  //
  // If the requested subentry does not exist, the ref will point to a nullptr
  // entry, and the original entry will be unref'ed.
  virtual Status ToSubEntryRef(CompilationCacheFetchTarget fetch_target);

 protected:
  TpuCompilationCacheInterface* parent_;  // Not owned.
  // A reference to entry_ is acquired in the constructor and released via
  // parent->DiscardEntryRefs in the destructor.
  CompiledSubgraph* entry_;
  // The index of the program in entry_ that is returned by the get method.
  int index_;
};

class TpuCompilationCacheInterface : public ResourceBase {
 public:
  explicit TpuCompilationCacheInterface(int64_t max_cache_size);
  ~TpuCompilationCacheInterface() override;

  // Ensures there is an entry for key present in the cache. By the time
  // CompileIfKeyAbsent returns there is guaranteed to be an entry in the cache
  // for key, and that entry will remain valid at least until
  // per_step_ref_holder is deleted. The first call to CompileIfKeyAbsent with a
  // key that is not in the cache will evaluate compile_function to compute the
  // value to use in the entry. Subsequent calls with the same key will block
  // until compile_function completes. Other cache reads and inserts may proceed
  // on other threads while compile_function is executing. If
  // per_step_ref_holder is nullptr then the caller is responsible for calling
  // Release(subgraph_key) to manually discard its reference to the compiled
  // program, once the caller will not look up the compiled program again.
  //
  // compile_function should compile the subgraph represented by key and fill in
  // one TPUExecutableProto per model-parallel core into its passed argument. It
  // should return OK if and only if compilation succeeds. The executable proto
  // vector will be discarded on non-OK status.
  Status CompileIfKeyAbsent(
      const TpuCompilationCacheKey& subgraph_key,
      const SessionMetadata* session_metadata,
      CompilationRefHolder* per_step_ref_holder, int64_t* uid,
      std::vector<std::string>* proto_key,
      std::vector<std::string>* sharding_key,
      std::vector<bool>* may_modify_variables,
      absl::Span<const xla::HloProto* const>* hlo_metadatas,
      const std::function<Status(TpuProgramGroupInterface*)>& compile_function);

  // Differences between MarkEntryForEviction and Release:
  // There are two modes of managing cache entries:
  // 1) LRU eviction + pinning; 2) manual.
  // We use mode 1) if CompilationRefHolder is provided to CompileIfKeyAbsent.
  // Otherwise it is manual mode (mainly used by XRT).
  // MarkEntryForEviction should only be used in mode 1) to eagerly evict cache
  // entries when callers know that they do not need them anymore.
  // Release should only be used in mode 2) to explicitly remove an entry.

  // Mark the entry indexed by `subgraph_uid` for eviction. This should only be
  // called if per_step_ref_holder was NOT nullptr in the corresponding call to
  // CompileIfKeyAbsent(subgraph_key, ...). Otherwise, use Release(int64
  // subgraph_uid).
  Status MarkEntryForEviction(int64_t subgraph_uid);

  // Manually discards a reference to the compiled subgraph. This should only be
  // called if per_step_ref_holder was nullptr in the corresponding call to
  // CompileIfKeyAbsent(subgraph_key, ...).
  Status Release(int64_t subgraph_uid);

  // Looks up an executable corresponding to the model-parallel core index of
  // the subgraph represented by key. On success a pointer to an EntryRef
  // holding the program is returned in entry.
  Status Lookup(const std::string& proto_key,
                std::unique_ptr<CompilationCacheEntryRef>* entry);

  // Looks up an executable corresponding to the model-parallel core index of
  // the subgraph represented by uid. On success a pointer to an EntryRef
  // holding the program is returned in entry.
  Status Lookup(int64_t uid, int proto_index,
                std::unique_ptr<CompilationCacheEntryRef>* entry);

  // Looks up the subgraph represented by uid, and returns the vector of keys,
  // one per core, corresponding to that subgraph.
  Status GetKeysFromUid(int64_t uid, std::vector<std::string>* keys);

  // Makes a reference holder for this cache, that can be stored in the per-step
  // resource manager and will ensure that compiled entries persist until the
  // end of a step.
  CompilationRefHolder* MakePerStepRefHolder();

  // Convenience method called by ~RefHolder without mu_ held. Calls
  // DiscardEntryRef on every element of entries.
  void DiscardEntryRefs(gtl::ArraySlice<CompiledSubgraph*> entries);

  std::string DebugString() const override { return "TpuCompilationCacheBase"; }

 protected:
  std::string ConstructCompilationCacheKey(const TpuCompilationCacheKey& key) {
    if (!key.has_guaranteed_const) {
      return key.prefix;
    }
    return absl::StrCat(key.prefix, "|", key.session_handle, "|",
                        key.guaranteed_const_fingerprint());
  }

  // Private implementation of the generic CompilationRefHolder that knows about
  // CompiledSubgraph entries.
  class RefHolder : public CompilationRefHolder {
   public:
    explicit RefHolder(TpuCompilationCacheInterface* parent);
    ~RefHolder() override;

    // Adds entry to the list of entries that will be released when the
    // RefHolder is destroyed. Each entry is released via a call to
    // parent_->DiscardEntryRefs.
    void AddRef(CompiledSubgraph* entry);

    std::string DebugString() const override;

   private:
    TpuCompilationCacheInterface* parent_;  // Not owned.
    std::vector<CompiledSubgraph*> entries_;
  };

  // The bulk of implementation of CompileIfKeyAbsent() with the exception
  // of unloading programs that corresponds to possibly removed cache
  // entries. The split helps to manage locking since we prefer to perform
  // unloading without holding extra locks.
  Status CompileIfKeyAbsentHelper(
      const TpuCompilationCacheKey& subgraph_key,
      const SessionMetadata* session_metadata,
      CompilationRefHolder* per_step_ref_holder, int64_t* uid,
      std::vector<std::string>* proto_key,
      std::vector<std::string>* sharding_key,
      std::vector<bool>* may_modify_variables,
      std::vector<CompiledSubgraph*>* removed_entries,
      absl::Span<const xla::HloProto* const>* hlo_metadatas,
      const std::function<Status(TpuProgramGroupInterface*)>& compile_function);

  // This is called by the cache when entry is marked for eviction; by
  // a RefHolder (via DiscardEntryRefs) when a step completes; and by
  // an EntryRefImpl when it is destroyed. Releases one reference to entry
  // if more than 1 remains. If only one reference is left, the entry is removed
  // from cache_ and is returned to the caller; which must eventually call
  // UnloadAndDestroy(). We do not call UnloadAndDestroy within DiscardEntryRef
  // to avoid holding the lock during program unloading.
  ABSL_MUST_USE_RESULT CompiledSubgraph* DiscardEntryRef(
      CompiledSubgraph* entry) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Marks the oldest unmarked entry for eviction. Requires that there is at
  // least one such entry. In case the evicted entry had only 1 reference it
  // is removed from the cache and returned to the caller which must eventually
  // call UnloadAndDestroy.
  ABSL_MUST_USE_RESULT CompiledSubgraph* MarkOldestEntryForEviction()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Updates datastructures to indicate that entry, which had been marked for
  // eviction, has been looked up. This is called by CompileIfKeyAbsent when an
  // entry is newly created, or an entry that has been marked for eviction but
  // not yet evicted is looked up.
  //
  // First the entry is unmarked for eviction, i.e. the cache gains a reference
  // to entry, entry's last_use field is set to be the most recent value of
  // use_counter_ and entries_by_last_use_ is updated accordingly.
  //
  // Next, the size of the cache is examined to see if any other entries need to
  // be marked for eviction now that entry has been unmarked. While the total
  // size of unmarked cached entries is greater than max_cache_size_, entries
  // are marked for eviction in LRU order. The most recently used entry is never
  // marked for eviction, so an entry larger than the max cache size will remain
  // in the cache until it is replaced by something else. In case some entries
  // actually were removed from the cache, they are a returned to the caller via
  // removed_entries. The caller must eventually delete them by calling
  // UnloadAndDestroy.
  void LookupEntryMarkedForEviction(
      CompiledSubgraph* entry, std::vector<CompiledSubgraph*>* removed_entries)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Removes the entry with given key from cache.
  size_t RemoveEntry(const std::string& key) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Inserts the given key and entry to cache.
  void InsertEntry(const std::string& key, CompiledSubgraph* entry)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns the cache key matching given subgraph_key.
  std::string FindCacheKey(const TpuCompilationCacheKey& subgraph_key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Creates a new entry by running initialize_programs and places it in the
  // cache to be looked up by key. The new entry is in the 'marked for eviction'
  // state (not present in entries_by_last_use_) and the caller is expected to
  // call LookupEntryMarkedForEviction after InitializeEntry.
  //
  // **InitializeEntry releases mu_ during the call to initialize_programs.**
  virtual CompiledSubgraph* InitializeEntry(
      const std::string& key,
      const std::function<Status(TpuProgramGroupInterface*)>&
          initialize_programs,
      const TpuCompilationCacheKey& subgraph_key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  // Unloads the program associated with the entry from all local devices
  // and deletes the entry itself. It is assumed no one else has a reference
  // to it and all related keys had already been removed from the cache.
  // The call can perform device IO so no locks should be held while calling it.
  void UnloadAndDestroy(CompiledSubgraph* entry) ABSL_LOCKS_EXCLUDED(mu_);

  // The maximum size of entries that are stored in the cache before entries are
  // marked for eviction.
  const int64_t max_cache_size_;
  // Mutex to protect access to shared resources under multi-threading
  // environment.
  absl::Mutex mu_;
  // The total size of entries that are stored and not marked for eviction.
  int64_t cache_size_ ABSL_GUARDED_BY(mu_) = 0;
  // The total size of entries that are marked for eviction.
  int64_t marked_for_eviction_size_ ABSL_GUARDED_BY(mu_) = 0;
  // The value to assign to the last_use field of the next entry that is looked
  // up.
  int64_t use_counter_ ABSL_GUARDED_BY(mu_) = 0;
  // session_key_map_ and fingerprint_key_map_ are used for looking up the
  // cache_ key matching a given subgraph key. When doing a lookup, check
  // session_key_map_ first to avoid unnecessay fingerprint computation.
  // Map from key prefix + session_handle to a cache_ key.
  absl::node_hash_map<std::string, std::string> session_key_map_
      ABSL_GUARDED_BY(mu_);
  // Map from key prefix + fingerprint to a cache_ key.
  absl::node_hash_map<std::string, std::string> fingerprint_key_map_
      ABSL_GUARDED_BY(mu_);
  // All the subgraph entries that can be looked up in the cache. An entry is
  // marked for eviction iff it is present in cache_ and not in
  // entries_by_last_use_.
  std::unordered_map<std::string, CompiledSubgraph*> cache_
      ABSL_GUARDED_BY(mu_);
  // All the subgraph entries that can be looked up in the cache, indexed by
  // uid.
  absl::flat_hash_map<int64_t, CompiledSubgraph*> entries_by_uid_
      ABSL_GUARDED_BY(mu_);
  // All the protos that can be looked up in the cache, indexed by proto
  // key. The value of the map is a subgraph and the index of the proto compiled
  // for that subgraph.
  std::unordered_map<std::string, std::pair<CompiledSubgraph*, int>>
      entries_by_proto_key_ ABSL_GUARDED_BY(mu_);
  // Map from last_use to entry, used to mark entries for eviction in LRU
  // order. If an entry's last_use counter is not present as a key in
  // entries_by_last_use_ then the entry has been marked for eviction.
  std::map<int64_t, CompiledSubgraph*> entries_by_last_use_
      ABSL_GUARDED_BY(mu_);

  TpuCompilationMetrics tpu_compilation_metrics_;

 private:
  TpuCompilationCacheInterface(const TpuCompilationCacheInterface&) = delete;
  TpuCompilationCacheInterface& operator=(const TpuCompilationCacheInterface&) =
      delete;
};
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_INTERFACE_H_
