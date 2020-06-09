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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_external.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_metrics.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_program.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/kernels/trace_util.h"

namespace tensorflow {
namespace tpu {

namespace {

using CompilationEntry = TpuCompilationCacheInterface::CompilationEntry;

int64 get_uid() {
  uint64 unsigned_rand = random::New64() & INT64_MAX;
  return static_cast<int64>(unsigned_rand);
}

void PopulateEntry(const std::string& key, CompilationEntry* entry,
                   std::unique_ptr<TpuProgram> tpu_program) {
  // Make the unique keys for each cached proto.
  for (int i = 0; i < tpu_program->program_count(); ++i) {
    entry->proto_key.push_back(ProtoKeyForComputation(key, i));
  }

  entry->tpu_program = std::move(tpu_program);
  entry->initialized = true;
}

std::string ConstructCompilationCacheKey(const TpuCompilationCacheKey& key) {
  if (!key.has_guaranteed_const) {
    return key.prefix;
  }
  return absl::StrCat(key.prefix, "|", key.session_handle, "|",
                      key.guaranteed_const_fingerprint());
}

// Return fingerprint_in_metadata if it's not empty; otherwise read input tensor
// data to compute the fingerprint.
std::string GuaranteedConstFingerprint(
    const string& fingerprint_in_metadata,
    const OpInputList& guaranteed_constants) {
  if (fingerprint_in_metadata.empty()) {
    uint64_t fingerprint = 0;
    for (const auto& constant : guaranteed_constants) {
      fingerprint = TpuCompile_CreateGuaranteedConstFingerprint(
          fingerprint, constant.tensor_data().data(),
          constant.tensor_data().size());
    }
    return std::to_string(fingerprint);
  } else {
    return fingerprint_in_metadata;
  }
}

std::string CreateShapePrefix(
    const std::vector<tensorflow::TensorShape>& dynamic_shapes) {
  std::string shapes_prefix;
  for (const TensorShape& shape : dynamic_shapes) {
    for (int64 size : shape.dim_sizes()) {
      absl::StrAppend(&shapes_prefix, size, ",");
    }
    absl::StrAppend(&shapes_prefix, ";");
  }
  return shapes_prefix;
}

// Include compilation configurations of the arguments that are not captured
// by the called graph.
std::string CreateConfigPrefix(const TPUCompileMetadataProto& metadata) {
  std::string config_prefix;
  for (const auto& arg : metadata.args()) {
    if (arg.is_same_data_across_replicas()) {
      absl::StrAppend(&config_prefix, ":s");
      // Same.
    } else {
      // Different.
      absl::StrAppend(&config_prefix, ":");
    }
    if (arg.enable_xla_sharding() ==
        tpu::TPUCompileMetadataProto::Arg::ALLOWED) {
      // Enabled.
      absl::StrAppend(&config_prefix, "e");
    }
    if (arg.unrestricted_layout()) {
      // Unrestricted.
      absl::StrAppend(&config_prefix, ":u");
    }
    absl::StrAppend(&config_prefix, ",type(", arg.dtype(), ")");
    if (arg.has_shape()) {
      absl::StrAppend(&config_prefix, ",shape(");
      for (const auto& dim : arg.shape().dim()) {
        absl::StrAppend(&config_prefix, dim.size(), ",");
      }
      absl::StrAppend(&config_prefix, ")");
    }
  }
  return config_prefix;
}

}  // namespace

TpuCompilationCacheInterface::TpuCompilationCacheInterface(
    int64_t max_cache_size)
    : max_cache_size_(max_cache_size) {
  if (max_cache_size < 0) {
    LOG(FATAL) << "`max_cache_size` value must be greater than equal to 0";
  }
  VLOG(1) << "Created compilation cache size " << max_cache_size_ << " bytes.";
}

TpuCompilationCacheInterface::~TpuCompilationCacheInterface() {
  VLOG(1) << "TpuCompilationCacheInterface::~TpuCompilationCacheInterface()";
  // A buggy client may be holding onto a reference, or a client might have
  // crashed while holding onto a reference. In either case, discard all
  // outstanding client references to avoid leaking storage.
  for (const auto& entry : entries_by_uid_) {
    while (entry.second->external_references > 0) {
      TF_CHECK_OK(Release(entry.first));
    }
  }
  while (!entries_by_last_use_.empty()) {
    UnloadAndDestroy(MarkOldestEntryForEviction());
  }
  // By the time the cache is deleted all reference holders should have already
  // been deleted, since they were holding references to the cache. So all
  // entries should be gone at this point.
  CHECK_EQ(cache_store_.size(), 0);
  CHECK_EQ(entries_by_uid_.size(), 0);
  CHECK_EQ(entries_by_proto_key_.size(), 0);
  CHECK_EQ(cache_size_, 0);
  CHECK_EQ(marked_for_eviction_size_, 0);
}

std::string TpuCompilationCacheInterface::FindCacheKey(
    const TpuCompilationCacheKey& subgraph_key) const {
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
  VLOG(1) << "No matching cache key found for key "
          << ConstructCompilationCacheKey(subgraph_key);
  return "";
}

void TpuCompilationCacheInterface::InsertEntry(
    const std::string& cache_key, const TpuCompilationCacheKey& subgraph_key,
    CompilationEntry* entry) {
  entry->parent = this;
  entry->subgraph_key = cache_key;
  entry->uid = get_uid();
  TpuCompilationCacheMetrics::SetCacheEntryCount(cache_store_.size());
  entry->cache_entry_debug_string = subgraph_key.prefix;
  VLOG(1) << "Cache Initializing Entry Session Debug "
          << entry->cache_entry_debug_string;

  if (!subgraph_key.has_guaranteed_const) {
    return;
  }
  session_key_map_.insert(std::make_pair(
      strings::StrCat(subgraph_key.prefix, subgraph_key.session_handle),
      cache_key));
  fingerprint_key_map_.insert(std::make_pair(
      strings::StrCat(subgraph_key.prefix,
                      subgraph_key.guaranteed_const_fingerprint()),
      cache_key));
}

CompilationEntry* TpuCompilationCacheInterface::InitializeEntry(
    const string& key,
    const std::function<Status(TpuProgram*)>& initialize_program,
    const TpuCompilationCacheKey& subgraph_key) {
  CompilationEntry* main_entry = new CompilationEntry();

  // Add the entry to the cache, with size zero since there are no compiled
  // programs in it. Once the subgraph has been compiled,
  // UpdateEntryAfterCompilation will be called to potentially mark old entries
  // that don't fit any more for eviction.
  //
  // At this point there is one reference to entry, which is owned by the caller
  // who created the entry. A second reference, owned by the cache, will be
  // added below since we leave the entry in the 'marked for eviction' state
  // here.
  InsertEntry(key, subgraph_key, main_entry);

  // Initialize the programs outside the lock so that other cache operations
  // can proceed during the (potentially lengthy) initialization.
  Status initialization_status;

  auto tpu_program = absl::make_unique<TpuProgram>();
  {
    mu_.Unlock();
    {
      profiler::TraceMe compile_programs_traceme(
          "TPU compilation cache compile",
          /*level=*/2);
      initialization_status = initialize_program(tpu_program.get());
    }
    mu_.Lock();
  }

  main_entry->initialization_status = initialization_status;

  // Add the entry to the uid index.
  auto uid_inserted = entries_by_uid_.insert(
      std::pair<int64, CompilationEntry*>(main_entry->uid, main_entry));
  CHECK(uid_inserted.second);

  if (initialization_status.ok()) {
    // Compute the entries total size once all members are initialized.
    main_entry->total_size = tpu_program->program_size();
  }

  // TODO(henrytan): handle sharding/unsharding.
  PopulateEntry(key, main_entry, std::move(tpu_program));

  for (int64 i = 0; i < main_entry->proto_key.size(); ++i) {
    auto entry_inserted = entries_by_proto_key_.insert(
        std::pair<string, std::pair<CompilationEntry*, int>>(
            main_entry->proto_key[i], std::make_pair(main_entry, i)));
    CHECK(entry_inserted.second);
  }

  // Add the size to marked_for_eviction_size_ since it will be adjusted down
  // again when the newly-created entry gets unmarked.
  marked_for_eviction_size_ += main_entry->total_size;
  return main_entry;
}

/*static*/ TpuCompilationCacheKey
TpuCompilationCacheInterface::CreateCompilationCacheKey(
    absl::string_view function_name, uint64 function_library_fingerprint,
    absl::string_view mlir_module,
    const tensorflow::OpInputList& guaranteed_constants,
    const std::vector<tensorflow::TensorShape>& dynamic_shapes,
    const tensorflow::tpu::TPUCompileMetadataProto& metadata,
    const TpuMeshStateInterface& mesh_state) {
  VLOG(1) << "FunctionLibraryFingerprint:" << function_library_fingerprint;
  std::string shapes_prefix = CreateShapePrefix(dynamic_shapes);
  VLOG(1) << "shapes_prefix = " << shapes_prefix;
  std::string config_prefix = CreateConfigPrefix(metadata);
  VLOG(1) << "config_prefix = " << config_prefix;
  std::vector<int32_t> flattened_device_ids;
  if (metadata.has_device_assignment()) {
    for (const auto& device :
         metadata.device_assignment().computation_devices()) {
      flattened_device_ids.insert(flattened_device_ids.end(),
                                  device.replica_device_ids().begin(),
                                  device.replica_device_ids().end());
    }
  }
  // TODO(henrytan): return the debug_string.
  const char* prefix =
      TpuCompile_CreateCompilationCacheKey(CompilationCacheKeyProperty{
          config_prefix.data(),
          shapes_prefix.data(),
          function_name.data(),
          mlir_module.data(),
          flattened_device_ids.data(),
          flattened_device_ids.size(),
          guaranteed_constants.size(),
          function_library_fingerprint,
          metadata.num_cores_per_replica(),
          metadata.num_replicas(),
          mesh_state.data(),
      });
  auto buffer_cleanup = gtl::MakeCleanup([prefix]() { delete[] prefix; });
  TpuCompilationCacheKey key;
  key.prefix = prefix;

  // Guaranteed constants can be different across sessions. Use session_handle
  // and guaranteed_const fingerprint to guarantee no collision.
  if (guaranteed_constants.size() > 0) {
    key.has_guaranteed_const = true;
    key.session_handle = metadata.session_handle();
    // Both `metadata` and `guaranteed_constants` lifetime are captured by
    // reference based on the assumption that these variables lifetime is
    // managed through the `TPUCompileOpKernelImpl` that outlives the
    // lifetime of the compilation cache lookups.
    string fingerprint;
    key.guaranteed_const_fingerprint = [&metadata, &guaranteed_constants,
                                        fingerprint]() mutable {
      if (fingerprint.empty()) {
        fingerprint = GuaranteedConstFingerprint(
            metadata.guaranteed_const_fingerprint(), guaranteed_constants);
      }
      return fingerprint;
    };
  }
  return key;
}

TpuCompilationRefHolder* TpuCompilationCacheInterface::MakePerStepRefHolder() {
  return new RefHolder(this);
}

Status TpuCompilationCacheInterface::MarkEntryForEviction(int64 subgraph_uid) {
  profiler::TraceMe key_release_traceme(
      "TPU compilation cache possibly evict uid",
      /*level=*/2);
  CompilationEntry* deleted_entry = nullptr;
  {
    absl::MutexLock lock(&mu_);
    auto iter = entries_by_uid_.find(subgraph_uid);
    if (iter == entries_by_uid_.end()) {
      // If already evicted, return ok.
      return Status::OK();
    }

    // Mark entry for eviction.
    CompilationEntry* subgraph_to_evict = iter->second;
    // If there are external references, should not use this API.
    if (subgraph_to_evict->external_references != 0) {
      return errors::Internal("Subgraph ", subgraph_to_evict->subgraph_key,
                              " external_references greater than zero. Should "
                              "use TpuCompilationCache::Release.");
    }

    VLOG(1) << "Marking " << subgraph_to_evict->subgraph_key << " for eviction";
    entries_by_last_use_.erase(subgraph_to_evict->last_use);
    cache_size_ -= subgraph_to_evict->total_size;
    marked_for_eviction_size_ += subgraph_to_evict->total_size;

    // Evict if refcount exactly one, otherwise only discard cache's reference
    // to the entry while the actual eviction will happen when refholder's
    // references go away.
    deleted_entry = DiscardEntryRef(subgraph_to_evict);

    VLOG(1) << "After possibly evicting entry " << subgraph_uid
            << " refs cache is " << cache_store_.size() << " entries ("
            << cache_size_ + marked_for_eviction_size_
            << " bytes), marked for eviction "
            << (cache_store_.size() - entries_by_last_use_.size())
            << " entries (" << marked_for_eviction_size_ << " bytes).";
  }

  // Unload from device cache if entry is evicted from host cache.
  UnloadAndDestroy(deleted_entry);
  return Status::OK();
}

Status TpuCompilationCacheInterface::Release(int64 subgraph_uid) {
  profiler::TraceMe key_release_traceme("TPU compilation cache release uid",
                                        /*level=*/2);

  CompilationEntry* deleted_entry = nullptr;
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
            << cache_store_.size() << " entries ("
            << cache_size_ + marked_for_eviction_size_
            << " bytes), marked for eviction "
            << (cache_store_.size() - entries_by_last_use_.size())
            << " entries (" << marked_for_eviction_size_ << " bytes).";
  }
  UnloadAndDestroy(deleted_entry);
  return Status::OK();
}

void TpuCompilationCacheInterface::UnloadAndDestroy(CompilationEntry* entry) {
  if (!entry) return;

  CHECK(entry->RefCountIsOne());
  entry->tpu_program->UnloadAndDestroyPrograms();
  entry->Unref();
}

size_t TpuCompilationCacheInterface::RemoveEntry(const string& key) {
  auto erased = cache_store_.erase(key);
  TpuCompilationCacheMetrics::SetCacheEntryCount(cache_store_.size());
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

ABSL_MUST_USE_RESULT CompilationEntry*
TpuCompilationCacheInterface::DiscardEntryRef(CompilationEntry* entry) {
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

void TpuCompilationCacheInterface::DiscardEntryRefs(
    gtl::ArraySlice<CompilationEntry*> entries) {
  std::vector<CompilationEntry*> removed_entries;
  {
    absl::MutexLock lock(&mu_);

    for (auto entry : entries) {
      removed_entries.push_back(DiscardEntryRef(entry));
    }

    VLOG(1) << "After discarding entry refs cache is " << cache_store_.size()
            << " entries (" << cache_size_ + marked_for_eviction_size_
            << " bytes), marked for eviction "
            << (cache_store_.size() - entries_by_last_use_.size())
            << " entries (" << marked_for_eviction_size_ << " bytes).";
  }
  for (auto removed_entry : removed_entries) {
    UnloadAndDestroy(removed_entry);
  }
}

ABSL_MUST_USE_RESULT CompilationEntry*
TpuCompilationCacheInterface::MarkOldestEntryForEviction() {
  CompilationEntry* entry_to_mark = entries_by_last_use_.begin()->second;
  VLOG(1) << "Marking " << entry_to_mark->subgraph_key << " for eviction";
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
    CompilationEntry* entry, std::vector<CompilationEntry*>* removed_entries) {
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

Status TpuCompilationCacheInterface::ToSubEntryRef(
    CompilationCacheEntryRef* entry,
    CompilationCacheFetchTarget fetch_target) const {
  return static_cast<EntryRefImpl*>(entry)->ToSubEntryRef(fetch_target);
}

TpuCompilationCacheInterface::EntryRefImpl::EntryRefImpl(
    TpuCompilationCacheInterface* parent, CompilationEntry* entry, int index)
    : parent_(parent), entry_(entry), index_(index) {
  if (entry_ == nullptr) {
    return;
  }
  if (entry_->main_entry == nullptr) {
    entry_->Ref();
  } else {
    // This is a sharding/unsharding entry nested in a main entry. Only refcount
    // the main entry.
    entry_->main_entry->Ref();
  }
}

TpuCompilationCacheInterface::EntryRefImpl::~EntryRefImpl() {
  if (entry_ == nullptr) {
    return;
  }
  if (entry_->main_entry == nullptr) {
    parent_->DiscardEntryRefs({entry_});
  } else {
    parent_->DiscardEntryRefs({entry_->main_entry});
  }
}

CompilationCacheEntry TpuCompilationCacheInterface::EntryRefImpl::get() {
  if (entry_ == nullptr) {
    // Create an empty entry if the entry is nullptr. This corresponds to
    // non-existing sharding/unsharding entries.
    return CompilationCacheEntry();
  }
  return CompilationCacheEntry(std::move(entry_->tpu_program));
}

Status TpuCompilationCacheInterface::EntryRefImpl::ToSubEntryRef(
    CompilationCacheFetchTarget fetch_target) {
  CompilationEntry* target = nullptr;
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
  // Otherwise, since the refcount is always on the main entry, we don't need
  // ref/unref.
  entry_ = target;
  return Status::OK();
}

Status TpuCompilationCacheInterface::Lookup(
    int64 uid, int proto_index,
    std::unique_ptr<CompilationCacheEntryRef>* entry) {
  entry->reset();

  profiler::TraceMe proto_lookup_traceme(
      "TPU compilation cache proto lookup by uid",
      /*level=*/2);

  absl::MutexLock lock(&mu_);
  const auto iter = entries_by_uid_.find(uid);
  if (iter == entries_by_uid_.end()) {
    return errors::NotFound("No subgraph found for uid ", uid);
  }
  CompilationEntry* cache_entry = iter->second;
  if (proto_index < 0 ||
      proto_index >= cache_entry->tpu_program->program_size()) {
    return errors::NotFound("No proto found for core index ", proto_index,
                            " in subgraph with uid ", uid);
  }
  *entry = std::unique_ptr<CompilationCacheEntryRef>(
      new EntryRefImpl(this, cache_entry, proto_index));
  return Status::OK();
}

Status TpuCompilationCacheInterface::Lookup(
    const string& proto_key, std::unique_ptr<CompilationCacheEntryRef>* entry) {
  entry->reset();

  profiler::TraceMe proto_lookup_traceme("TPU compilation cache proto lookup",
                                         /*level=*/2);

  absl::MutexLock lock(&mu_);
  const auto iter = entries_by_proto_key_.find(proto_key);
  if (iter == entries_by_proto_key_.end()) {
    return errors::NotFound("No proto found for key ", proto_key);
  }
  CompilationEntry* cache_entry = iter->second.first;
  int proto_index = iter->second.second;
  *entry = std::unique_ptr<CompilationCacheEntryRef>(
      new EntryRefImpl(this, cache_entry, proto_index));
  return Status::OK();
}

Status TpuCompilationCacheInterface::CompileIfKeyAbsentHelper(
    const TpuCompilationCacheKey& subgraph_key,
    const SessionMetadata* session_metadata,
    TpuCompilationRefHolder* per_step_ref_holder, int64* uid,
    std::vector<string>* proto_key, std::vector<bool>* may_modify_variables,
    std::vector<CompilationEntry*>* removed_entries,
    std::vector<std::shared_ptr<const xla::HloProto>>* hlo_metadata,
    const std::function<Status(TpuProgram*)>& compile_function) {
  profiler::TraceMe subgraph_lookup_traceme(
      "TPU compilation cache subgraph lookup",
      /*level=*/2);

  // NOTE: In spite of the fact that we use MutexLock, we do not hold the lock
  // for the lifetime of the object, see InitializeEntry() call below.
  absl::MutexLock lock(&mu_);

  std::string cache_key = FindCacheKey(subgraph_key);
  auto iter = cache_store_.find(cache_key);
  bool is_new_key = iter == cache_store_.end();

  const std::string session_name = SessionNameFromMetadata(session_metadata);

  CompilationEntry* entry = nullptr;
  if (is_new_key) {
    cache_key = ConstructCompilationCacheKey(subgraph_key);
    TpuCompilationCacheMetrics::IncrementCacheLookupCount(
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
        for (auto it = cache_store_.begin(); it != cache_store_.end(); ++it) {
          // TODO(henrytan): add DebugKey as cache_entry_debug_string to
          // TpuCompilationCacheKey.
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
            << "; cache is " << cache_store_.size() << " entries ("
            << cache_size_ + marked_for_eviction_size_ << " bytes), "
            << " marked for eviction "
            << (cache_store_.size() - entries_by_last_use_.size())
            << " entries (" << marked_for_eviction_size_ << " bytes).";
    // Note that InitializeEntry() will Release/Reacquire mu_.
    entry = InitializeEntry(cache_key, compile_function, subgraph_key);
    TRACELITERAL("TPU host compilation cache: compilation done.");

    LOG(INFO) << strings::StrCat(
        "TPU host compilation cache: compilation done for cache_key(",
        cache_key, "), session_name(", session_name, ")");
    // If session_name is present, log some additional stats related to HBM
    // here, so that they can be associated directly to the session.
    if (!session_name.empty()) {
      entry->tpu_program->LogProgramMemorySummary();
    }
  } else {
    TpuCompilationCacheMetrics::IncrementCacheLookupCount(true, session_name);
    const string msg =
        strings::StrCat("TPU host compilation cache hit: cache_key(", cache_key,
                        "), session_name(", session_name, ")");
    TRACESTRING(msg);
    VLOG(1) << msg;
    VLOG(1) << "Before refreshing entry for key " << cache_key
            << " with session_name( " << session_name << "); cache is "
            << cache_store_.size() << " entries ("
            << cache_size_ + marked_for_eviction_size_ << " bytes), "
            << " marked for eviction "
            << (cache_store_.size() - entries_by_last_use_.size())
            << " entries (" << marked_for_eviction_size_ << " bytes).";
    entry = iter->second;
    // Make a new reference that is owned by the caller.
    entry->Ref();
    // Block if necessary until the subgraph has been initialized.
    mu_.Await(absl::Condition(
        +[](CompilationEntry* e) { return e->initialized; }, entry));
  }

  // Let the caller know the uid of the entry.
  *uid = entry->uid;
  // Let the caller know the keys for each of the cached protos.
  *proto_key = entry->proto_key;
  *may_modify_variables = entry->tpu_program->may_modify_variables();
  *hlo_metadata = entry->hlo_metadata;

  // If the caller didn't supply a per_step_ref_holder then the caller is going
  // to manually release the reference later via a call to Release().
  if (per_step_ref_holder == nullptr) {
    ++entry->external_references;
  } else {
    // The caller wants its reference to be handed off to a per-step holder that
    // will discard the reference when the step completes.
    RefHolder* cast_ref_holder = static_cast<RefHolder*>(per_step_ref_holder);
    TF_RET_CHECK(cast_ref_holder != nullptr);
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
              << session_name << " cache is " << cache_store_.size()
              << " entries (" << cache_size_ + marked_for_eviction_size_
              << " bytes), "
              << " marked for eviction "
              << (cache_store_.size() - entries_by_last_use_.size())
              << " entries (" << marked_for_eviction_size_ << " bytes).";
  }
  return entry->initialization_status;
}

tensorflow::Status TpuCompilationCacheInterface::CompileIfKeyAbsent(
    const TpuCompilationCacheKey& cache_key,
    const tensorflow::SessionMetadata* session_metadata,
    TpuCompilationRefHolder* per_step_ref_holder, int64* uid,
    std::vector<string>* proto_key, std::vector<bool>* may_modify_variables,
    std::vector<std::shared_ptr<const xla::HloProto>>* hlo_metadata,
    const std::function<tensorflow::Status(TpuProgram*)>& compile_function) {
  std::vector<CompilationEntry*> removed_entries;
  auto status = CompileIfKeyAbsentHelper(
      cache_key, session_metadata, per_step_ref_holder, uid, proto_key,
      may_modify_variables, &removed_entries, hlo_metadata, compile_function);
  for (auto entry : removed_entries) {
    UnloadAndDestroy(entry);
  }
  return status;
}

}  // namespace tpu
}  // namespace tensorflow
