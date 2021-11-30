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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.h"

#include "grpcpp/security/credentials.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_support.h"

namespace tensorflow {
namespace tpu {
namespace {

#if defined(LIBTPU_ON_GCE)
using ResponseType = GetTpuProgramResponseExternal;
#else
using ResponseType = GetTpuProgramResponse;
#endif

static constexpr absl::Duration kProtoTimeout = absl::Minutes(15);
static gpr_timespec TimeToGprTimespec(absl::Time time) {
  if (time == absl::InfiniteFuture()) {
    return gpr_inf_future(GPR_CLOCK_REALTIME);
  }
  if (time == absl::InfinitePast()) {
    return gpr_inf_past(GPR_CLOCK_REALTIME);
  }

  gpr_timespec spec;
  timespec t = absl::ToTimespec(time);
  spec.tv_sec = t.tv_sec;
  spec.tv_nsec = static_cast<int32_t>(t.tv_nsec);
  spec.clock_type = GPR_CLOCK_REALTIME;
  return spec;
}
}  // namespace
TpuCompilationCacheRpcLookup::TpuCompilationCacheRpcLookup(
    const std::string& server_address, int64_t max_cache_size)
    : max_cache_size_(max_cache_size) {
  // Ensure that large TPU program can get sent over the channel.
  ::grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());
  auto channel =
      ::grpc::CreateCustomChannel(absl::StrCat("dns:///", server_address),
                                  CreateChannelCredentials(), args);
  stub_ = tpu::grpc::TpuCompilationCacheService::NewStub(channel);
  VLOG(1) << "Created RPC lookup cache size " << max_cache_size_ << " bytes.";
}

Status TpuCompilationCacheRpcLookup::Lookup(
    const std::string& proto_key,
    std::unique_ptr<CompilationCacheEntryRef>* entry,
    tpu::CompilationCacheFetchTarget fetch_target) {
  profiler::TraceMe proto_lookup_traceme("Remote TPU proto cache lookup",
                                         /*level=*/2);
  entry->reset();
  std::shared_ptr<CacheEntry> cache_entry;
  // Keep a reference to CacheEntry objects evicted from the cache so that the
  // potential deletion happens outside the lock upon method exit.
  std::vector<std::shared_ptr<CacheEntry>> removed_entries;

  std::string local_proto_key = absl::StrCat(
      proto_key, "_", tpu::CompilationCacheFetchTarget_Name(fetch_target));

  {
    absl::MutexLock lock(&mu_);
    auto iter = cache_.find(local_proto_key);
    if (iter == cache_.end()) {
      tpu::GetTpuProgramRequest request;
      request.set_key(proto_key);
      request.set_fetch_target(fetch_target);
      TF_RETURN_IF_ERROR(
          RemoteLookupLocked(local_proto_key, request, &cache_entry));
    } else {
      VLOG(1) << "Found key " << local_proto_key << " in local proto cache.";
      cache_entry = iter->second;
      auto erased = entries_by_last_use_.erase(cache_entry->last_use);
      CHECK_EQ(erased, 1);
    }
    PostLookupLocked(&cache_entry, entry, &removed_entries);
  }
  return Status::OK();
}

Status TpuCompilationCacheRpcLookup::Lookup(
    int64_t uid, int proto_index,
    std::unique_ptr<CompilationCacheEntryRef>* entry,
    tpu::CompilationCacheFetchTarget fetch_target) {
  profiler::TraceMe proto_lookup_traceme("Remote TPU proto cache lookup by uid",
                                         /*level=*/2);
  entry->reset();
  std::shared_ptr<CacheEntry> cache_entry;
  // Keep a reference to CacheEntry objects evicted from the cache so that the
  // potential deletion happens outside the lock upon method exit.
  std::vector<std::shared_ptr<CacheEntry>> removed_entries;

  // Make a string key so that we can uniformly store cached entries under
  // string keys whether they are looked up by proto_key or uid+index. The
  // expectation is that any given executable will only ever be looked up
  // *either* by proto_key *or* by uid+index, so we are not concerned that the
  // same proto could be placed in the cache twice if it is looked up by both
  // methods.
  std::string local_proto_key =
      absl::StrCat(" _ ", uid, ":", proto_index, "_",
                   tpu::CompilationCacheFetchTarget_Name(fetch_target));
  {
    absl::MutexLock lock(&mu_);
    auto iter = cache_.find(local_proto_key);
    if (iter == cache_.end()) {
      tpu::GetTpuProgramRequest request;
      tpu::TpuCompilationUidAndIndex* uid_and_index =
          request.mutable_uid_and_index();
      uid_and_index->set_uid(uid);
      uid_and_index->set_proto_index(proto_index);
      request.set_fetch_target(fetch_target);
      TF_RETURN_IF_ERROR(
          RemoteLookupLocked(local_proto_key, request, &cache_entry));
    } else {
      VLOG(1) << "Found uid " << uid << " and index " << proto_index
              << " in local proto cache.";
      cache_entry = iter->second;
      auto erased = entries_by_last_use_.erase(cache_entry->last_use);
      CHECK_EQ(erased, 1);
    }
    PostLookupLocked(&cache_entry, entry, &removed_entries);
  }
  return Status::OK();
}

Status TpuCompilationCacheRpcLookup::RemoteLookupLocked(
    const std::string& local_proto_key,
    const tpu::GetTpuProgramRequest& request,
    std::shared_ptr<CacheEntry>* cache_entry) {
  profiler::TraceMe proto_lookup_traceme("Remote TPU proto cache fetch",
                                         /*level=*/2);
  // Perform the RPC while holding the lock unless it is demonstrated that
  // this causes a performance problem.
  ::grpc::ClientContext client_context;
  client_context.set_deadline(TimeToGprTimespec(::absl::Now() + kProtoTimeout));
  client_context.set_compression_algorithm(GRPC_COMPRESS_GZIP);

  ResponseType response;
  Status s =
      FromGrpcStatus(stub_->GetTpuProgram(&client_context, request, &response));
  VLOG(1) << "Looked up key " << local_proto_key
          << " in remote subgraph cache status " << s;
  TF_RETURN_IF_ERROR(s);

  TF_RETURN_IF_ERROR(DeserializeRpcResponseToCacheEntry(
      local_proto_key, &response, cache_entry));
  cache_.emplace(local_proto_key, (*cache_entry));
  cache_size_ += (*cache_entry)->size;

  return Status::OK();
}

void TpuCompilationCacheRpcLookup::PostLookupLocked(
    std::shared_ptr<CacheEntry>* cache_entry,
    std::unique_ptr<CompilationCacheEntryRef>* entry,
    std::vector<std::shared_ptr<CacheEntry>>* removed_entries) {
  (*cache_entry)->last_use = use_counter_++;
  entries_by_last_use_[(*cache_entry)->last_use] = cache_entry->get();
  *entry =
      std::unique_ptr<CompilationCacheEntryRef>(new CacheWrapper(*cache_entry));

  // Evict overflowing entries if necessary, but never evict the most recently
  // used entry.
  while (entries_by_last_use_.size() > 1 && cache_size_ > max_cache_size_) {
    auto entry_to_evict = entries_by_last_use_.begin()->second;
    entries_by_last_use_.erase(entry_to_evict->last_use);
    CHECK_GE(cache_size_, entry_to_evict->size);
    cache_size_ -= entry_to_evict->size;
    // Delete the cache's reference to the entry, though clients may still be
    // holding onto references. We use 'removed_entries' to delay the possible
    // CacheEntry destruction until the mu_ lock is released.
    auto entry_to_evict_it = cache_.find(entry_to_evict->key);
    CHECK(entry_to_evict_it != cache_.end())
        << "Missing entry key: " << entry_to_evict->key;
    removed_entries->push_back(entry_to_evict_it->second);
    cache_.erase(entry_to_evict_it);
  }
}

std::string TpuCompilationCacheRpcLookup::DebugString() const {
  return "TpuCompilationCacheRpcLookup";
}
}  // namespace tpu
}  // namespace tensorflow
