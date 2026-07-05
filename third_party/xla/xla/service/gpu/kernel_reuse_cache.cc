/*Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/kernel_reuse_cache.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/codegen/emitters/computation_fingerprint.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/kernel_reuse_cache.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

// Collisions are expected. Perfoming atomic file write.
absl::Status SetFileContent(absl::string_view path, absl::string_view content) {
  tsl::Env* env = tsl::Env::Default();
  absl::InsecureBitGen gen;
  std::string tmppath =
      absl::StrCat(path, ".tmp.",
                   absl::uniform_int_distribution<int>(
                       0, std::numeric_limits<int>::max())(gen));
  if (!env->CreateUniqueFileName(&tmppath, "")) {
    return absl::InternalError(
        absl::StrCat("Unable to create tempfile name for :", path));
  }
  bool has_atomic_move;
  RETURN_IF_ERROR(env->HasAtomicMove(tmppath, &has_atomic_move));
  if (!has_atomic_move) {
    return absl::InternalError(
        absl::StrCat("Atomic move is not supported for :", path));
  }

  std::unique_ptr<tsl::WritableFile> file;
  RETURN_IF_ERROR(env->NewWritableFile(tmppath, &file));
  RETURN_IF_ERROR(file->Append(content));
  RETURN_IF_ERROR(file->Close());

  return env->RenameFile(tmppath, std::string(path));
}
}  // namespace

constexpr int kCacheCompatibilityVersion = 3;

absl::Status KernelReuseCache::Load(const CompilationCacheProto& proto) {
  if (proto.compatibility_version() != kCacheCompatibilityVersion) {
    LOG(WARNING) << "Provided CompilationCacheProto contains no longer "
                    "compatible data and needs to be regenerated.";
    return absl::OkStatus();
  }
  absl::MutexLock lock(m_);
  for (const auto& [name, entry] : proto.entries()) {
    std::optional<se::ClusterDim> cluster_dim;
    if (entry.has_cluster_dim()) {
      cluster_dim =
          se::ClusterDim{entry.cluster_dim().x(), entry.cluster_dim().y(),
                         entry.cluster_dim().z()};
    }
    std::vector<uint8_t> binary(entry.binary().data(),
                                entry.binary().data() + entry.binary().size());
    TF_RET_CHECK(
        cache_
            .insert(
                {entry.fingerprint(),
                 Entry{name,
                       LaunchDimensions{
                           entry.launch_dimensions().num_blocks(),
                           entry.launch_dimensions().num_threads_per_block()},
                       cluster_dim, entry.shmem_bytes(), std::move(binary)}})
            .second);
  }

  return absl::OkStatus();
}

CompilationCacheProto KernelReuseCache::Export() const {
  absl::MutexLock lock(m_);
  CompilationCacheProto proto;
  proto.set_compatibility_version(kCacheCompatibilityVersion);
  for (const auto& [fingerprint, future] : cache_) {
    const absl::StatusOr<Entry>& cache_entry = future.Await();
    if (!cache_entry.ok()) {
      // If a generator failed, the Future will hold an error.
      // We skip exporting these failed entries. Consumers of GetWithStatus
      // are responsible for handling potential errors from the Future.
      continue;
    }
    if (!hits_.contains(fingerprint)) {
      VLOG(5) << "Not exporting unused " << cache_entry->kernel_name;
      continue;
    }
    auto [it, inserted] = proto.mutable_entries()->emplace(
        cache_entry->kernel_name, CompilationCacheEntryProto{});
    CHECK(inserted) << cache_entry->kernel_name;
    CompilationCacheEntryProto& proto_entry = it->second;
    proto_entry.set_fingerprint(fingerprint);
    CompilationCacheEntryProto::LaunchDimensionsProto launch_dimensions_proto;
    launch_dimensions_proto.set_num_blocks(
        cache_entry->launch_dimensions.num_blocks());
    launch_dimensions_proto.set_num_threads_per_block(
        cache_entry->launch_dimensions.num_threads_per_block());
    *proto_entry.mutable_launch_dimensions() = launch_dimensions_proto;
    if (cache_entry->cluster_dim.has_value()) {
      CompilationCacheEntryProto::ClusterDimProto cluster_dim_proto;
      cluster_dim_proto.set_x(cache_entry->cluster_dim->x);
      cluster_dim_proto.set_y(cache_entry->cluster_dim->y);
      cluster_dim_proto.set_z(cache_entry->cluster_dim->z);
      *proto_entry.mutable_cluster_dim() = cluster_dim_proto;
    }
    proto_entry.set_shmem_bytes(cache_entry->shmem_bytes);
    proto_entry.set_binary(absl::string_view(
        reinterpret_cast<const char*>(cache_entry->binary.data()),
        cache_entry->binary.size()));
  }
  return proto;
}

absl::Status UpdateDiskKernelCache(absl::string_view path, const bool do_append,
                                   const CompilationCacheProto& current_cache) {
  CompilationCacheProto disk_cache;
  if (do_append) {
    RETURN_IF_ERROR(tsl::ReadBinaryProto(tsl::Env::Default(), std::string(path),
                                         &disk_cache));
    if (disk_cache.compatibility_version() != kCacheCompatibilityVersion) {
      LOG(WARNING) << "Provided CompilationCacheProto contains no longer "
                      "compatible data and needs to be regenerated.";
      disk_cache.Clear();
    }
  }

  absl::flat_hash_set<std::string> kernel_fingerprints;
  for (const auto& [_, entry] : disk_cache.entries()) {
    kernel_fingerprints.insert(entry.fingerprint());
  }

  int stored_kernel_count = 0;
  for (const auto& [name, entry] : current_cache.entries()) {
    if (kernel_fingerprints.contains(entry.fingerprint())) {
      continue;
    }
    (*disk_cache.mutable_entries())[name] = entry;
    stored_kernel_count++;
  }

  disk_cache.set_compatibility_version(kCacheCompatibilityVersion);
  if (stored_kernel_count) {
    RETURN_IF_ERROR(gpu::SetFileContent(path, disk_cache.SerializeAsString()));
    VLOG(2) << "Stored " << stored_kernel_count
            << " kernels in the cache file.";
  }
  return absl::OkStatus();
}

std::pair<tsl::Future<const KernelReuseCache::Entry*>, bool>
KernelReuseCache::GetWithStatus(
    const HloComputation* fused_computation,
    absl::Span<const emitters::KernelArgument> kernel_arguments,
    absl::string_view discriminator,
    absl::FunctionRef<tsl::Future<KernelReuseCache::Entry>()> generator) {
  std::string fingerprint = emitters::GetComputationFingerprint(
      fused_computation, kernel_arguments, discriminator);
  VLOG(4) << "Fingerprint: ";
  XLA_VLOG_LINES(4, fingerprint);
  return GetWithStatus(std::move(fingerprint), generator);
}

std::pair<tsl::Future<const KernelReuseCache::Entry*>, bool>
KernelReuseCache::GetWithStatus(
    std::string fingerprint,
    absl::FunctionRef<tsl::Future<KernelReuseCache::Entry>()> generator) {
  absl::MutexLock lock(m_);
  hits_.insert(fingerprint);

  // Probe cache before invoking generator() to avoid unnecessary work if entry
  // already exists.
  auto it = cache_.find(fingerprint);
  bool cached = true;
  if (it == cache_.end()) {
    cached = false;
    it = cache_.insert({std::move(fingerprint), generator()}).first;
  }

  return {it->second.Map(
              [](const KernelReuseCache::Entry& entry) { return &entry; }),
          cached};
}

}  // namespace xla::gpu
