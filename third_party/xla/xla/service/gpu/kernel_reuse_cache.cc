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

#include <functional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {
namespace {

// Calculates a fingerprint of the kernel arguments, which can be used for
// checking reusability.
//
// For example 2 arguments that are aligned to 16 bytes, aliased and also
// written by the kernel will be represented as "16aw,16aw".
//
// Overlapping arguments are only marked aliased, if at least one of them is
// written and their buffers are not exactly the same. If 2 arguments'
// buffers are exactly the same, then they are not marked aliased, but marked
// as duplicates, for example like this: "16,=0,16w,=2". The example means
// that the 1st argument is the same as the 0th and the 3rd is the same as
// the 2nd. These duplicated parameters are passed to the kernel only once.
std::string GetArgumentFingerprint(
    absl::Span<const emitters::KernelArgument> kernel_arguments) {
  return absl::StrJoin(kernel_arguments, ",",
                       [](std::string* s, const emitters::KernelArgument& arg) {
                         if (arg.first_with_same_slice().has_value()) {
                           absl::StrAppend(s, "=",
                                           arg.first_with_same_slice().value());
                           return;
                         }
                         absl::StrAppend(s, arg.alignment());
                         if (arg.aliased()) {
                           absl::StrAppend(s, "a");
                         }
                         if (arg.written()) {
                           absl::StrAppend(s, "w");
                         }
                       });
}

}  // namespace

std::string GetComputationFingerprint(
    const HloComputation* fused_computation,
    absl::Span<const emitters::KernelArgument> kernel_arguments,
    absl::string_view discriminator) {
  // We have to print constants, because otherwise we would accidentally reuse
  // kernels which have different builtin constants.
  //
  // It is not a problem to recursively print sub-computations, because we don't
  // have them at this point.
  auto print_options = HloPrintOptions::Fingerprint()
                           .set_print_only_essential_constants(false)
                           .set_print_operand_shape(false);

  return absl::StrCat(discriminator, "(",
                      GetArgumentFingerprint(kernel_arguments), ")",
                      fused_computation->ToString(print_options));
}

absl::Status KernelReuseCache::Load(const CompilationCacheProto& proto) {
  for (const auto& [name, entry] : proto.entries()) {
    std::optional<se::ClusterDim> cluster_dim;
    if (entry.has_cluster_dim()) {
      cluster_dim =
          se::ClusterDim{entry.cluster_dim().x(), entry.cluster_dim().y(),
                         entry.cluster_dim().z()};
    }
    TF_RET_CHECK(
        cache_
            .insert(
                {entry.fingerprint(),
                 Entry{name,
                       LaunchDimensions{
                           entry.launch_dimensions().num_blocks(),
                           entry.launch_dimensions().num_threads_per_block()},
                       cluster_dim, entry.shmem_bytes(), entry.binary()}})
            .second);
  }

  return absl::OkStatus();
}

CompilationCacheProto KernelReuseCache::Export() const {
  CompilationCacheProto proto;
  for (const auto& [fingerprint, cache_entry] : cache_) {
    if (!hits_.contains(fingerprint)) {
      VLOG(5) << "Not exporting unused " << cache_entry.kernel_name;
      continue;
    }
    auto [it, inserted] = proto.mutable_entries()->emplace(
        cache_entry.kernel_name, CompilationCacheEntryProto{});
    CHECK(inserted) << cache_entry.kernel_name;
    CompilationCacheEntryProto& proto_entry = it->second;
    proto_entry.set_fingerprint(fingerprint);
    LaunchDimensionsProto launch_dimensions_proto;
    launch_dimensions_proto.set_num_blocks(
        cache_entry.launch_dimensions.num_blocks());
    launch_dimensions_proto.set_num_threads_per_block(
        cache_entry.launch_dimensions.num_threads_per_block());
    *proto_entry.mutable_launch_dimensions() = launch_dimensions_proto;
    if (cache_entry.cluster_dim.has_value()) {
      ClusterDimProto cluster_dim_proto;
      cluster_dim_proto.set_x(cache_entry.cluster_dim->x);
      cluster_dim_proto.set_y(cache_entry.cluster_dim->y);
      cluster_dim_proto.set_z(cache_entry.cluster_dim->z);
      *proto_entry.mutable_cluster_dim() = cluster_dim_proto;
    }
    proto_entry.set_shmem_bytes(cache_entry.shmem_bytes);
    proto_entry.set_binary(cache_entry.binary);
  }
  return proto;
}

absl::Status UpdateDiskKernelCache(
    absl::string_view path, const bool do_append,
    const CompilationCacheProto& current_cache,
    absl::Span<const KernelReuseCache::NamedBinary> binaries_to_cache) {
  CompilationCacheProto disk_cache;
  if (do_append) {
    std::string serialized;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                             std::string(path), &serialized));
    if (!disk_cache.ParseFromString(std::string(serialized))) {
      return Internal("Failed to parse serialized CompilationCacheProto.");
    }
  }
  auto entries = disk_cache.mutable_entries();
  int stored_kernel_count = 0;
  for (const auto& [name, binary] : binaries_to_cache) {
    auto it_current = current_cache.entries().find(name);
    TF_RET_CHECK(it_current != current_cache.entries().end());
    auto [it_disk, inserted] = entries->insert({name, it_current->second});
    TF_RET_CHECK(inserted);
    TF_RET_CHECK(!binary.empty());
    it_disk->second.set_binary(reinterpret_cast<const char*>(binary.data()),
                               binary.size());
    VLOG(5) << "Cached kernel: " << name << ": " << binary.size();
    ++stored_kernel_count;
  }
  if (stored_kernel_count > 0) {
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(),
                                              std::string(path),
                                              disk_cache.SerializeAsString()));
    VLOG(2) << "Stored " << stored_kernel_count << " / "
            << binaries_to_cache.size() << " kernels in the cache file.";
  }
  return absl::OkStatus();
}

std::pair<absl::StatusOr<const KernelReuseCache::Entry*>, bool>
KernelReuseCache::GetWithStatus(
    const HloComputation* fused_computation,
    absl::Span<const emitters::KernelArgument> kernel_arguments,
    absl::string_view discriminator,
    const std::function<absl::StatusOr<KernelReuseCache::Entry>()>& generator) {
  std::string fingerprint = GetComputationFingerprint(
      fused_computation, kernel_arguments, discriminator);
  VLOG(4) << "Fingerprint: ";
  XLA_VLOG_LINES(4, fingerprint);
  return GetWithStatus(std::move(fingerprint), generator);
}

std::pair<absl::StatusOr<const KernelReuseCache::Entry*>, bool>
KernelReuseCache::GetWithStatus(
    std::string fingerprint,
    const std::function<absl::StatusOr<KernelReuseCache::Entry>()>& generator) {
  hits_.insert(fingerprint);
  auto it = cache_.find(fingerprint);
  if (it != cache_.end()) {
    return {&it->second, /*was_cached=*/true};
  }

  absl::StatusOr<Entry> entry = generator();
  if (entry.ok()) {
    it =
        cache_.insert({std::move(fingerprint), std::move(entry.value())}).first;
    return {&it->second, /*was_cached=*/false};
  }

  return {entry.status(), /*was_cached=*/false};
}

}  // namespace gpu
}  // namespace xla
