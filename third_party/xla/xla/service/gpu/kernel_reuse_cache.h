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
#ifndef XLA_SERVICE_GPU_KERNEL_REUSE_CACHE_H_
#define XLA_SERVICE_GPU_KERNEL_REUSE_CACHE_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla {
namespace gpu {

// Caches identical Kernels for deduplication.
// Thread-compatible.
class KernelReuseCache {
 public:
  struct Entry {
    std::string kernel_name;
    LaunchDimensions launch_dimensions;
    std::optional<se::ClusterDim> cluster_dim;
    int64_t shmem_bytes = 0;
    std::string binary;
  };
  struct NamedBinary {
    std::string name;
    std::vector<uint8_t> binary;
  };

  absl::Status Load(const CompilationCacheProto& proto);
  // Exporting skips kernels that were loaded but not used during emission.
  // See comment for hits_ below.
  CompilationCacheProto Export() const;
  bool IsEmpty() const { return cache_.empty(); }
  void Clear() {
    cache_.clear();
    hits_.clear();
  }

  // Retrieves the cache entry for the given computation, or generates it using
  // the given generator function and stores it in the cache.
  //
  // The returned pointer is never nullptr.
  //
  // A non-OK status is returned if the entry is not found and the generator
  // failed.
  std::pair<absl::StatusOr<const Entry*>, bool /*was_cached*/> GetWithStatus(
      const HloComputation* fused_computation,
      absl::Span<const KernelArgument> kernel_arguments,
      absl::string_view discriminator,
      const std::function<absl::StatusOr<Entry>()>& generator);

  // Retrieves the cache entry for the given fingerprint, or generates it using
  // the given generator function and stores it in the cache.
  //
  // The returned pointer is never nullptr.
  //
  // A non-OK status is returned if the entry is not found and the generator
  // failed.
  std::pair<absl::StatusOr<const Entry*>, bool /*was_cached*/> GetWithStatus(
      std::string fingerprint,
      const std::function<absl::StatusOr<Entry>()>& generator);

 private:
  absl::flat_hash_map<std::string /*fingerprint*/, Entry> cache_;
  // Track which fingerprints are in use. Unused ones can appear from loading a
  // partially compatible cache file. These should not be exported to avoid
  // linking the corresponding kernels later.
  absl::flat_hash_set<std::string> hits_;
};

// Add kernels to the cache file. Binaries are taken from binaries_to_cache,
// all other kernel properties are taken from current_cache.
// do_append makes an existing file be loaded first.
absl::Status UpdateDiskKernelCache(
    absl::string_view path, bool do_append,
    const CompilationCacheProto& current_cache,
    absl::Span<const KernelReuseCache::NamedBinary> binaries_to_cache);

// Calculates the fingerprint of a (fused_computation, kernel_arguments,
// discriminator) tuple.
//
// If a given fusion is implemented using multiple kernels, then for each
// kernel we should provide a discriminator, such as "init" and "impl".
//
// If the same fingerprint is returned twice, then we can reuse the kernel
// generated for the first computation.
std::string GetComputationFingerprint(
    const HloComputation* fused_computation,
    absl::Span<const KernelArgument> kernel_arguments,
    absl::string_view discriminator = "");

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_KERNEL_REUSE_CACHE_H_
