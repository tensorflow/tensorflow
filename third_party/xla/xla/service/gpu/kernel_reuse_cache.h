/*Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/launch_dimensions.h"

namespace xla {
namespace gpu {

// Caches identical Kernels for deduplication.
// Thread-compatible.
class KernelReuseCache {
 public:
  struct Entry {
    std::string kernel_name;
    LaunchDimensions launch_dimensions;
    int64_t shmem_bytes;
  };

  // Retrieves the cache entry for the given computation, or generates it using
  // the given generator function and stores it in the cache.
  std::pair<Entry, bool /*was_cached*/> Get(
      const HloComputation* fused_computation,
      absl::Span<const KernelArgument> kernel_arguments,
      absl::string_view discriminator, const std::function<Entry()>& generator);

  // Like `Get`, but for generator functions that can fail.
  std::pair<StatusOr<Entry>, bool /*was_cached*/> GetWithStatus(
      const HloComputation* fused_computation,
      absl::Span<const KernelArgument> kernel_arguments,
      absl::string_view discriminator,
      const std::function<StatusOr<Entry>()>& generator);

 private:
  absl::flat_hash_map<std::string /*fingerprint*/, Entry> cache_;
};

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
