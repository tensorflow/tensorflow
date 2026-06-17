/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_REQUESTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_REQUESTS_H_

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"

namespace xla::gpu {

// Allows collective operations to request a scratch memory.
class ScratchMemoryRequests {
 public:
  void RequestScratchMemory(GpuCliqueKey key, size_t size) {
    if (!allocations_.contains(key)) {
      allocations_[key] = size;
    } else {
      allocations_[key] = std::max(allocations_[key], size);
    }
  }

  size_t Size() const { return allocations_.size(); }

  std::vector<std::pair<GpuCliqueKey, size_t>> OrderedRequests() const {
    std::vector<std::pair<GpuCliqueKey, size_t>> sorted_allocations(
        allocations_.begin(), allocations_.end());
    std::sort(sorted_allocations.begin(), sorted_allocations.end());
    return sorted_allocations;
  }

 private:
  absl::flat_hash_map<GpuCliqueKey, size_t> allocations_;
};

};  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_REQUESTS_H_
