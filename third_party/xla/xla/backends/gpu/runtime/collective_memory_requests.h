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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_REQUESTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_REQUESTS_H_

#include <cstddef>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {

// Collective thunks (including collective FFI calls) can request certain
// argument and result buffers to be symmetric on the given GPU clique. XLA
// runtime is responsible for collecting such requests during the prepare stage
// and creating symmetric memories (see `xla::SymmetricMemory` for details) for
// all requested allocations during the initialize stage.
class CollectiveMemoryRequests {
 public:
  explicit CollectiveMemoryRequests(const BufferAllocations& buffers);

  // A set of buffer allocations that must have a corresponding symmetric memory
  // on the given clique. Allocations must be allocated from a compatible
  // collective memory allocator (typically have a memory space S(1) in the HLO
  // program).
  //
  // We use an synthethic id that is generated automatically when symmetric
  // allocations is requested that guarantees determenisitc iteration order
  // across all ranks. See `collective_clique_requests` for more details.
  struct SymmetricAllocations {
    size_t id;
    GpuCliqueKey key;
    absl::btree_set<BufferAllocation::Index> allocations;
  };

  // Adds a request to make the given allocation symmetric on the given clique.
  absl::Status RequestSymmetricAllocation(const GpuCliqueKey& clique_key,
                                          BufferAllocation::Index allocation);

  // Adds a request to make the given address range symmetric on the given
  // clique. If address does not correcpond to any of the buffer allocations in
  // the `buffers_`, it will return an error.
  absl::Status RequestSymmetricAddress(const GpuCliqueKey& clique_key,
                                       const se::DeviceAddressBase& addr);

  // Returns all cliques that have symmetric allocation requests.
  std::vector<GpuCliqueKey> RequestedCliques() const;

  // Returns all requested symmetric allocations ordered by clique.
  std::vector<SymmetricAllocations> OrderedSymmetricAllocations() const;

  size_t size() const { return allocations_.size(); }

 private:
  const BufferAllocations& buffers_;
  absl::flat_hash_map<GpuCliqueKey, SymmetricAllocations> allocations_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_REQUESTS_H_
