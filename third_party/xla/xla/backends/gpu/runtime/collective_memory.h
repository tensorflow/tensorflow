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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/multicast_memory.h"

namespace xla::gpu {

// A collection of symmetric memories acquired based on the collective memory
// requests collected from all thunks at prepare stage.
class CollectiveMemory {
 public:
  // Collective memory maps whole allocation at the given index to one of the
  // supported collective memory types.
  using Key = std::pair<GpuCliqueKey, BufferAllocation::Index>;

  // A multicast memory (shared ownership by all clique members) and a multimem
  // pointer for the current rank.
  struct MulticastMemory {
    std::shared_ptr<se::gpu::MulticastMemory> multicast_memory;
    void* multimem_ptr;
  };

  // Peer memory addresses for all participating ranks.
  struct PeerMemory {
    absl::btree_map<RankId, se::DeviceAddressBase> addrs;
  };

  CollectiveMemory(
      const BufferAllocations& buffers,
      absl::flat_hash_map<Key, std::unique_ptr<SymmetricMemory>> sym_memories,
      absl::flat_hash_map<Key, MulticastMemory> mcast_memories,
      absl::flat_hash_map<Key, PeerMemory> peer_memories);

  // Returns a symmetric memory and offset in that symmetric memory that
  // corresponds to the given buffer allocation index.
  std::pair<SymmetricMemory*, size_t> FindSymmetricMemory(
      const GpuCliqueKey& clique, BufferAllocation::Index allocation) const;

  // Returns a symmetric memory and offset in that symmetric memory that
  // corresponds to the given device address.
  std::pair<SymmetricMemory*, size_t> FindSymmetricMemory(
      const GpuCliqueKey& clique, se::DeviceAddressBase addr) const;

  // Returns a multimem address and offset from that multimem address that
  // corresponds to the given buffer allocation index.
  std::pair<void*, size_t> FindMultimemAddress(
      const GpuCliqueKey& clique, BufferAllocation::Index allocation) const;

  // Returns a multimem address and offset from that multimem address that
  // corresponds to the given device address.
  std::pair<void*, size_t> FindMultimemAddress(
      const GpuCliqueKey& clique, se::DeviceAddressBase addr) const;

  // Returns a peer address that corresponds to the given buffer allocation.
  std::optional<se::DeviceAddressBase> FindPeerAddress(
      const GpuCliqueKey& clique, RankId rank,
      BufferAllocation::Index allocation) const;

  // Returns a peer address corresponds to the given device address.
  std::optional<se::DeviceAddressBase> FindPeerAddress(
      const GpuCliqueKey& clique, RankId rank,
      se::DeviceAddressBase addr) const;

  // Returns a peer address corresponds to the given device address.
  template <typename T>
  std::optional<se::DeviceAddress<T>> FindPeerAddress(
      const GpuCliqueKey& clique, RankId rank, se::DeviceAddress<T> addr) const;

 public:
  const BufferAllocations& buffers_;
  absl::flat_hash_map<Key, std::unique_ptr<SymmetricMemory>> sym_memories_;
  absl::flat_hash_map<Key, MulticastMemory> mcast_memories_;
  absl::flat_hash_map<Key, PeerMemory> peer_memories_;
};

// Acquires collective memory using the given collective parameters for all
// requested memories.
//
// WARNING: This is a collective operation, that must be called by all
// participating ranks in the requested memories (cliques), otherwise it will
// lead to a deadlock.
absl::StatusOr<CollectiveMemory> AcquireCollectiveMemory(
    const CollectiveParams& params, const CollectiveCliques& cliques,
    const CollectiveMemoryRequests& requests);

//===----------------------------------------------------------------------===//
// CollectiveMemory templates implementation.
//===----------------------------------------------------------------------===//

template <typename T>
std::optional<se::DeviceAddress<T>> CollectiveMemory::FindPeerAddress(
    const GpuCliqueKey& clique, RankId rank, se::DeviceAddress<T> addr) const {
  if (auto p = FindPeerAddress(clique, rank, se::DeviceAddressBase(addr))) {
    return se::DeviceAddress<T>(*p);
  }
  return std::nullopt;
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_H_
