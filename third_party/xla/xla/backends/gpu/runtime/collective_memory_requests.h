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
// argument and result buffers to be collective on the given GPU clique, which
// makes these buffers accessible to all devices in the clique. XLA runtime is
// responsible for collecting such requests during the prepare stage and
// creating collective memories for all requested allocations during the
// initialize stage.
//
// There are two kinds of collective memory supported by XLA runtime:
//
//   (1) Symmetric memory allocated via the `Collectives` API and implemented in
//       the underlying collective communication library (i.e. NCCL), see
//       `xla::SymmetricMemory` for more details.
//
//   (2) Multicast memory implemented directly on top of multicast objects
//       supported by the low level GPU driver programming interface. Access to
//       such objects can be done via the multimem addresses and multimem
//       instruction.
//
//   (3) Peer memory implemented as device address exchange between ranks
//       running within same process (local GPU cliques). Within single host
//       multiple GPU devices can access peer memory via pointers, and
//       collective operations could be implemented as simple memory reads
//       and writes plus a cross-device barrier.
//
// Symmetric memory allocated via the underlying collective communication
// library can span multiple processes running on different hosts, connected via
// the host network. Multicast objects and peer memory require all devices to be
// connected with NVLINK (for CUDA platform).
//
// For CUDA symmetric memory see official NCCL documentation:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg
//
// For CUDA multicast object management see official documentation:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html
class CollectiveMemoryRequests {
 public:
  explicit CollectiveMemoryRequests(const BufferAllocations& buffers);

  // For both kinds of collective memory we use a synthetic id that is
  // generated automatically when collective allocations is requested, and
  // guarantees deterministic iteration order across all ranks. See
  // `collective_clique_requests` for more details on how exactly this id
  // is generated and why it provides a deterministic iteration order.

  // A set of buffer allocations that must have a corresponding symmetric memory
  // on the given clique. Allocations must be allocated from a compatible
  // collective memory allocator (typically have a memory space S(1) in the HLO
  // program).
  struct SymmetricAllocations {
    size_t id;  // see synthetic id documentation above
    GpuCliqueKey key;
    absl::btree_set<BufferAllocation::Index> allocations;
  };

  // A set of buffer allocation that must be mapped for multicast access.
  // Allocations must be allocated from a compatible collective memory allocator
  // (typically have a memory space S(1) in the HLO program).
  struct MulticastAllocations {
    // See synthetic id documentation above.
    size_t id;
    GpuCliqueKey key;

    // Buffer assigner can fit multiple buffers into one pre-allocated memory.
    // With current API users are allowed to map any of those buffers with
    // the multicast object. However CUDA requires the memory address mapped to
    // the multicast object to be aligned with specific alignment:
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html
    // This alignment can't currently be guaranteed by the buffer assigner,
    // that's why range_mapped option allows the user to map the entire
    // pre-allocated memory range with the multicast object.
    bool range_mapped;
    absl::btree_set<BufferAllocation::Index> allocations;
  };

  // A set of buffer allocation that must be exchanged with clique peers.
  struct PeerAllocations {
    size_t id;  // see synthetic id documentation above
    GpuCliqueKey key;
    absl::btree_set<BufferAllocation::Index> allocations;
  };

  // Adds a request to make the given allocation symmetric on the given clique.
  absl::Status RequestSymmetricAllocation(const GpuCliqueKey& clique_key,
                                          BufferAllocation::Index allocation);

  // Adds a request to make the given address range symmetric on the given
  // clique. If address does not correspond to any of the buffer allocations in
  // the `buffers_`, it will return an error.
  absl::Status RequestSymmetricAddress(const GpuCliqueKey& clique_key,
                                       const se::DeviceAddressBase& addr);

  // Adds a request to map the given allocation to multicast object on the given
  // clique.
  // If the method is called several times for different allocation in the same
  // clique, `range_mapped` flag will be used if it was set at any call.
  absl::Status RequestMulticastAllocation(const GpuCliqueKey& clique_key,
                                          BufferAllocation::Index allocation,
                                          bool range_mapped = false);

  // Adds a request to map the given address to multicast object on the given
  // clique. If address does not correspond to any of the buffer allocations in
  // the `buffers_`, it will return an error.
  absl::Status RequestMulticastAddress(const GpuCliqueKey& clique_key,
                                       const se::DeviceAddressBase& addr,
                                       bool range_mapped = false);

  // Adds a request to exchange the given allocation with clique peers.
  absl::Status RequestPeerAllocation(const GpuCliqueKey& clique_key,
                                     BufferAllocation::Index allocation);

  // Adds a request to exchange the given address with clique peers. If address
  // does not correspond to any of the buffer allocations in the `buffers_`, it
  // will return an error.
  absl::Status RequestPeerAddress(const GpuCliqueKey& clique_key,
                                  const se::DeviceAddressBase& addr);

  // Returns all cliques that have symmetric or multicast allocation requests.
  std::vector<GpuCliqueKey> RequestedCliques() const;

  // Returns all requested symmetric allocations ordered by clique.
  std::vector<SymmetricAllocations> OrderedSymmetricAllocations() const;

  // Returns all requested multicast allocations ordered by clique.
  std::vector<MulticastAllocations> OrderedMulticastAllocations() const;

  // Returns all requested peer allocations ordered by clique.
  std::vector<PeerAllocations> OrderedPeerAllocations() const;

  size_t symmetric_size() const { return sym_allocations_.size(); }
  size_t multicast_size() const { return mcast_allocations_.size(); }
  size_t peer_size() const { return peer_allocations_.size(); }

  const BufferAllocations& buffers() const { return buffers_; }

 private:
  const BufferAllocations& buffers_;
  absl::flat_hash_map<GpuCliqueKey, SymmetricAllocations> sym_allocations_;
  absl::flat_hash_map<GpuCliqueKey, MulticastAllocations> mcast_allocations_;
  absl::flat_hash_map<GpuCliqueKey, PeerAllocations> peer_allocations_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_REQUESTS_H_
