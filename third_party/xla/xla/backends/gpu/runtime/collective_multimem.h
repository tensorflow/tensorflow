/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MULTIMEM_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MULTIMEM_H_

#include <any>
#include <memory>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"  // IWYU pragma: keep

namespace xla::gpu {

// CollectiveMultimem is a collection of per-device virtual memory ranges
// registered with the multicast memory.
class CollectiveMultimem {
 public:
  // Allocates a CollectiveMultimem for the given clique key and rank.
  //
  // This is a collective operation that must be called concurrently by all
  // participating devices in the clique. Implementation relies on the
  // rendezvous synchronization to ensure that all ranks arrive to the barrier.
  // The result is collectively owned by all participants.
  //
  // Rendezvous leader creates a multicast memory and maps all per-device
  // memories passed as `map_to` argument to the created multicast memory. Each
  // rank then gets a virtual memory address bound to the multicast memory, and
  // operations performed via this pointer gets broadcasted to all participating
  // devices.
  //
  // The optional `payload` argument is captured by the returned shared pointer
  // to allow callers to associate arbitrary data with the collective multimem.
  static absl::StatusOr<std::shared_ptr<CollectiveMultimem>> Allocate(
      se::StreamExecutor& executor, const GpuCliqueKey& clique_key, RankId rank,
      se::DeviceAddressBase map_to);

  // Allocates a CollectiveMultimem for the given global device id.
  static absl::StatusOr<std::shared_ptr<CollectiveMultimem>> Allocate(
      se::StreamExecutor& executor, const GpuCliqueKey& clique_key,
      GlobalDeviceId global_device_id, se::DeviceAddressBase map_to);

  const GpuCliqueKey& clique_key() const { return clique_key_; }

  // Returns the device pointer to the multicast memory for the given rank.
  void* mapped_ptr(RankId rank) const { return mapped_ptrs_.at(rank); }

 private:
  CollectiveMultimem(
      GpuCliqueKey clique_key, absl::btree_map<RankId, void*> mapped_ptrs,
      std::unique_ptr<se::gpu::MulticastMemory> multicast_memory);

  // All devices in this clique will have access to the multicast memory.
  GpuCliqueKey clique_key_;

  // A mapping from a participating rank to the mapped virtual memory pointer.
  absl::btree_map<RankId, void*> mapped_ptrs_;

  // A mapping from a participating rank to the payload passed to the Allocate.
  absl::btree_map<RankId, std::any> payload_;

  // All virtual memory pointers are registered with this multicast memory.
  std::unique_ptr<se::gpu::MulticastMemory> multicast_memory_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MULTIMEM_H_
