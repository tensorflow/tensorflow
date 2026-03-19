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

#include "xla/backends/gpu/runtime/collective_memory_requests.h"

#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"
#include "xla/util.h"

namespace xla::gpu {

CollectiveMemoryRequests::CollectiveMemoryRequests(
    const BufferAllocations& buffers)
    : buffers_(buffers) {}

absl::Status CollectiveMemoryRequests::RequestSymmetricAllocation(
    const GpuCliqueKey& clique_key, BufferAllocation::Index allocation) {
  VLOG(5) << "Add collective allocation request: " << clique_key
          << "; allocation=" << allocation;

  // If symmetric allocation requests already exists add allocation index to it.
  if (auto it = sym_allocations_.find(clique_key);
      it != sym_allocations_.end()) {
    SymmetricAllocations& allocs = it->second;
    allocs.allocations.insert(allocation);
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // symmetric allocation requests to guarantee deterministic execution order.
  SymmetricAllocations alloc{
      /*id=*/sym_allocations_.size(), clique_key, {allocation}};

  sym_allocations_.try_emplace(clique_key, std::move(alloc));
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestSymmetricAddress(
    const GpuCliqueKey& clique_key, const se::DeviceAddressBase& addr) {
  VLOG(5) << "Add collective address request: " << clique_key
          << "; address=" << addr.opaque();

  if (auto allocation = buffers_.FindAllocationIndex(addr)) {
    return RequestSymmetricAllocation(clique_key, *allocation);
  }
  return Internal("Can't find buffer allocation index for a device address");
}

absl::Status CollectiveMemoryRequests::RequestMulticastAllocation(
    const GpuCliqueKey& clique_key, BufferAllocation::Index allocation,
    bool range_mapped) {
  VLOG(5) << "Add multicast allocation request: " << clique_key
          << "; allocation=" << allocation;

  // If multicast allocation requests already exists add allocation index to it.
  if (auto it = mcast_allocations_.find(clique_key);
      it != mcast_allocations_.end()) {
    it->second.range_mapped |= range_mapped;
    MulticastAllocations& allocs = it->second;
    allocs.allocations.insert(allocation);
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // multicast allocation requests to guarantee deterministic execution order.
  MulticastAllocations alloc{
      /*id=*/mcast_allocations_.size(), clique_key, range_mapped, {allocation}};

  mcast_allocations_.try_emplace(clique_key, std::move(alloc));
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestMulticastAddress(
    const GpuCliqueKey& clique_key, const se::DeviceAddressBase& addr,
    bool range_mapped) {
  VLOG(5) << "Add multicast address request: " << clique_key.ToString()
          << "; address=" << addr.opaque();

  if (auto allocation = buffers_.FindAllocationIndex(addr)) {
    return RequestMulticastAllocation(clique_key, *allocation, range_mapped);
  }
  return Internal("Can't find buffer allocation index for a device address");
}

absl::Status CollectiveMemoryRequests::RequestPeerAllocation(
    const GpuCliqueKey& clique_key, BufferAllocation::Index allocation) {
  VLOG(5) << "Add peer allocation request: " << clique_key
          << "; allocation=" << allocation;

  // If peer allocation requests already exists add allocation index to it.
  if (auto it = peer_allocations_.find(clique_key);
      it != peer_allocations_.end()) {
    PeerAllocations& allocs = it->second;
    allocs.allocations.insert(allocation);
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // peer allocation requests to guarantee deterministic execution order.
  PeerAllocations alloc{
      /*id=*/peer_allocations_.size(), clique_key, {allocation}};

  peer_allocations_.try_emplace(clique_key, std::move(alloc));
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestPeerAddress(
    const GpuCliqueKey& clique_key, const se::DeviceAddressBase& addr) {
  VLOG(5) << "Add peer address request: " << clique_key.ToString()
          << "; address=" << addr.opaque();

  if (auto allocation = buffers_.FindAllocationIndex(addr)) {
    return RequestPeerAllocation(clique_key, *allocation);
  }
  return Internal("Can't find buffer allocation index for a device address");
}

std::vector<GpuCliqueKey> CollectiveMemoryRequests::RequestedCliques() const {
  std::vector<GpuCliqueKey> clique_keys;
  clique_keys.reserve(sym_allocations_.size() + mcast_allocations_.size());
  for (const auto& [key, _] : sym_allocations_) {
    clique_keys.push_back(key);
  }
  for (const auto& [key, _] : mcast_allocations_) {
    clique_keys.push_back(key);
  }
  return clique_keys;
}

std::vector<CollectiveMemoryRequests::SymmetricAllocations>
CollectiveMemoryRequests::OrderedSymmetricAllocations() const {
  std::vector<SymmetricAllocations> sym_allocations;
  sym_allocations.reserve(sym_allocations_.size());
  for (const auto& [_, allocs] : sym_allocations_) {
    sym_allocations.push_back(allocs);
  }

  absl::c_sort(sym_allocations, [](const SymmetricAllocations& a,
                                   const SymmetricAllocations& b) {
    // Create symmetric memory for larger cliques first.
    if (a.key.devices().size() > b.key.devices().size()) {
      return true;
    }
    if (b.key.devices().size() > a.key.devices().size()) {
      return false;
    }

    // Prefer cliques with smaller id (comes earlier in execution order).
    return a.id < b.id;
  });

  return sym_allocations;
}

std::vector<CollectiveMemoryRequests::MulticastAllocations>
CollectiveMemoryRequests::OrderedMulticastAllocations() const {
  std::vector<MulticastAllocations> mcast_allocations;
  mcast_allocations.reserve(mcast_allocations_.size());
  for (const auto& [_, allocs] : mcast_allocations_) {
    mcast_allocations.push_back(allocs);
  }

  absl::c_sort(mcast_allocations, [](const MulticastAllocations& a,
                                     const MulticastAllocations& b) {
    // Create multicast memory for larger cliques first.
    if (a.key.devices().size() > b.key.devices().size()) {
      return true;
    }
    if (b.key.devices().size() > a.key.devices().size()) {
      return false;
    }

    // Prefer cliques with smaller id (comes earlier in execution order).
    return a.id < b.id;
  });

  return mcast_allocations;
}

std::vector<CollectiveMemoryRequests::PeerAllocations>
CollectiveMemoryRequests::OrderedPeerAllocations() const {
  std::vector<PeerAllocations> peer_allocations;
  peer_allocations.reserve(peer_allocations_.size());
  for (const auto& [_, allocs] : peer_allocations_) {
    peer_allocations.push_back(allocs);
  }

  absl::c_sort(peer_allocations,
               [](const PeerAllocations& a, const PeerAllocations& b) {
                 // Create multicast memory for larger cliques first.
                 if (a.key.devices().size() > b.key.devices().size()) {
                   return true;
                 }
                 if (b.key.devices().size() > a.key.devices().size()) {
                   return false;
                 }

                 // Prefer cliques with smaller id (comes earlier in execution
                 // order).
                 return a.id < b.id;
               });

  return peer_allocations;
}

}  // namespace xla::gpu
