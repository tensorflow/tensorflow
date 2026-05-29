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

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
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
    const GpuCliqueKey& clique, BufferAllocation::Index allocation) {
  VLOG(5) << "Add collective allocation request: " << clique
          << "; allocation=" << allocation;

  // If symmetric allocation requests already exists add allocation index to it.
  if (auto it = sym_allocations_.find(clique); it != sym_allocations_.end()) {
    CollectiveAllocations& allocs = it->second;
    allocs.allocations.insert(allocation);
    return absl::OkStatus();
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // symmetric allocation requests to guarantee deterministic execution order.
  CollectiveAllocations alloc{
      /*id=*/sym_allocations_.size(), clique, {allocation}};

  sym_allocations_.try_emplace(clique, std::move(alloc));
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestSymmetricAllocationSlice(
    const GpuCliqueKey& clique, BufferAllocation::Slice slice) {
  return RequestSymmetricAllocation(clique, slice.index());
}

absl::Status CollectiveMemoryRequests::RequestSymmetricAllocationSlices(
    const GpuCliqueKey& clique,
    absl::Span<const BufferAllocation::Slice> slices) {
  for (const BufferAllocation::Slice& slice : slices) {
    RETURN_IF_ERROR(RequestSymmetricAllocationSlice(clique, slice));
  }
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestSymmetricAddress(
    const GpuCliqueKey& clique, const se::DeviceAddressBase& addr) {
  VLOG(5) << "Add collective address request: " << clique
          << "; address=" << addr.opaque();

  if (auto allocation = buffers_.FindAllocationIndex(addr)) {
    return RequestSymmetricAllocation(clique, *allocation);
  }
  return Internal("Can't find buffer allocation index for a device address");
}

absl::Status CollectiveMemoryRequests::RequestSymmetricAddresses(
    const GpuCliqueKey& clique, absl::Span<const se::DeviceAddressBase> addrs) {
  for (const se::DeviceAddressBase& addr : addrs) {
    RETURN_IF_ERROR(RequestSymmetricAddress(clique, addr));
  }
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestMulticastAllocation(
    const GpuCliqueKey& clique, BufferAllocation::Index allocation) {
  VLOG(5) << "Add multicast allocation request: " << clique
          << "; allocation=" << allocation;

  // If multicast allocation requests already exists add allocation index to it.
  if (auto it = mcast_allocations_.find(clique);
      it != mcast_allocations_.end()) {
    CollectiveAllocations& allocs = it->second;
    allocs.allocations.insert(allocation);
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // multicast allocation requests to guarantee deterministic execution order.
  CollectiveAllocations alloc{
      /*id=*/mcast_allocations_.size(), clique, {allocation}};

  mcast_allocations_.try_emplace(clique, std::move(alloc));
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestMulticastAllocationSlice(
    const GpuCliqueKey& clique, BufferAllocation::Slice slice) {
  return RequestMulticastAllocation(clique, slice.index());
}

absl::Status CollectiveMemoryRequests::RequestMulticastAddress(
    const GpuCliqueKey& clique, const se::DeviceAddressBase& addr) {
  VLOG(5) << "Add multicast address request: " << clique.ToString()
          << "; address=" << addr.opaque();

  if (auto allocation = buffers_.FindAllocationIndex(addr)) {
    return RequestMulticastAllocation(clique, *allocation);
  }
  return Internal("Can't find buffer allocation index for a device address");
}

absl::Status CollectiveMemoryRequests::RequestPeerAllocation(
    const GpuCliqueKey& clique, BufferAllocation::Index allocation) {
  VLOG(5) << "Add peer allocation request: " << clique
          << "; allocation=" << allocation;

  // If peer allocation requests already exists add allocation index to it.
  if (auto it = peer_allocations_.find(clique); it != peer_allocations_.end()) {
    CollectiveAllocations& allocs = it->second;
    allocs.allocations.insert(allocation);
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // peer allocation requests to guarantee deterministic execution order.
  CollectiveAllocations alloc{
      /*id=*/peer_allocations_.size(), clique, {allocation}};

  peer_allocations_.try_emplace(clique, std::move(alloc));
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestPeerAllocationSlice(
    const GpuCliqueKey& clique, BufferAllocation::Slice slice) {
  return RequestPeerAllocation(clique, slice.index());
}

absl::Status CollectiveMemoryRequests::RequestPeerAddress(
    const GpuCliqueKey& clique, const se::DeviceAddressBase& addr) {
  VLOG(5) << "Add peer address request: " << clique.ToString()
          << "; address=" << addr.opaque();

  if (auto allocation = buffers_.FindAllocationIndex(addr)) {
    return RequestPeerAllocation(clique, *allocation);
  }
  return Internal("Can't find buffer allocation index for a device address");
}

std::vector<GpuCliqueKey> CollectiveMemoryRequests::RequestedCliques() const {
  std::vector<GpuCliqueKey> cliques;
  cliques.reserve(sym_allocations_.size() + mcast_allocations_.size());
  for (const auto& [clique, _] : sym_allocations_) {
    cliques.push_back(clique);
  }
  for (const auto& [clique, _] : mcast_allocations_) {
    cliques.push_back(clique);
  }
  return cliques;
}

std::vector<CollectiveMemoryRequests::CollectiveAllocations>
CollectiveMemoryRequests::OrderedSymmetricAllocations() const {
  std::vector<CollectiveAllocations> sym_allocations;
  sym_allocations.reserve(sym_allocations_.size());
  for (const auto& [_, allocs] : sym_allocations_) {
    sym_allocations.push_back(allocs);
  }

  absl::c_sort(sym_allocations, [](const CollectiveAllocations& a,
                                   const CollectiveAllocations& b) {
    // Create symmetric memory for larger cliques first.
    if (a.clique.devices().size() > b.clique.devices().size()) {
      return true;
    }
    if (b.clique.devices().size() > a.clique.devices().size()) {
      return false;
    }

    // Prefer cliques with smaller id (comes earlier in execution order).
    return a.id < b.id;
  });

  return sym_allocations;
}

std::vector<CollectiveMemoryRequests::CollectiveAllocations>
CollectiveMemoryRequests::OrderedMulticastAllocations() const {
  std::vector<CollectiveAllocations> mcast_allocations;
  mcast_allocations.reserve(mcast_allocations_.size());
  for (const auto& [_, allocs] : mcast_allocations_) {
    mcast_allocations.push_back(allocs);
  }

  absl::c_sort(mcast_allocations, [](const CollectiveAllocations& a,
                                     const CollectiveAllocations& b) {
    // Create multicast memory for larger cliques first.
    if (a.clique.devices().size() > b.clique.devices().size()) {
      return true;
    }
    if (b.clique.devices().size() > a.clique.devices().size()) {
      return false;
    }

    // Prefer cliques with smaller id (comes earlier in execution order).
    return a.id < b.id;
  });

  return mcast_allocations;
}

std::vector<CollectiveMemoryRequests::CollectiveAllocations>
CollectiveMemoryRequests::OrderedPeerAllocations() const {
  std::vector<CollectiveAllocations> peer_allocations;
  peer_allocations.reserve(peer_allocations_.size());
  for (const auto& [_, allocs] : peer_allocations_) {
    peer_allocations.push_back(allocs);
  }

  absl::c_sort(peer_allocations, [](const CollectiveAllocations& a,
                                    const CollectiveAllocations& b) {
    // Create multicast memory for larger cliques first.
    if (a.clique.devices().size() > b.clique.devices().size()) {
      return true;
    }
    if (b.clique.devices().size() > a.clique.devices().size()) {
      return false;
    }

    // Prefer cliques with smaller id (comes earlier in execution
    // order).
    return a.id < b.id;
  });

  return peer_allocations;
}

}  // namespace xla::gpu
