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
  VLOG(5) << "Add collective allocation request: " << clique_key.ToString()
          << "; allocation=" << allocation;

  // If symmetric allocation requests already exists add allocation index to it.
  if (auto it = allocations_.find(clique_key); it != allocations_.end()) {
    SymmetricAllocations& allocs = it->second;
    allocs.allocations.insert(allocation);
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // symmetric allocation requests to guarantee detemenistic execution order.
  SymmetricAllocations alloc{
      /*id=*/allocations_.size(), clique_key, {allocation}};

  allocations_.try_emplace(clique_key, std::move(alloc));
  return absl::OkStatus();
}

absl::Status CollectiveMemoryRequests::RequestSymmetricAddress(
    const GpuCliqueKey& clique_key, const se::DeviceAddressBase& addr) {
  VLOG(5) << "Add collective address request: " << clique_key.ToString()
          << "; address=" << addr.opaque();

  if (auto allocation = buffers_.FindAllocationIndex(addr)) {
    return RequestSymmetricAllocation(clique_key, *allocation);
  }
  return Internal("Can't find buffer allocation index for a device address");
}

std::vector<GpuCliqueKey> CollectiveMemoryRequests::RequestedCliques() const {
  std::vector<GpuCliqueKey> clique_keys;
  clique_keys.reserve(allocations_.size());
  for (const auto& [key, _] : allocations_) {
    clique_keys.push_back(key);
  }
  return clique_keys;
}

std::vector<CollectiveMemoryRequests::SymmetricAllocations>
CollectiveMemoryRequests::OrderedSymmetricAllocations() const {
  std::vector<SymmetricAllocations> allocations;
  allocations.reserve(allocations_.size());
  for (const auto& [_, allocs] : allocations_) {
    allocations.push_back(allocs);
  }

  absl::c_sort(allocations, [](const SymmetricAllocations& a,
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

  return allocations;
}

}  // namespace xla::gpu
