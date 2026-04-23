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

#include "xla/backends/gpu/runtime/collective_memory_cache.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {

std::shared_ptr<SymmetricMemory> CollectiveMemoryCache::FindSymmetricMemory(
    const GpuCliqueKey& clique, stream_executor::DeviceAddressBase addr) {
  absl::MutexLock lock(mutex_);
  auto it = sym_memories_.find(Key{clique, addr});
  if (it == sym_memories_.end()) {
    return nullptr;
  }
  return it->second.Lock();
}

std::shared_ptr<SymmetricMemory> CollectiveMemoryCache::AddSymmetricMemory(
    const GpuCliqueKey& clique, stream_executor::DeviceAddressBase addr,
    tsl::TiedRef<SymmetricMemory> sym_memory) {
  absl::MutexLock lock(mutex_);
  auto [it, _] =
      sym_memories_.insert_or_assign(Key{clique, addr}, std::move(sym_memory));
  return it->second.Lock();
}

std::shared_ptr<stream_executor::gpu::MulticastMemory>
CollectiveMemoryCache::AddMulticastMemory(
    const GpuCliqueKey& clique, stream_executor::DeviceAddressBase addr,
    tsl::TiedRef<stream_executor::gpu::MulticastMemory> multicast_memory) {
  absl::MutexLock lock(mutex_);
  auto [it, _] = multicast_memories_.insert_or_assign(
      Key{clique, addr}, std::move(multicast_memory));
  return it->second.Lock();
}

std::shared_ptr<stream_executor::gpu::MulticastMemory>
CollectiveMemoryCache::FindMulticastMemory(
    const GpuCliqueKey& clique, stream_executor::DeviceAddressBase addr) {
  absl::MutexLock lock(mutex_);
  auto it = multicast_memories_.find(Key{clique, addr});
  if (it == multicast_memories_.end()) {
    return nullptr;
  }
  return it->second.Lock();
}

std::pair<std::shared_ptr<stream_executor::MemoryAllocation>,
          std::shared_ptr<SymmetricMemory>>
CollectiveMemoryCache::FindScratchMemory(int64_t device_ordinal) {
  absl::MutexLock lock(mutex_);
  auto it = scratch_memories_.find(device_ordinal);
  if (it == scratch_memories_.end()) {
    return {nullptr, nullptr};
  }
  return std::make_pair(it->second.allocation.Lock(),
                        it->second.symmetric_memory.Lock());
}

std::pair<std::shared_ptr<stream_executor::MemoryAllocation>,
          std::shared_ptr<SymmetricMemory>>
CollectiveMemoryCache::AddScratchMemory(
    int64_t device_ordinal,
    tsl::TiedRef<stream_executor::MemoryAllocation> memory_allocation,
    tsl::TiedRef<SymmetricMemory> symmetric_memory) {
  absl::MutexLock lock(mutex_);
  auto [it, _] = scratch_memories_.insert_or_assign(
      device_ordinal,
      ScratchEntry{std::move(memory_allocation), std::move(symmetric_memory)});
  return std::make_pair(it->second.allocation.Lock(),
                        it->second.symmetric_memory.Lock());
}

}  // namespace xla::gpu
