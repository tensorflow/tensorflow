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

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {

std::shared_ptr<SymmetricMemory> CollectiveMemoryCache::FindSymmetricMemory(
    int device_ordinal, const GpuCliqueKey& clique,
    stream_executor::DeviceAddressBase addr) {
  absl::StatusOr<DeviceCache*> device_cache =
      device_caches_.GetOrCreate(device_ordinal);
  CHECK_OK(device_cache.status());
  absl::MutexLock lock((*device_cache)->mutex);
  auto it = (*device_cache)->sym_memories.find(Key{clique, addr});
  if (it == (*device_cache)->sym_memories.end()) {
    return nullptr;
  }
  return it->second.Lock();
}

std::shared_ptr<SymmetricMemory> CollectiveMemoryCache::AddSymmetricMemory(
    int device_ordinal, const GpuCliqueKey& clique,
    stream_executor::DeviceAddressBase addr,
    tsl::TiedRef<SymmetricMemory> sym_memory) {
  absl::StatusOr<DeviceCache*> device_cache =
      device_caches_.GetOrCreate(device_ordinal);
  CHECK_OK(device_cache.status());
  absl::MutexLock lock((*device_cache)->mutex);
  auto [it, _] = (*device_cache)
                     ->sym_memories.insert_or_assign(Key{clique, addr},
                                                     std::move(sym_memory));
  return it->second.Lock();
}

std::shared_ptr<stream_executor::gpu::MulticastMemory>
CollectiveMemoryCache::AddMulticastMemory(
    int device_ordinal, const GpuCliqueKey& clique,
    stream_executor::DeviceAddressBase addr,
    tsl::TiedRef<stream_executor::gpu::MulticastMemory> multicast_memory) {
  absl::StatusOr<DeviceCache*> device_cache =
      device_caches_.GetOrCreate(device_ordinal);
  CHECK_OK(device_cache.status());
  absl::MutexLock lock((*device_cache)->mutex);
  auto [it, _] = (*device_cache)
                     ->multicast_memories.insert_or_assign(
                         Key{clique, addr}, std::move(multicast_memory));
  return it->second.Lock();
}

std::shared_ptr<stream_executor::gpu::MulticastMemory>
CollectiveMemoryCache::FindMulticastMemory(
    int device_ordinal, const GpuCliqueKey& clique,
    stream_executor::DeviceAddressBase addr) {
  absl::StatusOr<DeviceCache*> device_cache =
      device_caches_.GetOrCreate(device_ordinal);
  CHECK_OK(device_cache.status());
  absl::MutexLock lock((*device_cache)->mutex);
  auto it = (*device_cache)->multicast_memories.find(Key{clique, addr});
  if (it == (*device_cache)->multicast_memories.end()) {
    return nullptr;
  }
  return it->second.Lock();
}

}  // namespace xla::gpu
