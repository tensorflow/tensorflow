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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {

// Caches collective memories (symmetric, multicast) across executable
// invocations. On the first invocation, collective memory windows are
// registered with the communication library (a collective operation). On
// subsequent invocations, the cached windows are reused to avoid redundant
// registration calls and to keep signal counters consistent.
//
// Correctness note: window registration is a collective operation — all ranks
// must call it together. Caching is safe because all ranks share the same
// buffer assignment, so they all get cache hits or misses for a given address.
//
// Lifetime: owned by GpuExecutable, persists across executions. Cached entries
// hold TiedRefs to the global GpuClique, so the underlying windows remain
// valid as long as the clique exists.
class CollectiveMemoryCache {
 public:
  // Returns a locked shared_ptr to the cached memory, or nullptr if not cached
  // or if the underlying TiedRef has expired.
  std::shared_ptr<SymmetricMemory> FindSymmetricMemory(
      const GpuCliqueKey& clique, stream_executor::DeviceAddressBase addr);

  std::shared_ptr<stream_executor::gpu::MulticastMemory> FindMulticastMemory(
      const GpuCliqueKey& clique, stream_executor::DeviceAddressBase addr);

  // Inserts or replaces a cached entry. Returns a locked shared_ptr.
  std::shared_ptr<SymmetricMemory> AddSymmetricMemory(
      const GpuCliqueKey& clique, stream_executor::DeviceAddressBase addr,
      tsl::TiedRef<SymmetricMemory> sym_memory);

  std::shared_ptr<stream_executor::gpu::MulticastMemory> AddMulticastMemory(
      const GpuCliqueKey& clique, stream_executor::DeviceAddressBase addr,
      tsl::TiedRef<stream_executor::gpu::MulticastMemory> multicast_memory);

  // Returns the cached scratch allocation+window pair, or {nullptr, nullptr}
  // on cache miss.
  std::pair<std::shared_ptr<stream_executor::MemoryAllocation>,
            std::shared_ptr<SymmetricMemory>>
  FindScratchMemory(int64_t device_ordinal) ABSL_LOCKS_EXCLUDED(mutex_);

  // Inserts or replaces a scratch entry. Returns locked shared_ptrs to the
  // cached allocation and symmetric memory window.
  std::pair<std::shared_ptr<stream_executor::MemoryAllocation>,
            std::shared_ptr<SymmetricMemory>>
  AddScratchMemory(
      int64_t device_ordinal,
      tsl::TiedRef<stream_executor::MemoryAllocation> memory_allocation,
      tsl::TiedRef<SymmetricMemory> symmetric_memory)
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  using Key = std::pair<GpuCliqueKey, stream_executor::DeviceAddressBase>;

  // Scratch symmetric memory is a bundle: the allocation provides the raw
  // device memory, and the symmetric memory is the registered window over it.
  struct ScratchEntry {
    tsl::TiedRef<stream_executor::MemoryAllocation> allocation;
    tsl::TiedRef<SymmetricMemory> symmetric_memory;
  };

  absl::Mutex mutex_;

  absl::flat_hash_map<Key, tsl::TiedRef<SymmetricMemory>> sym_memories_
      ABSL_GUARDED_BY(mutex_);

  absl::flat_hash_map<Key, tsl::TiedRef<stream_executor::gpu::MulticastMemory>>
      multicast_memories_ ABSL_GUARDED_BY(mutex_);

  absl::flat_hash_map<int64_t, ScratchEntry> scratch_memories_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_MEMORY_CACHE_H_
