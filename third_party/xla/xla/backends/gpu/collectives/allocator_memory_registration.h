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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_ALLOCATOR_MEMORY_REGISTRATION_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_ALLOCATOR_MEMORY_REGISTRATION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {

// Tracks allocator-owned GPU memory ranges and registers them with
// newly-created GPU cliques.
//
// This is intended for the BFC preallocation path where suballocator visitors
// report long-lived arena chunks before the first collective clique is created.
// It deliberately tracks allocator ranges, not individual PjRt buffers, and it
// does not discover cliques or allocations that existed before the callbacks
// were installed.
//
// When a clique is created, each recorded range is registered only with the GPU
// communicator that runs on the same local device ordinal. The resulting
// registrations are tied to the clique and retained with the allocation record;
// freeing the allocation releases those registrations.
class AllocatorMemoryRegistration
    : public std::enable_shared_from_this<AllocatorMemoryRegistration> {
 public:
  AllocatorMemoryRegistration() = default;

  // Returns visitors compatible with tsl::SubAllocator. The visitor API passes
  // an int32_t device index; this class interprets it as a local device
  // ordinal.
  tsl::SubAllocator::Visitor alloc_visitor();
  tsl::SubAllocator::Visitor free_visitor();

  // Returns a clique-created callback that registers all currently recorded
  // allocator ranges with the newly-created GPU clique.
  GpuCliqueCreatedCallback CliqueCreatedCallback();

  // Registers each recorded allocator range with the `clique` communicator that
  // runs on the same local device ordinal as the range. Ranges on devices not
  // participating in `clique` are skipped.
  absl::Status RegisterWithClique(GpuClique& clique);

 private:
  void RecordAlloc(void* ptr, int32_t device_ordinal, size_t bytes);
  void RecordFree(void* ptr, int32_t device_ordinal, size_t bytes);

  struct Allocation {
    se::DeviceAddressBase range;
    std::vector<tsl::TiedRef<RegisteredMemory>> registrations;
  };

  absl::Mutex mu_;
  // Recorded allocator ranges keyed by the local device they live on. Ranges
  // are rare in the preallocated BFC path, so a linear scan of the per-device
  // vector is fine.
  absl::flat_hash_map<LocalDeviceId, std::vector<Allocation>> allocations_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_ALLOCATOR_MEMORY_REGISTRATION_H_
