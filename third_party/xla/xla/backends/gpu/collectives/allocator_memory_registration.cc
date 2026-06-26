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

#include "xla/backends/gpu/collectives/allocator_memory_registration.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/util.h"
#include "tsl/platform/numbers.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

tsl::SubAllocator::Visitor AllocatorMemoryRegistration::alloc_visitor() {
  return [self = shared_from_this()](void* ptr, int32_t device_ordinal,
                                     size_t bytes) {
    self->RecordAlloc(ptr, device_ordinal, bytes);
  };
}

tsl::SubAllocator::Visitor AllocatorMemoryRegistration::free_visitor() {
  return [self = shared_from_this()](void* ptr, int32_t device_ordinal,
                                     size_t bytes) {
    self->RecordFree(ptr, device_ordinal, bytes);
  };
}

GpuCliqueCreatedCallback AllocatorMemoryRegistration::CliqueCreatedCallback() {
  return [self = shared_from_this()](GpuClique& clique) {
    if (absl::Status status = self->RegisterWithClique(clique); !status.ok()) {
      LOG_FIRST_N(WARNING, 10) << "Failed to register GPU memory with clique "
                               << clique.key() << ": " << status;
    }
  };
}

void AllocatorMemoryRegistration::RecordAlloc(void* ptr, int32_t device_ordinal,
                                              size_t bytes) {
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "Record GPU allocation " << ptr << " of "
      << tsl::strings::HumanReadableNumBytes(bytes);
  if (ptr == nullptr || bytes == 0) {
    return;
  }

  absl::MutexLock lock(mu_);
  allocations_[LocalDeviceId(device_ordinal)].push_back(
      Allocation{se::DeviceAddressBase(ptr, bytes), {}});
}

void AllocatorMemoryRegistration::RecordFree(void* ptr, int32_t device_ordinal,
                                             size_t bytes) {
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "Record GPU free " << ptr << " of "
      << tsl::strings::HumanReadableNumBytes(bytes);
  if (ptr == nullptr) {
    return;
  }

  absl::MutexLock lock(mu_);
  // Match on the base pointer only: the size reported to the free visitor is
  // not always consistent with what was recorded on allocation. Erasing the
  // allocation releases its TiedRefs, deregistering the range from every clique
  // it was registered with.
  auto& allocs = allocations_[LocalDeviceId(device_ordinal)];
  allocs.erase(std::remove_if(allocs.begin(), allocs.end(),
                              [&](const Allocation& a) {
                                return a.range.opaque() == ptr;
                              }),
               allocs.end());
}

absl::Status AllocatorMemoryRegistration::RegisterWithClique(
    GpuClique& clique) {
  tsl::profiler::TraceMe trace(
      "AllocatorMemoryRegistration::RegisterWithClique");

  auto register_on_comm = [&](RankId, Communicator* comm) -> absl::Status {
    auto* gpu_comm = dynamic_cast<GpuCommunicator*>(comm);
    if (gpu_comm == nullptr) {
      return absl::OkStatus();
    }

    se::StreamExecutor* executor = gpu_comm->stream_executor();
    if (executor == nullptr) {
      return absl::OkStatus();
    }

    absl::MutexLock lock(mu_);
    auto& allocs = allocations_[LocalDeviceId(executor->device_ordinal())];

    XLA_VLOG_DEVICE(3, executor->device_ordinal())
        << "Registering " << allocs.size()
        << " recorded allocations with GPU clique " << clique.key();

    for (Allocation& allocation : allocs) {
      ASSIGN_OR_RETURN(std::unique_ptr<RegisteredMemory> registered,
                       gpu_comm->CreateRegisteredMemory(allocation.range));
      ASSIGN_OR_RETURN(tsl::TiedRef<RegisteredMemory> tied,
                       clique.Tie(std::move(registered)));
      allocation.registrations.push_back(std::move(tied));
    }

    return absl::OkStatus();
  };

  return clique.ForEachCommWithStatus(register_on_comm);
}

}  // namespace xla::gpu
