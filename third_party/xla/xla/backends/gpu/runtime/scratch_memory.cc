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

#include "xla/backends/gpu/runtime/scratch_memory.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

absl::StatusOr<ScratchMemory> AcquireScratchMemory(
    const CollectiveParams& collective_params,
    const ScratchMemoryRequests& scratch_memory_requests,
    CollectiveMemoryCache& collective_memory_cache,
    stream_executor::StreamExecutor* executor, CollectiveCliques& cliques) {
  std::shared_ptr<stream_executor::MemoryAllocation> scratch_memory_allocation;
  std::shared_ptr<xla::SymmetricMemory> scratch_symmetric_memory;

  if (scratch_memory_requests.Size() == 0) {
    return ScratchMemory(std::move(scratch_memory_allocation),
                         std::move(scratch_symmetric_memory));
  }

  int64_t device_ordinal = executor->device_ordinal();
  auto [cached_alloc, cached_sym] =
      collective_memory_cache.FindScratchMemory(device_ordinal);
  if (cached_alloc != nullptr) {
    return ScratchMemory(std::move(cached_alloc), std::move(cached_sym));
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<stream_executor::MemoryAllocator> collective_allocator,
      executor->CreateMemoryAllocator(
          stream_executor::MemorySpace::kCollective));

  auto ordered_requests = scratch_memory_requests.OrderedRequests();
  auto [clique_key, size] = ordered_requests[0];

  for (const auto& [key, s] : ordered_requests) {
    if (key.num_devices() > clique_key.num_devices()) {
      clique_key = key;
    }

    if (s > size) {
      size = s;
    }
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<stream_executor::MemoryAllocation> memory_allocation,
      collective_allocator->Allocate(size));

  const std::optional<RankId> rank =
      clique_key.rank(collective_params.global_device_id);
  ASSIGN_OR_RETURN(auto* comm, cliques.GetComm(clique_key, rank.value()));
  ASSIGN_OR_RETURN(auto symmetric_memory,
                   comm->CreateSymmetricMemory(memory_allocation->address()));

  XLA_VLOG_DEVICE(1, device_ordinal)
      << "Allocated scratch memory (address="
      << memory_allocation->address().opaque()
      << "; size=" << memory_allocation->address().size() << ")"
      << " associated with symmetric memory: ("
      << symmetric_memory->addr().opaque()
      << "; size=" << symmetric_memory->addr().size() << ")"
      << " for clique: " << clique_key << " rank: " << rank.value();
  ASSIGN_OR_RETURN(auto tied_symmetric_memory,
                   cliques.Tie(clique_key, std::move(symmetric_memory)));

  ASSIGN_OR_RETURN(
      tsl::TiedRef<stream_executor::MemoryAllocation> tied_memory_allocation,
      cliques.Tie(clique_key, std::move(memory_allocation)));

  auto [alloc, sym_mem] = collective_memory_cache.AddScratchMemory(
      device_ordinal, std::move(tied_memory_allocation),
      std::move(tied_symmetric_memory));
  return ScratchMemory(std::move(alloc), std::move(sym_mem));
}

}  // namespace gpu
}  // namespace xla
