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

#include "xla/backends/gpu/runtime/collective_memory.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/rendezvous.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

CollectiveMemory::CollectiveMemory(
    const BufferAllocations& buffers,
    absl::flat_hash_map<Key, std::unique_ptr<SymmetricMemory>> sym_memories,
    absl::flat_hash_map<Key, MulticastMemory> mcast_memories,
    absl::flat_hash_map<Key, PeerMemory> peer_memories)
    : buffers_(buffers),
      sym_memories_(std::move(sym_memories)),
      mcast_memories_(std::move(mcast_memories)),
      peer_memories_(std::move(peer_memories)) {}

std::pair<SymmetricMemory*, size_t> CollectiveMemory::FindSymmetricMemory(
    const GpuCliqueKey& clique, BufferAllocation::Index allocation) const {
  auto it = sym_memories_.find(std::make_pair(clique, allocation));
  if (it == sym_memories_.end()) {
    return std::make_pair(nullptr, 0);
  }
  return std::make_pair(it->second.get(), 0);
}

std::pair<SymmetricMemory*, size_t> CollectiveMemory::FindSymmetricMemory(
    const GpuCliqueKey& clique, se::DeviceAddressBase addr) const {
  auto allocation = buffers_.FindAllocationIndex(addr);
  if (!allocation.has_value()) {
    return std::make_pair(nullptr, 0);
  }

  // Find offset from the base allocation.
  se::DeviceAddressBase base = buffers_.GetDeviceAddress(*allocation);
  size_t offset = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque()) -
                  tsl::safe_reinterpret_cast<uintptr_t>(base.opaque());

  auto [sym, sym_offset] = FindSymmetricMemory(clique, *allocation);
  return std::make_pair(sym, sym_offset + offset);
}

std::pair<void*, size_t> CollectiveMemory::FindMultimemAddress(
    const GpuCliqueKey& clique, BufferAllocation::Index allocation) const {
  auto it = mcast_memories_.find(std::make_pair(clique, allocation));
  if (it == mcast_memories_.end()) {
    return std::make_pair(nullptr, 0);
  }

  return std::make_pair(it->second.multimem_ptr, 0);
}

std::pair<void*, size_t> CollectiveMemory::FindMultimemAddress(
    const GpuCliqueKey& clique, se::DeviceAddressBase addr) const {
  auto allocation = buffers_.FindAllocationIndex(addr);
  if (!allocation.has_value()) {
    return std::make_pair(nullptr, 0);
  }

  // Find offset from the base allocation.
  se::DeviceAddressBase base = buffers_.GetDeviceAddress(*allocation);
  size_t offset = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque()) -
                  tsl::safe_reinterpret_cast<uintptr_t>(base.opaque());

  auto [mmem, mmem_offset] = FindMultimemAddress(clique, *allocation);
  return std::make_pair(mmem, mmem_offset + offset);
}

std::optional<se::DeviceAddressBase> CollectiveMemory::FindPeerAddress(
    const GpuCliqueKey& clique, RankId rank,
    BufferAllocation::Index allocation) const {
  auto it = peer_memories_.find(std::make_pair(clique, allocation));
  if (it == peer_memories_.end()) {
    return std::nullopt;
  }

  auto addr = it->second.addrs.find(rank);
  if (addr == it->second.addrs.end()) {
    return std::nullopt;
  }

  return addr->second;
}

std::optional<se::DeviceAddressBase> CollectiveMemory::FindPeerAddress(
    const GpuCliqueKey& clique, RankId rank, se::DeviceAddressBase addr) const {
  auto allocation = buffers_.FindAllocationIndex(addr);
  if (!allocation.has_value()) {
    return std::nullopt;
  }

  // Find offset from the base allocation.
  se::DeviceAddressBase base = buffers_.GetDeviceAddress(*allocation);
  size_t offset = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque()) -
                  tsl::safe_reinterpret_cast<uintptr_t>(base.opaque());

  // Find device address for peer allocation.
  auto peer_alloc = FindPeerAddress(clique, rank, *allocation);
  if (!peer_alloc.has_value()) {
    return std::nullopt;
  }

  return peer_alloc->GetByteSlice(offset, addr.size());
}

//===----------------------------------------------------------------------===//
// Local rendezvous parameters.
//===----------------------------------------------------------------------===//

namespace {

// Wrap GpuCliqueKey into a unique struct to guarantee we do not accidentally
// try to run multiple unrelated rendezvous for a same key.
struct RendezvousKey {
  GpuCliqueKey clique_key;

  bool operator==(const RendezvousKey& other) const {
    return clique_key == other.clique_key;
  }

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& key) {
    return H::combine(std::move(h), key.clique_key);
  }
};

// Parameters passed to the rendezvous callback from all ranks.
struct RendezvousParams {
  RankId rank;
  se::StreamExecutor* executor;
  const BufferAllocations& buffers;
};

struct RankCmp {
  bool operator()(const RendezvousParams* a, const RendezvousParams* b) const {
    return a->rank < b->rank;
  }
};

struct DeviceOrdinalFormatter {
  void operator()(std::string* out, const RendezvousParams* param) const {
    absl::StrAppend(out, param->executor->device_ordinal());
  }
};

struct RankFormatter {
  void operator()(std::string* out, const RendezvousParams* param) const {
    absl::StrAppend(out, param->rank.value());
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Symmetric memory acquisition.
//===----------------------------------------------------------------------===//

// Acquire symmetric memory for all requested allocation.
static absl::StatusOr<absl::flat_hash_map<CollectiveMemory::Key,
                                          std::unique_ptr<SymmetricMemory>>>
AcquireSymmetricMemory(
    const CollectiveParams& params, const CollectiveCliques& cliques,
    const BufferAllocations& buffers,
    absl::Span<const CollectiveMemoryRequests::SymmetricAllocations> allocs) {
  absl::flat_hash_map<CollectiveMemory::Key, std::unique_ptr<SymmetricMemory>>
      sym_memories;

  for (const CollectiveMemoryRequests::SymmetricAllocations& r : allocs) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);

    if (!rank.has_value()) {
      return Internal("Can't find global device id %v in clique key %s",
                      params.global_device_id, r.key.ToString());
    }

    // TODO(ezhulenev): All of the buffer allocations that we make symmetric
    // are created from the same underlying memory allocator. We can
    // significantly improve performance with a few tricks:
    //
    // 1. Coalesce adjacent allocations and create one large symmetric region.
    // 2. Create one big symmetric region from [start, end] addresses, we might
    //    have unused gaps in the middle, but it doesn't matter, we will ignore
    //    them.
    // 3. Cache symmetric memories in a process-level cache.
    //
    // Currently it's very simple proof of concept.

    ASSIGN_OR_RETURN(GpuCommunicator * comm, cliques.GetComm(r.key, *rank));
    for (BufferAllocation::Index i : r.allocations) {
      ASSIGN_OR_RETURN(
          std::unique_ptr<SymmetricMemory> symm,
          comm->CreateSymmetricMemory(buffers.GetDeviceAddress(i)));
      sym_memories[std::make_pair(r.key, i)] = std::move(symm);
    }
  }

  return sym_memories;
}

//===----------------------------------------------------------------------===//
// Multicast memory acquisition.
//===----------------------------------------------------------------------===//

namespace {

using MulticastMemoryMap =
    absl::flat_hash_map<CollectiveMemory::Key,
                        CollectiveMemory::MulticastMemory>;

struct MappedPtrFormatter {
  void operator()(std::string* out,
                  const std::pair<RankId, void*>& mapped_ptr) const {
    auto& [rank, ptr] = mapped_ptr;
    absl::StrAppend(out, absl::StrFormat("%d:%p", rank.value(), ptr));
  }
};

// A multicast object and a multimem mapping for all participating ranks.
struct MappedMulticastMemory {
  std::shared_ptr<se::gpu::MulticastMemory> multicast_memory;
  absl::btree_map<RankId, void*> rank_multimem_ptr;
};

using MappedMulticastMemoryMap =
    absl::flat_hash_map<CollectiveMemory::Key, MappedMulticastMemory>;

}  // namespace

// Acquire multicast memory for all requested allocation.
absl::StatusOr<MulticastMemoryMap> AcquireMulticastMemory(
    const CollectiveParams& params, const CollectiveCliques& cliques,
    const BufferAllocations& buffers,
    absl::Span<const CollectiveMemoryRequests::MulticastAllocations> allocs) {
  int32_t device_ordinal = params.executor->device_ordinal();

  MulticastMemoryMap mcast_memories;

  for (const CollectiveMemoryRequests::MulticastAllocations& r : allocs) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);

    if (!rank.has_value()) {
      return Internal("[%d] Can't find global device id %v in clique key %v",
                      device_ordinal, params.global_device_id, r.key);
    }

    // We rely on in-process rendezvous to allocate the multicast memory and set
    // up memory mapping on all ranks, and don't support multi-process mode.
    if (!r.key.is_local()) {
      return Unimplemented(
          "[%d] Multicast is not supported in multi-process mode in clique %v",
          device_ordinal, r.key);
    }

    std::string rendezvous_name = absl::StrFormat(
        "[%d] [rank=%v] AcquireMulticastMemory for clique %v: allocs=[%s]",
        device_ordinal, *rank, r.key, absl::StrJoin(r.allocations, ","));

    // Collect device addresses for mapped allocations.
    std::vector<se::DeviceAddressBase> map_to;
    map_to.reserve(r.allocations.size());
    for (BufferAllocation::Index i : r.allocations) {
      map_to.emplace_back(buffers.GetDeviceAddress(i));
    }

    // A callback for rendezvous to allocate and map the multicast memory. We
    // do one round of rendezvous for each clique.
    auto allocate = [&](absl::Span<const RendezvousParams*> params)
        -> absl::StatusOr<MappedMulticastMemoryMap> {
      // Sort all participants by rank to get deterministic execution.
      absl::c_sort(params, RankCmp{});

      VLOG(3) << absl::StrFormat(
          "[%s] [ranks=%s] Allocate collective multimem for clique: %v",
          absl::StrJoin(params, ",", DeviceOrdinalFormatter{}),
          absl::StrJoin(params, ",", RankFormatter{}), r.key);

      // We deterministically choose the first device to create the
      // multicast memory. We will map the rest of participants to it later.
      auto* gpu_executor =
          tsl::down_cast<se::gpu::GpuExecutor*>(params[0]->executor);
      if (gpu_executor == nullptr) {
        return Unimplemented("Unsupported stream executor type");
      }

      // As a result of rendezvous we collective multicast memory and mapping
      // for all participating ranks to the multimem address.
      MappedMulticastMemoryMap clique_mcast_memories;

      for (BufferAllocation::Index i : r.allocations) {
        // Allocate a multicast object for all participating devices.
        size_t multicast_size;

        // TODO(b/486104046): Add a separate API for range mapped allocations,
        // instead of using the size of the range mapped allocation.
        if (r.range_mapped) {
          ASSIGN_OR_RETURN(se::DeviceAddressBase address_range,
                           gpu_executor->GetMemoryRange(
                               params[0]->buffers.GetDeviceAddress(i)));
          multicast_size = address_range.size();
        } else {
          multicast_size = params[0]->buffers.GetDeviceAddress(i).size();
        }
        ASSIGN_OR_RETURN(
            std::unique_ptr<se::gpu::MulticastMemory> multicast_memory,
            gpu_executor->CreateMulticastMemory(multicast_size, params.size()));

        // For all participating devices, subscribe to the multicast object.
        for (const auto* param : params) {
          RETURN_IF_ERROR(multicast_memory->SubscribeDevice(
              param->executor->device_ordinal()));
        }

        // For all participating devices, map to the multicast memory.
        absl::btree_map<RankId, void*> mapped_ptrs;
        for (const auto* param : params) {
          ASSIGN_OR_RETURN(
              mapped_ptrs[param->rank],
              multicast_memory->MapMemory(
                  param->buffers.GetDeviceAddress(i),
                  tsl::down_cast<se::gpu::GpuExecutor*>(param->executor)));
        }

        VLOG(3) << absl::StrFormat(
            "[%s] [ranks=%s] Allocated collective multimem for clique: %v; "
            "mapped_ptrs=[%s]",
            absl::StrJoin(params, ",", DeviceOrdinalFormatter{}),
            absl::StrJoin(params, ",", RankFormatter{}), r.key,
            absl::StrJoin(mapped_ptrs, ", ", MappedPtrFormatter{}));

        clique_mcast_memories[std::make_pair(r.key, i)] = MappedMulticastMemory{
            std::move(multicast_memory), std::move(mapped_ptrs)};
      }
      return clique_mcast_memories;
    };

    // We expect that all local participants will collectively allocate the
    // multicast memory. We do one rendezvous for each clique, and from the
    // rendezvous callback allocate multicast memory for all allocations.
    RendezvousKey rendezvous_key = {r.key};
    RendezvousParams allocate_params = {*rank, params.executor, buffers};

    int64_t num_participants = r.key.num_local_participants();
    ASSIGN_OR_RETURN(
        std::shared_ptr<MappedMulticastMemoryMap> clique_mcast_memories,
        Rendezvous<MappedMulticastMemoryMap>(rendezvous_name, rendezvous_key,
                                             allocate_params, num_participants,
                                             allocate));

    // Copy clique multicast memory to each participating thread.
    for (auto& [k, v] : *clique_mcast_memories) {
      mcast_memories[k] = {v.multicast_memory, v.rank_multimem_ptr.at(*rank)};
    }
  }

  return mcast_memories;
}

//===----------------------------------------------------------------------===//
// Peer memory acquisition.
//===----------------------------------------------------------------------===//

namespace {

using PeerMemoryMap =
    absl::flat_hash_map<CollectiveMemory::Key, CollectiveMemory::PeerMemory>;

}  // namespace

// Acquire peer memory for all requested allocation.
absl::StatusOr<PeerMemoryMap> AcquirePeerMemory(
    const CollectiveParams& params, const CollectiveCliques& cliques,
    const BufferAllocations& buffers,
    absl::Span<const CollectiveMemoryRequests::PeerAllocations> allocs) {
  int32_t device_ordinal = params.executor->device_ordinal();

  PeerMemoryMap peer_memories;

  for (const CollectiveMemoryRequests::PeerAllocations& r : allocs) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);

    if (!rank.has_value()) {
      return Internal("[%d] Can't find global device id %v in clique key %v",
                      device_ordinal, params.global_device_id, r.key);
    }

    // We rely on in-process rendezvous to exchange peer memory with all ranks.
    if (!r.key.is_local()) {
      return Unimplemented(
          "[%d] Peer memory is not supported in multi-process mode in clique "
          "%v",
          device_ordinal, r.key);
    }

    std::string rendezvous_name = absl::StrFormat(
        "[%d] [rank=%v] AcquirePeerMemory for clique %v: allocs=[%s]",
        device_ordinal, *rank, r.key, absl::StrJoin(r.allocations, ","));

    // A callback for rendezvous to exchange peer allocation addresses with
    // all participating ranks.
    auto exchange = [&](absl::Span<const RendezvousParams*> params)
        -> absl::StatusOr<PeerMemoryMap> {
      // Sort all participants by rank to get deterministic execution.
      absl::c_sort(params, RankCmp{});

      VLOG(3) << absl::StrFormat(
          "[%s] [ranks=%s] Exchange collective peer memory for clique: %v",
          absl::StrJoin(params, ",", DeviceOrdinalFormatter{}),
          absl::StrJoin(params, ",", RankFormatter{}), r.key);

      PeerMemoryMap clique_peer_memories;
      for (BufferAllocation::Index i : r.allocations) {
        // For all participating devices get allocation device address.
        absl::btree_map<RankId, se::DeviceAddressBase> addrs;
        for (const RendezvousParams* param : params) {
          addrs[param->rank] = param->buffers.GetDeviceAddress(i);
        }
        clique_peer_memories[std::make_pair(r.key, i)] =
            CollectiveMemory::PeerMemory{std::move(addrs)};
      }
      return clique_peer_memories;
    };

    // We expect that all local participants will collectively allocate the
    // multicast memory. We do one rendezvous for each clique, and from the
    // rendezvous callback allocate multicast memory for all allocations.
    RendezvousKey rendezvous_key = {r.key};
    RendezvousParams allocate_params = {*rank, params.executor, buffers};

    int64_t num_participants = r.key.num_local_participants();
    ASSIGN_OR_RETURN(
        std::shared_ptr<PeerMemoryMap> clique_mcast_memories,
        Rendezvous<PeerMemoryMap>(rendezvous_name, rendezvous_key,
                                  allocate_params, num_participants, exchange));

    // Copy clique peer memory to each participating thread.
    for (auto& [k, v] : *clique_mcast_memories) {
      peer_memories[k] = v;
    }
  }

  return peer_memories;
}

//===----------------------------------------------------------------------===//
// Collective memory acquisition.
//===----------------------------------------------------------------------===//

absl::StatusOr<CollectiveMemory> AcquireCollectiveMemory(
    const CollectiveParams& params, const CollectiveCliques& cliques,
    const CollectiveMemoryRequests& requests) {
  // We rely on deterministic order of memory requests, to guarantee that all
  // ranks create collective memory in identical order, otherwise we can get
  // a deadlock.
  std::vector<CollectiveMemoryRequests::SymmetricAllocations> sym_allocs =
      requests.OrderedSymmetricAllocations();
  std::vector<CollectiveMemoryRequests::MulticastAllocations> mcast_allocs =
      requests.OrderedMulticastAllocations();
  std::vector<CollectiveMemoryRequests::PeerAllocations> peer_allocs =
      requests.OrderedPeerAllocations();

  // Short-circuit if we have nothing to allocate.
  if (sym_allocs.empty() && mcast_allocs.empty() && peer_allocs.empty()) {
    return CollectiveMemory(requests.buffers(), /*sym_memories=*/{},
                            /*mcast_memories=*/{}, /*peer_memories=*/{});
  }

  XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
      " Acquire collective memory for global device id %v: run_id=%v "
      "symmetric=%d multicast=%d peer=%d",
      params.global_device_id, params.run_id, sym_allocs.size(),
      mcast_allocs.size(), peer_allocs.size());
  absl::Time start = absl::Now();

  for (size_t i = 0; i < sym_allocs.size(); ++i) {
    const CollectiveMemoryRequests::SymmetricAllocations& r = sym_allocs[i];
    XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
        "    symmetric memory #%d (global device %v): id=%d; clique=%v; "
        "allocations=[%s]",
        i, params.global_device_id, r.id, r.key,
        absl::StrJoin(r.allocations, ", "));
  }

  for (size_t i = 0; i < mcast_allocs.size(); ++i) {
    const CollectiveMemoryRequests::MulticastAllocations& r = mcast_allocs[i];
    XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
        "    multicast memory #%d (global device %v): id=%d; clique=%v; "
        "allocations=[%s]",
        i, params.global_device_id, r.id, r.key,
        absl::StrJoin(r.allocations, ", "));
  }

  for (size_t i = 0; i < peer_allocs.size(); ++i) {
    const CollectiveMemoryRequests::PeerAllocations& r = peer_allocs[i];
    XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
        "    peer memory #%d (global device %v): id=%d; clique=%v; "
        "allocations=[%s]",
        i, params.global_device_id, r.id, r.key,
        absl::StrJoin(r.allocations, ", "));
  }

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("AcquireCollectiveMemory",
                                        {{"sym_allocs", sym_allocs.size()},
                                         {"mcast_allocs", mcast_allocs.size()},
                                         {"peer_allocs", peer_allocs.size()}});
  });

  ASSIGN_OR_RETURN(
      auto sym_memories,
      AcquireSymmetricMemory(params, cliques, requests.buffers(), sym_allocs));

  ASSIGN_OR_RETURN(auto mcast_memories,
                   AcquireMulticastMemory(params, cliques, requests.buffers(),
                                          mcast_allocs));

  ASSIGN_OR_RETURN(
      auto peer_memories,
      AcquirePeerMemory(params, cliques, requests.buffers(), peer_allocs));

  XLA_VLOG_DEVICE(2, params.executor->device_ordinal()) << absl::StreamFormat(
      "Acquired collective memory in %s for global device id %v; "
      "run_id=%v symmetric=%d multicast=%d peer=%d",
      absl::FormatDuration(absl::Now() - start), params.global_device_id,
      params.run_id, sym_allocs.size(), mcast_allocs.size(),
      peer_allocs.size());

  return CollectiveMemory(requests.buffers(), std::move(sym_memories),
                          std::move(mcast_memories), std::move(peer_memories));
}

}  // namespace xla::gpu
