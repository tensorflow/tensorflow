/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_multimem.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/rendezvous.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

CollectiveMultimem::CollectiveMultimem(
    GpuCliqueKey clique_key, absl::btree_map<RankId, void*> mapped_ptrs,
    std::unique_ptr<se::gpu::MulticastMemory> multicast_memory)
    : clique_key_(std::move(clique_key)),
      mapped_ptrs_(std::move(mapped_ptrs)),
      multicast_memory_(std::move(multicast_memory)) {}

namespace {

// Wrap GpuCliqueKey into a unique struct to guarantee we do not accidentally
// try to run multiple unrelated rendezvous for a same key.
struct AllocateRendezvousKey {
  GpuCliqueKey clique_key;

  bool operator==(const AllocateRendezvousKey& other) const {
    return clique_key == other.clique_key;
  }

  template <typename H>
  friend H AbslHashValue(H h, const AllocateRendezvousKey& key) {
    return H::combine(std::move(h), key.clique_key);
  }
};

// Parameters passed to the rendezvous callback from all ranks.
struct AllocateParams {
  se::StreamExecutor* executor;
  RankId rank;
  se::DeviceAddressBase map_to;
};

struct RankCmp {
  bool operator()(const AllocateParams* a, const AllocateParams* b) const {
    return a->rank < b->rank;
  }
};

struct RankFormatter {
  void operator()(std::string* out, const AllocateParams* param) const {
    absl::StrAppend(out, param->rank.value());
  }
};

struct MappedPtrFormatter {
  void operator()(std::string* out,
                  const std::pair<RankId, void*>& mapped_ptr) const {
    auto& [rank, ptr] = mapped_ptr;
    absl::StrAppend(out, absl::StrFormat("%d:%p", rank.value(), ptr));
  }
};

}  // namespace

absl::StatusOr<std::shared_ptr<CollectiveMultimem>>
CollectiveMultimem::Allocate(se::StreamExecutor& executor,
                             const GpuCliqueKey& clique_key, RankId rank,
                             se::DeviceAddressBase map_to) {
  VLOG(3) << absl::StrFormat(
      "rank=[%d] Allocate collective multimem for clique: %s", rank.value(),
      clique_key.ToString());

  // We rely on in-process rendezvous to allocate the multicast memory and set
  // up memory mapping on all ranks, and don't support multi-process mode.
  if (!clique_key.is_local()) {
    return Unimplemented(
        "%sMultimem is not supported in multi-process mode in clique %s",
        XlaFormatDevice(executor.device_ordinal()), clique_key.ToString());
  }

  std::string rendezvous_name = absl::StrFormat(
      "CollectiveMultimem::Allocate for clique %s", clique_key.ToString());
  AllocateRendezvousKey rendezvous_key = {clique_key};
  AllocateParams params = {&executor, rank, map_to};

  // A callback for rendezvous to allocate and map the multicast memory.
  auto allocate = [&](absl::Span<const AllocateParams*> params)
      -> absl::StatusOr<CollectiveMultimem> {
    // Sort all participants by rank to get deterministic execution.
    absl::c_sort(params, RankCmp{});

    VLOG(3) << absl::StrFormat(
        "ranks=[%s] Allocate collective multimem for clique: %s",
        absl::StrJoin(params, ",", RankFormatter{}), clique_key.ToString());

    // We deterministically choose the first device to create the
    // multicast memory. We will map the rest of participants to it later.
    auto* gpu_executor =
        dynamic_cast<se::gpu::GpuExecutor*>(params[0]->executor);
    if (gpu_executor == nullptr) {
      return absl::UnimplementedError("Unsupported stream executor type");
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::gpu::MulticastMemory> multicast_memory,
        gpu_executor->CreateMulticastMemory(params[0]->map_to.size(),
                                            params.size()));

    // For all participating devices, subscribe to the multicast object.
    for (const auto* param : params) {
      TF_RETURN_IF_ERROR(
          multicast_memory->SubscribeDevice(param->executor->device_ordinal()));
    }

    // For all participating devices, map to the multicast memory.
    absl::btree_map<RankId, void*> mapped_ptrs;
    for (const auto* param : params) {
      TF_ASSIGN_OR_RETURN(
          mapped_ptrs[param->rank],
          multicast_memory->MapMemory(
              param->map_to,
              dynamic_cast<se::gpu::GpuExecutor*>(param->executor)));
    }

    VLOG(3) << absl::StrFormat(
        "Allocated collective multimem for clique: %s; mapped_ptrs: [%s]",
        clique_key.ToString(),
        absl::StrJoin(mapped_ptrs, ", ", MappedPtrFormatter{}));

    return CollectiveMultimem(clique_key, std::move(mapped_ptrs),
                              std::move(multicast_memory));
  };

  // We expect that all local participants will collectively allocate the
  // multicast memory.
  int64_t num_participants = clique_key.num_local_participants();
  return Rendezvous<CollectiveMultimem>(rendezvous_name, rendezvous_key, params,
                                        num_participants, allocate);
}

absl::StatusOr<std::shared_ptr<CollectiveMultimem>>
CollectiveMultimem::Allocate(se::StreamExecutor& executor,
                             const GpuCliqueKey& clique_key,
                             GlobalDeviceId global_device_id,
                             se::DeviceAddressBase map_to) {
  if (std::optional<RankId> rank = clique_key.rank(global_device_id)) {
    return Allocate(executor, clique_key, *rank, map_to);
  }
  return InvalidArgument("Rank not found for device %v", global_device_id);
}
}  // namespace xla::gpu
