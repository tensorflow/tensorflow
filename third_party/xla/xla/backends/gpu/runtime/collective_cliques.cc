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

#include "xla/backends/gpu/runtime/collective_cliques.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

CollectiveCliques::CollectiveCliques(AcquiredCliquesMap cliques_map)
    : cliques_map_(std::move(cliques_map)) {}

absl::StatusOr<GpuCommunicator*> CollectiveCliques::GetComm(
    const GpuCliqueKey& clique_key, RankId rank) const {
  // Check that we locked access to a clique for `clique_key`.
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %s",
                    clique_key.ToString());
  }

  // Check that clique has a communicator for our rank.
  auto communicator = (*clique->second)->comm(rank);
  if (!communicator.has_value()) {
    return Internal("Communicator for rank %v not found in a NCCL clique %s",
                    rank, clique_key.ToString());
  }

  auto* gpu_communicator = dynamic_cast<GpuCommunicator*>(*communicator);
  if (!gpu_communicator) {
    return Internal("Communicator for rank %v is not a GpuCommunicator", rank);
  }

  return gpu_communicator;
}

absl::StatusOr<GpuCommunicator*> CollectiveCliques::GetComm(
    const GpuCliqueKey& clique_key, GlobalDeviceId global_device_id) const {
  std::optional<RankId> rank = clique_key.rank(global_device_id);
  if (!rank.has_value()) {
    return InvalidArgument("Rank not found for device %v", global_device_id);
  }
  return GetComm(clique_key, *rank);
}

absl::StatusOr<bool> CollectiveCliques::peer_access_enabled(
    const GpuCliqueKey& clique_key) const {
  // Check that we locked access to a clique for `clique_key`.
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %s",
                    clique_key.ToString());
  }

  return (*clique->second)->peer_access_enabled();
}

absl::StatusOr<CollectiveCliques> AcquireCollectiveCliques(
    const CollectiveParams& params, const CollectiveCliqueRequests& cliques) {
  std::vector<CollectiveCliqueRequests::CliqueRequest> ordered_cliques =
      cliques.OrderedRequestedCliques();
  if (ordered_cliques.empty()) {
    return CollectiveCliques();
  }

  VLOG(2) << absl::StreamFormat(
      "[%d] [run=%v] Acquire %d collective cliques for global device id %v; "
      "max number of channels for collectives %d; max number of "
      "channels for p2p %d",
      params.executor->device_ordinal(), params.run_id, ordered_cliques.size(),
      params.global_device_id, params.collective_max_nchannels,
      params.p2p_max_nchannels);

  for (size_t i = 0; i < ordered_cliques.size(); ++i) {
    const CollectiveCliqueRequests::CliqueRequest& r = ordered_cliques[i];
    VLOG(2) << absl::StreamFormat(
        "[%d]    clique #%d (global device %v): "
        "num_local_participants=%d; id=%d; key=%s; dev_comms=[%s]",
        params.executor->device_ordinal(), i, params.global_device_id,
        r.key.num_local_participants(), r.id, r.key.ToString(),
        absl::StrJoin(r.dev_comms, ", "));
  }

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "AcquireCollectiveCliques", {{"num_cliques", ordered_cliques.size()}});
  });

  AcquiredCliquesMap cliques_map;
  auto start_micros = tsl::Env::Default()->NowMicros();

  for (const CollectiveCliqueRequests::CliqueRequest& r : ordered_cliques) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);

    if (!rank.has_value()) {
      return Internal("Can't find global device id %v in clique key %s",
                      params.global_device_id, r.key.ToString());
    }

    // Default clique id callback that generates a unique clique id for the
    // clique key. This callback supports only local cliques (all ranks belong
    // to the same process), as otherwise clique id should be exchanged across
    // multiple processes via an external storage (i.e. builtin KV store).
    //
    // IMPORTANT: This callback is called once for the clique key by the
    // rendezvous leader elected inside the `AcquireGpuClique` implementation.
    CliqueIdCallback default_clique_id_callback =
        [&](const CliqueKey& key) -> absl::StatusOr<CliqueIds> {
      VLOG(4) << absl::StrFormat("Get local NCCL clique ids: clique=%v", key);
      auto& gpu_key = tsl::down_cast<const GpuCliqueKey&>(key);
      if (!gpu_key.is_local()) {
        return Internal(
            "For non-local GPU cliques (cliques that span multiple processes) "
            "clique id callback must be passed via execution params");
      }
      TF_ASSIGN_OR_RETURN(CliqueId clique_id,
                          params.collectives->CreateUniqueCliqueId());
      return CliqueIds(clique_id);
    };

    int64_t max_channels = r.key.is_p2p() ? params.p2p_max_nchannels
                                          : params.collective_max_nchannels;

    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<LockableGpuClique::Lock> clique,
        AcquireGpuClique(params.collectives, params.executor, params.run_id,
                         r.key, r.device_groups,
                         params.clique_id_callback ? *params.clique_id_callback
                                                   : default_clique_id_callback,
                         *rank, cliques_map, max_channels));

    cliques_map[r.key] = std::move(clique);
  }

  auto end_micros = tsl::Env::Default()->NowMicros();
  VLOG(2) << absl::StreamFormat(
      "[%d] [global_device=%v] [run=%v] Acquired %d collective cliques in %s; ",
      params.executor->device_ordinal(), params.global_device_id, params.run_id,
      cliques_map.size(),
      absl::FormatDuration(absl::Microseconds(end_micros - start_micros)));

  // After we acquired all GPU cliques, check if they already have required
  // device communicators, and create them if needed. Creating device
  // communicators is a collective operation that must be executed by all ranks,
  // but luckily we already are inside the collective function, so we can safely
  // create missing communicators here.
  for (const CollectiveCliqueRequests::CliqueRequest& r : ordered_cliques) {
    std::optional<RankId> rank = r.key.rank(params.global_device_id);
    std::shared_ptr<LockableGpuClique::Lock> clique = cliques_map.at(r.key);

    for (const GpuDeviceCommunicator::Requirements& reqs : r.dev_comms) {
      // Device communicator already exists in the GPU clique.
      if ((*clique)->device_comm(*rank, reqs)) {
        continue;
      }

      VLOG(2) << absl::StreamFormat(
          "[%d] Create device communicator: rank=%v clique=%s",
          params.executor->device_ordinal(), *rank, r.key.ToString());

      auto* comm = dynamic_cast<GpuCommunicator*>(*(*clique)->comm(*rank));
      DCHECK(comm) << "Communicator must be in the acquired clique";
      TF_ASSIGN_OR_RETURN(std::unique_ptr<GpuDeviceCommunicator> dev_comm,
                          comm->CreateDeviceComm(reqs));
      TF_RETURN_IF_ERROR(
          (*clique)->AddDeviceComm(*rank, reqs, std::move(dev_comm)));
    }
  }

  return CollectiveCliques(std::move(cliques_map));
}

}  // namespace xla::gpu
