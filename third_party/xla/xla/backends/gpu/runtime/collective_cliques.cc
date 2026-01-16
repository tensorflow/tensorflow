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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

CollectiveCliques::CollectiveCliques(AcquiredCliquesMap cliques_map)
    : cliques_map_(std::move(cliques_map)) {}

absl::StatusOr<Communicator*> CollectiveCliques::GetComm(
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
    return Internal("Communicator for rank %d not found in a NCCL clique %s",
                    rank.value(), clique_key.ToString());
  }

  return *communicator;
}

absl::StatusOr<Communicator*> CollectiveCliques::GetComm(
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

  VLOG(2) << "Acquire " << ordered_cliques.size()
          << " collective cliques for global device id "
          << params.global_device_id.value()
          << "; run_id=" << params.run_id.ToInt()
          << "; max number of channels for collectives "
          << params.collective_max_nchannels
          << "; max number of channels for p2p " << params.p2p_max_nchannels;

  for (size_t i = 0; i < ordered_cliques.size(); ++i) {
    const CollectiveCliqueRequests::CliqueRequest& r = ordered_cliques[i];
    VLOG(2) << "  clique #" << i << " (for global device id "
            << params.global_device_id.value() << ")"
            << ": num_local_participants=" << r.key.num_local_participants()
            << "; id=" << r.id << "; key=" << r.key.ToString();
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
      return Internal("Can't find global device id %d in clique key %s",
                      params.global_device_id.value(), r.key.ToString());
    }

    TF_ASSIGN_OR_RETURN(const CliqueIdCallback* clique_id_callback,
                        params.collectives->GetCliqueIdCallback(
                            params.nccl_clique_id_callback, r.key.is_local()));

    int64_t max_channels = r.key.is_p2p() ? params.p2p_max_nchannels
                                          : params.collective_max_nchannels;

    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<LockableGpuClique::Lock> clique,
        AcquireGpuClique(params.collectives, params.executor, params.run_id,
                         r.key, *clique_id_callback, *rank, cliques_map,
                         max_channels));

    cliques_map[r.key] = std::move(clique);
  }

  auto end_micros = tsl::Env::Default()->NowMicros();
  VLOG(2) << "Acquired " << cliques_map.size()
          << " collective cliques for global device id "
          << params.global_device_id.value() << " in "
          << (end_micros - start_micros) << " μs"
          << "; run_id=" << params.run_id.ToInt();

  return CollectiveCliques(std::move(cliques_map));
}

}  // namespace xla::gpu
