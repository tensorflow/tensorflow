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

#include "xla/backends/gpu/runtime/collective_execution.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/communicator.h"
#include "xla/debug_options_flags.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

static int64_t GetNumLocalParticipants(
    const CollectiveParams& params,
    absl::Span<const GlobalDeviceId> participants) {
  if (!params.global_device_id_map) {
    return participants.size();
  }

  std::vector<GlobalDeviceId> local_devices;
  local_devices.reserve(params.global_device_id_map->size());
  for (const auto& entry : *params.global_device_id_map) {
    local_devices.push_back(entry.second);
  }

  return absl::c_count_if(participants, [&](const GlobalDeviceId& device_id) {
    return absl::c_linear_search(local_devices, device_id);
  });
}

absl::StatusOr<GpuCliqueKey> GetGpuCliqueKey(
    const CollectiveParams& params,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode, bool is_p2p,
    bool include_participant_groups) {
  TF_RET_CHECK(params.collectives) << "Collectives API is not provided";

  GlobalDeviceId global_device_id = params.global_device_id;

  if (params.device_assn == nullptr) {
    return InvalidArgument(
        "Device assignment is null, but must be specified when running a "
        "collective thunk. If running multi-device HLO , make sure you're not "
        "using a tool designed for only one device like run_hlo_module.");
  }

  // Get the list of all devices that are participating in the collective
  // operation.
  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              replica_groups, group_mode));

  // Get grouping of participating devices.
  std::vector<std::vector<GlobalDeviceId>> participant_groups;
  if (include_participant_groups) {
    // If splitting is enabled, participating groups must match in order for a
    // clique to be reused from the cache. We can ignore the participating
    // groups otherwise.
    static const bool enable_nccl_comm_splitting =
        xla::GetDebugOptionsFromFlags().xla_gpu_enable_nccl_comm_splitting();
    if (enable_nccl_comm_splitting) {
      TF_ASSIGN_OR_RETURN(participant_groups,
                          GetParticipatingDevicesGroups(
                              *params.device_assn, replica_groups, group_mode));
    }
  }

  // Remove trivial group that contains all participants, as we do not want to
  // create two sets of communicator handles for these cases.
  if (participant_groups.size() == 1 && participant_groups[0] == participants) {
    participant_groups.clear();
  }

  int64_t num_local_participants =
      GetNumLocalParticipants(params, participants);

  absl::flat_hash_set<IncarnationId> unique_incarnations;
  if (params.incarnations) {
    for (GlobalDeviceId id : participants) {
      auto it = params.incarnations->find(id);
      if (it == params.incarnations->end()) {
        return FailedPrecondition("Incarnation for device %d not found",
                                  id.value());
      }
      unique_incarnations.insert(it->second);
    }
  }
  std::vector<IncarnationId> incarnations(unique_incarnations.begin(),
                                          unique_incarnations.end());
  absl::c_sort(incarnations);

  return GpuCliqueKey(std::move(participants), num_local_participants, is_p2p,
                      std::move(participant_groups), incarnations);
}

}  // namespace xla::gpu
