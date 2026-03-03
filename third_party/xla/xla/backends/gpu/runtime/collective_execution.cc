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
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

static int64_t GetNumLocalParticipants(
    const CollectiveParams& params, absl::Span<const GlobalDeviceId> devices) {
  if (!params.global_device_id_map) {
    return devices.size();
  }

  std::vector<GlobalDeviceId> local_devices;
  local_devices.reserve(params.global_device_id_map->size());
  for (const auto& entry : *params.global_device_id_map) {
    local_devices.push_back(entry.second);
  }

  return absl::c_count_if(devices, [&](const GlobalDeviceId& device) {
    return absl::c_linear_search(local_devices, device);
  });
}

absl::StatusOr<GpuCliqueKey> GetGpuCliqueKey(
    const CollectiveParams& params,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode, bool is_p2p) {
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
      std::vector<GlobalDeviceId> devices,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              replica_groups, group_mode));

  int64_t num_local_participants = GetNumLocalParticipants(params, devices);

  absl::flat_hash_set<IncarnationId> unique_incarnations;
  if (params.incarnations) {
    for (GlobalDeviceId device : devices) {
      auto it = params.incarnations->find(device);
      if (it == params.incarnations->end()) {
        return FailedPrecondition("Incarnation for device %v not found",
                                  device);
      }
      unique_incarnations.insert(it->second);
    }
  }
  std::vector<IncarnationId> incarnations(unique_incarnations.begin(),
                                          unique_incarnations.end());
  absl::c_sort(incarnations);

  return GpuCliqueKey(std::move(devices), num_local_participants, is_p2p,
                      incarnations);
}

}  // namespace xla::gpu
