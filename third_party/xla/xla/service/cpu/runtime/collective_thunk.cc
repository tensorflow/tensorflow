/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/runtime/collective_thunk.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/global_device_id.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

CollectiveThunk::CollectiveThunk(Kind kind, Thunk::Info info,
                                 OpParams op_params)
    : Thunk(kind, info), op_params_(std::move(op_params)) {}

absl::StatusOr<RendezvousKey> CollectiveThunk::GetRendezvousKey(
    const Thunk::CollectiveExecuteParams& params) {
  TF_RET_CHECK(params.device_assignment) << "Device assignment is null";

  const DeviceAssignment& device_assignment = *params.device_assignment;
  RendezvousKey::CollectiveOpKind op_kind = op_params_.has_channel_id
                                                ? RendezvousKey::kCrossModule
                                                : RendezvousKey::kCrossReplica;

  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode group_mode,
      GetCollectiveOpGroupMode(op_params_.has_channel_id,
                               op_params_.use_global_device_ids));

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participating_devices,
      GetParticipatingDevices(params.global_device_id, device_assignment,
                              op_params_.group, group_mode));

  int num_local_participants = participating_devices.size();
  return RendezvousKey{params.run_id, std::move(participating_devices),
                       num_local_participants, op_kind, op_params_.op_id};
}

absl::StatusOr<int32_t> CollectiveThunk::RankInGlobalDevices(
    const RendezvousKey& key, GlobalDeviceId device) {
  auto it = absl::c_find(key.global_devices, device);
  if (it == key.global_devices.end()) {
    return InvalidArgument(
        "Device %d not present in global devices %s.", device.value(),
        absl::StrJoin(key.global_devices, ", ",
                      [](std::string* out, GlobalDeviceId id) {
                        absl::StrAppend(out, id.value());
                      }));
  }
  return std::distance(key.global_devices.begin(), it);
}

}  // namespace xla::cpu
