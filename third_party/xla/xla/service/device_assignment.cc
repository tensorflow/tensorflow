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

#include "xla/service/device_assignment.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/runtime/device_id.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

using absl::StrAppend;

namespace xla {

absl::StatusOr<DeviceAssignment::LogicalID>
DeviceAssignment::LogicalIdForDevice(GlobalDeviceId device_id) const {
  std::optional<LogicalID> res;
  int64_t id = device_id.value();
  for (int r = 0; r < replica_count(); ++r) {
    for (int c = 0; c < computation_count(); ++c) {
      if (operator()(r, c) == device_id.value()) {
        if (res.has_value()) {
          return Internal("Device %d not unique in %v", id, *this);
        }
        res = LogicalID{r, c};
      }
    }
  }
  if (!res.has_value()) {
    return Internal("Device %d doesn't appear in %v", id, *this);
  }
  return res.value();
}

absl::StatusOr<int> DeviceAssignment::ReplicaIdForDevice(
    GlobalDeviceId device_id) const {
  ABSL_ASSIGN_OR_RETURN(const LogicalID logical_id, LogicalIdForDevice(device_id));
  return logical_id.replica_id;
}

absl::StatusOr<int> DeviceAssignment::PartitionIdForDevice(
    GlobalDeviceId device_id) const {
  ABSL_ASSIGN_OR_RETURN(const LogicalID logical_id, LogicalIdForDevice(device_id));
  return logical_id.computation_id;
}

absl::flat_hash_map<GlobalDeviceId, DeviceAssignment::LogicalID>
DeviceAssignment::GetDeviceToLogicalIdMap() const {
  absl::flat_hash_map<GlobalDeviceId, DeviceAssignment::LogicalID>
      device_to_logical_id;
  for (int r = 0; r < replica_count(); ++r) {
    for (int c = 0; c < computation_count(); ++c) {
      GlobalDeviceId device_id((*this)(r, c));
      device_to_logical_id[device_id] = DeviceAssignment::LogicalID{r, c};
    }
  }
  return device_to_logical_id;
}

bool DeviceAssignment::IsIota() const {
  if (num_elements() == 0) {
    return true;
  }

  int64_t offset = data()[0];
  for (int i = 0; i < num_elements(); ++i) {
    if (data()[i] != i + offset) {
      return false;
    }
  }
  return true;
}

bool DeviceAssignment::IsAll(int64_t val) const {
  for (int i = 0; i < num_elements(); ++i) {
    if (data()[i] != val) {
      return false;
    }
  }
  return true;
}

void DeviceAssignment::Serialize(DeviceAssignmentProto* proto) const {
  proto->set_replica_count(replica_count());
  proto->set_computation_count(computation_count());
  for (int computation = 0; computation < computation_count(); ++computation) {
    DeviceAssignmentProto::ComputationDevice* computation_device =
        proto->add_computation_devices();
    for (int replica = 0; replica < replica_count(); ++replica) {
      computation_device->add_replica_device_ids((*this)(replica, computation));
    }
  }
}

namespace {
#define RET_CHECK_ARG(condition) \
  if (!(condition)) return absl::InvalidArgumentError(#condition);
}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<DeviceAssignment>>
DeviceAssignment::Deserialize(const DeviceAssignmentProto& proto) {
  RET_CHECK_ARG(proto.computation_devices_size() == proto.computation_count());
  RET_CHECK_ARG(proto.replica_count() > 0);
  RET_CHECK_ARG(proto.computation_count() > 0);
  auto da = std::make_unique<DeviceAssignment>(proto.replica_count(),
                                               proto.computation_count());
  for (int comp_id = 0; comp_id < proto.computation_count(); ++comp_id) {
    const auto& comp = proto.computation_devices(comp_id);
    RET_CHECK_ARG(comp.replica_device_ids_size() == proto.replica_count());
    for (int replica = 0; replica < proto.replica_count(); ++replica) {
      (*da)(replica, comp_id) = comp.replica_device_ids(replica);
    }
  }
  return std::move(da);
}

std::string DeviceAssignment::ToString() const {
  std::string output = absl::StrFormat(
      "DeviceAssignment{replica_count=%d, computation_count=%d,",
      replica_count(), computation_count());
  for (int computation = 0; computation < computation_count(); ++computation) {
    StrAppend(&output, " Computation", computation, "{");
    for (int replica = 0; replica < replica_count(); ++replica) {
      if (replica > 0) {
        StrAppend(&output, " ");
      }
      int device_id = operator()(replica, computation);
      StrAppend(&output, device_id);
    }
    StrAppend(&output, "}");
  }
  StrAppend(&output, "}");
  return output;
}

}  // namespace xla
