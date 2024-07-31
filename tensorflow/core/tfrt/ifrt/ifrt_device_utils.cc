/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_device_utils.h"

#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/service/computation_placer.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/grid.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {

static constexpr int kTpuTopologyRank = 4;  // (x, y, z, core).

absl::StatusOr<std::vector<xla::ifrt::Device*>> GetAssignedIfrtDevices(
    const xla::ifrt::Client& ifrt_client, int num_replicas,
    int num_cores_per_replica,
    std::optional<std::vector<int>> device_assignment) {
  const int num_devices = num_replicas * num_cores_per_replica;

  // No device coordinates from ifrt devices. This disallow the mapping from
  // device assignment attribute to ifrt devices.
  bool no_device_coordinates = false;
  for (auto* device : ifrt_client.devices()) {
    if (!device->Attributes().map().contains("coords") ||
        !device->Attributes().map().contains("core_on_chip")) {
      no_device_coordinates = true;
      break;
    }
  }

  // If there is no device assignment attribute or no device coordinates,
  // get the default device assignment from IFRT.
  if (!device_assignment || device_assignment->empty() ||
      no_device_coordinates) {
    TF_ASSIGN_OR_RETURN(xla::DeviceAssignment xla_device_assignment,
                        ifrt_client.GetDefaultDeviceAssignment(
                            num_replicas, num_cores_per_replica));
    VLOG(3) << "Getting default device lists";
    std::vector<xla::ifrt::Device*> devices;
    devices.reserve(num_devices);
    for (int replica_idx = 0; replica_idx < num_replicas; replica_idx++) {
      for (int core_idx = 0; core_idx < num_cores_per_replica; core_idx++) {
        // This relies on the IFRT implementation of GetDefaultDeviceAssignment
        // that keeps device id the same between device assignment  and ifrt
        // device list.
        auto device_id = xla_device_assignment(replica_idx, core_idx);
        TF_ASSIGN_OR_RETURN(
            xla::ifrt::Device * device,
            ifrt_client.LookupDevice(xla::ifrt::DeviceId(device_id)));
        devices.push_back(device);
      }
    }
    return devices;
  }

  // Devices ordered as in the device assignment attribute.
  absl::flat_hash_map<GridCoords, int> devices_from_attribute;

  // Each device is encoded by [x,y,z,c] at the attribute.
  std::vector<int> coord;
  coord.reserve(kTpuTopologyRank);
  int device_index = 0;

  for (auto coord_attr : *device_assignment) {
    coord.push_back(coord_attr);
    if (coord.size() == kTpuTopologyRank) {
      devices_from_attribute.insert(
          {GridCoords(coord[0], coord[1], coord[2], coord[3]), device_index});
      device_index++;
      coord.clear();
    }
  }
  if (!coord.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Device assignment attribute is expected to be a multiple "
                     "of 4, but got ",
                     device_assignment->size()));
  }

  if (devices_from_attribute.size() != num_devices) {
    return absl::FailedPreconditionError(
        absl::StrCat("Device assignment has ", devices_from_attribute.size(),
                     " devices, but expected ", num_devices));
  }

  struct IfrtDeviceGrid {
    xla::ifrt::Device* device;
    GridCoords grid;
    int index_at_attribute;
  };
  std::vector<IfrtDeviceGrid> ifrt_devices;
  ifrt_devices.reserve(num_devices);

  for (auto* device : ifrt_client.devices()) {
    GridCoords grid;
    auto coords_it = device->Attributes().map().find("coords");
    auto core_on_chip_it = device->Attributes().map().find("core_on_chip");
    if (coords_it != device->Attributes().map().end() &&
        core_on_chip_it != device->Attributes().map().end()) {
      VLOG(3) << "Adding coords and core_on_chip attributes:"
              << device->DebugString();
      auto coords_list =
          std::get<xla::ifrt::AttributeMap::Int64ListValue>(coords_it->second);
      auto core_on_chip = std::get<xla::ifrt::AttributeMap::Int64Value>(
          core_on_chip_it->second);

      if (coords_list.value.size() != 3) {
        return absl::InternalError(absl::StrCat(
            "Expected coords to be of size 3, but got ",
            coords_list.value.size(), " for device ", device->DebugString()));
      }
      grid = GridCoords(coords_list.value[0], coords_list.value[1],
                        coords_list.value[2], core_on_chip.value);
    } else {
      return absl::InternalError(
          absl::StrCat("Device ", device->DebugString(),
                       " does not have coords or core_on_chip attribute."));
    }

    auto device_it_from_attribute = devices_from_attribute.find(grid);
    if (device_it_from_attribute == devices_from_attribute.end()) {
      VLOG(1) << "Device coordinates " << grid.ToString()
              << " does not match any TPU device assigned "
              << absl::StrJoin(*device_assignment, " ");
      continue;
    }
    ifrt_devices.push_back(
        {.device = device,
         .grid = grid,
         .index_at_attribute = device_it_from_attribute->second});
  }

  if (ifrt_devices.size() != num_devices) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Match ", ifrt_devices.size(), " devices, but expected ", num_devices));
  }

  // Sort the devices by the order in the device assignment attribute.
  absl::c_sort(ifrt_devices, [&](const auto& lhs, const auto& rhs) {
    return lhs.index_at_attribute < rhs.index_at_attribute;
  });

  std::vector<xla::ifrt::Device*> result;
  result.reserve(ifrt_devices.size());
  for (auto& device_grid : ifrt_devices) {
    result.push_back(device_grid.device);
    VLOG(3) << "Device: " << device_grid.device->DebugString()
            << " is assigned";
  }
  return result;
}

}  // namespace ifrt_serving
}  // namespace tensorflow
