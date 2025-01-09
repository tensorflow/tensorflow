/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/framework/device_id_utils.h"

#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/framework/device_id_manager.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/str_util.h"

namespace tsl {
namespace {

absl::StatusOr<int> ParsePlatformDeviceIdString(
    absl::string_view platform_device_id_str, absl::string_view device_type) {
  int32_t platform_device_id;
  if (!absl::SimpleAtoi(platform_device_id_str, &platform_device_id)) {
    // Pluggable device would have both device type and id in the string.
    const std::vector<std::string> device_type_and_id =
        tsl::str_util::Split(platform_device_id_str, ':');  // non-absl ok
    if (device_type_and_id.size() != 2 ||
        !absl::SimpleAtoi(device_type_and_id[1], &platform_device_id)) {
      return tsl::errors::InvalidArgument(
          "Could not parse entry in 'visible_device_list': '",
          platform_device_id_str, "'.");
    }
    if (!device_type.empty() && device_type_and_id[0] != device_type) {
      return -1;  // Return -1 to indicate that the device type doesn't match.
    }
  }
  return platform_device_id;
}

}  // namespace

void CheckValidTfDeviceId(const DeviceType& type,
                          const int visible_device_count,
                          const TfDeviceId tf_device_id) {
  PlatformDeviceId platform_device_id;
  TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(type, tf_device_id,
                                                    &platform_device_id));
  CHECK_LT(platform_device_id.value(), visible_device_count)  // Crash OK
      << "platform_device_id is outside discovered device range."
      << " TF " << type << " id: " << tf_device_id << ", platform " << type
      << " id: " << platform_device_id
      << ", visible device count: " << visible_device_count;
}

absl::Status ParseVisibleDeviceList(
    const std::string& visible_device_list, const int visible_device_count,
    std::vector<PlatformDeviceId>* visible_device_order,
    absl::string_view device_type) {
  visible_device_order->clear();

  // If the user wants to remap the visible to virtual Device mapping,
  // check for that here.
  if (visible_device_list.empty()) {
    visible_device_order->resize(visible_device_count);
    // By default, visible to virtual mapping is unchanged.
    std::iota(visible_device_order->begin(), visible_device_order->end(), 0);
  } else {
    const std::vector<std::string> order_str =
        tsl::str_util::Split(visible_device_list, ',');  // non-absl ok
    for (const std::string& platform_device_id_str : order_str) {
      int32_t platform_device_id;
      TF_ASSIGN_OR_RETURN(
          platform_device_id,
          ParsePlatformDeviceIdString(platform_device_id_str, device_type));
      if (platform_device_id == -1) {
        continue;  // Skip the device if the device type doesn't match.
      }
      if (platform_device_id < 0 ||
          platform_device_id >= visible_device_count) {
        return tsl::errors::InvalidArgument(
            "'visible_device_list' listed an invalid Device id '",
            platform_device_id, "' but visible device count is ",
            visible_device_count);
      }
      visible_device_order->push_back(
          tsl::PlatformDeviceId(platform_device_id));
    }
  }

  // Validate no repeats.
  std::set<PlatformDeviceId> visible_device_set(visible_device_order->begin(),
                                                visible_device_order->end());
  if (visible_device_set.size() != visible_device_order->size()) {
    return tsl::errors::InvalidArgument(
        "visible_device_list contained a duplicate entry: ",
        visible_device_list);
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> GetNumberTfDevicesAndConfigurePlatformDeviceId(
    const absl::flat_hash_map<std::string, int64_t>&
        session_option_device_counts,
    absl::string_view device_type, absl::string_view visible_device_list,
    const int visible_device_count) {
  size_t num_tf_devices = INT_MAX;
  const auto iter = session_option_device_counts.find(device_type);
  if (iter != session_option_device_counts.end()) {
    num_tf_devices = iter->second;
  }
  if (num_tf_devices == 0) {
    return 0;
  }
  std::vector<PlatformDeviceId> visible_device_order;
  TF_RETURN_IF_ERROR(ParseVisibleDeviceList(
      std::string(visible_device_list), visible_device_count,
      &visible_device_order, device_type));
  if (num_tf_devices > visible_device_order.size()) {
    num_tf_devices = visible_device_order.size();
  }
  for (int i = 0; i < num_tf_devices; ++i) {
    const PlatformDeviceId platform_device_id = visible_device_order[i];
    const TfDeviceId tf_device_id(i);
    TF_RETURN_IF_ERROR(tsl::DeviceIdManager::InsertTfPlatformDeviceIdPair(
        DeviceType(device_type), tf_device_id, platform_device_id));
  }
  return num_tf_devices;
}

absl::StatusOr<int> GetPlatformDeviceIdFromDeviceParsedName(
    const DeviceNameUtils::ParsedName& device_name,
    const DeviceType& device_type) {
  const TfDeviceId tf_device_id(GetDeviceIdFromDeviceParsedName(device_name));
  PlatformDeviceId platform_device_id;
  absl::Status platform_id_status = DeviceIdManager::TfToPlatformDeviceId(
      device_type, tf_device_id, &platform_device_id);
  if (platform_id_status.ok()) {
    return platform_device_id.value();
  }
  return platform_id_status;
}

int GetDeviceIdFromDeviceParsedName(
    const DeviceNameUtils::ParsedName& device_name) {
  // This assumes that TF device ID is the same as PJRT local device ID.
  return device_name.id;
}

}  // namespace tsl
