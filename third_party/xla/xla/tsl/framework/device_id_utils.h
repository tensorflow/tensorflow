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

#ifndef XLA_TSL_FRAMEWORK_DEVICE_ID_UTILS_H_
#define XLA_TSL_FRAMEWORK_DEVICE_ID_UTILS_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/framework/device_type.h"
#include "xla/tsl/util/device_name_utils.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tsl {

// Utility methods for translation between TensorFlow device ids and platform
// device ids.

// Verify that the platform_device_id associated with a TfDeviceId is
// legitimate.
void CheckValidTfDeviceId(const DeviceType& type, int visible_device_count,
                          TfDeviceId tf_device_id);

// Parse `visible_device_list` into a list of platform Device ids.
// When parsing non-PluggableDevices, the `device_type` parameter is
// optional (can be empty) and ignored. When using this function to
// parse the `visible_device_list` for PluggableDevices, the pluggable
// device type will be included in the `visible_device_list`, e.g.
// "PluggableDeviceA:0,PluggableDeviceA:1,PluggableDeviceB:0".
// In this case, the `device_type` parameter should be set to the
// corresponding pluggable device type to be parsed, e.g.
// "PluggableDeviceA". And the other types of PluggableDevices
// in the `visible_device_list` will be ignored.
absl::Status ParseVisibleDeviceList(
    const std::string& visible_device_list, int visible_device_count,
    std::vector<PlatformDeviceId>* visible_device_order,
    absl::string_view device_type = "");

// Returns how many TF devices should be created, and generates the mapping
// between TfDeviceId and PlatformDeviceId. The number of TF devices is the
// minimum among the device count in `session_option_device_counts`,
// `visible_device_count` and the number of visible devices in
// `visible_device_list`. If `visible_device_list` is empty, the mapping
// between TfDeviceId and PlatformDeviceId is an identity mapping.
// Please refer to tensorflow/compiler/xla/tsl/framework/device_id.h and
// tensorflow/core/protobuf/config.proto about the relationship between
// TfDeviceId and PlatformDeviceId, and how `visible_device_list` is used.
absl::StatusOr<size_t> GetNumberTfDevicesAndConfigurePlatformDeviceId(
    const absl::flat_hash_map<std::string, int64_t>&
        session_option_device_counts,
    absl::string_view device_type, absl::string_view visible_device_list,
    int visible_device_count);

absl::StatusOr<int> GetPlatformDeviceIdFromDeviceParsedName(
    const DeviceNameUtils::ParsedName& device_name,
    const DeviceType& device_type);

// Returns the id in device_name.
int GetDeviceIdFromDeviceParsedName(
    const DeviceNameUtils::ParsedName& device_name);

}  // namespace tsl

#endif  // XLA_TSL_FRAMEWORK_DEVICE_ID_UTILS_H_
