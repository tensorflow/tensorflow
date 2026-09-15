/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/validate_function_devices.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

absl::Status ValidateFunctionDeviceConstraints(
    const FunctionDef& fdef,
    const std::vector<DeviceAttributes>& available_devices) {
  // Build a set of available device types and full device names.
  std::vector<DeviceType> supported_device_types;
  std::vector<std::string> available_device_names;
  for (const auto& dev : available_devices) {
    DeviceNameUtils::ParsedName parsed;
    if (DeviceNameUtils::ParseFullName(dev.name(), &parsed)) {
      supported_device_types.push_back(DeviceType(parsed.type));
    }
    available_device_names.push_back(dev.name());
  }

  for (const NodeDef& node : fdef.node_def()) {
    const std::string& device = node.device();
    if (device.empty()) {
      continue;  // No device constraint on this node.
    }

    DeviceNameUtils::ParsedName parsed_device;
    if (!DeviceNameUtils::ParseFullName(device, &parsed_device)) {
      // If the device name can't be parsed, skip it. Other validation passes
      // will catch malformed device names.
      continue;
    }

    // Check if the requested device type is available.
    bool device_type_available = false;
    for (const DeviceType& avail_type : supported_device_types) {
      if (absl::EqualsIgnoreCase(parsed_device.type, avail_type.type_string())) {
        device_type_available = true;
        break;
      }
    }

    if (!device_type_available) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Could not satisfy device specification '", device,
          "' for operation ", node.name(), " (", node.op(),
          "). Available devices [",
          absl::StrJoin(available_device_names, ", "), "]."));
    }

    // For fully-specified device names, check if an exact match exists.
    if (parsed_device.has_job && parsed_device.has_replica &&
        parsed_device.has_task && parsed_device.has_type &&
        parsed_device.has_id) {
      bool exact_match = false;
      for (const auto& dev : available_devices) {
        DeviceNameUtils::ParsedName avail_parsed;
        if (DeviceNameUtils::ParseFullName(dev.name(), &avail_parsed)) {
          if (DeviceNameUtils::IsCompleteSpecification(parsed_device,
                                                       avail_parsed)) {
            exact_match = true;
            break;
          }
        }
      }

      if (!exact_match) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Could not satisfy device specification '", device,
            "' for operation ", node.name(), " (", node.op(),
            "). Available devices [",
            absl::StrJoin(available_device_names, ", "), "]."));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
