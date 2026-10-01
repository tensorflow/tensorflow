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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

absl::Status ValidateFunctionDeviceConstraints(
    const FunctionDef& fdef,
    const std::vector<DeviceAttributes>& available_devices) {
  // Pre-parse available devices once upfront.
  std::vector<DeviceNameUtils::ParsedName> parsed_available_devices;
  parsed_available_devices.reserve(available_devices.size());
  for (const auto& dev : available_devices) {
    DeviceNameUtils::ParsedName parsed;
    if (DeviceNameUtils::ParseFullName(dev.name(), &parsed)) {
      parsed_available_devices.push_back(parsed);
    }
  }

  for (const NodeDef& node : fdef.node_def()) {
    const std::string& device = node.device();
    if (device.empty()) {
      continue;
    }

    DeviceNameUtils::ParsedName parsed_device;
    if (!DeviceNameUtils::ParseFullOrLocalName(device, &parsed_device)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Malformed device specification '", device, "' for operation ",
          node.name(), " (", node.op(), ")."));
    }

    if (!DeviceNameUtils::HasSomeDetails(parsed_device)) {
      continue;
    }

    // The constraint is satisfied if at least one available device matches
    // every component specified in the constraint. IsSpecification checks
    // all specified attributes without requiring the constraint to be fully
    // specified, so partial constraints (e.g. '/job:localhost') and local
    // names (e.g. 'CPU:0') are handled uniformly.
    bool satisfied = false;
    for (const auto& avail_parsed : parsed_available_devices) {
      if (DeviceNameUtils::IsSpecification(parsed_device, avail_parsed)) {
        satisfied = true;
        break;
      }
    }

    if (!satisfied) {
      std::vector<std::string> available_device_names;
      available_device_names.reserve(available_devices.size());
      for (const auto& dev : available_devices) {
        available_device_names.push_back(dev.name());
      }
      return absl::InvalidArgumentError(absl::StrCat(
          "Could not satisfy device specification '", device,
          "' for operation ", node.name(), " (", node.op(),
          "). Available devices [",
          absl::StrJoin(available_device_names, ", "), "]."));
    }
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
