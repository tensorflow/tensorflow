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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

absl::Status ValidateFunctionDeviceConstraintsImpl(
    const FunctionDef& fdef, const FunctionLibraryDefinition* flib_def,
    const std::vector<DeviceAttributes>& available_devices,
    const std::vector<DeviceNameUtils::ParsedName>& parsed_available_devices,
    absl::flat_hash_set<std::string>& validated_devices,
    absl::flat_hash_set<std::string>& visited_functions) {
  for (const NodeDef& node : fdef.node_def()) {
    const std::string& device = node.device();
    if (!device.empty() && !validated_devices.contains(device)) {
      DeviceNameUtils::ParsedName parsed_device;
      if (!DeviceNameUtils::ParseFullOrLocalName(device, &parsed_device)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Malformed device specification '", device, "' for operation ",
            node.name(), " (", node.op(), ")."));
      }

      if (DeviceNameUtils::HasSomeDetails(parsed_device)) {
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
      validated_devices.insert(device);
    }

    // Recurse into nested functions referenced through PartitionedCall /
    // StatefulPartitionedCall nodes so their device constraints are validated
    // too. Regular ops are also looked up defensively: op-to-function mappings
    // can exist in the library (e.g. via function optimization passes), and
    // Find() simply returns nullptr when the op is a plain op.
    if (flib_def != nullptr) {
      std::vector<std::string> inner_funcs;
      inner_funcs.push_back(node.op());
      if (node.op() == "PartitionedCall" ||
          node.op() == "StatefulPartitionedCall") {
        auto it = node.attr().find(FunctionLibraryDefinition::kFuncAttr);
        if (it != node.attr().end() && it->second.has_func()) {
          inner_funcs.push_back(it->second.func().name());
        }
      }

      for (const std::string& func_name : inner_funcs) {
        const FunctionDef* inner_fdef = flib_def->Find(func_name);
        if (inner_fdef != nullptr &&
            visited_functions.insert(func_name).second) {
          absl::Status inner_status = ValidateFunctionDeviceConstraintsImpl(
              *inner_fdef, flib_def, available_devices,
              parsed_available_devices, validated_devices, visited_functions);
          if (!inner_status.ok()) {
            return inner_status;
          }
        }
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateFunctionDeviceConstraints(
    const FunctionDef& fdef, const FunctionLibraryDefinition* flib_def,
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

  absl::flat_hash_set<std::string> validated_devices;
  absl::flat_hash_set<std::string> visited_functions;
  visited_functions.insert(fdef.signature().name());

  return ValidateFunctionDeviceConstraintsImpl(
      fdef, flib_def, available_devices, parsed_available_devices,
      validated_devices, visited_functions);
}

}  // namespace tensorflow
