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
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

// Validates the device constraints of `fdef` and recurses into every nested
// function reachable through function-bearing node attributes. The
// visited-functions guard makes cyclic call graphs terminate.
absl::Status ValidateFunctionDeviceConstraintsImpl(
    const FunctionDef& fdef, const FunctionLibraryDefinition* flib_def,
    const std::vector<DeviceNameUtils::ParsedName>& parsed_available_devices,
    absl::flat_hash_set<absl::string_view>& validated_devices,
    absl::flat_hash_set<absl::string_view>& visited_functions) {
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
          // The available-device list is only needed for the error message,
          // so it is rendered here instead of on every validation.
          std::vector<std::string> available_device_names;
          available_device_names.reserve(parsed_available_devices.size());
          for (const auto& avail_parsed : parsed_available_devices) {
            available_device_names.push_back(
                DeviceNameUtils::ParsedNameToString(avail_parsed));
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

    // Recurse into nested functions. Function-bearing attributes are not
    // limited to the 'f' attribute of (Stateful)PartitionedCall: control flow
    // ops carry functions in attributes like 'then_branch'/'else_branch'
    // (If) or 'cond'/'body' (While), and list-of-function attributes exist
    // too (e.g. 'branches'). Every 'func' or 'list.func' attribute is
    // therefore followed, regardless of the op type. Plain ops are looked up
    // defensively as well: op-to-function mappings can exist in the library
    // (e.g. via function optimization passes). Names are cached in
    // visited_functions before the lookup, including negative results for
    // plain ops, so every distinct name is resolved at most once.
    if (flib_def != nullptr) {
      auto validate_inner = [&](absl::string_view func_name) -> absl::Status {
        if (!visited_functions.insert(func_name).second) {
          return absl::OkStatus();
        }
        const FunctionDef* inner_fdef = flib_def->Find(func_name);
        if (inner_fdef != nullptr) {
          return ValidateFunctionDeviceConstraintsImpl(
              *inner_fdef, flib_def, parsed_available_devices,
              validated_devices, visited_functions);
        }
        return absl::OkStatus();
      };

      absl::Status inner_status = validate_inner(node.op());
      if (!inner_status.ok()) {
        return inner_status;
      }
      for (const auto& attr : node.attr()) {
        if (attr.second.has_func()) {
          inner_status = validate_inner(attr.second.func().name());
          if (!inner_status.ok()) {
            return inner_status;
          }
        } else if (attr.second.has_list()) {
          for (const auto& func : attr.second.list().func()) {
            inner_status = validate_inner(func.name());
            if (!inner_status.ok()) {
              return inner_status;
            }
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
    const std::vector<Device*>& available_devices) {
  std::vector<DeviceNameUtils::ParsedName> parsed_available_devices;
  parsed_available_devices.reserve(available_devices.size());
  for (const Device* dev : available_devices) {
    if (dev == nullptr) {
      continue;
    }
    // Device parses its name into a ParsedName at construction time, so this
    // reuses the already-parsed form instead of re-parsing every name.
    parsed_available_devices.push_back(dev->parsed_name());
  }

  absl::flat_hash_set<absl::string_view> validated_devices;
  absl::flat_hash_set<absl::string_view> visited_functions;
  visited_functions.insert(fdef.signature().name());

  return ValidateFunctionDeviceConstraintsImpl(
      fdef, flib_def, parsed_available_devices, validated_devices,
      visited_functions);
}

absl::Status ValidateFunctionDeviceConstraints(
    const FunctionDef& fdef, const FunctionLibraryDefinition* flib_def,
    const std::vector<DeviceAttributes>& available_devices) {
  std::vector<DeviceNameUtils::ParsedName> parsed_available_devices;
  parsed_available_devices.reserve(available_devices.size());
  for (const auto& dev : available_devices) {
    DeviceNameUtils::ParsedName parsed;
    if (DeviceNameUtils::ParseFullName(dev.name(), &parsed)) {
      parsed_available_devices.push_back(parsed);
    }
  }

  absl::flat_hash_set<absl::string_view> validated_devices;
  absl::flat_hash_set<absl::string_view> visited_functions;
  visited_functions.insert(fdef.signature().name());

  return ValidateFunctionDeviceConstraintsImpl(
      fdef, flib_def, parsed_available_devices, validated_devices,
      visited_functions);
}

}  // namespace tensorflow
