/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/device_util.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace tensorflow {
namespace jit {
using xla::StatusOr;

StatusOr<const XlaOpRegistry::DeviceRegistration*>
DeviceInfoCache::GetCompilationDevice(absl::string_view device_name) {
  auto it = device_to_device_registration_.find(device_name);
  if (it != device_to_device_registration_.end()) {
    return it->second;
  }

  string device_name_str = string(device_name);
  TF_ASSIGN_OR_RETURN(const DeviceType& device_type,
                      GetDeviceTypeFor(device_name_str));
  const XlaOpRegistry::DeviceRegistration* registration;
  if (!XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration)) {
    registration = nullptr;
  }

  device_to_device_registration_.insert(
      {std::move(device_name_str), registration});

  return registration;
}

StatusOr<std::reference_wrapper<const DeviceType>>
DeviceInfoCache::GetDeviceTypeFor(absl::string_view device_name) {
  auto it = device_to_device_type_.find(device_name);
  if (it != device_to_device_type_.end()) {
    return std::cref(*it->second);
  }

  string device_name_str = string(device_name);
  auto device_type = absl::make_unique<DeviceType>("");
  TF_RETURN_IF_ERROR(
      DeviceNameToDeviceType(device_name_str, device_type.get()));

  it = device_to_device_type_
           .insert({std::move(device_name_str), std::move(device_type)})
           .first;
  return std::cref(*it->second);
}
}  // namespace jit

Status DeviceNameToDeviceType(const string& device, DeviceType* device_type) {
  DeviceNameUtils::ParsedName parsed;
  if (!DeviceNameUtils::ParseFullName(device, &parsed)) {
    return errors::Internal("Malformed assigned device '", device, "'");
  }
  *device_type = DeviceType(parsed.type);
  return Status::OK();
}

Status PickDeviceForXlaImpl(absl::Span<const string> device_names,
                            bool allow_mixing_unknown_and_cpu,
                            bool* out_can_pick_device,
                            string* out_device_picked) {
  if (out_can_pick_device) {
    *out_can_pick_device = true;
  }

#define FAILED_TO_PICK_DEVICE(failing_status) \
  do {                                        \
    if (out_can_pick_device) {                \
      *out_can_pick_device = false;           \
      return Status::OK();                    \
    } else {                                  \
      return failing_status;                  \
    }                                         \
  } while (false)

  TF_RET_CHECK(!device_names.empty()) << "No devices to choose from";
  DCHECK_NE(out_can_pick_device == nullptr, out_device_picked == nullptr);

  absl::flat_hash_set<absl::string_view> device_names_set;
  for (absl::string_view device_name : device_names) {
    if (!device_name.empty()) {
      // TODO(sanjoy): Figure out if this is necessary.
      device_names_set.insert(device_name);
    }
  }

  absl::optional<absl::string_view> maybe_gpu_device;
  absl::optional<absl::string_view> maybe_cpu_device;
  absl::optional<absl::string_view> maybe_unknown_device;

  for (absl::string_view device_name : device_names_set) {
    DeviceNameUtils::ParsedName parsed_name;
    TF_RET_CHECK(DeviceNameUtils::ParseFullName(device_name, &parsed_name))
        << device_name;
    if (parsed_name.type == "GPU") {
      if (maybe_gpu_device) {
        FAILED_TO_PICK_DEVICE(errors::Internal(
            "Multiple GPU devices ", absl::StrJoin(device_names, ", ")));
      }
      maybe_gpu_device = device_name;
    } else if (parsed_name.type == "CPU") {
      if (maybe_cpu_device) {
        FAILED_TO_PICK_DEVICE(errors::Internal(
            "Multiple CPU devices ", absl::StrJoin(device_names, ", ")));
      }
      maybe_cpu_device = device_name;
    } else {
      if (maybe_unknown_device) {
        FAILED_TO_PICK_DEVICE(errors::Internal(
            "Multiple unknown devices ", absl::StrJoin(device_names, ", ")));
      }
      maybe_unknown_device = device_name;
    }
  }

  if (maybe_unknown_device && maybe_gpu_device) {
    FAILED_TO_PICK_DEVICE(errors::Internal(
        "Found both unknown and GPU devices: ", *maybe_unknown_device, ", ",
        *maybe_gpu_device));
  }

  if (!allow_mixing_unknown_and_cpu) {
    if (maybe_unknown_device && maybe_cpu_device) {
      FAILED_TO_PICK_DEVICE(errors::Internal(
          "Found both unknown and CPU devices: ", *maybe_unknown_device, ", ",
          *maybe_cpu_device));
    }
  }

  if (out_device_picked) {
    if (maybe_gpu_device) {
      *out_device_picked = string(*maybe_gpu_device);
    } else if (maybe_unknown_device) {
      *out_device_picked = string(*maybe_unknown_device);
    } else {
      *out_device_picked = string(*maybe_cpu_device);
    }
  }

  return Status::OK();

#undef FAILED_TO_PICK_DEVICE
}

Status PickDeviceForXla(absl::Span<const string> device_names,
                        bool allow_mixing_unknown_and_cpu,
                        string* out_device_picked) {
  return PickDeviceForXlaImpl(device_names, allow_mixing_unknown_and_cpu,
                              /*out_can_pick_device=*/nullptr,
                              out_device_picked);
}

Status CanPickDeviceForXla(absl::Span<const string> device_names,
                           bool allow_mixing_unknown_and_cpu,
                           bool* out_can_pick_device) {
  return PickDeviceForXlaImpl(device_names, allow_mixing_unknown_and_cpu,
                              out_can_pick_device,
                              /*out_device_picked=*/nullptr);
}
}  // namespace tensorflow
