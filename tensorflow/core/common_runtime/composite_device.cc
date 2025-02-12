/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/composite_device.h"

#include "absl/strings/str_join.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

const char* const kCompositeDeviceType = "COMPOSITE";

std::unique_ptr<CompositeDevice> CompositeDevice::MakeDevice(
    const std::vector<string>& underlying_devices, const int unique_device_id,
    const DeviceNameUtils::ParsedName& host_name, absl::Status* status) {
  DeviceNameUtils::ParsedName parsed_name = host_name;
  parsed_name.type = kCompositeDeviceType;
  parsed_name.id = unique_device_id;
  const string device_name = DeviceNameUtils::ParsedNameToString(parsed_name);
  return CompositeDevice::MakeDevice(underlying_devices, device_name, status);
}

std::unique_ptr<CompositeDevice> CompositeDevice::MakeDevice(
    const std::vector<string>& underlying_devices, const string& device_name,
    absl::Status* status) {
  if (underlying_devices.empty()) {
    status->Update(
        errors::InvalidArgument("underlying_devices should not be empty."));
    return nullptr;
  }
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(underlying_devices.at(0), &parsed_name)) {
    status->Update(tensorflow::errors::InvalidArgument(
        "Cannot parse device name ", underlying_devices.at(0),
        " when creating CompositeDevice."));
    return nullptr;
  }
  const string& underlying_type = parsed_name.type;
  for (int i = 1; i < underlying_devices.size(); ++i) {
    DeviceNameUtils::ParsedName name;
    if (!DeviceNameUtils::ParseFullName(underlying_devices.at(i), &name)) {
      status->Update(tensorflow::errors::InvalidArgument(
          "Cannot parse device name ", underlying_devices.at(i),
          " when creating CompositeDevice."));
      return nullptr;
    }
    if (name.type != underlying_type) {
      status->Update(tensorflow::errors::InvalidArgument(
          "Expect device type ", parsed_name.type, "; but got type ", name.type,
          " from device: ", underlying_devices.at(i),
          " when creating CompositeDevice."));
      return nullptr;
    }
  }

  DeviceAttributes device_attributes;
  device_attributes.set_name(device_name);
  device_attributes.set_device_type(kCompositeDeviceType);

  return absl::WrapUnique(
      new CompositeDevice(device_attributes, underlying_devices));
}

}  // namespace tensorflow
