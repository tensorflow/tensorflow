/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device.h"
#include "tensorflow/tsl/platform/errors.h"

namespace tensorflow {

Status NextPluggableDeviceFactory::ListPhysicalDevices(
    std::vector<string>* devices) {
  TF_Status* c_status = TF_NewStatus();
  int32_t device_count = api_->TFNPD_GetDeviceCount(c_status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status));
  TF_DeleteStatus(c_status);

  for (int i = 0; i < device_count; ++i) {
    const string device_name =
        absl::StrCat("/physical_device:", device_type_, ":", i);
    devices->push_back(device_name);
  }

  return OkStatus();
}

Status NextPluggableDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const std::string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  TF_Status* c_status = TF_NewStatus();

  // Setup per-device states or resources that are internal to plugin.
  api_->TFNPD_InitPluginInternalDeviceStates(c_status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status));

  int32_t device_count = api_->TFNPD_GetDeviceCount(c_status);
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status));
  TF_DeleteStatus(c_status);

  for (int i = 0; i < device_count; ++i) {
    NextPluggableDevice::Options options;
    options.device_name_prefix = name_prefix;
    options.device_name = device_type_;
    options.compilation_device_name = compilation_device_name_;
    options.device_ordinal = i;

    auto device =
        std::make_unique<NextPluggableDevice>(session_options, options);
    devices->push_back(std::move(device));
  }

  LOG(INFO) << "Created " << device_count
            << " TensorFlow NextPluggableDevices. "
            << "Physical device type: " << device_type_;
  return OkStatus();
}

}  // namespace tensorflow
