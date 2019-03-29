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

#include "tensorflow/compiler/jit/device_info_cache.h"

#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace tensorflow {
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
  TF_RETURN_IF_ERROR(DeviceToDeviceType(device_name_str, device_type.get()));

  it = device_to_device_type_
           .insert({std::move(device_name_str), std::move(device_type)})
           .first;
  return std::cref(*it->second);
}
}  // namespace tensorflow
