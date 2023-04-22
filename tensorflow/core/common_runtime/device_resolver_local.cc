/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/device_resolver_local.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

Status DeviceResolverLocal::GetDeviceAttributes(const string& device,
                                                DeviceAttributes* attributes) {
  Device* dev;
  // LookupDevice returns InvalidArgument if the device is not found.
  Status s = dev_mgr_->LookupDevice(device, &dev);
  if (errors::IsInvalidArgument(s)) {
    return errors::NotFound(device, " not found");
  } else if (!s.ok()) {
    return s;
  }
  *attributes = dev->attributes();
  return Status::OK();
}

Status DeviceResolverLocal::GetAllDeviceAttributes(
    const string& task, std::vector<DeviceAttributes>* attributes) {
  return errors::Internal(
      "GetTaskCached is not supposed to be called in local collectives");
}

Status DeviceResolverLocal::UpdateDeviceAttributes(
    const std::vector<DeviceAttributes>& attributes) {
  return errors::Internal(
      "UpdateDeviceAttributes shouldn't be called with local collectives");
}

}  // namespace tensorflow
