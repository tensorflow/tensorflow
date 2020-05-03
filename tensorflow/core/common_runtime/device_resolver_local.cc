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

namespace tensorflow {

void DeviceResolverLocal::GetAllDeviceAttributesAsync(
    const std::vector<string>& devices, const std::vector<string>& tasks,
    std::vector<DeviceAttributes>* attributes, const StatusCallback& done) {
  attributes->clear();
  for (const string& device_name : devices) {
    Device* dev;
    Status s = dev_mgr_->LookupDevice(device_name, &dev);
    if (!s.ok()) {
      done(s);
      return;
    }
    attributes->push_back(dev->attributes());
  }
  done(Status::OK());
}

void DeviceResolverLocal::GetDeviceAttributesAsync(const string& device,
                                                   const string& task,
                                                   DeviceAttributes* attributes,
                                                   const StatusCallback& done) {
  Device* dev;
  Status s = dev_mgr_->LookupDevice(device, &dev);
  if (s.ok()) {
    *attributes = dev->attributes();
  }
  done(s);
}

}  // namespace tensorflow
