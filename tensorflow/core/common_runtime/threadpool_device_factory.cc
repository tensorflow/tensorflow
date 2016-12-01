/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Register a factory that provides CPU devices.
#include "tensorflow/core/common_runtime/threadpool_device.h"

#include <vector>
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// TODO(zhifengc/tucker): Figure out the bytes of available RAM.
class ThreadPoolDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    // TODO(zhifengc/tucker): Figure out the number of available CPUs
    // and/or NUMA configuration.
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/cpu:", i);
      devices->push_back(new ThreadPoolDevice(
          options, name, Bytes(256 << 20), DeviceLocality(), cpu_allocator()));
    }

    return Status::OK();
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory);

}  // namespace tensorflow
