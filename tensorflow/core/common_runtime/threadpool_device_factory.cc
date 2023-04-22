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

#include <vector>

// Register a factory that provides CPU devices.
#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// TODO(zhifengc/tucker): Figure out the bytes of available RAM.
class ThreadPoolDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back("/physical_device:CPU:0");

    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    int num_numa_nodes = port::NUMANumNodes();
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:CPU:", i);
      std::unique_ptr<ThreadPoolDevice> tpd;
      if (options.config.experimental().use_numa_affinity()) {
        int numa_node = i % num_numa_nodes;
        if (numa_node != i) {
          LOG(INFO) << "Only " << num_numa_nodes
                    << " NUMA nodes visible in system, "
                    << " assigning device " << name << " to NUMA node "
                    << numa_node;
        }
        DeviceLocality dev_locality;
        dev_locality.set_numa_node(numa_node);
        tpd = absl::make_unique<ThreadPoolDevice>(
            options, name, Bytes(256 << 20), dev_locality,
            ProcessState::singleton()->GetCPUAllocator(numa_node));
      } else {
        tpd = absl::make_unique<ThreadPoolDevice>(
            options, name, Bytes(256 << 20), DeviceLocality(),
            ProcessState::singleton()->GetCPUAllocator(port::kNUMANoAffinity));
      }
      devices->push_back(std::move(tpd));
    }

    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory, 60);

}  // namespace tensorflow
