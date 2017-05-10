/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

VirtualPlacer::VirtualPlacer(Cluster* cluster) : has_gpu_(false) {
  devices_["CPU"] = GetLocalCPUInfo();
  if (GetNumAvailableGPUs() > 0) {
    has_gpu_ = true;
    devices_["GPU"] = GetLocalGPUInfo(0);
  }
  unknown_device_.set_type("UNKNOWN");
}

const OpInfo::DeviceProperties& VirtualPlacer::get_device(
    const NodeDef& node) const {
  string device_type;
  DeviceNameUtils::ParsedName parsed;
  if (!node.device().empty() &&
      DeviceNameUtils::ParseFullName(node.device(), &parsed)) {
    device_type = parsed.type;
  } else {
    if (has_gpu_) {
      device_type = "GPU";
    } else {
      device_type = "CPU";
    }
  }
  auto it = devices_.find(device_type);
  if (it == devices_.end()) {
    return unknown_device_;
  }
  return it->second;
}

}  // end namespace grappler
}  // end namespace tensorflow
