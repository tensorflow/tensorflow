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
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

VirtualPlacer::VirtualPlacer(const Cluster* cluster) : has_gpu_(false) {
  CHECK(cluster);
  devices_ = cluster->GetDevices();
  for (const auto& device : devices_) {
    if (str_util::Lowercase(device.first).find("gpu") != string::npos) {
      has_gpu_ = true;
    }
  }

  unknown_device_.set_type("UNKNOWN");
}

const DeviceProperties& VirtualPlacer::get_device(const NodeDef& node) const {
  string device;
  if (!node.device().empty()) {
    auto it = devices_.find(node.device());
    if (it != devices_.end()) {
      return it->second;
    }
    DeviceNameUtils::ParsedName parsed_name;
    bool parsed = DeviceNameUtils::ParseFullName(node.device(), &parsed_name);
    if (!parsed) {
      parsed = DeviceNameUtils::ParseLocalName(node.device(), &parsed_name);
      parsed_name.job = "localhost";
    }
    if (!parsed) {
      if (node.device() == "GPU" || node.device() == "CPU" ||
          node.device() == "gpu" || node.device() == "cpu") {
        parsed_name.job = "localhost";
        parsed_name.type = node.device();
        parsed = true;
      }
    }
    if (!parsed) {
      return unknown_device_;
    } else {
      device = strings::StrCat(
          "/job:", parsed_name.job, "/replica:", parsed_name.replica,
          "/task:", parsed_name.task, "/",
          str_util::Lowercase(parsed_name.type), ":", parsed_name.id);
    }
  } else {
    if (has_gpu_) {
      device = "/job:localhost/replica:0/task:0/gpu:0";
    } else {
      device = "/job:localhost/replica:0/task:0/cpu:0";
    }
  }
  auto it = devices_.find(device);
  if (it == devices_.end()) {
    return unknown_device_;
  }
  return it->second;
}

}  // end namespace grappler
}  // end namespace tensorflow
