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

VirtualPlacer::VirtualPlacer(const Cluster* cluster) {
  CHECK(cluster);
  devices_ = cluster->GetDevices();

  if (devices_.empty()) {
    // If there are no devices in the cluster, add a single device, "UNKNOWN" to
    // the cluster.
    default_device_ = "UNKNOWN";
    DeviceProperties& prop = devices_["UNKNOWN"];
    prop.set_type("UNKNOWN");

  } else {
    default_device_ = devices_.begin()->first;
    for (const auto& device : devices_) {
      if (str_util::Lowercase(device.first).find("gpu") != string::npos) {
        default_device_ = device.first;
      }
      break;
    }
  }
}

const DeviceProperties& VirtualPlacer::get_device(const NodeDef& node) const {
  string device = get_canonical_device_name(node);
  auto it = devices_.find(device);
  DCHECK(it != devices_.end());
  return it->second;
}

string VirtualPlacer::get_canonical_device_name(const NodeDef& node) const {
  string device;
  if (!node.device().empty()) {
    if (devices_.find(node.device()) != devices_.end()) {
      return node.device();
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
      return get_default_device_name();
    } else {
      device = strings::StrCat(
          "/job:", parsed_name.job, "/replica:", parsed_name.replica,
          "/task:", parsed_name.task, "/",
          str_util::Lowercase(parsed_name.type), ":", parsed_name.id);
    }
  } else {
    return get_default_device_name();
  }
  if (devices_.find(device) == devices_.end()) {
    return get_default_device_name();
  }
  return device;
}

const string& VirtualPlacer::get_default_device_name() const {
  return default_device_;
}

}  // end namespace grappler
}  // end namespace tensorflow
