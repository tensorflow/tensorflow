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

  } else if (devices_.size() == 1) {
    // If there is only one device in the cluster, use it as default device,
    // whatever it is.
    default_device_ = devices_.begin()->first;
  } else {
    // Default device is set from the devices in the cluster in the following
    // priority: /gpu:0, /cpu:0, or any device.
    // TODO(dyoon): This logic assumes single machine with CPU and GPU devices.
    // Make it more general to support multiple machines, job types, and devices
    // other than CPU and GPU.
    std::map<int, string> cpu_devices;  // CPU device map: id -> device name.
    std::map<int, string> gpu_devices;  // GPU device map: id -> device name.
    for (const auto& device : devices_) {
      DeviceNameUtils::ParsedName parsed_name;
      bool parsed = DeviceNameUtils::ParseFullName(device.first, &parsed_name);
      if (parsed) {
        // Parsed devices are stored to cpu_devices or gpu_devices map,
        // addressed (and orderd) by device id.
        if (str_util::Lowercase(parsed_name.type) == "gpu") {
          gpu_devices[parsed_name.id] = device.first;
        } else if (str_util::Lowercase(parsed_name.type) == "cpu") {
          cpu_devices[parsed_name.id] = device.first;
        }
      }
    }
    if (!gpu_devices.empty()) {
      // GPU:0 (or GPU with smallest device id).
      default_device_ = gpu_devices.begin()->second;
    } else if (!cpu_devices.empty()) {
      // CPU:0 (or CPU with smallest device id).
      default_device_ = cpu_devices.begin()->second;
    } else {
      default_device_ = devices_.begin()->first;  // Any device.
    }
  }

  // Default job name for canonical device name.
  default_job_name_ = "localhost";
  // Scan the device names from the cluster, and if there is one job name used,
  // use it for canonical device name.
  std::unordered_set<string> job_names_from_cluster;
  for (const auto& device : devices_) {
    const auto& device_name = device.first;
    DeviceNameUtils::ParsedName parsed_name;
    bool parsed = DeviceNameUtils::ParseFullName(device_name, &parsed_name);
    if (parsed && !parsed_name.job.empty()) {
      job_names_from_cluster.insert(parsed_name.job);
    }
  }
  // If there is only  type of job name in all the devices in the cluster, use
  // that one as default job name; otherwise, use localhost.
  // TODO(dyoon): this should be improved, especially when the cluster is
  // composed of multiple worker, PS, and other types of jobs.
  if (job_names_from_cluster.size() == 1) {
    auto it = job_names_from_cluster.begin();
    default_job_name_ = *it;
  }
}

const DeviceProperties& VirtualPlacer::get_device(const NodeDef& node) const {
  string device = get_canonical_device_name(node);
  VLOG(3) << "Device name: " << device;
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
      if (parsed_name.job.empty()) {
        parsed_name.job = default_job_name_;
      }
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
