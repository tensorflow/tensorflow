/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device_set.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

DeviceSet::DeviceSet() {}

DeviceSet::~DeviceSet() {}

void DeviceSet::AddDevice(Device* device) {
  devices_.push_back(device);
  device_by_name_.insert({device->name(), device});
}

void DeviceSet::FindMatchingDevices(const DeviceNameUtils::ParsedName& spec,
                                    std::vector<Device*>* devices) const {
  // TODO(jeff): If we are going to repeatedly lookup the set of devices
  // for the same spec, maybe we should have a cache of some sort
  devices->clear();
  for (Device* d : devices_) {
    if (DeviceNameUtils::IsCompleteSpecification(spec, d->parsed_name())) {
      devices->push_back(d);
    }
  }
}

Device* DeviceSet::FindDeviceByName(const string& name) const {
  return gtl::FindPtrOrNull(device_by_name_, name);
}

// Higher result implies lower priority.
static int Order(const DeviceType& d) {
  if (StringPiece(d.type()) == DEVICE_CPU) {
    return 3;
  } else if (StringPiece(d.type()) == DEVICE_GPU) {
    return 2;
  } else {
    return 1;
  }
}

static bool ByPriority(const DeviceType& a, const DeviceType& b) {
  // Order by "order number"; break ties lexicographically.
  return std::make_pair(Order(a), StringPiece(a.type())) <
         std::make_pair(Order(b), StringPiece(b.type()));
}

std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList() const {
  std::vector<DeviceType> result;
  std::set<string> seen;
  for (Device* d : devices_) {
    auto t = d->device_type();
    if (seen.insert(t).second) {
      result.emplace_back(DeviceType(t));
    }
  }
  std::sort(result.begin(), result.end(), ByPriority);
  return result;
}

}  // namespace tensorflow
