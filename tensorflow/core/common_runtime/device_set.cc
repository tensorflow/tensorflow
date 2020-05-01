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

#include "tensorflow/core/common_runtime/device_set.h"

#include <set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

DeviceSet::DeviceSet() {}

DeviceSet::~DeviceSet() {}

void DeviceSet::AddDevice(Device* device) {
  mutex_lock l(devices_mu_);
  devices_.push_back(device);
  prioritized_devices_.clear();
  prioritized_device_types_.clear();
  for (const string& name :
       DeviceNameUtils::GetNamesForDeviceMappings(device->parsed_name())) {
    device_by_name_.insert({name, device});
  }
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

// static
int DeviceSet::DeviceTypeOrder(const DeviceType& d) {
  return DeviceFactory::DevicePriority(d.type_string());
}

static bool DeviceTypeComparator(const DeviceType& a, const DeviceType& b) {
  // First sort by prioritized device type (higher is preferred) and
  // then by device name (lexicographically).
  auto a_priority = DeviceSet::DeviceTypeOrder(a);
  auto b_priority = DeviceSet::DeviceTypeOrder(b);
  if (a_priority != b_priority) {
    return a_priority > b_priority;
  }

  return StringPiece(a.type()) < StringPiece(b.type());
}

std::vector<DeviceType> DeviceSet::PrioritizedDeviceTypeList() const {
  std::vector<DeviceType> result;
  std::set<string> seen;
  for (Device* d : devices_) {
    const auto& t = d->device_type();
    if (seen.insert(t).second) {
      result.emplace_back(t);
    }
  }
  std::sort(result.begin(), result.end(), DeviceTypeComparator);
  return result;
}

void DeviceSet::SortPrioritizedDeviceTypeVector(
    PrioritizedDeviceTypeVector* vector) {
  if (vector == nullptr) return;

  auto device_sort = [](const PrioritizedDeviceTypeVector::value_type& a,
                        const PrioritizedDeviceTypeVector::value_type& b) {
    // First look at set priorities.
    if (a.second != b.second) {
      return a.second > b.second;
    }
    // Then fallback to default priorities.
    return DeviceTypeComparator(a.first, b.first);
  };

  std::sort(vector->begin(), vector->end(), device_sort);
}

void DeviceSet::SortPrioritizedDeviceVector(PrioritizedDeviceVector* vector) {
  auto device_sort = [](const std::pair<Device*, int32>& a,
                        const std::pair<Device*, int32>& b) {
    if (a.second != b.second) {
      return a.second > b.second;
    }

    const string& a_type_name = a.first->device_type();
    const string& b_type_name = b.first->device_type();
    if (a_type_name != b_type_name) {
      auto a_priority = DeviceFactory::DevicePriority(a_type_name);
      auto b_priority = DeviceFactory::DevicePriority(b_type_name);
      // First sort by prioritized device type (higher is preferred) and
      // then by device name (lexicographically).
      if (a_priority != b_priority) {
        return a_priority > b_priority;
      }
    }
    return StringPiece(a.first->name()) < StringPiece(b.first->name());
  };
  std::sort(vector->begin(), vector->end(), device_sort);
}

namespace {

void UpdatePrioritizedVectors(
    const std::vector<Device*>& devices,
    PrioritizedDeviceVector* prioritized_devices,
    PrioritizedDeviceTypeVector* prioritized_device_types) {
  if (prioritized_devices->size() != devices.size()) {
    for (Device* d : devices) {
      prioritized_devices->emplace_back(
          d, DeviceSet::DeviceTypeOrder(DeviceType(d->device_type())));
    }
    DeviceSet::SortPrioritizedDeviceVector(prioritized_devices);
  }

  if (prioritized_device_types != nullptr &&
      prioritized_device_types->size() != devices.size()) {
    std::set<DeviceType> seen;
    for (const std::pair<Device*, int32>& p : *prioritized_devices) {
      DeviceType t(p.first->device_type());
      if (seen.insert(t).second) {
        prioritized_device_types->emplace_back(t, p.second);
      }
    }
  }
}

}  // namespace

const PrioritizedDeviceVector& DeviceSet::prioritized_devices() const {
  mutex_lock l(devices_mu_);
  UpdatePrioritizedVectors(devices_, &prioritized_devices_,
                           /* prioritized_device_types */ nullptr);
  return prioritized_devices_;
}

const PrioritizedDeviceTypeVector& DeviceSet::prioritized_device_types() const {
  mutex_lock l(devices_mu_);
  UpdatePrioritizedVectors(devices_, &prioritized_devices_,
                           &prioritized_device_types_);
  return prioritized_device_types_;
}

}  // namespace tensorflow
