/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/device.h"

#include <memory>
#include <utility>
#include <vector>

#include "xla/python/ifrt/types.pb.h"

namespace xla {
namespace ifrt {

DeviceList::DeviceList(Devices devices) {
  if (devices.size() <= kInlineDeviceSize) {
    state_ = State{std::move(devices)};
  } else {
    state_ = std::make_shared<State>(State{std::move(devices)});
  }
}

absl::StatusOr<DeviceList> DeviceList::FromProto(LookupDeviceFunc lookup_device,
                                                 const DeviceListProto& proto) {
  DeviceList::Devices devices;
  devices.reserve(proto.device_ids_size());
  for (int device_id : proto.device_ids()) {
    TF_ASSIGN_OR_RETURN(Device * device, lookup_device(device_id));
    devices.push_back(device);
  }
  return DeviceList(std::move(devices));
}

DeviceListProto DeviceList::ToProto() const {
  DeviceListProto proto;
  proto.mutable_device_ids()->Reserve(devices().size());
  for (Device* device : devices()) {
    proto.mutable_device_ids()->AddAlreadyReserved(device->id());
  }
  return proto;
}

std::vector<int> GetDeviceIds(DeviceList device_list) {
  std::vector<int> ids;
  ids.reserve(device_list.devices().size());
  for (const Device* device : device_list.devices()) {
    ids.push_back(device->id());
  }
  return ids;
}

}  // namespace ifrt
}  // namespace xla
