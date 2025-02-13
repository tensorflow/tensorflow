/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/device_list.h"

#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char DeviceList::ID = 0;

absl::StatusOr<tsl::RCReference<DeviceList>> DeviceList::FromProto(
    xla::ifrt::Client* client, const DeviceListProto& proto) {
  absl::InlinedVector<Device*, 1> devices;
  devices.reserve(proto.device_ids_size());
  for (int device_id : proto.device_ids()) {
    TF_ASSIGN_OR_RETURN(Device* const device,
                        client->LookupDevice(DeviceId(device_id)));
    devices.push_back(device);
  }
  return client->MakeDeviceList(devices);
}

DeviceListProto DeviceList::ToProto() const {
  DeviceListProto proto;
  proto.mutable_device_ids()->Reserve(devices().size());
  for (Device* device : devices()) {
    proto.mutable_device_ids()->AddAlreadyReserved(device->Id().value());
  }
  return proto;
}

std::vector<DeviceId> GetDeviceIds(
    const tsl::RCReference<DeviceList>& device_list) {
  std::vector<DeviceId> ids;
  ids.reserve(device_list->devices().size());
  for (const Device* device : device_list->devices()) {
    ids.push_back(device->Id());
  }
  return ids;
}

}  // namespace ifrt
}  // namespace xla
