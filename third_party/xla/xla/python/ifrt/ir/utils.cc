/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/utils.h"

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

// Returns a DeviceList for the given device ids.
absl::StatusOr<DeviceListRef> LookUpDevices(Client* client,
                                            absl::Span<const DeviceId> ids) {
  std::vector<Device*> devices;
  devices.reserve(ids.size());
  for (DeviceId id : ids) {
    TF_ASSIGN_OR_RETURN(devices.emplace_back(), client->LookupDevice(id));
  }
  return client->MakeDeviceList(devices);
}

}  // namespace ifrt
}  // namespace xla
