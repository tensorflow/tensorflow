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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char DeviceList::ID = 0;
char BasicDeviceList::ID = 0;

absl::StatusOr<tsl::RCReference<DeviceList>> DeviceList::FromProto(
    LookupDeviceFunc lookup_device, const DeviceListProto& proto) {
  // TODO(hyeontaek): Define SerDes for `DeviceList` and use it to remove this
  // layering inversion.
  BasicDeviceList::Devices devices;
  devices.reserve(proto.device_ids_size());
  for (int device_id : proto.device_ids()) {
    TF_ASSIGN_OR_RETURN(Device * device, lookup_device(DeviceId(device_id)));
    devices.push_back(device);
  }
  return BasicDeviceList::Create(std::move(devices));
}

DeviceListProto DeviceList::ToProto() const {
  DeviceListProto proto;
  proto.mutable_device_ids()->Reserve(devices().size());
  for (Device* device : devices()) {
    proto.mutable_device_ids()->AddAlreadyReserved(device->Id().value());
  }
  return proto;
}

tsl::RCReference<DeviceList> BasicDeviceList::Create(Devices devices) {
  return tsl::MakeRef<BasicDeviceList>(std::move(devices));
}

BasicDeviceList::BasicDeviceList(Devices devices) : hash_(kUnsetHash) {
  if (devices.size() <= kInlineDeviceSize) {
    state_ = State{std::move(devices)};
  } else {
    state_ = std::make_shared<State>(State{std::move(devices)});
  }
}

uint64_t BasicDeviceList::hash() const {
  uint64_t hash = hash_.load(std::memory_order_relaxed);
  if (ABSL_PREDICT_FALSE(hash == kUnsetHash)) {
    hash = absl::HashOf(devices());
    if (ABSL_PREDICT_FALSE(hash == kUnsetHash)) {
      ++hash;
    }
    hash_.store(hash, std::memory_order_relaxed);
  }
  return hash;
}

std::string BasicDeviceList::ToString() const {
  return absl::StrCat("BasicDeviceList([",
                      absl::StrJoin(state().devices, ",",
                                    [](std::string* out, Device* device) {
                                      absl::StrAppend(out,
                                                      device->DebugString());
                                    }),
                      "])");
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
