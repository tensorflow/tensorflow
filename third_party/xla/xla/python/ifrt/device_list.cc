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

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

class FingerprintPrinter {
 public:
  FingerprintPrinter() : hash_(kDefaultKey64) {}
  void Append(const absl::AlphaNum& a) { hash_.Append(a.data(), a.size()); }
  uint64_t Fingerprint() && {
    highwayhash::HHResult64 result;
    hash_.Finalize(&result);
    return result;
  }

 private:
  static constexpr highwayhash::HHKey kDefaultKey64 = {
      0x4ea9929a25d561c6,
      0x98470d187b523e8f,
      0x592040a2da3c4b53,
      0xbff8b246e3c587a2,
  };
  highwayhash::HighwayHashCatT<HH_TARGET> hash_;
};

}  // namespace

char DeviceList::ID = 0;

absl::StatusOr<DeviceListRef> DeviceList::FromProto(
    xla::ifrt::Client* client, const DeviceListProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for DeviceList deserialization"));
  }

  absl::InlinedVector<Device*, 1> devices;
  devices.reserve(proto.device_ids_size());
  for (int device_id : proto.device_ids()) {
    TF_ASSIGN_OR_RETURN(Device* const device,
                        client->LookupDevice(DeviceId(device_id)));
    devices.push_back(device);
  }
  return client->MakeDeviceList(devices);
}

void DeviceList::ToProto(DeviceListProto& proto, SerDesVersion version) const {
  // TODO(b/423702568): Change the return type to `absl::Status` for graceful
  // error handling.
  CHECK_GE(version.version_number(), SerDesVersionNumber(0))
      << "Unsupported " << version.version_number()
      << " for DeviceList serialization";

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());

  proto.mutable_device_ids()->Reserve(devices().size());
  for (Device* device : devices()) {
    proto.mutable_device_ids()->AddAlreadyReserved(device->Id().value());
  }
}

uint64_t DeviceList::fingerprint() const {
  FingerprintPrinter printer;
  for (Device* device : devices()) {
    printer.Append(device->Id().value());
  }
  return std::move(printer).Fingerprint();
}

std::vector<DeviceId> GetDeviceIds(const DeviceListRef& device_list) {
  std::vector<DeviceId> ids;
  ids.reserve(device_list->devices().size());
  for (const Device* device : device_list->devices()) {
    ids.push_back(device->Id());
  }
  return ids;
}

}  // namespace ifrt
}  // namespace xla
