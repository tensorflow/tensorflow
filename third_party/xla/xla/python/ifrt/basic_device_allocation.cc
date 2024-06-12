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

#include "xla/python/ifrt/basic_device_allocation.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/topology.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char BasicDeviceAllocation::ID = 0;

namespace {

// Makes a name for a `BasicDeviceAllocation` based on its `devices`.
std::string MakeBasicDeviceAllocationName(const DeviceList& devices) {
  if (devices.size() == 1) {
    return absl::StrCat("BasicDeviceAllocation-",
                        devices.front()->Id().value());
  } else {
    // TODO(hyeontaek): Change the hash to fingerprinting. A hash value is more
    // likely to collide if the user creates many multi-device DeviceLists.
    return absl::StrCat("BasicDeviceAllocation-multi-", devices.hash());
  }
}

// Returns the default memory kind of `device`.
// TODO(hyeontaek): Make this a method of `Device`.
absl::StatusOr<MemoryKind> GetDefaultMemoryKindFromDevice(
    const Device* device) {
  MemoryKind default_memory_kind;
  TF_ASSIGN_OR_RETURN(Memory * default_memory, device->DefaultMemory());
  if (default_memory != nullptr) {
    default_memory_kind = default_memory->Kind();
  }
  return default_memory_kind;
}

// Returns a list of addressable devices in `devices`. The relative orders of
// devices is preserved.
DeviceList GetAddressableDevices(const DeviceList& devices) {
  int num_addressable_devices = 0;
  for (Device* device : devices) {
    if (device->IsAddressable()) {
      ++num_addressable_devices;
    }
  }
  if (devices.size() == num_addressable_devices) {
    return devices;
  } else {
    DeviceList::Devices addessable_device_items;
    addessable_device_items.reserve(num_addressable_devices);
    for (Device* device : devices) {
      if (device->IsAddressable()) {
        addessable_device_items.push_back(device);
      }
    }
    return DeviceList(std::move(addessable_device_items));
  }
}

// Returns all memory kinds from `devices`. It also checks if the default memory
// kind of all devices is the same as `expected_default_memory_kind`.
absl::StatusOr<std::vector<MemoryKind>> GetAllMemoryKindsFromDevices(
    const DeviceList& devices, MemoryKind expected_default_memory_kind) {
  std::vector<MemoryKind> all_memory_kinds;
  absl::flat_hash_set<MemoryKind> all_memory_kinds_set;
  for (const Device* device : devices) {
    TF_ASSIGN_OR_RETURN(MemoryKind default_memory_kind,
                        GetDefaultMemoryKindFromDevice(device));
    if (default_memory_kind != expected_default_memory_kind) {
      return absl::InvalidArgumentError(
          absl::StrCat("Default memory kinds of devices must be the same: ",
                       default_memory_kind.DebugString(), " vs. ",
                       expected_default_memory_kind.DebugString()));
    }
    for (Memory* memory : device->Memories()) {
      if (all_memory_kinds_set.insert(memory->Kind()).second) {
        all_memory_kinds.push_back(memory->Kind());
      }
    }
  }
  return all_memory_kinds;
}

}  // namespace

absl::StatusOr<tsl::RCReference<BasicDeviceAllocation>>
BasicDeviceAllocation::Create(DeviceList devices) {
  if (devices.empty()) {
    return absl::InvalidArgumentError(
        "BasicDeviceAllocation requires at least one device");
  }

  Client* client = devices.front()->client();
  std::string name = MakeBasicDeviceAllocationName(devices);
  DeviceList addressable_devices = GetAddressableDevices(devices);

  TF_ASSIGN_OR_RETURN(MemoryKind first_default_memory_kind,
                      GetDefaultMemoryKindFromDevice(devices.front()));
  TF_ASSIGN_OR_RETURN(
      std::vector<MemoryKind> all_memory_kinds,
      GetAllMemoryKindsFromDevices(devices, first_default_memory_kind));

  // Attributes for BasicDeviceAllocation is empty at the moment.
  AttributeMap attributes({});

  return tsl::MakeRef<BasicDeviceAllocation>(
      client, std::move(name), std::move(devices),
      std::move(addressable_devices), first_default_memory_kind,
      std::move(all_memory_kinds), std::move(attributes));
}

absl::StatusOr<tsl::RCReference<BasicDeviceAllocation>>
BasicDeviceAllocation::Create(Client* client,
                              absl::Span<const DeviceId> device_ids) {
  DeviceList::Devices devices;
  devices.reserve(device_ids.size());
  for (DeviceId device_id : device_ids) {
    TF_ASSIGN_OR_RETURN(Device * device, client->LookupDevice(device_id));
    devices.push_back(device);
  }
  return Create(DeviceList(std::move(devices)));
}

absl::StatusOr<std::shared_ptr<Topology>> BasicDeviceAllocation::GetTopology()
    const {
  return client_->GetTopologyForDevices(devices_);
}

std::string BasicDeviceAllocation::DebugString() const {
  return absl::StrCat("BasicDeviceAllocation(name=\"", name_,
                      "\",devices=", devices_.DebugString(), ")");
}

}  // namespace ifrt
}  // namespace xla
