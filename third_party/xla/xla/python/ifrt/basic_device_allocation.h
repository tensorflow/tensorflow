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

#ifndef XLA_PYTHON_IFRT_BASIC_DEVICE_ALLOCATION_H_
#define XLA_PYTHON_IFRT_BASIC_DEVICE_ALLOCATION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_allocation.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/topology.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// Basic implementation of `DeviceAllocation` that simply wraps a `DeviceList`.
// The list must contain at least one device.
class BasicDeviceAllocation final
    : public llvm::RTTIExtends<BasicDeviceAllocation, DeviceAllocation> {
 public:
  // Creates a `BasicDeviceAllocation` from devices. There must be at least one
  // device.
  static absl::StatusOr<tsl::RCReference<BasicDeviceAllocation>> Create(
      DeviceList devices);

  // Creates a `BasicDeviceAllocation` from device ids. `client` must have
  // devices identified by `device_ids`. There must be at least one device.
  static absl::StatusOr<tsl::RCReference<BasicDeviceAllocation>> Create(
      Client* client, absl::Span<const DeviceId> device_ids);

  Client* client() const override { return client_; }

  absl::string_view name() const override { return name_; }

  DeviceList GetDeviceList() const override { return devices_; }

  DeviceList GetAddressableDeviceList() const override {
    return addressable_devices_;
  }

  MemoryKind GetDefaultMemoryKind() const override {
    return default_memory_kind_;
  }

  std::vector<MemoryKind> GetAllMemoryKinds() const override {
    return all_memory_kinds_;
  }

  absl::StatusOr<std::shared_ptr<Topology>> GetTopology() const override;

  const AttributeMap& Attributes() const override { return attributes_; }

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  BasicDeviceAllocation(Client* client, std::string name, DeviceList devices,
                        DeviceList addressable_devices,
                        MemoryKind default_memory_kind,
                        std::vector<MemoryKind> all_memory_kinds,
                        AttributeMap attributes)
      : client_(client),
        name_(std::move(name)),
        devices_(std::move(devices)),
        addressable_devices_(std::move(addressable_devices)),
        default_memory_kind_(default_memory_kind),
        all_memory_kinds_(std::move(all_memory_kinds)),
        attributes_(std::move(attributes)) {}

  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  Client* client_;
  std::string name_;
  DeviceList devices_;
  DeviceList addressable_devices_;
  MemoryKind default_memory_kind_;
  std::vector<MemoryKind> all_memory_kinds_;
  AttributeMap attributes_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_BASIC_DEVICE_ALLOCATION_H_
