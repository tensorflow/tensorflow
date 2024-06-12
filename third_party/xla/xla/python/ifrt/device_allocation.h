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

#ifndef XLA_PYTHON_IFRT_DEVICE_ALLOCATION_H_
#define XLA_PYTHON_IFRT_DEVICE_ALLOCATION_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/topology.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class Client;

// Abstract interface for device allocation.
// TODO(hyeontaek): It is unclear if we need RCReference and/or RTTI. We might
// just use shared_ptr and a simple base class.
class DeviceAllocation
    : public tsl::ReferenceCounted<DeviceAllocation>,
      public llvm::RTTIExtends<DeviceAllocation, llvm::RTTIRoot> {
 public:
  DeviceAllocation() = default;

  // Not copyable or movable.
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation(DeviceAllocation&&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(DeviceAllocation&&) = delete;

  bool operator==(const DeviceAllocation& other) const {
    return this == &other;
  }
  bool operator!=(const DeviceAllocation& other) const {
    return this != &other;
  }

  virtual Client* client() const = 0;

  // Name of `DeviceAllocation`.
  virtual absl::string_view name() const = 0;

  // Returns a `DeviceList`. Devices in the list are valid only during the
  // lifetime of this `DeviceAllocation`.
  virtual DeviceList GetDeviceList() const = 0;

  // Returns a `DeviceList` that contains only addressable devices. Devices in
  // the list are valid only during the lifetime of this `DeviceAllocation`.
  virtual DeviceList GetAddressableDeviceList() const = 0;

  // Returns the default memory kind for this `DeviceAllocation`.
  virtual MemoryKind GetDefaultMemoryKind() const = 0;

  // Returns all available memory kinds for this `DeviceAllocation`.
  virtual std::vector<MemoryKind> GetAllMemoryKinds() const = 0;

  // Returns a `Topology` that is associated with this `DeviceAllocation`. Some
  // platforms may have no topology defined and return an error.
  virtual absl::StatusOr<std::shared_ptr<Topology>> GetTopology() const = 0;

  // Returns implementation-specific attributes about this `DeviceAllocation`.
  virtual const AttributeMap& Attributes() const = 0;

  virtual std::string DebugString() const = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_DEVICE_ALLOCATION_H_
