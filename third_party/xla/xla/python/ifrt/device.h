/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_DEVICE_H_
#define XLA_PYTHON_IFRT_DEVICE_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {
namespace ifrt {

class Client;
class Memory;

// Globally unique device IDs.
TSL_LIB_GTL_DEFINE_INT_TYPE(DeviceId, int32_t);

// `Device` represents a single device that can run computations. The types of
// supported computations depend on the runtime.
class Device : public llvm::RTTIExtends<Device, llvm::RTTIRoot> {
 public:
  Device() = default;

  // Not copyable or movable.
  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

  virtual Client* client() const = 0;

  // The ID of this device. Globally unique across all processes.
  virtual DeviceId Id() const = 0;

  // Returns vendor specific attributes about the device. For example the model
  // number of a GPU, or the mesh coordinates of a TPU device. The returned
  // reference will remain valid for the lifetime of the Device.
  virtual const AttributeMap& Attributes() const = 0;

  // A vendor-dependent string that uniquely identifies the kind of device,
  // e.g., "Tesla V100-SXM2-16GB". May be used to determine whether two GPUs are
  // compatible compilation.
  virtual absl::string_view Kind() const = 0;

  // Debug string suitable for reading by end users, should be reasonably terse,
  // for example: "CpuDevice(id=0)".
  virtual absl::string_view ToString() const = 0;

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current device unambiguously.
  virtual absl::string_view DebugString() const = 0;

  // Returns the default memory space attached to this device.
  virtual absl::StatusOr<Memory*> DefaultMemory() const = 0;

  // Returns all memory spaces attached to this device.
  // The memory spaces are in no particular order.
  virtual absl::Span<Memory* const> Memories() const = 0;

  // Whether client can issue commands to this device.
  virtual bool IsAddressable() const = 0;

  // The index of the process that this device belongs to, i.e. is addressable
  // from. This is not always identical to Client::process_index() in a
  // multi-process setting, where each client can see devices from all
  // processes, but only a subset of them are addressable and have the same
  // process_index as the client.
  virtual int ProcessIndex() const = 0;

  template <class Sink>
  friend void AbslStringify(Sink& sink, const Device& device) {
    sink.Append(device.ToString());
  }

  template <class Sink>
  friend void AbslStringify(Sink& sink, const Device* device) {
    if (device == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(device->ToString());
    }
  }

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_DEVICE_H_
