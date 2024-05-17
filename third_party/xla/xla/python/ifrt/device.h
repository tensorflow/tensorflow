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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/device.pb.h"
#include "tsl/lib/gtl/int_type.h"

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
  virtual const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
  Attributes() const = 0;

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

  static char ID;  // NOLINT
};

// Ordered list of devices.
class DeviceList {
 public:
  using value_type = Device*;

  // Number of devices to inline in `Devices`.
  static constexpr int kInlineDeviceSize = 1;

  // TODO(hyeontaek): Consider using variant<Device*, std::vector<Device*>> for
  // better performance.
  using Devices = absl::InlinedVector<Device*, kInlineDeviceSize>;

  DeviceList() : DeviceList(Devices()) {}

  // Constructor with a pre-populated `devices`.
  explicit DeviceList(Devices devices);

  DeviceList(const DeviceList& other);
  DeviceList(DeviceList&& other);
  DeviceList& operator=(const DeviceList& other);
  DeviceList& operator=(DeviceList&& other);

  // Function that matches the semantics of `Client::LookupDevice()`.
  using LookupDeviceFunc = absl::FunctionRef<absl::StatusOr<Device*>(DeviceId)>;

  // Constructs `DeviceList` from `DeviceListProto`. Devices are looked up using
  // `lookup_device`. Device ids in the proto must be consistent with the
  // devices returned by `lookup_device`.
  static absl::StatusOr<DeviceList> FromProto(LookupDeviceFunc lookup_device,
                                              const DeviceListProto& proto);

  // Returns a `DeviceListProto` representation.
  DeviceListProto ToProto() const;

  absl::Span<Device* const> devices() const { return state().devices; }

  bool operator==(const DeviceList& other) const {
    const std::shared_ptr<State>* lhs =
        std::get_if<std::shared_ptr<State>>(&state_);
    const std::shared_ptr<State>* rhs =
        std::get_if<std::shared_ptr<State>>(&other.state_);
    if (lhs != nullptr && rhs != nullptr && lhs->get() == rhs->get()) {
      return true;
    }
    return devices() == other.devices();
  }
  bool operator!=(const DeviceList& other) const { return !(*this == other); }

  // Returns the hash of devices. This hash is stable only within the process.
  uint64_t hash() const;

  int size() const { return state().devices.size(); }
  bool empty() const { return state().devices.empty(); }

  Device* operator[](int i) const { return state().devices[i]; }
  Device* at(int i) const { return state().devices.at(i); }
  Device* front() const { return state().devices.front(); }
  Device* back() const { return state().devices.back(); }

  auto begin() const { return state().devices.begin(); }
  auto cbegin() const { return state().devices.cbegin(); }
  auto end() const { return state().devices.end(); }
  auto cend() const { return state().devices.cend(); }

  std::string DebugString() const;

 private:
  // Internal state that may be shared across `DeviceList` instances.
  struct State {
    Devices devices;
  };

  State& state() {
    return std::visit(
        [](auto& state) -> State& {
          using T = std::decay_t<decltype(state)>;
          if constexpr (std::is_same_v<T, State>) {
            return state;
          } else if constexpr (std::is_same_v<T, std::shared_ptr<State>>) {
            return *state;
          }
        },
        state_);
  }

  const State& state() const {
    return std::visit(
        [](auto& state) -> const State& {
          using T = std::decay_t<decltype(state)>;
          if constexpr (std::is_same_v<T, State>) {
            return state;
          } else if constexpr (std::is_same_v<T, std::shared_ptr<State>>) {
            return *state;
          }
        },
        state_);
  }

  std::variant<State, std::shared_ptr<State>> state_;

  // Cached hash. 0 indicates the hash needs to be computed and cached.
  // May be written multiple times with the same non-zero value.
  static constexpr uint64_t kUnsetHash = 0;
  mutable std::atomic<uint64_t> hash_;
};

// Returns the id of each device in `device_list`.
std::vector<DeviceId> GetDeviceIds(DeviceList device_list);

// Hash function for `DeviceList`. Assumes that every unique device has a unique
// `Device` object, not duplicate `Device` objects ("d1 == d2 if d1->id() ==
// d2->id()").
template <typename H>
H AbslHashValue(H h, const DeviceList& devices) {
  return H::combine(std::move(h), devices.hash());
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_DEVICE_H_
