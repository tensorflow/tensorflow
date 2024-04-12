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

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/device.pb.h"

namespace xla {
namespace ifrt {

class Client;

// Short-term alias to reuse `xla::PjRtDevice` without a separate abstract type.
using Device = ::xla::PjRtDevice;

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
  using LookupDeviceFunc = absl::FunctionRef<absl::StatusOr<Device*>(int)>;

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
std::vector<int> GetDeviceIds(DeviceList device_list);

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
