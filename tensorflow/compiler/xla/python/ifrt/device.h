/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_DEVICE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_DEVICE_H_

#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/types.pb.h"

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

  // Constructor with a pre-populated `devices`.
  explicit DeviceList(Devices devices);

  DeviceList(const DeviceList& devices) = default;
  DeviceList(DeviceList&& devices) = default;
  DeviceList& operator=(const DeviceList& other) = default;
  DeviceList& operator=(DeviceList&& other) = default;

  // Function that matches the semantics of `Client::LookupDevice()`.
  using LookupDeviceFunc = absl::FunctionRef<StatusOr<Device*>(int)>;

  // Constructs `DeviceList` from `DeviceListProto`. Devices are looked up using
  // `lookup_device`. Device ids in the proto must be consistent with the
  // devices returned by `lookup_device`.
  static StatusOr<DeviceList> FromProto(LookupDeviceFunc lookup_device,
                                        const DeviceListProto& proto);

  // Returns a `DeviceListProto` representation.
  DeviceListProto ToProto() const;

  absl::Span<Device* const> devices() const { return state_->devices; }

  bool operator==(const DeviceList& other) const {
    return devices() == other.devices();
  }
  bool operator!=(const DeviceList& other) const {
    return devices() != other.devices();
  }

  int size() const { return state_->devices.size(); }
  bool empty() const { return state_->devices.empty(); }

  Device* operator[](int i) const { return state_->devices[i]; }
  Device* at(int i) const { return state_->devices.at(i); }
  Device* front() const { return state_->devices.front(); }
  Device* back() const { return state_->devices.back(); }

  auto begin() const { return state_->devices.begin(); }
  auto cbegin() const { return state_->devices.cbegin(); }
  auto end() const { return state_->devices.end(); }
  auto cend() const { return state_->devices.cend(); }

 private:
  // Internal state that may be shared across `DeviceList` instances.
  struct State {
    Devices devices;
  };

  std::shared_ptr<State> state_;
};

// Returns the id of each device in `device_list`.
std::vector<int> GetDeviceIds(DeviceList device_list);

// Hash function for `DeviceList`. Assumes that every unique device has a unique
// `Device` object, not duplicate `Device` objects ("d1 == d2 if d1->id() ==
// d2->id()").
template <typename H>
H AbslHashValue(H h, const DeviceList& devices) {
  return H::combine(std::move(h), devices.devices());
}

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_DEVICE_H_
