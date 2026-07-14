/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_BASIC_DEVICE_LIST_H_
#define XLA_PYTHON_IFRT_BASIC_DEVICE_LIST_H_

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "absl/base/call_once.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// Simple implementation of `DeviceList` that contains a list of devices without
// creating any runtime object in the IFRT implementation.
//
// This is a transitory type that will be replaced with (1) a non-IFRT container
// defined by the user code (e.g., `std::vector<Device*>`) or (2) IFRT
// implementation's own `DeviceList` constructed from its `xla::ifrt::Client`
// API implementation.
//
// Note for IFRT API users: This class is primarily intended for IFRT
// implementations. Please use `Client::MakeDeviceList()` instead.
class BasicDeviceList : public llvm::RTTIExtends<BasicDeviceList, DeviceList> {
 public:
  // Number of devices to inline in `Devices`.
  static constexpr int kInlineDeviceSize = 1;

  // TODO(hyeontaek): Consider using variant<Device*, std::vector<Device*>> for
  // better performance.
  using Devices = absl::InlinedVector<Device*, kInlineDeviceSize>;

  // Constructor with a pre-populated `devices`.
  static DeviceListRef Create(Devices devices);
  static DeviceListRef Create(absl::Span<Device* const> devices);
  static DeviceListRef Create(std::initializer_list<Device*> devices);

  ~BasicDeviceList() override = default;

  absl::Span<Device* const> devices() const override { return devices_; }

  DeviceList* AddressableDeviceList() const override;

  bool operator==(const DeviceList& other) const override {
    if (this == &other) {
      return true;
    }
    const auto* other_basic_device_list =
        llvm::dyn_cast<BasicDeviceList>(&other);
    if (other_basic_device_list == nullptr) {
      return false;
    }
    return devices_ == other_basic_device_list->devices_;
  }

  uint64_t hash() const override;

  static char ID;  // NOLINT

 private:
  explicit BasicDeviceList(Devices devices);

  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  std::string ToString() const override;

  Devices devices_;

  // Addressable device list is dynamically computed and cached.
  struct AddressableDeviceListCache {
    absl::once_flag once_flag;
    DeviceList* device_list = nullptr;
    DeviceListRef device_list_holder;
  };
  mutable AddressableDeviceListCache addressable_device_list_cache_;

  // Cached hash. 0 indicates the hash needs to be computed and cached.
  // May be written multiple times with the same non-zero value.
  static constexpr uint64_t kUnsetHash = 0;
  mutable std::atomic<uint64_t> hash_ = kUnsetHash;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_BASIC_DEVICE_LIST_H_
