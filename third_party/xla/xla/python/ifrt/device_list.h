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

#ifndef XLA_PYTHON_IFRT_DEVICE_LIST_H_
#define XLA_PYTHON_IFRT_DEVICE_LIST_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// Ordered immutable list of devices.
class DeviceList : public tsl::ReferenceCounted<DeviceList>,
                   public llvm::RTTIExtends<DeviceList, llvm::RTTIRoot> {
 public:
  // Not copyable or movable. `DeviceList` is a runtime object that may contain
  // runtime-specific state that cannot be trivially copied or moved.
  DeviceList(const DeviceList&) = delete;
  DeviceList(DeviceList&&) = delete;
  DeviceList& operator=(const DeviceList&) = delete;
  DeviceList& operator=(DeviceList&&) = delete;

  // Constructs `DeviceList` from `DeviceListProto`. Devices are looked up using
  // `client`. Device ids in the proto must be consistent with the devices
  // returned by `client`.
  static absl::StatusOr<tsl::RCReference<DeviceList>> FromProto(
      xla::ifrt::Client* client, const DeviceListProto& proto);

  // Returns a `DeviceListProto` representation.
  DeviceListProto ToProto() const;

  // Returns the number of devices.
  // TODO(hyeontaek): Make this a virtual method and make it possible for a
  // subclass to lazily materialize devices for `devices()`.
  int size() const { return devices().size(); }

  // Returns a list of `Devices*` represented by this `DeviceList`.
  virtual absl::Span<Device* const> devices() const = 0;

  // Returns a `DeviceList*` containing only addressable devices from this
  // `DeviceList`. It returns itself if all devices are addressable. It points
  // to a heap-allocated object; the pointer is valid at least until this
  // `DeviceList` is destroyed, and it can be persisted beyond this
  // `DeviceList`'s lifetime by using `tsl::FormRef()`.
  virtual DeviceList* AddressableDeviceList() const = 0;

  // Returns true if all devices are addressable.
  bool IsFullyAddressable() const { return AddressableDeviceList() == this; }

  virtual bool operator==(const DeviceList& other) const = 0;
  bool operator!=(const DeviceList& other) const { return !(*this == other); }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DeviceList& device_list) {
    sink.Append(device_list.ToString());
  }

  template <class Sink>
  friend void AbslStringify(Sink& sink,
                            const tsl::RCReference<DeviceList>& device_list) {
    if (device_list == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(device_list->ToString());
    }
  }

  // Returns the hash of devices. This hash is stable only within the process.
  virtual uint64_t hash() const = 0;

  // TODO(hyeontaek): Remove this method in favor of AbslStringify.
  std::string DebugString() const { return ToString(); }

  static char ID;  // NOLINT

 protected:
  DeviceList() = default;

  virtual std::string ToString() const = 0;
};

// Returns the id of each device in `device_list`.
std::vector<DeviceId> GetDeviceIds(
    const tsl::RCReference<DeviceList>& device_list);

// Hash function for `DeviceList`. Assumes that every unique device has a unique
// `Device` object, not duplicate `Device` objects ("d1 == d2 if d1->id() ==
// d2->id()").
template <typename H>
H AbslHashValue(H h, const tsl::RCReference<DeviceList>& devices) {
  return H::combine(std::move(h), devices->hash());
}

// Prevent comparing two tsl::RCReference<DeviceList>. If attempted, the
// compilation will fail with an ambiguous use of these operators.
inline bool operator==(const tsl::RCReference<DeviceList>& lhs,
                       const tsl::RCReference<DeviceList>& rhs) {
  CHECK(false) << "Comparing two tsl::RCReference<DeviceList> directly is "
                  "typically unintended. Do a comparison after deferenceing "
                  "them, or compare their raw pointers.";
}
inline bool operator!=(const tsl::RCReference<DeviceList>& lhs,
                       const tsl::RCReference<DeviceList>& rhs) {
  CHECK(false) << "Comparing two tsl::RCReference<DeviceList> directly is "
                  "typically unintended. Do a comparison after deferenceing "
                  "them, or compare their raw pointers.";
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_DEVICE_LIST_H_
