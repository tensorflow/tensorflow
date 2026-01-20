/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_MAYBE_OWNING_DEVICE_ADDRESS_H_
#define XLA_SERVICE_MAYBE_OWNING_DEVICE_ADDRESS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include "absl/base/macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla {

// MaybeOwningDeviceAddress represents either an owned or unowned device
// address. Like std::variant<se::ScopedDeviceAddress<uint8_t>, DeviceMemory>.
// When the object goes output of scope, it will free the underlying device
// address if it owns it.
class MaybeOwningDeviceAddress {
 public:
  MaybeOwningDeviceAddress() = default;
  MaybeOwningDeviceAddress(MaybeOwningDeviceAddress&&) = default;
  MaybeOwningDeviceAddress& operator=(MaybeOwningDeviceAddress&&) = default;

  explicit MaybeOwningDeviceAddress(se::ScopedDeviceAddress<uint8_t> owned)
      : mem_(std::move(owned)) {}

  explicit MaybeOwningDeviceAddress(se::DeviceAddressBase unowned)
      : mem_(unowned) {}

  MaybeOwningDeviceAddress& operator=(se::DeviceAddressBase unowned) {
    mem_ = unowned;
    return *this;
  }

  MaybeOwningDeviceAddress& operator=(se::ScopedDeviceAddress<uint8_t> owned) {
    mem_ = std::move(owned);
    return *this;
  }

  // Fetches the underlying DeviceAddressBase. The caller of this function is
  // *not* responsible for freeing the address.
  se::DeviceAddressBase AsDeviceAddress() const;

  // Release the se::ScopedDeviceAddress<uint8_t> without freeing
  // it, and moves the ownership of the address from the object to the caller.
  //
  // A nullopt is returned if the HasOwnership() == false;
  std::optional<se::ScopedDeviceAddress<uint8_t>> Release();

  // If the device address is owned, returns a pointer to the internal
  // ScopedDeviceAddress, otherwise nullptr is returned.
  const se::ScopedDeviceAddress<uint8_t>* AsScopedDeviceAddress() const;

  ABSL_DEPRECATE_AND_INLINE()
  se::DeviceAddressBase AsDeviceMemoryBase() const { return AsDeviceAddress(); }

  ABSL_DEPRECATE_AND_INLINE()
  const se::ScopedDeviceAddress<uint8_t>* AsOwningDeviceMemory() const {
    return AsScopedDeviceAddress();
  }

  // Returns true if has ownership over underlying address.
  bool HasOwnership() const;

 private:
  std::variant<se::DeviceAddressBase, se::ScopedDeviceAddress<uint8_t>> mem_;
};

}  // namespace xla

#endif  // XLA_SERVICE_MAYBE_OWNING_DEVICE_ADDRESS_H_
