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

#include "xla/service/maybe_owning_device_address.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"

namespace xla {

se::DeviceAddressBase MaybeOwningDeviceAddress::AsDeviceAddress() const {
  if (HasOwnership()) {
    return *std::get<se::ScopedDeviceAddress<uint8_t>>(mem_);
  }
  return std::get<se::DeviceAddressBase>(mem_);
}

bool MaybeOwningDeviceAddress::HasOwnership() const {
  return std::holds_alternative<se::ScopedDeviceAddress<uint8_t>>(mem_);
}

std::optional<se::ScopedDeviceAddress<uint8_t>>
MaybeOwningDeviceAddress::Release() {
  if (!HasOwnership()) {
    return {};
  }
  return std::move(std::get<se::ScopedDeviceAddress<uint8_t>>(mem_));
}

const se::ScopedDeviceAddress<uint8_t>*
MaybeOwningDeviceAddress::AsScopedDeviceAddress() const {
  return HasOwnership() ? &std::get<se::ScopedDeviceAddress<uint8_t>>(mem_)
                        : nullptr;
}

}  // namespace xla
