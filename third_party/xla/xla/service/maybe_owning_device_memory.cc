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

#include "xla/service/maybe_owning_device_memory.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"

namespace xla {

stream_executor::DeviceAddressBase MaybeOwningDeviceMemory::AsDeviceMemoryBase()
    const {
  if (HasOwnership()) {
    return *std::get<stream_executor::ScopedDeviceAddress<uint8_t>>(mem_);
  }
  return std::get<stream_executor::DeviceAddressBase>(mem_);
}

bool MaybeOwningDeviceMemory::HasOwnership() const {
  return std::holds_alternative<stream_executor::ScopedDeviceAddress<uint8_t>>(
      mem_);
}

std::optional<stream_executor::ScopedDeviceAddress<uint8_t>>
MaybeOwningDeviceMemory::Release() {
  if (!HasOwnership()) {
    return {};
  }
  return std::move(
      std::get<stream_executor::ScopedDeviceAddress<uint8_t>>(mem_));
}

const stream_executor::ScopedDeviceAddress<uint8_t>*
MaybeOwningDeviceMemory::AsOwningDeviceMemory() const {
  return HasOwnership()
             ? &std::get<stream_executor::ScopedDeviceAddress<uint8_t>>(mem_)
             : nullptr;
}

}  // namespace xla
