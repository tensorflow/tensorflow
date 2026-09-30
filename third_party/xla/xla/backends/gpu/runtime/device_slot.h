/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_DEVICE_SLOT_H_
#define XLA_BACKENDS_GPU_RUNTIME_DEVICE_SLOT_H_

#include <memory>

#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"

namespace xla::gpu {

// Type-erased, cache-line-aligned base for a per-device state slot.
// Aligned so that states of different devices never share a cache line.
struct alignas(ABSL_CACHELINE_SIZE) DeviceSlot {
  virtual ~DeviceSlot() = default;

  template <typename T>
  static std::unique_ptr<DeviceSlot> Create();

  template <typename T>
  static T* Unwrap(DeviceSlot* slot);
};

template <typename T>
struct TypedDeviceSlot final : DeviceSlot {
  T value;
};

template <typename T>
std::unique_ptr<DeviceSlot> DeviceSlot::Create() {
  return std::make_unique<TypedDeviceSlot<T>>();
}

template <typename T>
T* DeviceSlot::Unwrap(DeviceSlot* slot) {
  return slot != nullptr ? &(static_cast<TypedDeviceSlot<T>*>(slot)->value)
                         : nullptr;
}

// No argument function that creates a DeviceSlot.
using DeviceSlotFactoryRef = absl::FunctionRef<std::unique_ptr<DeviceSlot>()>;

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_DEVICE_SLOT_H_
