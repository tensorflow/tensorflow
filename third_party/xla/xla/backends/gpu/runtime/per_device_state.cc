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

#include "xla/backends/gpu/runtime/per_device_state.h"

#include <memory>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/cow_storage.h"
#include "xla/backends/gpu/runtime/device_slot.h"
#include "xla/backends/gpu/runtime/vector_storage.h"

namespace xla::gpu {
namespace {

std::vector<std::unique_ptr<DeviceSlot>> CreateSlots(
    int num_devices, DeviceSlotFactoryRef factory) {
  if (num_devices <= 0) {
    return {};
  }
  std::vector<std::unique_ptr<DeviceSlot>> slots;
  slots.reserve(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    slots.push_back(factory());
  }
  return slots;
}

}  // namespace

class UntypedPerDeviceState::Impl {
 public:
  Impl(int num_devices, DeviceSlotFactoryRef factory)
      : vector_storage_(CreateSlots(num_devices, factory)) {}

  int num_device_slots() const { return vector_storage_.size(); }

  DeviceSlot* Find(int device_ordinal) const {
    if (device_ordinal < 0) {
      return nullptr;
    }
    if (DeviceSlot* slot = vector_storage_.Find(device_ordinal)) {
      return slot;
    }
    return cow_storage_.Find(device_ordinal);
  }

  absl::StatusOr<DeviceSlot*> GetOrCreate(int device_ordinal,
                                          DeviceSlotFactoryRef factory) {
    if (device_ordinal < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Negative device ordinal: ", device_ordinal));
    }
    if (DeviceSlot* slot = vector_storage_.Find(device_ordinal)) {
      return slot;
    }
    return cow_storage_.GetOrCreate(device_ordinal, factory);
  }

  void ForEach(absl::FunctionRef<void(DeviceSlot&)> fn) const {
    vector_storage_.ForEach(fn);
    cow_storage_.ForEach(fn);
  }

 private:
  VectorStorage vector_storage_;
  CowStorage cow_storage_;
};

UntypedPerDeviceState::UntypedPerDeviceState(int num_devices,
                                             DeviceSlotFactoryRef factory)
    : impl_(std::make_unique<Impl>(num_devices, factory)) {}

UntypedPerDeviceState::~UntypedPerDeviceState() = default;

int UntypedPerDeviceState::num_device_slots() const {
  return impl_->num_device_slots();
}

DeviceSlot* UntypedPerDeviceState::Find(int device_ordinal) const {
  return impl_->Find(device_ordinal);
}

absl::StatusOr<DeviceSlot*> UntypedPerDeviceState::GetOrCreate(
    int device_ordinal, DeviceSlotFactoryRef factory) {
  return impl_->GetOrCreate(device_ordinal, factory);
}

void UntypedPerDeviceState::ForEach(
    absl::FunctionRef<void(DeviceSlot&)> fn) const {
  impl_->ForEach(fn);
}

}  // namespace xla::gpu
