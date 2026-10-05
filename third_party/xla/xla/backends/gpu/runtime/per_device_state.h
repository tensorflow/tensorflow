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

#ifndef XLA_BACKENDS_GPU_RUNTIME_PER_DEVICE_STATE_H_
#define XLA_BACKENDS_GPU_RUNTIME_PER_DEVICE_STATE_H_

#include <memory>

#include "absl/base/call_once.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/device_slot.h"

namespace xla::gpu {

// Type-erased backing container for `PerDeviceState<T>`.
class UntypedPerDeviceState {
 public:
  UntypedPerDeviceState(int num_devices, DeviceSlotFactoryRef factory);
  ~UntypedPerDeviceState();

  UntypedPerDeviceState(const UntypedPerDeviceState&) = delete;
  UntypedPerDeviceState& operator=(const UntypedPerDeviceState&) = delete;

  int num_device_slots() const;

  DeviceSlot* Find(int device_ordinal) const;
  absl::StatusOr<DeviceSlot*> GetOrCreate(int device_ordinal,
                                          DeviceSlotFactoryRef factory);
  void ForEach(absl::FunctionRef<void(DeviceSlot&)> fn) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Per-device states of type `T`, keyed by device ordinal.
//
// States for ordinals in [0, num_devices) are built in the constructor, and
// `Find` is an array index. States for any other non-negative ordinal are
// built by the first `GetOrCreate`, and `Find` is a lock-free scan over the
// ordinals built so far. `num_devices <= 0` (no topology) builds every state
// on first touch.
//
// Every state has its own cache line and keeps its address for the lifetime of
// the storage. `T` must be default-constructible. It is never copied or moved.
// The storage synchronizes only construction and publication across devices;
// callers must not mutate the same ordinal's `T` concurrently unless `T` is
// internally synchronized.
template <typename T>
class PerDeviceState : private UntypedPerDeviceState {
 public:
  PerDeviceState() : PerDeviceState(0) {}
  explicit PerDeviceState(int num_devices)
      : UntypedPerDeviceState(num_devices, &DeviceSlot::Create<T>) {}

  using UntypedPerDeviceState::num_device_slots;

  // Lock-free. Returns nullptr for an ordinal outside [0, num_devices) whose
  // state was not built yet.
  T* Find(int device_ordinal) const {
    return DeviceSlot::Unwrap<T>(UntypedPerDeviceState::Find(device_ordinal));
  }

  // Returns the state for `device_ordinal`, building it on the first call if
  // the ordinal is outside [0, num_devices). Lock-free once the state exists.
  absl::StatusOr<T*> GetOrCreate(int device_ordinal) {
    ABSL_ASSIGN_OR_RETURN(DeviceSlot * slot,
                     UntypedPerDeviceState::GetOrCreate(
                         device_ordinal, &DeviceSlot::Create<T>));
    return DeviceSlot::Unwrap<T>(slot);
  }

  // Thread-safe. Calls `init_fn` exactly once per `device_ordinal` and returns
  // its status; subsequent calls for the same ordinal return the cached status.
  // If concurrent callers pass different `init_fn`s for the same ordinal, it is
  // unspecified which one is called.
  absl::Status GetOrCreateAndInitialize(
      int device_ordinal, absl::FunctionRef<absl::Status(T*)> init_fn) {
    ABSL_ASSIGN_OR_RETURN(DeviceSlot * slot,
                     UntypedPerDeviceState::GetOrCreate(
                         device_ordinal, &DeviceSlot::Create<T>));
    absl::call_once(slot->init_flag, [&]() {
      slot->init_status = init_fn(DeviceSlot::Unwrap<T>(slot));
    });
    return slot->init_status;
  }

  // Lock-free. Calls `fn` for every state in [0, num_devices) and every state
  // outside [0, num_devices) built before the call. States built concurrently
  // by `GetOrCreate` may or may not be visited. Order of iteration over device
  // states is not guaranteed. Mutating the state `T&` during iteration is not
  // thread-safe; objects must be synchronized externally by the caller for
  // such a use case.
  void ForEach(absl::FunctionRef<void(T&)> fn) const {
    UntypedPerDeviceState::ForEach(
        [&](DeviceSlot& slot) { fn(*DeviceSlot::Unwrap<T>(&slot)); });
  }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_PER_DEVICE_STATE_H_
