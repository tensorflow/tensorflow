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

#ifndef XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_HANDLE_H_
#define XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_HANDLE_H_

#include "absl/base/macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// An RAII-style container for the DeviceAddress that is deallocated via the
// owning StreamExecutor when the handle is destroyed.
class DeviceAddressHandle {
 public:
  DeviceAddressHandle() : address_(), executor_(nullptr) {}

  // A helper constructor to generate a scoped device address given an already
  // allocated address and a stream executor.
  DeviceAddressHandle(StreamExecutor* executor, DeviceAddressBase address);
  ~DeviceAddressHandle();

  // Moves ownership of the address from other to the constructed
  // object.
  DeviceAddressHandle(DeviceAddressHandle&& other) noexcept;

  // Moves ownership of the address from other to this object.
  DeviceAddressHandle& operator=(DeviceAddressHandle&& other) noexcept;

  // Accessors for the DeviceAddressBase.
  const DeviceAddressBase& address() const { return address_; }
  DeviceAddressBase* address_ptr() { return &address_; }

  ABSL_DEPRECATE_AND_INLINE()
  const DeviceAddressBase& memory() const { return address(); }
  ABSL_DEPRECATE_AND_INLINE()
  DeviceAddressBase* memory_ptr() { return address_ptr(); }

 private:
  // Frees the associated address.
  void Free();

  DeviceAddressBase address_;  // Value we wrap with scoped-release.
  StreamExecutor* executor_;   // Null if this object is inactive.
};
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_DEVICE_ADDRESS_HANDLE_H_
