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

#ifndef XLA_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
#define XLA_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_

#include <utility>
#include <variant>

#include "xla/stream_executor/device_memory.h"

namespace stream_executor {

// Allows to represent a value that is either a host scalar or a scalar stored
// on the device.
template <typename T>
class HostOrDeviceScalar {
 public:
  explicit HostOrDeviceScalar(T host_value) : value_(std::move(host_value)) {}
  explicit HostOrDeviceScalar(DeviceMemory<T> device_ptr)
      : value_(std::move(device_ptr)) {
    CHECK_EQ(1, device_ptr.ElementCount());
  }

  bool on_device() const {
    return std::holds_alternative<DeviceMemory<T>>(value_);
  }

  const void* opaque() const {
    return on_device() ? std::get<DeviceMemory<T>>(value_).opaque()
                       : &std::get<T>(value_);
  }

 private:
  std::variant<T, DeviceMemory<T>> value_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
