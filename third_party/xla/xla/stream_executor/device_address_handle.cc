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

#include "xla/stream_executor/device_address_handle.h"

#include <utility>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

DeviceAddressHandle::DeviceAddressHandle(StreamExecutor* executor,
                                         DeviceAddressBase address)
    : address_(std::move(address)), executor_(executor) {}

DeviceAddressHandle::DeviceAddressHandle(DeviceAddressHandle&& other) noexcept
    : address_(std::move(other.address_)), executor_(other.executor_) {
  other.address_ = DeviceAddressBase();
}

DeviceAddressHandle::~DeviceAddressHandle() { Free(); }

void DeviceAddressHandle::Free() {
  if (!address_.is_null()) {
    executor_->Deallocate(&address_);
  }
}

DeviceAddressHandle& DeviceAddressHandle::operator=(
    DeviceAddressHandle&& other) noexcept {
  Free();
  address_ = std::move(other.address_);
  other.address_ = DeviceAddressBase();
  executor_ = other.executor_;
  return *this;
}

}  // namespace stream_executor
