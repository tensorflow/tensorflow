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
#include "xla/stream_executor/device_memory_handle.h"

#include <utility>

#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace stream_executor {

DeviceMemoryHandle::DeviceMemoryHandle(StreamExecutorInterface *executor,
                                       DeviceMemoryBase memory)
    : memory_(std::move(memory)), executor_(executor) {}

DeviceMemoryHandle::DeviceMemoryHandle(DeviceMemoryHandle &&other) noexcept
    : memory_(std::move(other.memory_)), executor_(other.executor_) {
  other.memory_ = DeviceMemoryBase();
}

DeviceMemoryHandle::~DeviceMemoryHandle() { Free(); }

void DeviceMemoryHandle::Free() {
  if (!memory_.is_null()) {
    executor_->Deallocate(&memory_);
  }
}

DeviceMemoryHandle &DeviceMemoryHandle::operator=(
    DeviceMemoryHandle &&other) noexcept {
  Free();
  memory_ = std::move(other.memory_);
  other.memory_ = DeviceMemoryBase();
  executor_ = other.executor_;
  return *this;
}

}  // namespace stream_executor
