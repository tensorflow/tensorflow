/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/temporary_device_memory.h"

#include "xla/stream_executor/stream.h"

namespace stream_executor {

TemporaryDeviceMemoryBase::~TemporaryDeviceMemoryBase() {
  parent_->temporary_memory_manager()->MarkFinalized(device_memory_,
                                                     allocation_generation_,
                                                     /*must_exist=*/false);
}

DeviceMemoryBase* TemporaryDeviceMemoryBase::mutable_device_memory() {
  return &device_memory_;
}

const DeviceMemoryBase& TemporaryDeviceMemoryBase::device_memory() const {
  return device_memory_;
}

bool TemporaryDeviceMemoryBase::IsAllocated() const {
  return parent_->temporary_memory_manager()->HasAllocated(
      device_memory_, allocation_generation_);
}

TemporaryDeviceMemoryBase::TemporaryDeviceMemoryBase(
    Stream* parent, DeviceMemoryBase device_memory,
    uint64_t allocation_generation)
    : device_memory_(device_memory),
      allocation_generation_(allocation_generation),
      parent_(parent) {
  DCHECK(IsAllocated());
}

}  // namespace stream_executor
