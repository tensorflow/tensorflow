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

#include "xla/python/pjrt_ifrt/pjrt_memory.h"

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"

namespace xla {
namespace ifrt {

char PjRtCompatibleMemory::ID = 0;

char PjRtMemory::ID = 0;

PjRtMemory::PjRtMemory(PjRtClient* client, xla::PjRtMemorySpace* pjrt_memory)
    : client_(client), pjrt_memory_(pjrt_memory), kind_(pjrt_memory->kind()) {
  for (xla::PjRtDevice* device : pjrt_memory->devices()) {
    devices_.push_back(*client->LookupPjRtDevice(device));
  }
}

PjRtMemory::PjRtMemory(PjRtClient* client, const MemoryKind& kind,
                       Device* device)
    : client_(client), kind_(kind) {
  pjrt_memory_ = nullptr;
  devices_.push_back(device);
}

MemoryId PjRtMemory::Id() const {
  if (pjrt_memory_ == nullptr) {
    return MemoryId(-1);
  }
  return MemoryId(pjrt_memory_->id());
}

const MemoryKind& PjRtMemory::Kind() const { return kind_; }

absl::string_view PjRtMemory::ToString() const {
  if (pjrt_memory_ == nullptr) {
    return "UNADDRESSABLE_MEMORY_SPACE";
  }
  return pjrt_memory_->ToString();
}

absl::string_view PjRtMemory::DebugString() const {
  if (pjrt_memory_ == nullptr) {
    return "Unaddressable PjRtMemory";
  }
  return pjrt_memory_->DebugString();
}

absl::Span<Device* const> PjRtMemory::Devices() const { return devices_; }

MemoryKind CanonicalizeMemoryKindWithPjRtDevice(MemoryKind memory_kind,
                                                xla::PjRtDevice* device) {
  if (memory_kind.memory_kind().has_value()) {
    return memory_kind;
  }
  auto default_memory_space = device->default_memory_space();
  if (default_memory_space.ok()) {
    return MemoryKind((*default_memory_space)->kind());
  }
  return MemoryKind();
}

}  // namespace ifrt
}  // namespace xla
