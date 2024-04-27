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

#include "xla/python/pjrt_ifrt/pjrt_device.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char PjRtCompatibleDevice::ID = 0;

char PjRtDevice::ID = 0;

PjRtDevice::PjRtDevice(PjRtClient* client, xla::PjRtDevice* pjrt_device)
    : client_(client), pjrt_device_(pjrt_device) {}

DeviceId PjRtDevice::Id() const {
  return DeviceId(pjrt_device_->global_device_id().value());
}

absl::string_view PjRtDevice::Kind() const {
  return pjrt_device_->device_kind();
}

absl::string_view PjRtDevice::ToString() const {
  return pjrt_device_->ToString();
}

absl::string_view PjRtDevice::DebugString() const {
  return pjrt_device_->DebugString();
}

absl::StatusOr<Memory*> PjRtDevice::DefaultMemory() const {
  TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * pjrt_memory_space,
                      pjrt_device_->default_memory_space());
  return client_->LookupPjRtMemory(pjrt_memory_space);
}

bool PjRtDevice::IsAddressable() const { return pjrt_device_->IsAddressable(); }

absl::Span<Memory* const> PjRtDevice::Memories() const { return memories_; }

int PjRtDevice::ProcessIndex() const { return pjrt_device_->process_index(); }

const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
PjRtDevice::Attributes() const {
  return pjrt_device_->Attributes();
}

}  // namespace ifrt
}  // namespace xla
