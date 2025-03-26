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
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"

namespace xla {
namespace ifrt {

char PjRtCompatibleDevice::ID = 0;

char PjRtDevice::ID = 0;

PjRtDevice::PjRtDevice(
    PjRtClient* client, DeviceId id, std::string kind, std::string to_string,
    std::string debug_string, int process_index,
    absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes,
    xla::PjRtDevice* pjrt_device)
    : client_(client),
      id_(id),
      attributes_(FromPjRtAttributeMap(std::move(attributes))),
      kind_(std::move(kind)),
      to_string_(std::move(to_string)),
      debug_string_(std::move(debug_string)),
      process_index_(process_index),
      pjrt_device_(pjrt_device) {}

DeviceId PjRtDevice::Id() const { return id_; }

const AttributeMap& PjRtDevice::Attributes() const { return attributes_; }

absl::string_view PjRtDevice::Kind() const { return kind_; }

absl::string_view PjRtDevice::ToString() const { return to_string_; }

absl::string_view PjRtDevice::DebugString() const { return debug_string_; }

absl::StatusOr<Memory*> PjRtDevice::DefaultMemory() const {
  return default_memory_;
}

bool PjRtDevice::IsAddressable() const { return pjrt_device_ != nullptr; }

absl::Span<Memory* const> PjRtDevice::Memories() const { return memories_; }

int PjRtDevice::ProcessIndex() const { return process_index_; }

}  // namespace ifrt
}  // namespace xla
