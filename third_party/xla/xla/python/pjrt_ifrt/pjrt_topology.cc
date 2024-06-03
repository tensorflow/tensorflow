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

#include "xla/python/pjrt_ifrt/pjrt_topology.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"

namespace xla::ifrt {

char PjRtTopology::ID = 0;

PjRtTopology::PjRtTopology(
    std::shared_ptr<const xla::PjRtTopologyDescription> description)
    : description_(std::move(description)) {}

absl::string_view PjRtTopology::platform_name() const {
  return description_->platform_name();
}

absl::string_view PjRtTopology::platform_version() const {
  return description_->platform_version();
}

PjRtPlatformId PjRtTopology::platform_id() const {
  return description_->platform_id();
}

std::vector<std::unique_ptr<const PjRtDeviceDescription>>
PjRtTopology::DeviceDescriptions() const {
  return description_->DeviceDescriptions();
}

absl::StatusOr<xla::Layout> PjRtTopology::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) const {
  return description_->GetDefaultLayout(element_type, dims);
}

absl::StatusOr<std::string> PjRtTopology::Serialize() const {
  return description_->Serialize();
}

const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
PjRtTopology::Attributes() const {
  return description_->Attributes();
}

}  // namespace xla::ifrt
