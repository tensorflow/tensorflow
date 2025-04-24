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

#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_device_description.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"

namespace xla {

absl::StatusOr<Layout> CpuTopologyDescription::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) const {
  Shape shape = ShapeUtil::MakeValidatedShape(element_type, dims).value();
  return LayoutUtil::GetWithDefaultLayout(shape).layout();
}

absl::StatusOr<std::string> CpuTopologyDescription::Serialize() const {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(cpu_topology_.ToProto(), &result)) {
    return absl::InternalError("Failed to serialize cpu_topology");
  }
  return result;
}

std::vector<std::unique_ptr<const PjRtDeviceDescription>>
CpuTopologyDescription::DeviceDescriptions() const {
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices;
  devices.reserve(cpu_topology_.number_of_devices());
  for (const CpuTopology::CpuDevice& device : cpu_topology_.devices()) {
    devices.push_back(std::make_unique<CpuDeviceDescription>(
        device.process_id, device.local_device_id));
  }
  return devices;
}

}  // namespace xla
