/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/interpreter/interpreter_topology_description.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_topology_description_registry.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/primitive_util.h"
#include "xla/service/device_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

REGISTER_PJRT_TOPOLOGY_DESERIALIZER(
    Interpreter, xla::InterpreterId(), xla::InterpreterName(),
    [](const xla::PjRtTopologyDescriptionProto& proto) {
      return InterpreterTopologyDescription::FromProto(proto);
    });

const InterpreterDescription& InterpreterDescription::Singleton() {
  static const InterpreterDescription* const singleton =
      new InterpreterDescription;
  return *singleton;
}

std::vector<std::unique_ptr<const PjRtDeviceDescription>>
InterpreterTopologyDescription::DeviceDescriptions() const {
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices;
  devices.push_back(std::make_unique<InterpreterDescription>());
  return devices;
}

absl::StatusOr<uint64_t> InterpreterTopologyDescription::Fingerprint() const {
  return xla::InterpreterId();
}

absl::Span<const int> InterpreterTopologyDescription::GetMemorySpaceKindIds()
    const {
  static const int kInterpreterMemorySpaceKindIds[] = {0};
  return absl::MakeConstSpan(kInterpreterMemorySpaceKindIds);
}

absl::StatusOr<Layout> InterpreterTopologyDescription::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) const {
  if (!primitive_util::IsArrayType(element_type)) {
    return InvalidArgument("Element type %s does not support layout",
                           PrimitiveType_Name(element_type));
  }
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  return LayoutUtil::GetWithDefaultLayout(shape).layout();
}

absl::StatusOr<DeviceAssignment>
InterpreterTopologyDescription::GetDefaultDeviceAssignment(
    int process_index, int num_replicas,
    std::optional<int> num_replicas_per_slice, int num_partitions,
    const MultiSliceConfig* multi_slice_config) const {
  if (num_replicas_per_slice.has_value() || multi_slice_config) {
    return absl::UnimplementedError(
        "Multi-slice GetDefaultDeviceAssignment is not supported.");
  }
  if (num_replicas != 1 || num_partitions != 1) {
    return absl::UnimplementedError(
        "Interpreter only supports num_replicas=1 and num_partitions=1.");
  }
  DeviceAssignment assignment(1, 1);
  assignment(0, 0) = 0;
  return assignment;
}

absl::StatusOr<xla::PjRtTopologyDescriptionProto>
InterpreterTopologyDescription::ToProto() const {
  PjRtTopologyDescriptionProto proto;
  proto.set_platform_id(platform_id());
  proto.set_platform_name(platform_name());
  proto.set_platform_version(platform_version());
  proto.set_is_subslice_topology(is_subslice_topology());
  return proto;
}

/*static*/ absl::StatusOr<std::unique_ptr<InterpreterTopologyDescription>>
InterpreterTopologyDescription::FromProto(
    const xla::PjRtTopologyDescriptionProto& proto) {
  if (proto.platform_id() != xla::InterpreterId() &&
      proto.platform_name() != xla::InterpreterName()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The platform is not an Interpreter platform. platform_id: ",
        proto.platform_id(), ", platform_name: ", proto.platform_name()));
  }
  return std::make_unique<InterpreterTopologyDescription>();
}

}  // namespace xla
