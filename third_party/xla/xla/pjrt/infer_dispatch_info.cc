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

#include "xla/pjrt/infer_dispatch_info.h"

#include "xla/pjrt/utils.h"

namespace xla {

static std::vector<Shape> GetParameterShapes(const ComputationLayout& layout) {
  // For now, XLA programs compiled with multiple arguments for PJRT cannot use
  // tuples for any of their arguments, so we can assume that a tuple can only
  // arise when there is a single argument.
  std::vector<Shape> shapes;
  if (layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple()) {
    shapes.reserve(layout.parameter_shape(0).tuple_shapes().size());
    absl::c_copy(layout.parameter_shape(0).tuple_shapes(),
                 std::back_inserter(shapes));
  } else {
    shapes.reserve(layout.parameter_count());
    for (const ShapeLayout& sl : layout.parameter_layouts()) {
      shapes.push_back(sl.shape());
    }
  }
  return shapes;
}

absl::StatusOr<CommonPjRtLoadedExecutable::DispatchInfo> InferDispatchInfo(
    CommonPjRtClient* client, const ComputationLayout& layout,
    const HloInputOutputAliasConfig& alias_config,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<CommonPjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs) {
  CommonPjRtLoadedExecutable::DispatchInfo result{
      .parameter_device_shapes = GetParameterShapes(layout),
      .output_device_shape = layout.result_shape(),
      .addressable_devices = std::move(addressable_devices),
      .addressable_device_logical_ids =
          std::move(addressable_device_logical_ids),
      .device_assignment = std::move(device_assignment),
  };
  {
    absl::Span<const Shape> shapes =
        result.output_device_shape.IsTuple()
            ? absl::MakeSpan(result.output_device_shape.tuple_shapes())
            : absl::MakeSpan(&result.output_device_shape, 1);
    result.output_memory_space_kind_ids.reserve(shapes.size());
    for (const auto& shape : shapes) {
      TF_ASSIGN_OR_RETURN(int kind, client->GetMemorySpaceKindForShape(shape));
      result.output_memory_space_kind_ids.push_back(kind);
    }
  }
  // Initializes information about which arguments to which executables must
  // be donated due to aliases that were specified by the computation.
  TF_ASSIGN_OR_RETURN(
      result.parameters_that_must_be_donated,
      ComputeParametersThatMustBeDonated(
          alias_config, result.parameter_device_shapes.size(), tuple_inputs));
  result.input_buffer_sizes_in_bytes.reserve(
      result.parameter_device_shapes.size());
  for (const Shape& shape : result.parameter_device_shapes) {
    DCHECK(!shape.IsTuple());
    TF_ASSIGN_OR_RETURN(int kind, client->GetMemorySpaceKindForShape(shape));
    TF_ASSIGN_OR_RETURN(int64_t size_in_bytes,
                        client->GetOnDeviceBytesCount(kind, shape));
    result.input_buffer_sizes_in_bytes.push_back(size_in_bytes);
  }
  return result;
}

}  // namespace xla
