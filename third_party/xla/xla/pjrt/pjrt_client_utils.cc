/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_client_utils.h"

#include <cstddef>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<std::vector<Shape>> ConvertShapeSpecsToShapes(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts) {
  if (device_layouts.has_value() &&
      device_layouts->size() != shape_specs.size()) {
    return InvalidArgument(
        "Number of layouts %d does not match the number of shapes %d",
        device_layouts->size(), shape_specs.size());
  }
  std::vector<Shape> device_shapes;
  device_shapes.reserve(shape_specs.size());
  for (size_t i = 0; i < shape_specs.size(); ++i) {
    auto& shape_spec = shape_specs[i];
    Shape& device_shape = device_shapes.emplace_back(
        ShapeUtil::MakeShape(shape_spec.element_type, shape_spec.dims));
    if (device_layouts.has_value() && (*device_layouts)[i].has_value()) {
      *device_shape.mutable_layout() = *(*device_layouts)[i];
    }
  }
  return device_shapes;
}

}  // namespace xla
