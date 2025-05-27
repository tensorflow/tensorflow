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
#include "xla/pjrt/gpu/se_gpu_topology_description.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::StatusOr<std::string> StreamExecutorGpuTopologyDescription::Serialize()
    const {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(gpu_topology_->ToProto(), &result)) {
    return absl::InternalError("Failed to serialize gpu_topology");
  }
  return result;
}

absl::StatusOr<Layout> StreamExecutorGpuTopologyDescription::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) const {
  Shape shape = ShapeUtil::MakeValidatedShape(element_type, dims).value();
  Layout layout = LayoutUtil::GetWithDefaultLayout(shape).layout();
  // `GetWithDefaultLayout` returns a padded layout for sub-byte types since the
  // notion of "default" is context dependent and in this case means the default
  // for literals for historical reasons. Because of this, we need to manually
  // populate the `element_size_in_bits` for sub-byte types here.
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    layout.set_element_size_in_bits(primitive_util::BitWidth(element_type));
  }
  return layout;
}

}  // namespace xla
