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

#ifndef XLA_PJRT_PJRT_CLIENT_UTILS_H_
#define XLA_PJRT_PJRT_CLIENT_UTILS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/shape.h"

namespace xla {

absl::StatusOr<std::vector<Shape>> ConvertShapeSpecsToShapes(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts);

std::string GetPluginStablehloVersionOrDefault(
    const absl::flat_hash_map<std::string, PjRtValueType>& attributes);
std::string GetPluginStablehloVersionOrDefault(const PjRtClient* client);

std::string GetPluginSdyVersionOrDefault(
    const absl::flat_hash_map<std::string, PjRtValueType>& attributes);
std::string GetPluginSdyVersionOrDefault(const PjRtClient* client);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_CLIENT_UTILS_H_
