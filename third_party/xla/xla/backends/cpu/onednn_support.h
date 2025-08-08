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

#ifndef XLA_BACKENDS_CPU_ONEDNN_SUPPORT_H_
#define XLA_BACKENDS_CPU_ONEDNN_SUPPORT_H_

// oneDNN-fusion-related defines that don't depend on oneDNN Graph API.
// For anything dependent on Graph API, put it in onednn_fusion.h.

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

inline constexpr absl::string_view kOneDnnFusionKind = "__onednn_fusion";

// Returns true if the dot operation is supported by oneDNN. Returns an error
// if the dot operation shape is invalid.
absl::StatusOr<bool> IsOneDnnDotSupported(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_ONEDNN_SUPPORT_H_
