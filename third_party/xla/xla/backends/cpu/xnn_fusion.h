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

#ifndef XLA_BACKENDS_CPU_XNN_FUSION_H_
#define XLA_BACKENDS_CPU_XNN_FUSION_H_

#include "absl/status/statusor.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Returns true if the dot operation is supported by XNNPACK. Returns an error
// if the dot operation shape is invalid.
absl::StatusOr<bool> IsXnnDotSupported(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_XNN_FUSION_H_
