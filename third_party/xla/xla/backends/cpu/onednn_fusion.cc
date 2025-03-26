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

#include "xla/backends/cpu/onednn_fusion.h"

#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<bool> IsOneDnnDotSupported(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape) {
  // TODO(penporn): Support other element types.
  if (lhs_shape.element_type() != F32 || rhs_shape.element_type() != F32 ||
      out_shape.element_type() != F32) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  // Restrict support to no transposes and row-major layouts for now.
  return dot_canonical_dims.lhs_canonical && dot_canonical_dims.rhs_canonical &&
         !dot_canonical_dims.lhs_column_major &&
         !dot_canonical_dims.rhs_column_major;
}

}  // namespace xla::cpu
