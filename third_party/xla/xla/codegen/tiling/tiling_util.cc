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

#include "xla/codegen/tiling/tiling_util.h"

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"

namespace xla {

bool IsSameShapeMultiOutputFusion(absl::Span<const HloInstructionAdaptor> roots,
                                  Shape::Equal eq) {
  if (roots.size() <= 1) {
    return false;
  }
  return absl::c_all_of(roots.subspan(1),
                        [&](const HloInstructionAdaptor& root) {
                          return eq(roots[0].shape(), root.shape());
                        });
}

bool IsSameShapeMultiOutputFusion(const HloFusionInstruction& fusion,
                                  Shape::Equal eq) {
  const Shape& root_shape = fusion.shape();
  if (!root_shape.IsTuple() || root_shape.tuple_shapes().size() <= 1) {
    return false;
  }

  const Shape& first_subshape = root_shape.tuple_shapes()[0];
  return absl::c_all_of(
      absl::MakeSpan(root_shape.tuple_shapes()).subspan(1),
      [&](const Shape& shape) { return eq(first_subshape, shape); });
}

}  // namespace xla
