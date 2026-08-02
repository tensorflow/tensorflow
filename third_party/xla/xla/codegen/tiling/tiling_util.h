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

#ifndef XLA_CODEGEN_TILING_TILING_UTIL_H_
#define XLA_CODEGEN_TILING_TILING_UTIL_H_

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"

namespace xla {

// Returns true if `roots` represents a multi-output fusion (i.e. has more than
// one root) and all roots have the same shape according to `eq`.
bool IsSameShapeMultiOutputFusion(absl::Span<const HloInstructionAdaptor> roots,
                                  Shape::Equal eq);

bool IsSameShapeMultiOutputFusion(const HloFusionInstruction& fusion,
                                  Shape::Equal eq);

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_TILING_UTIL_H_
