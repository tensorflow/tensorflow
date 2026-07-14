/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/scatter_expander.h"

#include "absl/algorithm/container.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"

namespace xla {

bool GpuScatterExpander::InstructionMatchesPattern(HloInstruction* inst) {
  if (!HloPredicateIsOp<HloOpcode::kScatter>(inst)) {
    return false;
  }
  auto is_unsupported_element = [](const Shape& shape) {
    return primitive_util::BitWidth(shape.element_type()) > 64;
  };
  // TODO(b/129698548): Scattering elements larger than 64 bits is not
  // supported by XLA:GPU.
  // Note: Variadic scatter is only supported for unique indices.
  if (inst->shape().IsTuple()) {
    return !inst->unique_indices() ||
           absl::c_any_of(inst->shape().tuple_shapes(), is_unsupported_element);
  }
  return is_unsupported_element(inst->shape());
}

}  // namespace xla
