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

#include "xla/tests/hlo_runner_agnostic_reference_mixin.h"

#include <string>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"

namespace xla {

ProgramShape GetProgramShapeWithLayout(const HloModule& module) {
  ProgramShape program_shape;
  const auto* entry = module.entry_computation();
  for (const auto* param : entry->parameter_instructions()) {
    program_shape.AddParameter(param->shape(), std::string(param->name()));
  }
  *program_shape.mutable_result() = entry->root_instruction()->shape();
  return program_shape;
}

bool ProgramShapesEqual(const ProgramShape& lhs, const ProgramShape& rhs) {
  if (lhs.parameters_size() != rhs.parameters_size()) {
    return false;
  }
  for (int i = 0; i < lhs.parameters_size(); ++i) {
    if (!Shape::Equal().IgnoreElementSizeInLayout()(lhs.parameters(i),
                                                    rhs.parameters(i))) {
      return false;
    }
  }
  return Shape::Equal().IgnoreElementSizeInLayout()(lhs.result(), rhs.result());
}

}  // namespace xla
