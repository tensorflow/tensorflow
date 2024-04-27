/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_algebraic_simplifier.h"

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

bool GpuAlgebraicSimplifierVisitor::ShouldStrengthReduceDotToReduce(
    const HloInstruction* hlo) {
  if (!options_.enable_dot_strength_reduction()) {
    return false;
  }

  const HloDotInstruction* dot = DynCast<HloDotInstruction>(hlo);
  if (dot == nullptr) {
    return false;
  }

  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  DotDimensionNumbers dnums = dot->dot_dimension_numbers();
  bool lhs_is_vector = (dnums.lhs_batch_dimensions_size() +
                            dnums.lhs_contracting_dimensions_size() ==
                        lhs->shape().rank());
  bool rhs_is_vector = (dnums.rhs_batch_dimensions_size() +
                            dnums.rhs_contracting_dimensions_size() ==
                        rhs->shape().rank());
  // Strength-reduce vector-vector dots since they are not supported by
  // GemmFusion.
  return lhs_is_vector && rhs_is_vector;
}

}  // namespace xla::gpu
