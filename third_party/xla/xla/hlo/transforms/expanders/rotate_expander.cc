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

#include "xla/hlo/transforms/expanders/rotate_expander.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/status_macros.h"

namespace xla {

bool RotateExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kRotate;
}

absl::StatusOr<HloInstruction*> RotateExpander::ExpandInstruction(
    HloInstruction* instruction) {
  auto* rotate = Cast<HloRotateInstruction>(instruction);
  HloInstruction* result = rotate->mutable_operand(0);
  absl::Span<const int64_t> dimensions = rotate->dimensions();
  const std::vector<int64_t>& shifts = rotate->shifts();
  TF_RET_CHECK(dimensions.size() == shifts.size());
  TF_RET_CHECK(!result->shape().IsTuple());

  for (size_t idx = 0; idx < dimensions.size(); ++idx) {
    int64_t dim = dimensions[idx];
    int64_t shift = shifts[idx];
    const Shape& curr_shape = result->shape();
    int64_t rank = curr_shape.dimensions().size();
    TF_RET_CHECK(dim >= 0 && dim < rank);

    int64_t dim_size = curr_shape.dimensions(dim);
    if (dim_size <= 1) {
      continue;
    }
    int64_t norm_shift = ((shift % dim_size) + dim_size) % dim_size;
    if (norm_shift == 0) {
      continue;
    }

    std::vector<int64_t> start_indices_0(rank, 0);
    std::vector<int64_t> limit_indices_0(curr_shape.dimensions().begin(),
                                         curr_shape.dimensions().end());
    std::vector<int64_t> strides(rank, 1);

    start_indices_0[dim] = norm_shift;

    std::vector<int64_t> start_indices_1(rank, 0);
    std::vector<int64_t> limit_indices_1(curr_shape.dimensions().begin(),
                                         curr_shape.dimensions().end());
    limit_indices_1[dim] = norm_shift;

    ASSIGN_OR_RETURN(HloInstruction * slice0,
                     MakeSliceHlo(result, start_indices_0, limit_indices_0,
                                  strides, &rotate->metadata()));
    ASSIGN_OR_RETURN(HloInstruction * slice1,
                     MakeSliceHlo(result, start_indices_1, limit_indices_1,
                                  strides, &rotate->metadata()));

    ASSIGN_OR_RETURN(result,
                     MakeConcatHlo({slice0, slice1}, dim, &rotate->metadata()));
  }

  return result;
}

}  // namespace xla
