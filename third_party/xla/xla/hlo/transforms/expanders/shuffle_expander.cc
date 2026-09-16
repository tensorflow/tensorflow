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

#include "xla/hlo/transforms/expanders/shuffle_expander.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shuffle.h"
#include "xla/util.h"

namespace xla {

bool ShuffleExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kShuffle) {
    return false;
  }
  auto* shuffle = Cast<HloShuffleInstruction>(instruction);
  switch (shuffle->mode()) {
    case ShuffleMode::kRotate:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<HloInstruction*> ShuffleExpander::ExpandInstruction(
    HloInstruction* instruction) {
  auto* shuffle = Cast<HloShuffleInstruction>(instruction);
  HloInstruction* result = shuffle->mutable_operand(0);
  absl::Span<const int64_t> dimensions = shuffle->dimensions();
  switch (shuffle->mode()) {
    case ShuffleMode::kRotate: {
      const auto& shifts = shuffle->rotate().shifts();
      for (size_t idx = 0; idx < dimensions.size(); ++idx) {
        int64_t dim = dimensions[idx];
        int64_t shift = shifts[idx];
        const Shape& curr_shape = result->shape();
        int64_t rank = curr_shape.dimensions().size();

        int64_t dim_size = curr_shape.dimensions(dim);
        int64_t norm_shift = shuffle::NormalizeShift(shift, dim_size);
        if (norm_shift == 0) {
          continue;
        }

        DimensionVector start_indices_0(rank, 0);
        DimensionVector limit_indices_0(curr_shape.dimensions().begin(),
                                        curr_shape.dimensions().end());
        DimensionVector strides(rank, 1);

        start_indices_0[dim] = norm_shift;

        DimensionVector start_indices_1(rank, 0);
        DimensionVector limit_indices_1(curr_shape.dimensions().begin(),
                                        curr_shape.dimensions().end());
        limit_indices_1[dim] = norm_shift;

        ABSL_ASSIGN_OR_RETURN(HloInstruction * slice0,
                         MakeSliceHlo(result, start_indices_0, limit_indices_0,
                                      strides, &shuffle->metadata()));
        ABSL_ASSIGN_OR_RETURN(HloInstruction * slice1,
                         MakeSliceHlo(result, start_indices_1, limit_indices_1,
                                      strides, &shuffle->metadata()));

        ABSL_ASSIGN_OR_RETURN(
            result, MakeConcatHlo({slice0, slice1}, dim, &shuffle->metadata()));
      }

      return result;
    }
    default:
      return nullptr;
  }
}

}  // namespace xla
