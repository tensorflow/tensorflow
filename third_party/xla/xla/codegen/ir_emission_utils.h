#include <optional>

#include "absl/functional/any_invocable.h"
#include "xla/hlo/utils/hlo_traversal.h"
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

#ifndef XLA_CODEGEN_IR_EMISSION_UTILS_H_
#define XLA_CODEGEN_IR_EMISSION_UTILS_H_

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Returns the bitwidth of the given primitive type. Unfortunately,
// primitive_util::BitWidth(PRED) return 1 instead of 8.
int GetBitwidth(PrimitiveType type);

/// Description of how to emit a given transposition.
struct TransposeDescription {
  // Transpose instruction.
  const HloInstruction* instr;

  // Normalized transpose dimensions.
  absl::InlinedVector<int64_t, 3> dimensions;

  // Permutations of normalized transpose dimensions.
  absl::InlinedVector<int64_t, 3> permutation;

  // Required amount of shared memory in bytes.
  int64_t shmem_usage = 0;

  TransposeDescription(const HloInstruction* instr,
                       absl::InlinedVector<int64_t, 3> dimensions,
                       absl::InlinedVector<int64_t, 3> permutation,
                       int64_t shmem_usage)
      : instr(instr),
        dimensions(dimensions),
        permutation(permutation),
        shmem_usage(shmem_usage) {}

  // Transpose instruction input shape.
  const Shape& input_shape() const { return instr->operand(0)->shape(); }

  // Returns true, if both descriptions have the same dimensions and
  // permutation, even if they're produced by different instructions.
  bool IsEquivalent(const TransposeDescription& other) const {
    return dimensions == other.dimensions && permutation == other.permutation &&
           GetBitwidth(instr->shape().element_type()) ==
               GetBitwidth(other.instr->shape().element_type());
  }
};

// Checks if the instruction is elementwise.
bool IsIntermediate(const HloInstruction* instr, int allowed_operand_count = 1);

// Find the first gero that statises the given predicate.
std::optional<HloInstructionAdaptor> FindHero(
    const HloInstructionAdaptor& root,
    absl::AnyInvocable<bool(const HloInstruction&)> predicate);

}  // namespace xla

#endif  // XLA_CODEGEN_IR_EMISSION_UTILS_H_
