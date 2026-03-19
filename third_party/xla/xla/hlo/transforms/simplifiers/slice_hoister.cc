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

#include "xla/hlo/transforms/simplifiers/slice_hoister.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {

// Helper function to attempt hoisting a slice through a transpose operation.
// Returns true if a change was made.
absl::StatusOr<bool> TryHoistSliceThroughTranspose(
    HloSliceInstruction* slice_instruction, HloComputation* computation) {
  HloInstruction* transpose = slice_instruction->mutable_operand(0);
  if (transpose->opcode() != HloOpcode::kTranspose) {
    return false;
  }

  // All checks passed, perform the hoisting.
  auto dimensions_permutation = transpose->dimensions();
  auto inversed_dimensions_permutation =
      InversePermutation(dimensions_permutation);

  std::vector<int64_t> new_slice_starts = Permute(
      slice_instruction->slice_starts(), inversed_dimensions_permutation);
  std::vector<int64_t> new_slice_limits = Permute(
      slice_instruction->slice_limits(), inversed_dimensions_permutation);
  std::vector<int64_t> new_slice_strides = Permute(
      slice_instruction->slice_strides(), inversed_dimensions_permutation);

  HloInstruction* transpose_operand = transpose->mutable_operand(0);
  Shape new_slice_shape = ShapeUtil::PermuteDimensions(
      inversed_dimensions_permutation, slice_instruction->shape());

  HloInstruction* new_slice =
      computation->AddInstruction(HloInstruction::CreateSlice(
          new_slice_shape, transpose_operand, new_slice_starts,
          new_slice_limits, new_slice_strides));
  HloInstruction* new_transpose =
      computation->AddInstruction(HloInstruction::CreateTranspose(
          slice_instruction->shape(), new_slice, dimensions_permutation));
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(slice_instruction, new_transpose));
  return true;
}

// Helper function to attempt hoisting a slice through an element-wise binary
// operation. Returns true if a change was made.
absl::StatusOr<bool> TryHoistSliceThroughElementwiseBinaryOperation(
    HloSliceInstruction* slice_instruction, HloComputation* computation) {
  HloInstruction* slice_operand = slice_instruction->mutable_operand(0);
  if (!slice_operand->IsElementwiseBinary()) {
    return false;
  }

  HloInstruction* lhs = slice_operand->mutable_operand(0);
  HloInstruction* rhs = slice_operand->mutable_operand(1);

  if (lhs->shape() != rhs->shape()) {
    VLOG(1) << " Operand shapes do not match: " << lhs->shape() << " and "
            << rhs->shape();
    return false;
  }
  if (lhs->shape().element_type() !=
      slice_instruction->shape().element_type()) {
    VLOG(1) << " Slice element type does not match operand element type: "
            << lhs->shape().element_type() << " and "
            << slice_instruction->shape().element_type();
    return false;
  }

  // All checks passed, perform the hoisting.
  HloInstruction* lhs_slice =
      computation->AddInstruction(HloInstruction::CreateSlice(
          slice_instruction->shape(), lhs, slice_instruction->slice_starts(),
          slice_instruction->slice_limits(),
          slice_instruction->slice_strides()));
  HloInstruction* rhs_slice =
      computation->AddInstruction(HloInstruction::CreateSlice(
          slice_instruction->shape(), rhs, slice_instruction->slice_starts(),
          slice_instruction->slice_limits(),
          slice_instruction->slice_strides()));
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      slice_instruction, HloInstruction::CreateBinary(
                             slice_instruction->shape(),
                             slice_operand->opcode(), lhs_slice, rhs_slice)));
  return true;
}

// Helper function that goes through a list of potential hoisting scenarios.
// Returns true if a change was made.
absl::StatusOr<bool> TryHoistingSlice(HloInstruction* instruction,
                                      HloComputation* computation) {
  if (instruction->opcode() != HloOpcode::kSlice) {
    return false;
  }
  HloSliceInstruction* slice_instruction =
      Cast<HloSliceInstruction>(instruction);
  auto hoisting_functions = {TryHoistSliceThroughElementwiseBinaryOperation,
                             TryHoistSliceThroughTranspose};
  for (auto hoisting_function : hoisting_functions) {
    TF_ASSIGN_OR_RETURN(bool changed,
                        hoisting_function(slice_instruction, computation));
    if (changed) {
      return true;
    }
  }
  return false;
}

// As slices reduce the size of the input, it can be beneficial to hoist
// slices as high in the graph as possible, ideally right after parameter
// reads, which could reduce both compute and memory costs.
//
// Currently, this pass hoists slice operations through element-wise binary
// operations. Note that this pass can create redundant slices, which can be
// removed by running CSE.
//
// Note that algebraic simplifier also has `HandleSlice` function.
absl::StatusOr<bool> HoistSliceOperations(HloComputation* computation) {
  bool changed = false;
  bool changed_on_last_iteration = false;
  // TODO(b/434724820): Consider also other operations that aren't handled in
  // algebraic simplifier.
  // TODO(b/434724820): Make this more efficient by e.g. using a worklist or a
  // topological sort.
  do {
    changed |= changed_on_last_iteration;
    changed_on_last_iteration = false;
    std::vector<HloInstruction*> instructions =
        computation->MakeInstructionPostOrder();
    for (HloInstruction* instruction : instructions) {
      TF_ASSIGN_OR_RETURN(bool instruction_changed,
                          TryHoistingSlice(instruction, computation));
      if (instruction_changed) {
        changed_on_last_iteration = true;
        break;
      }
    }
  } while (changed_on_last_iteration);

  return changed;
}
}  // anonymous namespace

absl::StatusOr<bool> SliceHoister::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool changed_computation,
                        HoistSliceOperations(computation));
    changed |= changed_computation;
  }
  return changed;
}

}  // namespace xla
