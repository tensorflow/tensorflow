/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_layout_assignment.h"

#include <numeric>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace cpu {

// We want to change the layout of constant arrays to be column major when all
// of their users are dot operations that can be made faster with the flipped
// layout.  To avoid going quadriatic over the # of instructions, we cache this
// property in should_make_rhs_col_major -- it maps a constant to true if all of
// the users of said constant are dot operations that can be sped up.  This
// cache is populated lazily as we encounter dot operations traversing the
// instruction stream.

namespace {
using ::tensorflow::gtl::nullopt;
using ::tensorflow::gtl::optional;

using ShouldMakeOperandColMajorCache =
    tensorflow::gtl::FlatMap<const HloInstruction*, bool>;
}  // namespace

static bool ShouldMakeAllUsersColMajor(const HloInstruction* instruction) {
  for (auto* user : instruction->users()) {
    optional<int64> operand_idx = ProfitableToMakeDotOperandColumnMajor(*user);
    if (!operand_idx || user->operand(*operand_idx) != instruction ||
        std::count(user->operands().begin(), user->operands().end(),
                   instruction) != 1) {
      return false;
    }
  }
  return true;
}

static optional<int64> ShouldMakeOperandColumnMajor(
    ShouldMakeOperandColMajorCache* cache, const HloInstruction& instruction) {
  optional<int64> operand_idx =
      ProfitableToMakeDotOperandColumnMajor(instruction);
  if (!operand_idx) {
    return nullopt;
  }

  const HloInstruction* operand = instruction.operand(*operand_idx);
  if (operand->opcode() != HloOpcode::kConstant) {
    return nullopt;
  }

  auto it = cache->find(operand);
  if (it == cache->end()) {
    auto insert_result =
        cache->insert({operand, ShouldMakeAllUsersColMajor(operand)});
    CHECK(insert_result.second);
    it = insert_result.first;
  }

  return it->second ? operand_idx : nullopt;
}

static Shape RowMajorShape(const Shape& old_shape) {
  Shape new_shape(old_shape);
  std::vector<int64> dimension_order(new_shape.dimensions_size());
  std::iota(dimension_order.rbegin(), dimension_order.rend(), 0);
  *new_shape.mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
  return new_shape;
}

static Shape ColMajorShape(const Shape& old_shape) {
  Shape new_shape(old_shape);
  std::vector<int64> dimension_order(new_shape.dimensions_size());
  std::iota(dimension_order.begin(), dimension_order.end(), 0);
  *new_shape.mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
  return new_shape;
}

Status CpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  ShouldMakeOperandColMajorCache cache;

  const HloComputation* computation = constraints->computation();
  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kConvolution &&
        PotentiallyImplementedAsEigenConvolution(*instruction,
                                                 target_machine_features_)) {
      const HloInstruction* convolution = instruction;
      const HloInstruction* lhs_instruction = convolution->operand(0);
      const HloInstruction* rhs_instruction = convolution->operand(1);

      // In order to implement `convolution` with Eigen convolution, the layouts
      // of the input, filter, and output need to be row-major.
      //
      // These constraints are not hard constraints. Ideally, we should decide
      // which layouts to choose according to some cost model.
      Shape output_shape(RowMajorShape(convolution->shape()));
      Shape input_shape(RowMajorShape(lhs_instruction->shape()));
      Shape filter_shape(RowMajorShape(rhs_instruction->shape()));

      // Set layouts of the instructions' shapes.
      TF_RETURN_IF_ERROR(
          constraints->SetOperandLayout(input_shape, convolution, 0));
      TF_RETURN_IF_ERROR(
          constraints->SetOperandLayout(filter_shape, convolution, 1));
      TF_RETURN_IF_ERROR(
          constraints->SetInstructionLayout(output_shape, convolution));
    } else if (optional<int64> op_idx =
                   ShouldMakeOperandColumnMajor(&cache, *instruction)) {
      const HloInstruction* op = instruction->operand(*op_idx);
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
          ColMajorShape(op->shape()), instruction, *op_idx));
    } else if (PotentiallyImplementedAsEigenDot(*instruction,
                                                target_machine_features_)) {
      const HloInstruction* dot = instruction;
      // In order to implement `dot` with Eigen dot, the layouts of the lhs,
      // rhs, and output need to be row-major.
      //
      // These constraints are not hard constraints. Ideally, we should decide
      // which layouts to choose according to some cost model.
      Shape output_shape(RowMajorShape(dot->shape()));

      const HloInstruction* lhs_instruction = dot->operand(0);
      Shape lhs_shape(RowMajorShape(lhs_instruction->shape()));
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(lhs_shape, dot, 0));

      const HloInstruction* rhs_instruction = dot->operand(1);
      Shape rhs_shape(RowMajorShape(rhs_instruction->shape()));
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(rhs_shape, dot, 1));

      // Set layouts of the instructions' shapes.
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(output_shape, dot));
    } else {
      for (int64 operand_no = 0; operand_no < instruction->operand_count();
           ++operand_no) {
        // Skip operands which already have a constraint.
        if (constraints->OperandLayout(instruction, operand_no) != nullptr) {
          continue;
        }
        // Skip over forwarded operands.
        if (constraints->OperandBufferForwarded(instruction, operand_no)) {
          continue;
        }
        // Skip operands with non-array shapes.
        if (!ShapeUtil::IsArray(instruction->operand(operand_no)->shape())) {
          continue;
        }
        Shape operand_shape(
            RowMajorShape(instruction->operand(operand_no)->shape()));
        TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
            operand_shape, instruction, operand_no));
      }
      // Skip over the root instruction for the top-level computation.
      if (computation->parent()->entry_computation() == computation &&
          computation->root_instruction() == instruction) {
        continue;
      }
      // Skip instructions which don't produce array shapes (tuples, opaque,
      // etc.).
      if (!ShapeUtil::IsArray(instruction->shape())) {
        continue;
      }
    }
  }
  return Status::OK();
}
}  // namespace cpu
}  // namespace xla
