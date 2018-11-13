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

#include "tensorflow/compiler/xla/service/dot_decomposer.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

// TODO(b/69062148) Remove this code when all backends support BatchDot
// natively.
Status DecomposeBatchDot(HloInstruction* dot) {
  auto computation = dot->parent();
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& dot_shape = dot->shape();

  // ShapeInference should guarantee that lhs/rhs batch dimensions match.
  CHECK_EQ(dnums.lhs_batch_dimensions_size(),
           dnums.rhs_batch_dimensions_size());
  const int64 num_batch_dims = dnums.lhs_batch_dimensions_size();
  // Calculate total batch size (note that ShapeInference requires that
  // the batch dimensions are most-major).
  int64 batch_size = 1;
  for (int i = 0; i < num_batch_dims; ++i) {
    CHECK_EQ(lhs_shape.dimensions(dnums.lhs_batch_dimensions(i)),
             rhs_shape.dimensions(dnums.rhs_batch_dimensions(i)));
    batch_size *= lhs_shape.dimensions(dnums.lhs_batch_dimensions(i));
  }

  // Set lhs/rhs_transpose.
  CHECK_EQ(1, dnums.lhs_contracting_dimensions_size());
  const int64 lhs_contracting_dim_number = dnums.lhs_contracting_dimensions(0);
  const bool lhs_transpose = (lhs_contracting_dim_number - num_batch_dims) == 0;

  CHECK_EQ(1, dnums.rhs_contracting_dimensions_size());
  const int64 rhs_contracting_dim_number = dnums.rhs_contracting_dimensions(0);
  const bool rhs_transpose = (rhs_contracting_dim_number - num_batch_dims) == 1;

  // Compute R3 and R3 shapes for lhs.
  PrimitiveType lhs_type = lhs_shape.element_type();
  const int64 lhs_rows = lhs_shape.dimensions(num_batch_dims + 0);
  const int64 lhs_cols = lhs_shape.dimensions(num_batch_dims + 1);
  Shape lhs_shape_r3 =
      ShapeUtil::MakeShape(lhs_type, {batch_size, lhs_rows, lhs_cols});
  Shape lhs_slice_shape_r3 =
      ShapeUtil::MakeShape(lhs_type, {1, lhs_rows, lhs_cols});
  Shape lhs_slice_shape_r2 =
      ShapeUtil::MakeShape(lhs_type, {lhs_rows, lhs_cols});

  // Compute R3 and R3 shapes for rhs.
  PrimitiveType rhs_type = rhs_shape.element_type();
  const int64 rhs_rows = rhs_shape.dimensions(num_batch_dims + 0);
  const int64 rhs_cols = rhs_shape.dimensions(num_batch_dims + 1);
  Shape rhs_shape_r3 =
      ShapeUtil::MakeShape(rhs_type, {batch_size, rhs_rows, rhs_cols});
  Shape rhs_slice_shape_r3 =
      ShapeUtil::MakeShape(rhs_type, {1, rhs_rows, rhs_cols});
  Shape rhs_slice_shape_r2 =
      ShapeUtil::MakeShape(rhs_type, {rhs_rows, rhs_cols});

  // Compute R3 and R3 shapes for dot output.
  PrimitiveType dot_type = dot_shape.element_type();
  const int64 dot_rows = dot_shape.dimensions(num_batch_dims + 0);
  const int64 dot_cols = dot_shape.dimensions(num_batch_dims + 1);
  Shape dot_shape_r2 = ShapeUtil::MakeShape(dot_type, {dot_rows, dot_cols});
  Shape dot_shape_r3 = ShapeUtil::MakeShape(dot_type, {1, dot_rows, dot_cols});
  Shape concat_shape_r3 =
      ShapeUtil::MakeShape(dot_type, {batch_size, dot_rows, dot_cols});

  // Reshape lhs/rhs into R3.
  auto lhs_r3 = computation->AddInstruction(
      HloInstruction::CreateReshape(lhs_shape_r3, lhs));
  auto rhs_r3 = computation->AddInstruction(
      HloInstruction::CreateReshape(rhs_shape_r3, rhs));

  // Loop through batch size, slicing out required lhs/rhs to compute each Dot.
  std::vector<HloInstruction*> output_slices(batch_size);
  for (int64 i = 0; i < batch_size; ++i) {
    // Slice R3 shape from 'lhs' and reshape to R2.
    auto lhs_slice_r3 = computation->AddInstruction(
        HloInstruction::CreateSlice(lhs_slice_shape_r3, lhs_r3, {i, 0, 0},
                                    {i + 1, lhs_rows, lhs_cols}, {1, 1, 1}));
    auto lhs_slice_r2 = computation->AddInstruction(
        HloInstruction::CreateReshape(lhs_slice_shape_r2, lhs_slice_r3));

    // Slice R3 shape from 'rhs' and reshape to R2.
    auto rhs_slice_r3 = computation->AddInstruction(
        HloInstruction::CreateSlice(rhs_slice_shape_r3, rhs_r3, {i, 0, 0},
                                    {i + 1, rhs_rows, rhs_cols}, {1, 1, 1}));
    auto rhs_slice_r2 = computation->AddInstruction(
        HloInstruction::CreateReshape(rhs_slice_shape_r2, rhs_slice_r3));

    // Transpose lhs/rhs (if needed).
    if (lhs_transpose) {
      Shape lhs_slice_shape_r2_transpose =
          ShapeUtil::MakeShape(lhs_type, {lhs_cols, lhs_rows});
      lhs_slice_r2 =
          computation->AddInstruction(HloInstruction::CreateTranspose(
              lhs_slice_shape_r2_transpose, lhs_slice_r2, {1, 0}));
    }
    if (rhs_transpose) {
      Shape rhs_slice_shape_r2_transpose =
          ShapeUtil::MakeShape(rhs_type, {rhs_cols, rhs_rows});
      rhs_slice_r2 =
          computation->AddInstruction(HloInstruction::CreateTranspose(
              rhs_slice_shape_r2_transpose, rhs_slice_r2, {1, 0}));
    }

    // Compute Dot of lhs/rhs R2 slices.
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    auto dot_r2 = computation->AddInstruction(HloInstruction::CreateDot(
        dot_shape_r2, lhs_slice_r2, rhs_slice_r2, dot_dnums));
    dot_r2->set_precision_config(dot->precision_config());

    // Reshape Dot to R3 so we can concat along batch dimension.
    auto dot_r3 = computation->AddInstruction(
        HloInstruction::CreateReshape(dot_shape_r3, dot_r2));

    output_slices[i] = dot_r3;
  }

  // Concatenate slices from 'output_slices' along batch dimension.
  auto concat = computation->AddInstruction(
      HloInstruction::CreateConcatenate(concat_shape_r3, output_slices, 0));
  // Reshape output 'new_dot' to original dimensions.
  auto new_dot = computation->AddInstruction(
      HloInstruction::CreateReshape(dot_shape, concat));

  // Replace all uses of 'dot' in 'computation' with 'new_dot'.
  return computation->ReplaceInstruction(dot, new_dot);
}

}  // namespace

StatusOr<bool> DotDecomposer::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "DotDecomposer ENTRY\n" + module->ToString());
  // Gather all batch Dot operations.
  std::vector<HloInstruction*> batch_dots;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kDot) {
        continue;
      }
      const DotDimensionNumbers& dnums = instruction->dot_dimension_numbers();
      if (dnums.lhs_batch_dimensions_size() > 0 && decompose_batch_dot_) {
        batch_dots.push_back(instruction);
      }
    }
  }
  // Decompose each batch Dot in 'batch_dots'.
  bool changed = false;
  for (auto* dot : batch_dots) {
    TF_RETURN_IF_ERROR(DecomposeBatchDot(dot));
    changed = true;
  }
  XLA_VLOG_LINES(2, "DotDecompose EXIT\n" + module->ToString());
  return changed;
}

}  // namespace xla
