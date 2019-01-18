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

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
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
    auto dot_r2 = computation->AddInstruction(
        HloInstruction::CreateDot(dot_shape_r2, lhs_slice_r2, rhs_slice_r2,
                                  dot_dnums, dot->precision_config()));

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

// Convert a dot into a canonical form where non-contracting and contracting
// dimensions are reshaped together and batch dimensions are the most major
// dimensions. The requires transposing and reshapes the lhs and rhs and
// reshaping the output batch to the original shape.
Status CanonicalizeDot(HloInstruction* original_dot) {
  auto computation = original_dot->parent();
  const auto& original_dnums = original_dot->dot_dimension_numbers();
  const int64 num_batch_dims = original_dnums.lhs_batch_dimensions_size();
  const int64 num_contracting_dims =
      original_dnums.lhs_contracting_dimensions_size();

  const auto& lhs_shape = original_dot->operand(0)->shape();
  const int64 lhs_rank = lhs_shape.rank();
  const int64 num_lhs_non_contracting_dims =
      lhs_rank - num_batch_dims - num_contracting_dims;

  std::vector<int64> lhs_non_contracting_dims;
  lhs_non_contracting_dims.reserve(num_lhs_non_contracting_dims);
  int64 lhs_contracting_size = 1;
  int64 lhs_non_contracting_size = 1;
  std::vector<int64> batch_dim_sizes;
  batch_dim_sizes.reserve(num_batch_dims);
  for (int64 i = 0; i < lhs_rank; ++i) {
    if (absl::c_linear_search(original_dnums.lhs_contracting_dimensions(), i)) {
      lhs_contracting_size *= lhs_shape.dimensions(i);
    } else if (absl::c_linear_search(original_dnums.lhs_batch_dimensions(),
                                     i)) {
      batch_dim_sizes.push_back(lhs_shape.dimensions(i));
    } else {
      lhs_non_contracting_dims.push_back(i);
      lhs_non_contracting_size *= lhs_shape.dimensions(i);
    }
  }
  // The canonical form of the lhs is
  // [BatchDims, NonContractingDims, ContractingsDims]
  std::vector<int64> lhs_transpose;
  lhs_transpose.reserve(lhs_rank);
  lhs_transpose.insert(lhs_transpose.end(),
                       original_dnums.lhs_batch_dimensions().begin(),
                       original_dnums.lhs_batch_dimensions().end());
  lhs_transpose.insert(lhs_transpose.end(), lhs_non_contracting_dims.begin(),
                       lhs_non_contracting_dims.end());
  lhs_transpose.insert(lhs_transpose.end(),
                       original_dnums.lhs_contracting_dimensions().begin(),
                       original_dnums.lhs_contracting_dimensions().end());
  HloInstruction* transposed_lhs =
      computation->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::PermuteDimensions(InversePermutation(lhs_transpose),
                                       lhs_shape),
          original_dot->mutable_operand(0), lhs_transpose));
  std::vector<int64> lhs_reshape_dims = batch_dim_sizes;
  lhs_reshape_dims.push_back(lhs_non_contracting_size);
  lhs_reshape_dims.push_back(lhs_contracting_size);
  // Reshape the contracting and non-contracting dimensions together.
  HloInstruction* reshaped_lhs =
      computation->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(lhs_shape.element_type(), lhs_reshape_dims),
          transposed_lhs));

  const auto& rhs_shape = original_dot->operand(1)->shape();
  const int64 rhs_rank = rhs_shape.rank();
  const int64 num_rhs_non_contracting_dims =
      rhs_rank - num_batch_dims - num_contracting_dims;
  std::vector<int64> rhs_non_contracting_dims;
  rhs_non_contracting_dims.reserve(num_rhs_non_contracting_dims);
  int64 rhs_non_contracting_size = 1;
  int64 rhs_contracting_size = 1;
  for (int64 i = 0; i < rhs_rank; ++i) {
    if (absl::c_linear_search(original_dnums.rhs_contracting_dimensions(), i)) {
      rhs_contracting_size *= rhs_shape.dimensions(i);
    } else if (!absl::c_linear_search(original_dnums.rhs_batch_dimensions(),
                                      i)) {
      rhs_non_contracting_dims.push_back(i);
      rhs_non_contracting_size *= rhs_shape.dimensions(i);
    }
  }

  // The canonical form of the rhs is
  // [BatchDims, ContractingsDims, NonContractingDims]
  std::vector<int64> rhs_transpose;
  rhs_transpose.reserve(rhs_rank);
  rhs_transpose.insert(rhs_transpose.end(),
                       original_dnums.rhs_batch_dimensions().begin(),
                       original_dnums.rhs_batch_dimensions().end());
  rhs_transpose.insert(rhs_transpose.end(),
                       original_dnums.rhs_contracting_dimensions().begin(),
                       original_dnums.rhs_contracting_dimensions().end());
  rhs_transpose.insert(rhs_transpose.end(), rhs_non_contracting_dims.begin(),
                       rhs_non_contracting_dims.end());
  HloInstruction* transposed_rhs =
      computation->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::PermuteDimensions(InversePermutation(rhs_transpose),
                                       rhs_shape),
          original_dot->mutable_operand(1), rhs_transpose));

  std::vector<int64> rhs_reshape_dims = batch_dim_sizes;
  rhs_reshape_dims.push_back(rhs_contracting_size);
  rhs_reshape_dims.push_back(rhs_non_contracting_size);
  // Reshape the contracting and non-contracting dimensions together.
  HloInstruction* reshaped_rhs =
      computation->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(rhs_shape.element_type(), rhs_reshape_dims),
          transposed_rhs));

  std::vector<int64> dot_dims = batch_dim_sizes;
  dot_dims.push_back(lhs_non_contracting_size);
  dot_dims.push_back(rhs_non_contracting_size);

  DotDimensionNumbers dot_dnums;
  for (int64 i = 0; i < num_batch_dims; ++i) {
    dot_dnums.add_lhs_batch_dimensions(i);
    dot_dnums.add_rhs_batch_dimensions(i);
  }
  dot_dnums.add_lhs_contracting_dimensions(num_batch_dims + 1);
  dot_dnums.add_rhs_contracting_dimensions(num_batch_dims);

  HloInstruction* dot = computation->AddInstruction(HloInstruction::CreateDot(
      ShapeUtil::MakeShape(original_dot->shape().element_type(), dot_dims),
      reshaped_lhs, reshaped_rhs, dot_dnums, original_dot->precision_config()));

  return computation->ReplaceInstruction(
      original_dot, computation->AddInstruction(HloInstruction::CreateReshape(
                        original_dot->shape(), dot)));
}

}  // namespace

StatusOr<bool> DotDecomposer::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "DotDecomposer ENTRY\n" + module->ToString());
  // Gather all Non-canonical Dot operations.
  std::vector<HloInstruction*> non_canonical_dots;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kDot) {
        continue;
      }
      const DotDimensionNumbers& dnums = instruction->dot_dimension_numbers();
      // A dot it not canonical if there are more than one contracting
      // dimension.
      if (dnums.lhs_contracting_dimensions_size() > 1) {
        non_canonical_dots.push_back(instruction);
        continue;
      }
      if (dnums.lhs_batch_dimensions().empty() &&
          dnums.lhs_contracting_dimensions().empty()) {
        non_canonical_dots.push_back(instruction);
        continue;
      }
      if (dnums.lhs_batch_dimensions().empty()) {
        continue;
      }
      std::vector<int64> canonical_batch_dims(
          dnums.lhs_batch_dimensions_size());
      absl::c_iota(canonical_batch_dims, 0);
      if (!absl::c_equal(dnums.lhs_batch_dimensions(), canonical_batch_dims) ||
          !absl::c_equal(dnums.rhs_batch_dimensions(), canonical_batch_dims)) {
        non_canonical_dots.push_back(instruction);
      }
    }
  }
  bool changed = false;
  for (auto* dot : non_canonical_dots) {
    TF_RETURN_IF_ERROR(CanonicalizeDot(dot));
    changed = true;
  }

  if (decompose_batch_dot_) {
    std::vector<HloInstruction*> batch_dots;
    for (auto* computation : module->MakeNonfusionComputations()) {
      for (auto* instruction : computation->instructions()) {
        if (instruction->opcode() != HloOpcode::kDot) {
          continue;
        }
        const DotDimensionNumbers& dnums = instruction->dot_dimension_numbers();
        if (!dnums.lhs_batch_dimensions().empty()) {
          batch_dots.push_back(instruction);
        }
      }
    }
    // Decompose each batch Dot in 'batch_dots'.

    for (auto* dot : batch_dots) {
      TF_RETURN_IF_ERROR(DecomposeBatchDot(dot));
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "DotDecompose EXIT\n" + module->ToString());
  return changed;
}

}  // namespace xla
