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

#include "tensorflow/compiler/xla/service/transpose_folding.h"

#include <vector>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

TransposeFolding::OperandIndices CanFoldOperandsIntoDot(
    const HloInstruction& dot,
    const TransposeFolding::TransposableGemmOperandsFn&
        transposable_gemm_operands) {
  if (HloOpcode::kDot != dot.opcode() ||
      dot.dot_dimension_numbers().lhs_batch_dimensions_size() != 0) {
    return {};
  }

  TransposeFolding::OperandIndices operand_set;
  for (int64 i = 0; i < dot.operand_count(); ++i) {
    auto& operand = *dot.operand(i);
    if (operand.IsRank2Transpose()) {
      operand_set.push_back(i);
    } else if (operand.shape().rank() != 2) {
      return {};
    }
  }

  return transposable_gemm_operands(dot, operand_set);
}

TransposeFolding::OperandIndices CanFoldOperandsIntoConvolution(
    const HloInstruction& convolution,
    const TransposeFolding::TransposableConvOperandsFn&
        transposable_conv_operands) {
  if (HloOpcode::kConvolution != convolution.opcode()) {
    return {};
  }

  TransposeFolding::OperandIndices operand_set;
  for (int64 i = 0; i < convolution.operand_count(); ++i) {
    auto& operand = *convolution.operand(i);
    if (operand.opcode() == HloOpcode::kTranspose) {
      operand_set.push_back(i);
    }
  }

  return transposable_conv_operands(convolution, operand_set);
}

using InstructionOperandsPair =
    std::pair<HloInstruction*, TransposeFolding::OperandIndices>;

// Folds the operands of `dot` that are foldable transposes. `computation` is
// the parent HLO computation of `dot`.
Status FoldTransposeIntoDot(InstructionOperandsPair pair) {
  HloInstruction* dot = pair.first;

  DotDimensionNumbers new_dim_numbers = dot->dot_dimension_numbers();
  HloInstruction* new_lhs = dot->mutable_operand(0);
  HloInstruction* new_rhs = dot->mutable_operand(1);

  CHECK_EQ(new_dim_numbers.lhs_batch_dimensions_size(), 0);
  CHECK_EQ(new_dim_numbers.rhs_batch_dimensions_size(), 0);
  CHECK_EQ(new_dim_numbers.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(new_dim_numbers.rhs_contracting_dimensions_size(), 1);

  for (int64 operand_index : pair.second) {
    // We've checked that there aren't any batch dimensions and that the inputs
    // are rank 2, and shape inference guarantees that there is exactly one
    // contracting dimension.
    if (operand_index == 0) {
      CHECK_EQ(new_lhs->opcode(), HloOpcode::kTranspose);
      new_dim_numbers.set_lhs_contracting_dimensions(
          0, 1 - new_dim_numbers.lhs_contracting_dimensions(0));
      new_lhs = new_lhs->mutable_operand(0);
    } else {
      CHECK_EQ(operand_index, 1);
      CHECK_EQ(new_rhs->opcode(), HloOpcode::kTranspose);
      new_dim_numbers.set_rhs_contracting_dimensions(
          0, 1 - new_dim_numbers.rhs_contracting_dimensions(0));
      new_rhs = new_rhs->mutable_operand(0);
    }
  }

  std::unique_ptr<HloInstruction> new_dot = HloInstruction::CreateDot(
      dot->shape(), new_lhs, new_rhs, new_dim_numbers, dot->precision_config());
  return dot->parent()->ReplaceWithNewInstruction(dot, std::move(new_dot));
}

// Folds the operands of `convolution` that are foldable transposes.
// `computation` is the parent HLO computation of `convolution`.
//
// Returns whether the module is changed.
bool FoldTransposeIntoConvolution(InstructionOperandsPair pair) {
  auto& convolution = *pair.first;
  auto& operand_indices = pair.second;

  if (operand_indices.empty()) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  ConvolutionDimensionNumbers new_dnums = dnums;

  HloInstruction* new_lhs;
  const int64 kLhsIdx = 0;
  if (absl::c_linear_search(operand_indices, kLhsIdx)) {
    HloInstruction& transpose = *convolution.mutable_operand(kLhsIdx);
    const auto& transpose_dimensions = transpose.dimensions();
    HloInstruction& transpose_operand = *transpose.mutable_operand(0);

    // Everything remains the same except for the input/output dimension
    // numbers. We need to apply the transpose permutation to the original shape
    // to figure out what the new logical dimensions are.
    new_dnums.set_input_batch_dimension(
        transpose_dimensions[dnums.input_batch_dimension()]);
    new_dnums.set_input_feature_dimension(
        transpose_dimensions[dnums.input_feature_dimension()]);
    for (auto& input_spatial_dimension :
         *new_dnums.mutable_input_spatial_dimensions()) {
      input_spatial_dimension = transpose_dimensions[input_spatial_dimension];
    }
    new_lhs = &transpose_operand;
  } else {
    new_lhs = convolution.mutable_operand(kLhsIdx);
  }

  HloInstruction* new_rhs;
  const int64 kRhsIdx = 1;
  if (absl::c_linear_search(operand_indices, kRhsIdx)) {
    HloInstruction& transpose = *convolution.mutable_operand(kRhsIdx);
    const auto& transpose_dimensions = transpose.dimensions();
    HloInstruction& transpose_operand = *transpose.mutable_operand(0);

    // Everything remains the same except for the kernel dimension numbers. We
    // need to apply the transpose permutation to the original shape to figure
    // out what the new logical dimensions are.
    new_dnums.set_kernel_input_feature_dimension(
        transpose_dimensions[dnums.kernel_input_feature_dimension()]);
    new_dnums.set_kernel_output_feature_dimension(
        transpose_dimensions[dnums.kernel_output_feature_dimension()]);
    for (auto& kernel_spatial_dimension :
         *new_dnums.mutable_kernel_spatial_dimensions()) {
      kernel_spatial_dimension = transpose_dimensions[kernel_spatial_dimension];
    }
    new_rhs = &transpose_operand;
  } else {
    new_rhs = convolution.mutable_operand(kRhsIdx);
  }

  auto new_conv = HloInstruction::CreateConvolve(
      convolution.shape(), new_lhs, new_rhs, convolution.feature_group_count(),
      convolution.batch_group_count(), convolution.window(), new_dnums,
      convolution.precision_config());
  TF_CHECK_OK(convolution.parent()->ReplaceWithNewInstruction(
      &convolution, std::move(new_conv)));

  return true;
}

}  // namespace

TransposeFolding::TransposeFolding(
    TransposableGemmOperandsFn transposable_gemm_operands,
    TransposableConvOperandsFn transposable_conv_operands)
    : transposable_gemm_operands_(std::move(transposable_gemm_operands)),
      transposable_conv_operands_(std::move(transposable_conv_operands)) {}

StatusOr<bool> TransposeFolding::Run(HloModule* module) {
  // Modifying the graph while traversing is dangerous, so we find all folding
  // opportunities before actually folding them.
  std::vector<std::pair<HloInstruction*, OperandIndices>> foldable_dots;
  std::vector<std::pair<HloInstruction*, OperandIndices>> foldable_convolutions;
  FunctionVisitor visit_fn([this, &foldable_dots, &foldable_convolutions](
                               HloInstruction* instruction) {
    {
      OperandIndices operand_indices =
          CanFoldOperandsIntoDot(*instruction, transposable_gemm_operands_);
      if (!operand_indices.empty()) {
        foldable_dots.emplace_back(instruction, operand_indices);
      }
    }
    {
      OperandIndices operand_indices = CanFoldOperandsIntoConvolution(
          *instruction, transposable_conv_operands_);
      if (!operand_indices.empty()) {
        foldable_convolutions.emplace_back(
            std::make_pair(instruction, operand_indices));
      }
    }
    return Status::OK();
  });

  for (auto* comp : module->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(comp->Accept(&visit_fn));
  }

  bool changed = false;
  for (InstructionOperandsPair& pair : foldable_dots) {
    TF_RETURN_IF_ERROR(FoldTransposeIntoDot(pair));
    changed = true;
  }
  for (InstructionOperandsPair& pair : foldable_convolutions) {
    changed |= FoldTransposeIntoConvolution(pair);
  }
  return changed;
}

}  // namespace xla
