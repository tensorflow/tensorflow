/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/transpose_folding.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla {
namespace {

TransposeFolding::OperandIndices CanFoldOperandsIntoConvolution(
    const HloInstruction& convolution,
    const TransposeFolding::TransposableConvOperandsFn&
        transposable_conv_operands) {
  if (HloOpcode::kConvolution != convolution.opcode()) {
    return {};
  }

  TransposeFolding::OperandIndices operand_set;
  for (int64_t i = 0; i < convolution.operand_count(); ++i) {
    auto& operand = *convolution.operand(i);
    if (operand.opcode() == HloOpcode::kTranspose) {
      operand_set.push_back(i);
    }
  }

  return transposable_conv_operands(convolution, operand_set);
}

bool IsNonIdentityTranspose(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kTranspose) {
    for (int dim = 0; dim < instruction->dimensions().size(); ++dim) {
      if (dim != instruction->dimensions(dim)) {
        return true;
      }
    }
  }
  return false;
}

void TransposeDims(tsl::protobuf::RepeatedField<int64_t>& dims,
                   absl::Span<const int64_t> transpose_dims) {
  for (auto& dim : dims) {
    dim = transpose_dims[dim];
  }
}

using InstructionOperandsPair =
    std::pair<HloInstruction*, TransposeFolding::OperandIndices>;

// Folds the operands of `dot` that are foldable transposes.
absl::Status FoldTransposeIntoDot(InstructionOperandsPair& pair) {
  HloInstruction* dot = pair.first;

  DotDimensionNumbers new_dot_dims = dot->dot_dimension_numbers();
  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);

  for (int64_t operand_index : pair.second) {
    if (operand_index == 0) {
      TransposeDims(*new_dot_dims.mutable_lhs_contracting_dimensions(),
                    lhs->dimensions());
      TransposeDims(*new_dot_dims.mutable_lhs_batch_dimensions(),
                    lhs->dimensions());
      lhs = lhs->mutable_operand(0);
    } else {
      CHECK_EQ(operand_index, 1);
      TransposeDims(*new_dot_dims.mutable_rhs_contracting_dimensions(),
                    rhs->dimensions());
      TransposeDims(*new_dot_dims.mutable_rhs_batch_dimensions(),
                    rhs->dimensions());
      rhs = rhs->mutable_operand(0);
    }
  }
  HloInstruction* new_dot =
      dot->parent()->AddInstruction(HloInstruction::CreateDot(
          dot->shape(), lhs, rhs, new_dot_dims, dot->precision_config()));
  dot->SetupDerivedInstruction(new_dot);
  return dot->parent()->ReplaceInstruction(dot, new_dot);
}

// Folds the operands of `convolution` that are foldable transposes.
// `computation` is the parent HLO computation of `convolution`.
//
// Returns whether the module is changed.
bool FoldTransposeIntoConvolution(InstructionOperandsPair& pair) {
  auto& convolution = *pair.first;
  auto& operand_indices = pair.second;

  if (operand_indices.empty()) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  ConvolutionDimensionNumbers new_dnums = dnums;

  HloInstruction* new_lhs;
  const int64_t kLhsIdx = 0;
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
  const int64_t kRhsIdx = 1;
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
    CanFoldTransposeOperand dot_can_fold_transpose_operand,
    TransposableConvOperandsFn transposable_conv_operands)
    : dot_can_fold_transpose_operand_(
          std::move(dot_can_fold_transpose_operand)),
      transposable_conv_operands_(std::move(transposable_conv_operands)) {}

absl::StatusOr<bool> TransposeFolding::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Modifying the graph while traversing is dangerous, so we find all folding
  // opportunities before actually folding them.
  std::vector<InstructionOperandsPair> foldable_dots;
  std::vector<InstructionOperandsPair> foldable_convolutions;

  FunctionVisitor visit_fn([this, &foldable_dots, &foldable_convolutions](
                               HloInstruction* instruction) {
    if (instruction->opcode() == HloOpcode::kDot) {
      // Don't fold dots with a 1D operand.
      if ((instruction->operand(0)->shape().rank() < 2) ||
          (instruction->operand(1)->shape().rank() < 2)) {
        return absl::OkStatus();
      }

      OperandIndices operand_indices;
      for (int64_t i = 0; i < 2; ++i) {
        if (!IsNonIdentityTranspose(instruction->operand(i))) {
          continue;
        }

        TF_ASSIGN_OR_RETURN(bool can_fold_operand,
                            dot_can_fold_transpose_operand_(*instruction, i));

        if (can_fold_operand) {
          operand_indices.push_back(i);
        }
      }

      if (!operand_indices.empty()) {
        foldable_dots.emplace_back(instruction, operand_indices);
      }
    }

    {
      OperandIndices operand_indices = CanFoldOperandsIntoConvolution(
          *instruction, transposable_conv_operands_);
      if (!operand_indices.empty()) {
        foldable_convolutions.emplace_back(instruction, operand_indices);
      }
    }
    return absl::OkStatus();
  });

  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
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

/*static*/ absl::StatusOr<bool>
TransposeFolding::IsRowColumnTransposeDotOperand(const HloInstruction& dot,
                                                 int64_t operand_idx) {
  TF_RET_CHECK(dot.opcode() == HloOpcode::kDot);
  TF_RET_CHECK(dot.operand_count() > operand_idx);

  const HloInstruction& transpose = *dot.operand(operand_idx);
  TF_RET_CHECK(transpose.opcode() == HloOpcode::kTranspose);

  const DotDimensionNumbers& dot_dims = dot.dot_dimension_numbers();

  auto batch_dims = (operand_idx == 0) ? dot_dims.lhs_batch_dimensions()
                                       : dot_dims.rhs_batch_dimensions();

  auto contracting_dims = (operand_idx == 0)
                              ? dot_dims.lhs_contracting_dimensions()
                              : dot_dims.rhs_contracting_dimensions();

  return (batch_dims.size() == transpose.shape().rank() - 2) &&
         (contracting_dims.size() == 1) &&
         absl::c_all_of(batch_dims, [&](int64_t dim) {
           return transpose.dimensions(dim) == dim;
         });
}

}  // namespace xla
