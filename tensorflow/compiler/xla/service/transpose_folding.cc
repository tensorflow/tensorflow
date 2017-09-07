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
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
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
  if (HloOpcode::kDot != dot.opcode()) {
    return {};
  }

  TransposeFolding::OperandIndices operand_set;
  for (int64 i = 0; i < dot.operand_count(); ++i) {
    auto& operand = *dot.operand(i);
    if (operand.IsRank2Transpose() && operand.user_count() == 1) {
      operand_set.push_back(i);
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

  // We only support folding the RHS.
  const int64 kRhsOperandIndex = 1;
  auto& operand = *convolution.operand(kRhsOperandIndex);
  if (operand.opcode() == HloOpcode::kTranspose && operand.user_count() == 1) {
    return transposable_conv_operands(convolution, {kRhsOperandIndex});
  }

  return {};
}

using InstructionOperandsPair =
    std::pair<HloInstruction*, TransposeFolding::OperandIndices>;

// Folds the operands of `dot` that are foldable transposes. `computation` is
// the parent HLO computation of `dot`.
//
// Returns whether the module is changed.
bool FoldTransposeIntoDot(InstructionOperandsPair pair) {
  auto* dot = pair.first;
  std::vector<HloInstruction*> instructions_to_fuse(1, dot);
  for (const int64 operand_index : pair.second) {
    instructions_to_fuse.push_back(dot->mutable_operand(operand_index));
  }

  // Early-exit if no operands are foldable.
  if (instructions_to_fuse.size() == 1) {
    return false;
  }

  dot->parent()->CreateFusionInstruction(
      instructions_to_fuse, HloInstruction::FusionKind::kTransposeDot);
  return true;
}

// Folds the operands of `convolution` that are foldable transposes.
// `computation` is the parent HLO computation of `convolution`.
//
// Returns whether the module is changed.
bool FoldTransposeIntoConvolution(InstructionOperandsPair pair) {
  auto& convolution = *pair.first;

  // We only support fusing the RHS transpose into convolution.
  //
  // ConvolutionDimensionNumbers doesn't make enough of a distinction between
  // the output and the activations.
  //
  // TODO(b/37125184): Support transposing the LHS too.
  if (pair.second.size() != 1 || pair.second.front() != 1) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  HloInstruction& transpose = *convolution.mutable_operand(1);
  CHECK_EQ(transpose.opcode(), HloOpcode::kTranspose);
  const auto& transpose_dimensions = transpose.dimensions();
  HloInstruction& transpose_operand = *transpose.mutable_operand(0);

  // Everything remains the same except for the kernel dimension numbers. We
  // need to apply the transpose permutation to the original shape to figure out
  // what the new logical dimensions are.
  ConvolutionDimensionNumbers new_dnums = dnums;
  new_dnums.set_kernel_input_feature_dimension(
      transpose_dimensions[dnums.kernel_input_feature_dimension()]);
  new_dnums.set_kernel_output_feature_dimension(
      transpose_dimensions[dnums.kernel_output_feature_dimension()]);
  for (auto& kernel_spatial_dimension :
       *new_dnums.mutable_kernel_spatial_dimensions()) {
    kernel_spatial_dimension = transpose_dimensions[kernel_spatial_dimension];
  }

  auto new_conv = HloInstruction::CreateConvolve(
      convolution.shape(), convolution.mutable_operand(0), &transpose_operand,
      convolution.window(), new_dnums);
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
  auto visit_fn = [this, &foldable_dots,
                   &foldable_convolutions](HloInstruction* instruction) {
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
    return tensorflow::Status::OK();
  };

  std::vector<HloComputation*> computations;
  for (auto& computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    computations.push_back(computation.get());
  }
  for (auto& comp : computations) {
    TF_RETURN_IF_ERROR(comp->Accept(visit_fn));
  }

  bool changed = false;
  for (InstructionOperandsPair& pair : foldable_dots) {
    changed |= FoldTransposeIntoDot(pair);
  }
  for (InstructionOperandsPair& pair : foldable_convolutions) {
    changed |= FoldTransposeIntoConvolution(pair);
  }
  return changed;
}

}  // namespace xla
