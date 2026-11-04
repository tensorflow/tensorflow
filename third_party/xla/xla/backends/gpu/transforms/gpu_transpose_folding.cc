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

#include "xla/backends/gpu/transforms/gpu_transpose_folding.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/shape_tracker.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// Returns true for transposes and (type-preserving) bitcasts.
bool IsTransposeOrBitcast(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kTranspose ||
         (inst->opcode() == HloOpcode::kBitcast &&
          inst->shape().element_type() ==
              inst->operand(0)->shape().element_type());
}

bool IsSupportedByGemm(const DotOperandDims& dims,
                       bool requires_minor_contracting_dims) {
  if (requires_minor_contracting_dims &&
      !AreContractingDimsMinor(dims.shape().layout(),
                               dims.Indices(DotOperandDims::kContracting))) {
    return false;
  }
  return MatrixLayout::For(dims.shape(), dims.Indices(DotOperandDims::kBatch),
                           dims.Indices(DotOperandDims::kContracting),
                           dims.Indices(DotOperandDims::kNonContracting))
      .ok();
}

// Folds the transpose/bitcast chain above `operand` into the farthest producer
// for which the dot stays supported. Candidates are the chain's input and every
// producer above a transpose: bitcasts are free, so stopping between leading
// bitcasts gains nothing. Returns whether `operand` and `dims` changed.
absl::StatusOr<bool> FoldIntoDotOperand(bool requires_minor_contracting_dims,
                                        HloInstruction*& operand,
                                        DotOperandDims& dims) {
  const DotOperandDims consumer_dims = dims;
  ShapeTracker tracker(operand->shape());
  bool above_transpose = false;
  bool changed = false;
  for (HloInstruction* source = operand; IsTransposeOrBitcast(source);) {
    ABSL_RETURN_IF_ERROR(tracker.PrependInstruction(source));
    above_transpose |= source->opcode() == HloOpcode::kTranspose;
    source = source->mutable_operand(0);
    if (!above_transpose && IsTransposeOrBitcast(source)) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(std::optional<DotOperandDims> source_dims,
                     consumer_dims.FoldBackward(tracker));
    if (!source_dims.has_value() ||
        !IsSupportedByGemm(*source_dims, requires_minor_contracting_dims)) {
      continue;
    }
    dims = *std::move(source_dims);
    operand = source;
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> FoldIntoDot(HloInstruction* dot,
                                 const se::GpuComputeCapability& gpu_version) {
  ABSL_ASSIGN_OR_RETURN((std::array<DotOperandDims, 2> dims),
                   DotOperandDims::FromDot(dot));
  std::array<HloInstruction*, 2> operands = {dot->mutable_operand(0),
                                             dot->mutable_operand(1)};
  const bool requires_minor_contracting_dims =
      DotRequiresMinorContractingDims(*dot, gpu_version);
  bool changed = false;
  for (int64_t i = 0; i < 2; ++i) {
    ABSL_ASSIGN_OR_RETURN(bool folded,
                     FoldIntoDotOperand(requires_minor_contracting_dims,
                                        operands[i], dims[i]));
    changed |= folded;
  }
  if (!changed) {
    return false;
  }
  ABSL_ASSIGN_OR_RETURN(DotDimensionNumbers dim_nums,
                   DotOperandDims::CreateDotDimensionNumbers(dims));
  HloInstruction* new_dot = dot->AddInstruction(
      HloInstruction::CreateDot(dot->shape(), operands[0], operands[1],
                                dim_nums, dot->precision_config()));
  dot->SetupDerivedInstruction(new_dot);
  ABSL_RETURN_IF_ERROR(dot->parent()->ReplaceInstruction(dot, new_dot));
  return true;
}

absl::StatusOr<bool> FoldIntoConvolution(HloInstruction* conv) {
  ConvolutionDimensionNumbers dim_nums = conv->convolution_dimension_numbers();
  std::array<HloInstruction*, 2> operands = {conv->mutable_operand(0),
                                             conv->mutable_operand(1)};
  bool changed = false;
  while (operands[0]->opcode() == HloOpcode::kTranspose) {
    absl::Span<const int64_t> p = operands[0]->dimensions();
    dim_nums.set_input_batch_dimension(p[dim_nums.input_batch_dimension()]);
    dim_nums.set_input_feature_dimension(p[dim_nums.input_feature_dimension()]);
    for (int64_t& dim : *dim_nums.mutable_input_spatial_dimensions()) {
      dim = p[dim];
    }
    operands[0] = operands[0]->mutable_operand(0);
    changed = true;
  }
  while (operands[1]->opcode() == HloOpcode::kTranspose) {
    absl::Span<const int64_t> p = operands[1]->dimensions();
    dim_nums.set_kernel_input_feature_dimension(
        p[dim_nums.kernel_input_feature_dimension()]);
    dim_nums.set_kernel_output_feature_dimension(
        p[dim_nums.kernel_output_feature_dimension()]);
    for (int64_t& dim : *dim_nums.mutable_kernel_spatial_dimensions()) {
      dim = p[dim];
    }
    operands[1] = operands[1]->mutable_operand(0);
    changed = true;
  }
  if (!changed) {
    return false;
  }
  std::unique_ptr<HloInstruction> new_conv =
      conv->CloneWithNewOperands(conv->shape(), operands);
  new_conv->set_convolution_dimension_numbers(dim_nums);
  ABSL_RETURN_IF_ERROR(
      conv->parent()->ReplaceWithNewInstruction(conv, std::move(new_conv)));
  return true;
}

}  // namespace

absl::StatusOr<bool> GpuTransposeFolding::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      // Skip instructions with extra (e.g. scaling or sparsity) operands.
      if (instruction->operand_count() != 2) {
        continue;
      }
      if (instruction->opcode() == HloOpcode::kDot) {
        ABSL_ASSIGN_OR_RETURN(bool folded, FoldIntoDot(instruction, gpu_version_));
        changed |= folded;
      } else if (fold_into_convolutions_ &&
                 instruction->opcode() == HloOpcode::kConvolution) {
        ABSL_ASSIGN_OR_RETURN(bool folded, FoldIntoConvolution(instruction));
        changed |= folded;
      }
    }
  }
  return changed;
}

}  // namespace xla::gpu
