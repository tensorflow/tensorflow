/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/split_k_gemm_rewriter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/codegen/triton/support_legacy.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_tiling_propagation.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

bool HasDivisibleSuffixAllowingSplit(const absl::Span<int64_t const> span,
                                     const int64_t divisor) {
  CHECK_GE(divisor, 1);
  int64_t product = 1;
  // Note: Using reverse iterator.
  for (auto it = span.crbegin(); it != span.crend(); ++it) {
    product *= *it;
    if (product % divisor == 0) {
      return true;
    }
    if (divisor % product != 0) {
      return false;
    }
  }
  return false;
}

namespace {

// Copy source values into destination incrementing those >= threshold by 1.
void CopyIncrementingAboveThreshold(
    const google::protobuf::RepeatedField<int64_t>& source,
    google::protobuf::RepeatedField<int64_t>& destination, const int threshold) {
  destination.Reserve(source.size());
  for (int64_t x : source) {
    if (x >= threshold) {
      ++x;
    }
    destination.Add(x);
  }
}

// Copy source values into destination incrementing those >= threshold by 1.
void CopyIncrementingAboveThreshold(absl::Span<const int64_t> source,
                                    DimensionVector& destination,
                                    const int threshold) {
  destination.reserve(source.size());
  for (int64_t x : source) {
    if (x >= threshold) {
      ++x;
    }
    destination.push_back(x);
  }
}

absl::Status UncompilableMatmul(absl::string_view explanation) {
  absl::Status s = absl::CancelledError(explanation);
  s.SetPayload(kUncompilableFusion, absl::Cord(explanation));
  return s;
}

// Returns the padded K dimension so that it is a multiple of split_k and 16B.
int64_t GetPaddedK(HloInstruction& dot, int64_t k, int64_t split_k) {
  const int64_t alignment_in_bits = 16 * 8;
  int64_t min_element_size_in_bits = alignment_in_bits;
  for (const HloInstruction* p : dot.parent()->parameter_instructions()) {
    min_element_size_in_bits = std::min(
        min_element_size_in_bits, ShapeUtil::ElementSizeInBits(p->shape()));
  }
  return RoundUpTo(k, split_k * alignment_in_bits / min_element_size_in_bits);
}

}  // namespace

absl::StatusOr<HloInstruction*> MakeSplitKOperand(
    HloInstruction& dot, const TritonFusionAnalysis& analysis,
    const TritonGemmConfig& config, const int64_t contracting_dim_idx,
    TritonFusionAnalysis::Scope scope, const int operand_number,
    std::optional<int64_t> padded_k_size = std::nullopt) {
  HloInstruction* operand = dot.mutable_operand(operand_number);
  const int64_t k = operand->shape().dimensions(contracting_dim_idx);
  padded_k_size =
      std::max(GetPaddedK(dot, k, config.split_k), padded_k_size.value_or(0));
  const bool need_padding = k < *padded_k_size;

  auto check_if_supported = [&](const HloInstruction& hlo,
                                bool check_divisibility) {
    const TensorIterationSpec::DimIterationSpec* spec =
        analysis.IterSpec(scope, &hlo, contracting_dim_idx);
    if (spec == nullptr) {
      // No contracting dimension - no checks needed.
      return absl::OkStatus();
    }
    if (spec->size() != 1) {
      return UncompilableMatmul("Unsupported case.");
    }
    const TensorIterationSpec::IterationSpecFragment& fragment = spec->at(0);
    if (fragment.is_sliced()) {
      return UncompilableMatmul(
          "Sliced contracting dimension is not supported yet.");
    }
    if (check_divisibility && !HasDivisibleSuffixAllowingSplit(
                                  fragment.subfragments, config.split_k)) {
      return UncompilableMatmul("Contracting dimension is too fragmented.");
    }
    bool is_scale = scope == TritonFusionAnalysis::Scope::LHS_SCALE ||
                    scope == TritonFusionAnalysis::Scope::RHS_SCALE;
    if (!is_scale ? config.split_k > ceil(1.0 * fragment.count / config.block_k)
                  : config.split_k >= fragment.count) {
      return UncompilableMatmul(
          "Too small divisible part of the contracting dimension.");
    }
    return absl::OkStatus();
  };

  // The divisibility check is only used to ensure that the TritonFusionAnalysis
  // in IrEmitterTriton can propagate the fragments correctly after the split-k
  // transform. The contracting dimension is always contiguous so far.
  //
  // If padding is needed on the operand then the divisibility may not hold
  // up for the scope parameters. We just check some basics here, and we check
  // the full analysis after the split-k transform at the end of
  // MakeDotComputationSplitKBatch.
  TF_RETURN_IF_ERROR(
      check_if_supported(*operand, /*check_divisibility=*/!need_padding));
  for (const HloInstruction* param : analysis.ScopeParameters(scope)) {
    TF_RETURN_IF_ERROR(
        check_if_supported(*param, /*check_divisibility=*/!need_padding));
  }

  // Add padding if needed.
  if (need_padding) {
    HloInstruction* const zero =
        dot.parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(operand->shape().element_type())));

    int64_t padding = *padded_k_size - k;
    PaddingConfig padding_config =
        MakeNoPaddingConfig(operand->shape().dimensions().size());
    padding_config.mutable_dimensions(contracting_dim_idx)
        ->set_edge_padding_high(padding);

    TF_ASSIGN_OR_RETURN(HloInstruction * pad,
                        MakePadHlo(operand, zero, padding_config));
    *pad->mutable_shape()->mutable_layout() = operand->shape().layout();
    operand = pad;
  }
  CHECK_GE(operand->shape().dimensions(contracting_dim_idx), config.split_k);

  // Add bitcast.
  const Shape& shape = operand->shape();
  Shape new_shape(shape.element_type(), /*dimensions=*/{});

  for (int i = 0; i < shape.dimensions().size(); ++i) {
    const int64_t dimension_size = shape.dimensions(i);
    if (i == contracting_dim_idx) {
      new_shape.add_dimensions(config.split_k);
      new_shape.add_dimensions(dimension_size / config.split_k);
    } else {
      new_shape.add_dimensions(dimension_size);
    }
  }

  Layout* new_layout = new_shape.mutable_layout();
  // Iterate through the logical dimension numbers in their physical order;
  // copy them into the new layout incrementing by one those that get shifted
  // by the insertion of the new batch dimension.
  for (int64_t logical_dim_idx : shape.layout().minor_to_major()) {
    // When 'logical_dim_idx' == 'contracting_dim_idx' add both
    // 'logical_dim_idx'+1 and 'logical_dim_idx' because it gets split into two.
    if (logical_dim_idx >= contracting_dim_idx) {
      new_layout->add_minor_to_major(logical_dim_idx + 1);
    }
    if (logical_dim_idx <= contracting_dim_idx) {
      new_layout->add_minor_to_major(logical_dim_idx);
    }
  }
  return MakeBitcastHlo(operand, new_shape);
}

// Apply split K configuration from the tiling config to the fused dot()
// computation: bitcast the operands, change the output shape and the dot
// dimensions.
absl::Status MakeDotComputationSplitKBatch(HloComputation* computation,
                                           const TritonGemmConfig& config) {
  HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  if (dot == nullptr) {
    dot = hlo_query::GetFirstInstructionWithOpcode(*computation,
                                                   HloOpcode::kScaledDot);
    CHECK(dot != nullptr);
  }

  TF_ASSIGN_OR_RETURN(const auto analysis,
                      TritonFusionAnalysis::Execute(*computation));
  const DotDimensionNumbers& old_dim_numbers = dot->dot_dimension_numbers();
  DotDimensionNumbers new_dim_numbers;

  TF_ASSIGN_OR_RETURN(const int64_t lhs_contracting_idx,
                      ContractingDimensionIndex(*dot, 0));
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_contracting_dimensions(),
      *new_dim_numbers.mutable_lhs_contracting_dimensions(),
      lhs_contracting_idx);
  new_dim_numbers.mutable_lhs_batch_dimensions()->Add(lhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_batch_dimensions(),
      *new_dim_numbers.mutable_lhs_batch_dimensions(), lhs_contracting_idx);

  TF_ASSIGN_OR_RETURN(const int64_t rhs_contracting_idx,
                      ContractingDimensionIndex(*dot, 1));
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_contracting_dimensions(),
      *new_dim_numbers.mutable_rhs_contracting_dimensions(),
      rhs_contracting_idx);
  new_dim_numbers.mutable_rhs_batch_dimensions()->Add(rhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_batch_dimensions(),
      *new_dim_numbers.mutable_rhs_batch_dimensions(), rhs_contracting_idx);

  // Collect HLOs to transform between dot output and root. These will
  // get a new major most batch dimension sized as split K factor. Other inputs
  // of these HLOs will get broadcasted.
  std::stack<HloInstruction*> to_process;
  // Store the same HLOs also in a hash set for quick lookups.
  absl::flat_hash_set<HloInstruction*> to_process_set;
  HloInstruction* current = dot;
  do {
    to_process.push(current);
    CHECK(to_process_set.insert(current).second);
    if (current->users().empty()) {
      break;
    }
    CHECK_EQ(current->user_count(), 1);
    current = current->users()[0];
    if (!legacy_triton::IsDistributiveOverAddition(*current)) {
      return Cancelled("Operation non-distributive over addition after dot.");
    }
  } while (true);

  // Keep track of whether any of the operands were padded.
  bool did_pad = false;
  auto is_padded = [](const HloInstruction* op) {
    return op->opcode() == HloOpcode::kBitcast &&
           op->operand(0)->opcode() == HloOpcode::kPad;
  };

  // Process the collected HLOs from computation root to dot.
  HloDotInstruction* dot_cast = DynCast<HloDotInstruction>(dot);
  PrimitiveType accumulator_dtype =
      dot_cast != nullptr ? GetGemmAccumulatorType(dot_cast) : F32;
  while (!to_process.empty()) {
    HloInstruction* current = to_process.top();
    to_process.pop();
    // Add split-K dimension to `current`.
    HloInstruction* expanded;
    if (current == dot) {
      if (dot_cast != nullptr) {
        // Dot operation.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * lhs,
            MakeSplitKOperand(*dot, analysis, config, lhs_contracting_idx,
                              TritonFusionAnalysis::Scope::LHS, 0));
        TF_ASSIGN_OR_RETURN(
            HloInstruction * rhs,
            MakeSplitKOperand(*dot, analysis, config, rhs_contracting_idx,
                              TritonFusionAnalysis::Scope::RHS, 1));
        did_pad = is_padded(lhs) || is_padded(rhs);
        // Keep the precision of the accumulator type for the dot output.
        TF_ASSIGN_OR_RETURN(
            expanded, MakeDotHlo(lhs, rhs, new_dim_numbers,
                                 dot->precision_config(), accumulator_dtype));
      } else {
        // Scaled dot operation.
        // At least one scaling operand is not a scalar and will define the
        // contracting dimension size.
        std::optional<int64_t> padded_k_size;
        auto assign_scale_operand =
            [&](TritonFusionAnalysis::Scope scope, int contracting_idx,
                int operand_idx,
                int64_t block_size) -> absl::StatusOr<HloInstruction*> {
          TF_ASSIGN_OR_RETURN(
              HloInstruction * scale,
              MakeSplitKOperand(*dot, analysis, config, contracting_idx, scope,
                                operand_idx));
          padded_k_size = scale->shape().dimensions(contracting_idx + 1) *
                          block_size * config.split_k;
          return scale;
        };
        HloInstruction* lhs_scale = dot->mutable_operand(2);
        if (analysis.lhs_scale_block_size().has_value()) {
          TF_ASSIGN_OR_RETURN(
              lhs_scale,
              assign_scale_operand(TritonFusionAnalysis::Scope::LHS_SCALE,
                                   lhs_contracting_idx, 2,
                                   *analysis.lhs_scale_block_size()));
        }
        HloInstruction* rhs_scale = dot->mutable_operand(3);
        if (analysis.rhs_scale_block_size().has_value()) {
          if (padded_k_size.has_value()) {
            int64_t rhs_block_size = analysis.rhs_scale_block_size().value();
            if (*padded_k_size % (rhs_block_size * config.split_k) != 0) {
              return UncompilableMatmul("Unable to split-K block scaled dot.");
            }
            TF_ASSIGN_OR_RETURN(
                rhs_scale,
                MakeSplitKOperand(*dot, analysis, config, rhs_contracting_idx,
                                  TritonFusionAnalysis::Scope::RHS_SCALE, 3,
                                  *padded_k_size / rhs_block_size));
          } else {
            TF_ASSIGN_OR_RETURN(
                rhs_scale,
                assign_scale_operand(TritonFusionAnalysis::Scope::RHS_SCALE,
                                     rhs_contracting_idx, 3,
                                     *analysis.rhs_scale_block_size()));
          }
        }
        did_pad = is_padded(lhs_scale) || is_padded(rhs_scale);
        // Make LHS/RHS input operands with fixed contracting dimension size.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * lhs,
            MakeSplitKOperand(*dot, analysis, config, lhs_contracting_idx,
                              TritonFusionAnalysis::Scope::LHS, 0,
                              padded_k_size));
        TF_ASSIGN_OR_RETURN(
            HloInstruction * rhs,
            MakeSplitKOperand(*dot, analysis, config, rhs_contracting_idx,
                              TritonFusionAnalysis::Scope::RHS, 1,
                              padded_k_size));
        TF_ASSIGN_OR_RETURN(
            expanded,
            MakeScaledDotHlo(lhs, rhs, lhs_scale, rhs_scale, new_dim_numbers,
                             dot->precision_config(), accumulator_dtype));
      }
      // Make the added batch dimension the major-most, keep the order of the
      // original dimensions.
      expanded->mutable_shape()->mutable_layout()->clear_minor_to_major();
      CopyIncrementingAboveThreshold(dot->shape().layout().minor_to_major(),
                                     *expanded->mutable_shape()
                                          ->mutable_layout()
                                          ->mutable_minor_to_major(),
                                     0);
      expanded->mutable_shape()->mutable_layout()->add_minor_to_major(0);
      dot->SetupDerivedInstruction(expanded);
    } else {
      // Propagate the precision of the accumulator to the GEMM fusion root.
      expanded = computation->AddInstruction(
          current->CloneWithNewShape(ShapeUtil::PrependMajorDimension(
              config.split_k, ShapeUtil::ChangeElementType(
                                  current->shape(), accumulator_dtype))));
      if (expanded->opcode() == HloOpcode::kTranspose) {
        const auto* old_transpose = Cast<HloTransposeInstruction>(current);
        auto* new_transpose = Cast<HloTransposeInstruction>(expanded);
        new_transpose->mutable_dimensions()->clear();
        new_transpose->mutable_dimensions()->reserve(
            new_transpose->shape().dimensions().size());
        // The split-K batch dimension is always major.
        new_transpose->mutable_dimensions()->push_back(0);
        for (const int64_t dim : old_transpose->dimensions()) {
          new_transpose->mutable_dimensions()->push_back(dim + 1);
        }
      }
    }
    TF_RETURN_IF_ERROR(current->ReplaceAllUsesWithDifferentShape(expanded));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(current));
    // Broadcast operands.
    if (current == dot) {
      continue;
    }
    for (int i = 0; i < expanded->operands().size(); ++i) {
      HloInstruction* operand = expanded->mutable_operand(i);
      if (!to_process_set.contains(operand)) {
        // Broadcast the operand to the Split-K dimension and convert to the
        // accumulator dtype.
        HloInstruction* convert = MakeConvertToHlo(operand, accumulator_dtype);
        std::vector<int64_t> broadcast_dimensions(
            operand->shape().dimensions().size());
        absl::c_iota(broadcast_dimensions, 1);
        TF_RETURN_IF_ERROR(expanded->ReplaceOperandWithDifferentShape(
            i,
            MakeBroadcastHlo(convert, broadcast_dimensions,
                             ShapeUtil::PrependMajorDimension(
                                 config.split_k,
                                 ShapeUtil::ChangeElementType(
                                     operand->shape(), accumulator_dtype)))));
      }
    }
  }

  if (did_pad) {
    // Check if the analysis can work on the transformed HLO.
    // We can fail gracefully here, but not in IrEmitterTriton.
    // For the case without padding, we already checked this in
    // MakeSplitKOperand with the divisibility check.
    TF_RETURN_IF_ERROR(
        TritonFusionAnalysis::Execute(*computation, config.split_k).status());
  }

  return absl::OkStatus();
}

absl::Status MakeDotSplitKBatch(HloInstruction* dot_fusion,
                                const TritonGemmConfig& config) {
  CHECK_EQ(dot_fusion->opcode(), HloOpcode::kFusion);

  if (dot_fusion->shape().IsTuple()) {
    return Unimplemented("Tuple output is not supported with split-K yet.");
  }

  const PrimitiveType output_type = dot_fusion->shape().element_type();
  const Layout output_layout = dot_fusion->shape().layout();

  auto status = MakeDotComputationSplitKBatch(
      dot_fusion->fused_instructions_computation(), config);
  if (!status.ok()) {
    TF_RETURN_IF_ERROR(
        HloDCE()
            .RunOnComputation(dot_fusion->fused_instructions_computation())
            .status());
    return status;
  }
  const HloInstruction* root = dot_fusion->fused_expression_root();

  *dot_fusion->mutable_shape() = root->shape();
  HloInstruction* zero =
      dot_fusion->parent()->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(root->shape().element_type())));
  auto initial_dot_fusion_users = dot_fusion->users();
  // The batch dimension to reduce is the first one by construction.
  TF_ASSIGN_OR_RETURN(HloInstruction * reduce,
                      MakeReduceHlo(dot_fusion, zero, /*dimensions=*/{0},
                                    HloOpcode::kAdd, &dot_fusion->metadata()));

  // The output of the reduce has to have the layout of the original dot.
  *reduce->mutable_shape()->mutable_layout() = output_layout;

  HloInstruction* convert = MakeConvertToHlo(reduce, output_type);

  if (dot_fusion->IsRoot()) {
    dot_fusion->parent()->set_root_instruction(convert,
                                               /*accept_different_shape=*/true);
  } else {
    // Replace all users expect for convert created above to avoid cycles.
    TF_RETURN_IF_ERROR(dot_fusion->ReplaceAllUsesWithDifferentShape(
        initial_dot_fusion_users, convert));
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
