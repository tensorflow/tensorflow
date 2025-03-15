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

#include <cmath>
#include <cstdint>
#include <iterator>
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
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_tiling_propagation.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

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
    const tsl::protobuf::RepeatedField<int64_t>& source,
    tsl::protobuf::RepeatedField<int64_t>& destination, const int threshold) {
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

absl::StatusOr<HloInstruction*> MakeSparseMetaOperand(
    HloDotInstruction& dot, const TritonGemmConfig& config) {
  CHECK_EQ(dot.sparse_operands(), 1);
  CHECK_EQ(dot.sparsity().front().index(), 0);

  HloInstruction* meta = dot.mutable_operand(2);
  const Shape& shape = meta->shape();
  if (shape.dimensions().back() % config.split_k != 0) {
    return UncompilableMatmul("Sparsity metadata has incorrect shape.");
  }

  std::vector<int64_t> dimensions(shape.dimensions().begin(),
                                  shape.dimensions().end() - 1);
  dimensions.push_back(config.split_k);
  dimensions.push_back(shape.dimensions().back() / config.split_k);
  Shape new_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      shape.element_type(), dimensions);
  return MakeBitcastHlo(meta, new_shape);
}

}  // namespace

absl::StatusOr<HloInstruction*> MakeSplitKOperand(
    HloInstruction& dot, const TritonFusionAnalysis& analysis,
    const TritonGemmConfig& config, const int64_t contracting_dim_idx,
    const int operand_number) {
  HloInstruction* operand = dot.mutable_operand(operand_number);
  const int64_t k = operand->shape().dimensions(contracting_dim_idx);
  const bool need_padding = k % config.split_k != 0;

  TritonFusionAnalysis::Scope scope = (operand_number == 0)
                                          ? TritonFusionAnalysis::Scope::LHS
                                          : TritonFusionAnalysis::Scope::RHS;
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
    if (config.split_k > ceil(1.0 * fragment.count / config.block_k)) {
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

    PaddingConfig padding_config = MakeNoPaddingConfig(operand->shape().rank());
    padding_config.mutable_dimensions(contracting_dim_idx)
        ->set_edge_padding_high(config.split_k - k % config.split_k);

    TF_ASSIGN_OR_RETURN(HloInstruction * pad,
                        MakePadHlo(operand, zero, padding_config));
    *pad->mutable_shape()->mutable_layout() = operand->shape().layout();
    operand = pad;
  }
  CHECK_GE(operand->shape().dimensions(contracting_dim_idx), config.split_k);

  // Add bitcast.
  const Shape& shape = operand->shape();
  Shape new_shape(shape.element_type(), {}, {});

  for (int i = 0; i < shape.rank(); ++i) {
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
absl::Status MakeDotComputationSplitKBatch(
    HloComputation* computation, const TritonGemmConfig& config,
    bool disable_reduced_precision_reduction) {
  HloDotInstruction* dot = Cast<HloDotInstruction>(
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot));
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

  // Make sure we have a supported sparse dot.
  if (dot->sparse_operands()) {
    if (dot->sparsity().size() != 1 || dot->sparsity().front().index() != 0) {
      return UncompilableMatmul("Sparsity is only supported on left operand.");
    }
  }

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

  // Process the collected HLOs from computation root to dot.
  bool did_pad = false;
  while (!to_process.empty()) {
    HloInstruction* current = to_process.top();
    to_process.pop();
    // Add split-K dimension to `current`.
    HloInstruction* expanded;
    if (current == dot) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * lhs,
          MakeSplitKOperand(*dot, analysis, config, lhs_contracting_idx, 0));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * rhs,
          MakeSplitKOperand(*dot, analysis, config, rhs_contracting_idx, 1));
      if (lhs->operand(0)->opcode() == HloOpcode::kPad) {
        CHECK_EQ(rhs->operand(0)->opcode(), HloOpcode::kPad);
        did_pad = true;
      }
      std::vector<SparsityDescriptor> sparsity(dot->sparsity().begin(),
                                               dot->sparsity().end());
      std::vector<HloInstruction*> sparse_meta(sparsity.size());
      for (int i = 0; i < sparsity.size(); ++i) {
        // This is only correct for LHS sparse operand after dot decomposition.
        sparsity[i].set_dimension(sparsity[i].dimension() + 1);
        TF_ASSIGN_OR_RETURN(sparse_meta[i],
                            MakeSparseMetaOperand(*dot, config));
      }
      expanded = MakeDotHlo(lhs, rhs, new_dim_numbers, dot->precision_config(),
                            dot->shape().element_type(), sparsity, sparse_meta)
                     .value();
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
      expanded = computation->AddInstruction(current->CloneWithNewShape(
          ShapeUtil::PrependMajorDimension(config.split_k, current->shape())));
      if (expanded->opcode() == HloOpcode::kTranspose) {
        const auto* old_transpose = Cast<HloTransposeInstruction>(current);
        auto* new_transpose = Cast<HloTransposeInstruction>(expanded);
        new_transpose->mutable_dimensions()->clear();
        new_transpose->mutable_dimensions()->reserve(
            new_transpose->shape().rank());
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
        std::vector<int64_t> broadcast_dimensions(operand->shape().rank());
        absl::c_iota(broadcast_dimensions, 1);
        TF_RETURN_IF_ERROR(expanded->ReplaceOperandWithDifferentShape(
            i, MakeBroadcastHlo(operand, broadcast_dimensions,
                                ShapeUtil::PrependMajorDimension(
                                    config.split_k, operand->shape()))));
      }
    }
  }

  if (disable_reduced_precision_reduction) {
    PrimitiveType output_type =
        computation->root_instruction()->shape().element_type();
    PrimitiveType accumulator_type = output_type == PrimitiveType::F64
                                         ? PrimitiveType::F64
                                         : PrimitiveType::F32;

    computation->root_instruction()->mutable_shape()->set_element_type(
        accumulator_type);
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

  const bool disable_reduced_precision_reduction =
      dot_fusion->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_triton_gemm_disable_reduced_precision_reduction();
  const PrimitiveType output_type = dot_fusion->shape().element_type();
  const Layout output_layout = dot_fusion->shape().layout();

  TF_RETURN_IF_ERROR(MakeDotComputationSplitKBatch(
      dot_fusion->fused_instructions_computation(), config,
      disable_reduced_precision_reduction));
  const HloInstruction* root = dot_fusion->fused_expression_root();

  *dot_fusion->mutable_shape() = root->shape();
  HloInstruction* zero =
      dot_fusion->parent()->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(root->shape().element_type())));
  // The batch dimension to reduce is the first one by construction.
  TF_ASSIGN_OR_RETURN(HloInstruction * reduce,
                      MakeReduceHlo(dot_fusion, zero, /*dimensions=*/{0},
                                    HloOpcode::kAdd, &dot_fusion->metadata()));

  // The output of the reduce has to have the layout of the original dot.
  *reduce->mutable_shape()->mutable_layout() = output_layout;

  if (dot_fusion->IsRoot()) {
    dot_fusion->parent()->set_root_instruction(reduce,
                                               /*accept_different_shape=*/true);
  } else {
    TF_RETURN_IF_ERROR(dot_fusion->ReplaceAllUsesWithDifferentShape(reduce));
  }

  if (disable_reduced_precision_reduction) {
    HloInstruction* convert = MakeConvertToHlo(reduce, output_type);
    if (reduce->IsRoot()) {
      reduce->parent()->set_root_instruction(convert,
                                             /*accept_different_shape=*/true);
    } else {
      TF_RETURN_IF_ERROR(reduce->ReplaceAllUsesWithDifferentShape(convert));
    }
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
