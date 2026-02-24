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

#include "xla/backends/gpu/transforms/dot_strength_reduction.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/triton/support_legacy.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

HloInstruction* ConvertTo(HloInstruction* instruction, PrimitiveType type,
                          const OpMetadata* metadata) {
  if (instruction->shape().element_type() == type) {
    return instruction;
  }
  Shape new_shape = instruction->shape();
  new_shape.set_element_type(type);
  return instruction->parent()->AddInstruction(
      HloInstruction::CreateConvert(new_shape, instruction), metadata);
}

// Transposes the dot operand to make dimensions in "batch, non-contracting,
// contracting" order, sorted by index within each category.
HloInstruction* PermuteDotOperandDimensions(HloInstruction* operand,
                                            DotOperandDims* dims,
                                            const OpMetadata* metadata) {
  std::vector<int64_t> permutation;
  for (auto kind : {DotOperandDims::kBatch, DotOperandDims::kNonContracting,
                    DotOperandDims::kContracting}) {
    for (auto index : dims->DimensionIndices(kind)) {
      permutation.push_back(index);
    }
  }
  if (absl::c_is_sorted(permutation)) {
    return operand;
  }
  Shape new_shape = ShapeUtil::PermuteDimensions(permutation, operand->shape());
  operand = operand->parent()->AddInstruction(
      HloInstruction::CreateTranspose(new_shape, operand, permutation),
      metadata);
  dims->Permute(permutation);
  return operand;
}

HloInstruction* BroadcastDimensions(HloInstruction* operand,
                                    int64_t insert_before,
                                    absl::Span<const int64_t> bounds_to_insert,
                                    const OpMetadata* metadata) {
  if (bounds_to_insert.empty()) {
    return operand;
  }

  std::vector<int64_t> broadcast_dimensions(
      operand->shape().dimensions().size());
  std::iota(broadcast_dimensions.begin(),
            broadcast_dimensions.begin() + insert_before, 0);
  std::iota(broadcast_dimensions.begin() + insert_before,
            broadcast_dimensions.end(),
            insert_before + bounds_to_insert.size());

  Shape new_shape = ShapeUtil::InsertDimensionsAtIndex(
      operand->shape(), insert_before, bounds_to_insert);
  return operand->parent()->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape, operand, broadcast_dimensions),
      metadata);
}

HloComputation* CreateScalarAddComputation(HloModule* module,
                                           PrimitiveType type) {
  HloComputation::Builder b("scalar_add_computation");
  Shape shape = ShapeUtil::MakeShape(type, {});
  auto scalar_lhs =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
  auto scalar_rhs =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
  auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
  return module->AddEmbeddedComputation(b.Build(scalar_op));
}

// Reduces the last `num_dims_to_reduce` dimensions of `instruction`.
HloInstruction* ReduceDimensions(HloInstruction* instruction,
                                 size_t num_dims_to_reduce,
                                 PrimitiveType accumulator_type,
                                 const OpMetadata* metadata) {
  if (num_dims_to_reduce == 0) {
    return instruction;
  }
  HloComputation* computation = instruction->parent();

  std::vector<int64_t> reduce_dims(num_dims_to_reduce);
  absl::c_iota(reduce_dims,
               instruction->shape().dimensions().size() - num_dims_to_reduce);

  Shape reduce_shape =
      ShapeUtil::DeleteDimensions(reduce_dims, instruction->shape());
  reduce_shape.set_element_type(accumulator_type);

  Literal zero_literal = LiteralUtil::Zero(accumulator_type);
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(zero_literal)));

  HloComputation* add_computation =
      CreateScalarAddComputation(computation->parent(), accumulator_type);

  return computation->AddInstruction(
      HloInstruction::CreateReduce(
          reduce_shape, ConvertTo(instruction, accumulator_type, metadata),
          zero, reduce_dims, add_computation),
      metadata);
}

}  // namespace

absl::StatusOr<HloInstruction*> DotStrengthReduction::ExpandInstruction(
    HloInstruction* instruction) {
  HloDotInstruction* dot = Cast<HloDotInstruction>(instruction);
  const OpMetadata* metadata = &dot->metadata();
  TF_ASSIGN_OR_RETURN(auto dot_dims, DotOperandDims::FromDot(dot));

  std::array<HloInstruction*, 2> operands = {dot->mutable_operand(0),
                                             dot->mutable_operand(1)};
  for (int i = 0; i < 2; ++i) {
    DotOperandDims& our_dims = dot_dims[i];
    DotOperandDims& other_dims = dot_dims[1 - i];
    // Convert operands to the dot resulting type.
    operands[i] = ConvertTo(operands[i], dot->shape().element_type(), metadata);
    // Ensure dimensions are in "batch, non-contracting, contracting" order.
    operands[i] = PermuteDotOperandDimensions(operands[i], &our_dims, metadata);

    // Both lhs and rhs will have [batch, lhs non-contracting, rhs
    // non-contracting, contracting] dimensions.
    // Therefore, we insert other side's non-contracting dimensions before or
    // after our contracting depending on the operand.
    int insert_before = our_dims.DimensionCount(DotOperandDims::kBatch);
    if (i == 0) {
      insert_before += our_dims.DimensionCount(DotOperandDims::kNonContracting);
    }

    operands[i] = BroadcastDimensions(
        operands[i], insert_before,
        other_dims.DimensionSizes(DotOperandDims::kNonContracting), metadata);
  }

  // At this point, both operands have the same shape. Elementwise multiply.
  CHECK(operands[0]->shape().dimensions() == operands[1]->shape().dimensions());
  TF_ASSIGN_OR_RETURN(
      HloInstruction * flow,
      MakeMultiplyForDotPrecisionAlgorithm(
          operands[0], operands[1], dot->precision_config().algorithm()));
  flow->set_metadata(*metadata);

  // If there were any contracting dims, we need to reduce them.
  flow = ReduceDimensions(
      flow, dot_dims[0].DimensionCount(DotOperandDims::kContracting),
      GetGemmAccumulatorType(dot), metadata);

  // If the output type is different from what it was before (either because
  // reduction used a different accumulator type, or because types of operand
  // differed the output type for multiply), convert to the output type.
  flow = ConvertTo(flow, dot->shape().element_type(), metadata);
  return flow;
}

bool DotStrengthReduction::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDot) {
    return false;
  }

  const HloDotInstruction* dot = Cast<HloDotInstruction>(instruction);
  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();

  const bool lhs_is_vector = (dnums.lhs_batch_dimensions_size() +
                                  dnums.lhs_contracting_dimensions_size() ==
                              lhs->shape().dimensions().size());
  const bool rhs_is_vector = (dnums.rhs_batch_dimensions_size() +
                                  dnums.rhs_contracting_dimensions_size() ==
                              rhs->shape().dimensions().size());
  // For s32xs32->s32 dots, with RHS contracting dimension 1,
  // the loop emitter is slow and other backends don't support it.
  // Rewriting it as a faster alternative.
  const bool is_favourable_s32_dot = [&]() {
    if (lhs->shape().element_type() != S32 ||
        rhs->shape().element_type() != S32 ||
        dot->shape().element_type() != S32) {
      return false;
    }
    if (dnums.rhs_contracting_dimensions().size() != 1 ||
        dnums.rhs_contracting_dimensions()[0] != 1) {
      return false;
    }
    return true;
  }();
  if (!lhs_is_vector && !rhs_is_vector && !is_favourable_s32_dot) {
    return false;
  }
  // Strength-reduce vector-vector dots since they are not supported by
  // GemmFusion.
  if (lhs_is_vector && rhs_is_vector) {
    return true;
  }

  absl::StatusOr<bool> is_too_small =
      IsMatrixMultiplicationTooSmallForRewriting(*instruction,
                                                 /*threshold=*/10000000);
  CHECK_OK(is_too_small.status());
  if (is_too_small.value()) {
    return true;
  }

  // If GemmFusion cannot handle this dot, we should strength-reduce it so that
  // it can be handled by the fusion pipeline.
  return !legacy_triton::CanTritonHandleGEMM(*dot, compute_capability_);
}

}  // namespace gpu
}  // namespace xla
