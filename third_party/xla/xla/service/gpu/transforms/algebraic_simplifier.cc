/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/algebraic_simplifier.h"

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/codegen/triton/support_legacy.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace m = ::xla::match;

absl::StatusOr<bool>
GpuAlgebraicSimplifierVisitor::TryToSinkBroadcastOperandsOfChainedAdds(
    HloInstruction* add) {
  if (!options_.enable_sink_broadcast()) {
    return false;
  }

  HloInstruction *conv, *constant_0, *broadcast_0, *add_0, *constant_1,
      *broadcast_1;
  if (!Match(add, m::AddAnyOrder(
                      m::AddAnyOrder(
                          &add_0, m::Convolution(&conv, m::Op(), m::Op()),
                          m::Broadcast(&broadcast_0, m::Constant(&constant_0))),
                      m::Broadcast(&broadcast_1, m::Constant(&constant_1))))) {
    return false;
  }

  // Skip when the broadcast shapes and dimensions don't match.
  if (!ShapeUtil::Equal(constant_0->shape(), constant_1->shape()) ||
      broadcast_0->dimensions() != broadcast_1->dimensions()) {
    return false;
  }

  HloInstruction* new_constant_add =
      add->AddInstruction(HloInstruction::CreateBinary(
          constant_0->shape(), HloOpcode::kAdd, constant_0, constant_1));
  HloInstruction* new_bcast =
      add->AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_0->shape(), new_constant_add, broadcast_0->dimensions()));
  TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
      add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd,
                                        new_bcast, conv)));
  return true;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  TF_ASSIGN_OR_RETURN(bool replaced,
                      TryToSinkBroadcastOperandsOfChainedAdds(add));
  if (replaced) {
    return absl::OkStatus();
  }

  return AlgebraicSimplifierVisitor::HandleAdd(add);
}

bool GpuAlgebraicSimplifierVisitor::SupportedDotPrecisionConfig(
    const PrecisionConfig& config, bool has_contracting_dim) {
  if (!has_contracting_dim) {
    return config.algorithm() == PrecisionConfig::ALG_UNSET ||
           config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
  }
  return config.algorithm() == PrecisionConfig::ALG_UNSET ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
}

absl::StatusOr<HloInstruction*>
GpuAlgebraicSimplifierVisitor::MakeMultiplyForPrecisionAlgorithm(
    HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs) {
  const auto algorithm = dot->precision_config().algorithm();
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      return DotAlgorithmRewriter::MakeMultiplyForBF16BF16F32(lhs, rhs);
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
      return DotAlgorithmRewriter::MakeMultiplyForBF16BF16F32X3(lhs, rhs);
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
      return DotAlgorithmRewriter::MakeMultiplyForBF16BF16F32X6(lhs, rhs);
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      return DotAlgorithmRewriter::MakeMultiplyForBF16BF16F32X9(lhs, rhs);
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      return DotAlgorithmRewriter::MakeMultiplyForTF32TF32F32(lhs, rhs);
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      return DotAlgorithmRewriter::MakeMultiplyForTF32TF32F32X3(lhs, rhs);
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      return MakeBinaryHlo(HloOpcode::kMultiply, lhs, rhs);
    case PrecisionConfig::ALG_UNSET:
      return MakeBinaryHlo(HloOpcode::kMultiply, lhs, rhs);
    default:
      CHECK(false) << "Unsupported dot precision algorithm: " << algorithm;
  }
}

bool GpuAlgebraicSimplifierVisitor::ShouldStrengthReduceDotToReduce(
    const HloInstruction* hlo) {
  if (!options_.enable_dot_strength_reduction()) {
    return false;
  }

  const HloDotInstruction* dot = DynCast<HloDotInstruction>(hlo);
  if (dot == nullptr) {
    return false;
  }

  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  DotDimensionNumbers dnums = dot->dot_dimension_numbers();
  bool lhs_is_vector = (dnums.lhs_batch_dimensions_size() +
                            dnums.lhs_contracting_dimensions_size() ==
                        lhs->shape().dimensions_size());
  bool rhs_is_vector = (dnums.rhs_batch_dimensions_size() +
                            dnums.rhs_contracting_dimensions_size() ==
                        rhs->shape().dimensions_size());
  // Strength-reduce vector-vector dots since they are not supported by
  // GemmFusion.
  if (lhs_is_vector && rhs_is_vector) {
    return true;
  }

  absl::StatusOr<bool> is_too_small =
      IsMatrixMultiplicationTooSmallForRewriting(*hlo, /*threshold=*/10000000);
  CHECK_OK(is_too_small.status());
  if (is_too_small.value()) {
    return true;
  }

  // If GemmFusion cannot handle this dot, we should strength-reduce it so that
  // it can be handled by the fusion pipeline.
  return !legacy_triton::CanTritonHandleGEMM(*dot, compute_capability_);
}

}  // namespace xla::gpu
