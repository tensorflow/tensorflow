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

#include <cstdint>
#include <functional>

#include "absl/log/check.h"
#include "absl/log/log.h"
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
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
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
  return MakeMultiplyForDotPrecisionAlgorithm(
      lhs, rhs, dot->precision_config().algorithm());
}

bool GpuAlgebraicSimplifierVisitor::ShouldStrengthReduceDotToReduce(
    const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kDot) {
    return false;
  }

  if (hlo->operand(0)->shape().dimensions().size() <= 1 ||
      hlo->operand(1)->shape().dimensions().size() <= 1) {
    return true;
  }

  // Here we conservatively assume the operand is extracted from a tuple,
  // it is aliased and likely will require copy to resolve conflicting layout.
  auto may_require_copy = HloPredicateIsOp<HloOpcode::kGetTupleElement>;
  if (hlo->user_count() > 1 && (may_require_copy(hlo->operand(0)) ||
                                may_require_copy(hlo->operand(1)))) {
    VLOG(2) << "Layout Inefficient dot possibly incurring extra copies.\n";
    return true;
  }

  std::function<bool(const HloInstruction*, int)> layout_restrictive =
      [&](const HloInstruction* op, int level = 0) -> bool {
    switch (op->opcode()) {
      case HloOpcode::kCustomCall:
      case HloOpcode::kReshape:
        return true;
      default:
        // Use level to control how far to follow the operand chains to
        // identify layout restrictive ops. Choose to go < 5 recursive calls.
        if (level > 5) {
          return false;
        }
        for (const HloInstruction* operand : op->operands()) {
          if (layout_restrictive(operand, level + 1)) {
            return true;
          }
        }
        return false;
    }
  };

  if (layout_restrictive(hlo, 0)) {
    return true;
  }

  // TODO(appujee): Add support for DotCanonicalizer::SetConvDimNumbersFromDot.
  ConvolutionDimensionNumbers dnums;
  auto dimension_acceptable = [](int64_t dim, const Shape& shape) {
    if (dim >= shape.dimensions().size()) {
      // Added dimension has size 1.
      return false;
    }
    // TODO(appujee): TransferSizeUtil::LaneCount() not available for GPUs.
    return shape.dimensions()[dim] >= 128 / 2;
  };
  const Shape& output_shape = hlo->shape();
  if (!dimension_acceptable(dnums.output_batch_dimension(), output_shape) &&
      !dimension_acceptable(dnums.output_feature_dimension(), output_shape)) {
    VLOG(2) << "Layout inefficient dot whose output shape has small "
               "lane/sublane dimensions\n";
    return true;
  }
  const Shape& input_shape = hlo->operand(0)->shape();
  if (!dimension_acceptable(dnums.input_batch_dimension(), input_shape) &&
      !dimension_acceptable(dnums.input_feature_dimension(), input_shape)) {
    VLOG(2) << "Layout inefficient dot whose input shape has small "
               "lane/sublane dimensions\n";
    return true;
  }
  const Shape& kernel_shape = hlo->operand(1)->shape();
  if (!dimension_acceptable(dnums.kernel_input_feature_dimension(),
                            kernel_shape) &&
      !dimension_acceptable(dnums.kernel_output_feature_dimension(),
                            kernel_shape)) {
    VLOG(2) << "Layout inefficient dot whose kernel shape has small "
               "lane/sublane dimensions\n";
    return true;
  }

  return false;
}

}  // namespace xla::gpu
