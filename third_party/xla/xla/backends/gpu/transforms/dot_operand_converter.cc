/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/dot_operand_converter.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace xla::gpu {
namespace {

bool IsIntegralExactInBf16(PrimitiveType type) {
  return primitive_util::IsIntegralType(type) &&
         primitive_util::CastPreservesValues(type, BF16);
}

bool HasDefaultPrecision(const HloInstruction* dot) {
  const PrecisionConfig& precision_config = dot->precision_config();
  return precision_config.algorithm() == PrecisionConfig::ALG_UNSET &&
         absl::c_all_of(precision_config.operand_precision(),
                        [](int p) { return p == PrecisionConfig::DEFAULT; });
}

// Returns the target element type to which the operands of `dot` should be
// converted. Every <=8-bit integer value (S1, U1, S2, U2, S4, U4, S8, U8) is
// exactly representable in BF16. When a default-precision dot on a BF16-capable
// GPU with TF32 enabled multiplies an <=8-bit integer operand by an F32 operand
// with an F32 or BF16 output, converting both operands to BF16 avoids falling
// back to a TF32 dot while preserving the exact integer values.
PrimitiveType DesiredOperandType(const HloInstruction* dot,
                                 const se::GpuComputeCapability& gpu_version) {
  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  PrimitiveType lhs_type = lhs->shape().element_type();
  PrimitiveType rhs_type = rhs->shape().element_type();
  PrimitiveType out_type = dot->shape().element_type();
  if ((out_type == F32 || out_type == BF16) && HasDefaultPrecision(dot) &&
      tsl::tensor_float_32_execution_enabled() &&
      DotOperandConverter::SupportsBf16Dot(gpu_version) &&
      ((IsIntegralExactInBf16(lhs_type) && rhs_type == F32) ||
       (lhs_type == F32 && IsIntegralExactInBf16(rhs_type)))) {
    return BF16;
  }
  return ShapeUtil::HigherPrecisionElementType(lhs->shape(), rhs->shape());
}

}  // namespace

bool DotOperandConverter::SupportsBf16Dot(
    const se::GpuComputeCapability& gpu_version) {
  if (const auto* cuda_cc = gpu_version.cuda_compute_capability()) {
    return cuda_cc->IsAtLeastAmpere();
  }
  if (const auto* rocm_cc = gpu_version.rocm_compute_capability()) {
    return rocm_cc->has_hipblaslt() && rocm_cc->gfx9_mi300_series();
  }
  return false;
}

bool DotOperandConverter::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (HloPredicateIsNotOp<HloOpcode::kDot>(instruction)) {
    return false;
  }
  HloInstruction* lhs = instruction->mutable_operand(0);
  HloInstruction* rhs = instruction->mutable_operand(1);

  PrimitiveType lhs_type = lhs->shape().element_type();
  PrimitiveType rhs_type = rhs->shape().element_type();

  if (lhs_type == rhs_type) {
    return false;
  }

  // Exclude conversions between FP8 types.
  absl::flat_hash_set<PrimitiveType> non_converting = {F8E4M3FN, F8E5M2,
                                                       F8E4M3FNUZ, F8E5M2FNUZ};
  return !(non_converting.contains(lhs_type) &&
           non_converting.contains(rhs_type));
}

absl::StatusOr<HloInstruction*> DotOperandConverter::ExpandInstruction(
    HloInstruction* instruction) {
  PrimitiveType desired_type = DesiredOperandType(instruction, gpu_version_);
  for (int i : {0, 1}) {
    HloInstruction* operand = instruction->mutable_operand(i);
    if (operand->shape().element_type() == desired_type) {
      continue;
    }
    Shape new_shape =
        ShapeUtil::ChangeElementType(operand->shape(), desired_type);
    HloInstruction* convert_inst = instruction->AddInstruction(
        HloInstruction::CreateConvert(new_shape, operand));
    ABSL_RETURN_IF_ERROR(
        instruction->ReplaceOperandWithDifferentShape(i, convert_inst));
  }
  return nullptr;
}

}  // namespace xla::gpu
