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

#include "xla/backends/gpu/codegen/tensor_ir/support.h"

#include "absl/strings/str_cat.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::tensor_ir {
namespace {

bool IsSupportedPrimitiveType(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::PRED:
    case PrimitiveType::S8:
    case PrimitiveType::S16:
    case PrimitiveType::S32:
    case PrimitiveType::S64:
    case PrimitiveType::U8:
    case PrimitiveType::U16:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
    case PrimitiveType::F16:
    case PrimitiveType::BF16:
    case PrimitiveType::F32:
    case PrimitiveType::F64: {
      return true;
    }

    default: {
      return false;
    }
  }
}

bool IsSupportedFusionOpcode(HloOpcode opcode) {
  switch (opcode) {
    // Unary elementwise operations.
    case HloOpcode::kAbs:
    case HloOpcode::kCeil:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:

    // Binary elementwise operations.
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kCompare:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
    case HloOpcode::kClamp:

    // Layout modification operations.
    case HloOpcode::kReshape:
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:

    // Reduction operations.
    case HloOpcode::kDot:
    case HloOpcode::kReduce:

    // Miscellaneous operations.
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kIota:
    case HloOpcode::kConcatenate: {
      return true;
    }

    default: {
      return false;
    }
  }
}

bool IsSupportedReductionOpcode(HloOpcode opcode) {
  switch (opcode) {
    // Binary elementwise operations.
    case HloOpcode::kAdd:
    case HloOpcode::kCompare:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
    case HloOpcode::kClamp:

    // Miscellaneous operations.
    case HloOpcode::kParameter:
    case HloOpcode::kConstant: {
      return true;
    }

    default: {
      return false;
    }
  }
}

CodegenDecision CheckCompareInstruction(const HloInstruction& instr) {
  auto compare = Cast<HloCompareInstruction>(&instr);
  if (compare->order() == ComparisonOrder::kTotal &&
      primitive_util::IsFloatingPointType(
          compare->operand(0)->shape().element_type())) {
    return CodegenDecision::Forbid(
        "Total order comparison is not supported for floating-point types");
  }
  return CodegenDecision::Allow();
}

CodegenDecision IsSupportedReductionInstruction(const HloInstruction& instr) {
  if (!instr.shape().IsArray()) {
    return CodegenDecision::Forbid(absl::StrCat("Unsupported non-array shape: ",
                                                instr.shape().ToString()));
  }
  if (!IsSupportedPrimitiveType(instr.shape().element_type())) {
    return CodegenDecision::Forbid(
        absl::StrCat("Unsupported element type: ",
                     primitive_util::LowercasePrimitiveTypeName(
                         instr.shape().element_type())));
  }
  if (!IsSupportedReductionOpcode(instr.opcode())) {
    return CodegenDecision::Forbid(
        absl::StrCat("Unsupported reduction instruction: ",
                     HloOpcodeString(instr.opcode())));
  }
  if (instr.opcode() == HloOpcode::kCompare) {
    auto decision = CheckCompareInstruction(instr);
    if (!decision.IsAllowed()) {
      return decision;
    }
  }
  return CodegenDecision::Allow();
}

}  // namespace

CodegenDecision IsSupportedFusionComputation(const HloComputation& comp) {
  for (const HloInstruction* instruction : comp.instructions()) {
    auto decision = IsInstructionSupportedForFusion(*instruction);
    if (!decision.IsAllowed()) {
      return decision;
    }
  }
  return CodegenDecision::Allow();
}

CodegenDecision IsInstructionSupportedForFusion(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    return IsSupportedFusionComputation(
        *instr.fused_instructions_computation());
  }

  if (!instr.shape().IsArray()) {
    return CodegenDecision::Forbid(absl::StrCat("Unsupported non-array shape: ",
                                                instr.shape().ToString()));
  }

  if (!IsSupportedPrimitiveType(instr.shape().element_type())) {
    return CodegenDecision::Forbid(
        absl::StrCat("Unsupported element type: ",
                     primitive_util::LowercasePrimitiveTypeName(
                         instr.shape().element_type())));
  }
  if (!IsSupportedFusionOpcode(instr.opcode())) {
    return CodegenDecision::Forbid(absl::StrCat(
        "Unsupported instruction: ", HloOpcodeString(instr.opcode())));
  }

  switch (instr.opcode()) {
    case HloOpcode::kReduce: {
      auto reduce = Cast<HloReduceInstruction>(&instr);
      if (reduce->input_count() != 1) {
        return CodegenDecision::Forbid(
            absl::StrCat("Unsupported variadic reduction: ", instr.name()));
      }
      for (const HloInstruction* init_value : reduce->init_values()) {
        if (DynCast<HloConstantInstruction>(init_value) == nullptr) {
          return CodegenDecision::Forbid(absl::StrCat(
              "Unsupported reduction initial value: ", init_value->name()));
        }
      }
      for (const HloInstruction* inner : instr.to_apply()->instructions()) {
        auto decision = IsSupportedReductionInstruction(*inner);
        if (!decision.IsAllowed()) {
          return decision;
        }
      }
      break;
    }

    case HloOpcode::kConstant: {
      // Only scalar constants are supported.
      if (!instr.shape().dimensions().empty()) {
        return CodegenDecision::Forbid(
            absl::StrCat("Unsupported non-scalar constant: ", instr.name()));
      }
      break;
    }

    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kNot: {
      // Verify that logical operations are used on boolean types.
      if (instr.shape().element_type() != PrimitiveType::PRED) {
        return CodegenDecision::Forbid(absl::StrCat(
            "Unsupported type for logical operation: ", instr.name()));
      }
      break;
    }

    case HloOpcode::kCompare: {
      auto decision = CheckCompareInstruction(instr);
      if (!decision.IsAllowed()) {
        return decision;
      }
      break;
    }

    default: {
      break;
    }
  }
  return CodegenDecision::Allow();
}

}  // namespace xla::gpu::tensor_ir
