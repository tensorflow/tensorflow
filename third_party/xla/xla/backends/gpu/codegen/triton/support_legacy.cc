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

#include "xla/backends/gpu/codegen/triton/support_legacy.h"

#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/variant_visitor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace xla {
namespace gpu {
namespace legacy_triton {

bool IsDistributiveOverAddition(const HloInstruction& hlo) {
  // The list is most likely incomplete.
  // For example division can be added too but only for operand #0.
  if (hlo.opcode() == HloOpcode::kMultiply ||
      hlo.opcode() == HloOpcode::kNegate ||
      hlo.opcode() == HloOpcode::kBitcast ||
      hlo.opcode() == HloOpcode::kReshape || hlo.opcode() == HloOpcode::kCopy ||
      hlo.opcode() == HloOpcode::kTranspose ||
      hlo.opcode() == HloOpcode::kConvert ||
      hlo.opcode() == HloOpcode::kBroadcast ||
      hlo.opcode() == HloOpcode::kSlice) {
    return true;
  }
  return false;
}

// Types that are supported by Triton as dot output.
//
// BF16 is supported in a sense that all operations on it are implemented
// through F32 and converts have to be inserted into the HLO graph, but
// they can be missing during fusion.
bool IsTritonSupportedDotOutputType(
    const PrimitiveType t, const se::GpuComputeCapability& gpu_version) {
  switch (t) {
    case F16:
    case F32:
      return true;
    case F8E5M2:
      return std::visit(VariantVisitor{[](const se::CudaComputeCapability& cc) {
                                         return cc.IsAtLeastAmpere();
                                       },
                                       [](const se::RocmComputeCapability& cc) {
                                         return false;
                                       }},
                        gpu_version);

    case F8E4M3FN:
      return std::visit(VariantVisitor{[](const se::CudaComputeCapability& cc) {
                                         return cc.IsAtLeastHopper();
                                       },
                                       [](const se::RocmComputeCapability& cc) {
                                         return false;
                                       }},
                        gpu_version);
    case BF16:
      return std::visit(VariantVisitor{[](const se::CudaComputeCapability& cc) {
                                         return true;
                                       },
                                       [](const se::RocmComputeCapability& cc) {
                                         return cc.has_bf16_dtype_support();
                                       }},
                        gpu_version);
    case S32:
      return std::visit(VariantVisitor{[](const se::CudaComputeCapability& cc) {
                                         return cc.IsAtLeastAmpere();
                                       },
                                       [](const se::RocmComputeCapability& cc) {
                                         return false;
                                       }},
                        gpu_version);
    default:
      return false;
  }
};

// Data types that are supported by the Triton emitters.
// TODO(b/266862493): Support more data types (F8, F64, etc.).
bool IsTritonSupportedDataType(PrimitiveType type,
                               const se::GpuComputeCapability& gpu_version) {
  if (IsTritonSupportedDotOutputType(type, gpu_version)) {
    return true;
  }
  switch (type) {
    case PRED:
    case S8:
    case S16:
    case S32:
      return true;
    default:
      return false;
  }
}

CodegenDecision IsInstructionSupportsDataTypes(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version) {
  if (!IsTritonSupportedDataType(instr.shape().element_type(), gpu_version)) {
    return CodegenDecision::Forbid("Unsupported output data type.");
  }

  for (const HloInstruction* operand : instr.operands()) {
    const auto operand_type = operand->shape().element_type();
    switch (instr.opcode()) {
      case HloOpcode::kConvert:
        if (operand_type == S4) continue;
        [[fallthrough]];
      default:
        if (!IsTritonSupportedDataType(operand_type, gpu_version)) {
          return CodegenDecision::Forbid("Unsupported input data type.");
        }
    }
  }
  return CodegenDecision::Allow();
}

std::vector<HloOpcode> TritonSupportedUnaryElementwiseUpToFloatNormalization(
    PrimitiveType element_type) {
  std::vector<HloOpcode> ret = {HloOpcode::kConvert};
  if (element_type == PrimitiveType::PRED) {
    ret.push_back(HloOpcode::kNot);
    return ret;
  }
  ret.push_back(HloOpcode::kAbs);
  ret.push_back(HloOpcode::kNegate);
  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F16 ||
      element_type == PrimitiveType::F64) {
    absl::c_copy(std::vector<HloOpcode>{HloOpcode::kCos, HloOpcode::kExp,
                                        HloOpcode::kExpm1, HloOpcode::kFloor,
                                        HloOpcode::kCeil, HloOpcode::kLog,
                                        HloOpcode::kLog1p, HloOpcode::kRsqrt,
                                        HloOpcode::kSin, HloOpcode::kSqrt,
                                        HloOpcode::kCbrt, HloOpcode::kTan,
                                        HloOpcode::kTanh, HloOpcode::kErf},
                 std::back_inserter(ret));
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedBinaryElementwiseUpToFloatNormalization(
    PrimitiveType element_type) {
  if (element_type == PrimitiveType::PRED) {
    return {HloOpcode::kAnd, HloOpcode::kOr, HloOpcode::kXor,
            HloOpcode::kCompare};
  }
  std::vector<HloOpcode> ret = {HloOpcode::kAdd,      HloOpcode::kCompare,
                                HloOpcode::kMaximum,  HloOpcode::kMinimum,
                                HloOpcode::kMultiply, HloOpcode::kSubtract};
  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F16 ||
      element_type == PrimitiveType::F64) {
    ret.push_back(HloOpcode::kAtan2);
    ret.push_back(HloOpcode::kPower);
    if (element_type != PrimitiveType::F16) {
      ret.push_back(HloOpcode::kDivide);
    }
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedTernaryElementwiseUpToFloatNormalization(
    PrimitiveType element_type) {
  return {HloOpcode::kSelect, HloOpcode::kClamp};
}

bool IsTritonSupportedElementwiseUpToFloatNormalization(
    HloOpcode opcode, PrimitiveType element_type) {
  return absl::c_linear_search(
             TritonSupportedUnaryElementwiseUpToFloatNormalization(
                 element_type),
             opcode) ||
         absl::c_linear_search(
             TritonSupportedBinaryElementwiseUpToFloatNormalization(
                 element_type),
             opcode) ||
         absl::c_linear_search(
             TritonSupportedTernaryElementwiseUpToFloatNormalization(
                 element_type),
             opcode);
}

CodegenDecision CanTritonHandleElementwise(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version) {
  if (auto decision = IsInstructionSupportsDataTypes(instr, gpu_version);
      !decision.CanFuse()) {
    return decision;
  }
  if (instr.opcode() == HloOpcode::kConstant) {
    return CodegenDecision::Allow();
  } else if (!IsTritonSupportedElementwiseUpToFloatNormalization(
                 instr.opcode(), instr.operand(0)->shape().element_type())) {
    return CodegenDecision::Forbid("Unsupported elementwise operation.");
  }
  return CodegenDecision::Allow();
}

bool IsDotAlgorithmSupportedByTriton(
    PrecisionConfig::Algorithm algorithm,
    const se::GpuComputeCapability& gpu_version) {
  auto cuda_compute_capability =
      std::get_if<se::CudaComputeCapability>(&gpu_version);
  auto rocm_compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      if (cuda_compute_capability) {
        return true;
      }
      return false;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      if (cuda_compute_capability) {
        return true;
      }
      if (rocm_compute_capability) {
        return rocm_compute_capability->has_bf16_dtype_support();
      }
      return false;

    // TODO(b/326579472): Fix the support of this algorithm and maybe allow it
    // here.
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    default:
      return false;
  }
}

CodegenDecision AreDotInputAndOutputTypesSupportedAndCompatible(
    const HloDotInstruction& dot, const se::GpuComputeCapability& gpu_version) {
  auto output_type = dot.shape().element_type();
  auto lhs_type = dot.operand(0)->shape().element_type();
  auto rhs_type = dot.operand(1)->shape().element_type();

  // TODO(b/266862493): Support more output types.
  if (!IsTritonSupportedDotOutputType(output_type, gpu_version)) {
    return CodegenDecision::Forbid("Unsupported output data type for Dot op.");
  }

  if (!IsTritonSupportedDataType(lhs_type, gpu_version) ||
      !IsTritonSupportedDataType(rhs_type, gpu_version)) {
    return CodegenDecision::Forbid("Unsupported input data type for Dot op.");
  }

  if (output_type == PrimitiveType::S32 &&
      !(primitive_util::Is8BitIntegralType(lhs_type) &&
        primitive_util::Is8BitIntegralType(rhs_type))) {
    return CodegenDecision::Forbid(
        "Currently, S32 output is only supported for 8-bit integral inputs.");
  }

  return CodegenDecision::Allow();
}

// Filters GEMMs which can be handled using Triton.
CodegenDecision CanTritonHandleGEMM(
    const HloDotInstruction& dot, const se::GpuComputeCapability& gpu_version) {
  auto cuda_compute_capability =
      std::get_if<se::CudaComputeCapability>(&gpu_version);
  auto rocm_compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);

  CHECK(cuda_compute_capability || rocm_compute_capability);

  if (dot.precision_config().algorithm() == PrecisionConfig::ALG_UNSET) {
    if (!tsl::tensor_float_32_execution_enabled() ||
        absl::c_any_of(dot.precision_config().operand_precision(),
                       [](int x) { return x != PrecisionConfig::DEFAULT; })) {
      return CodegenDecision::Forbid(
          "Having non-default operand precisions or TensorFloat-32 disabled "
          "for Dot op with unset algorithm.");
    }
  } else {
    if (!IsDotAlgorithmSupportedByTriton(dot.precision_config().algorithm(),
                                         gpu_version)) {
      return CodegenDecision::Forbid(absl::StrFormat(
          "Unsupported algorithm on the current device(s): %s",
          PrecisionConfig::Algorithm_Name(dot.precision_config().algorithm())));
    }
  }

  if (auto decision =
          AreDotInputAndOutputTypesSupportedAndCompatible(dot, gpu_version);
      !decision.CanFuse()) {
    return decision;
  }

  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  // TODO(b/269580541): support multiple batch dimensions.
  if (dim_numbers.lhs_batch_dimensions().size() > 1) {
    return CodegenDecision::Forbid("Multiple batch dimensions.");
  }

  return CodegenDecision::Allow();
}

bool NoNonContractingDimension(const HloDotInstruction& dot) {
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();
  if (dim_numbers.lhs_batch_dimensions().size() +
              dim_numbers.lhs_contracting_dimensions().size() ==
          dot.operand(0)->shape().dimensions_size() ||
      dim_numbers.rhs_batch_dimensions().size() +
              dim_numbers.rhs_contracting_dimensions().size() ==
          dot.operand(1)->shape().dimensions_size()) {
    return true;
  }
  return false;
}

CodegenDecision IsTritonSupportedDynamicSlice(
    const HloDynamicSliceInstruction& instr) {
  for (const HloInstruction* index_operand : instr.index_operands()) {
    switch (index_operand->shape().element_type()) {
      case S8:
      case S16:
      case S32:
        break;  // supported
      default:
        return CodegenDecision::Forbid(
            "Dynamic slice is only supported with S8, S16, or S32 indices.");
    }
  }

  // Similar to normal slice, we cannot slice a non-major-most dimension as
  // that would introduce non-contiguous strides under tiling. The existing
  // check against this in GetRequirementsIfSupportedOrder is not suitable for
  // dynamic slices, so we instead check for this here.
  const HloInstruction* input = instr.operand(0);
  Layout in_layout = input->shape().layout();
  int64_t majormost_dim_id =
      in_layout.minor_to_major(in_layout.minor_to_major_size() - 1);

  for (int i = 0; i < input->shape().dimensions_size(); ++i) {
    if (i == majormost_dim_id) {
      continue;
    } else if (input->shape().dimensions(i) != instr.slice_sizes(i)) {
      return CodegenDecision::Forbid(
          "Unsupported dynamic slice on non-major-most dimension.");
    }
  }

  // TODO(b/343143854): Check the subtleties of which dynamic slices are
  // supported, for example that a fragmented dimension cannot be sliced.
  return CodegenDecision::Allow();
}

CodegenDecision IsTritonSupportedInstruction(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version) {
  if (instr.IsElementwise()) {
    return CanTritonHandleElementwise(instr, gpu_version);
  }

  switch (instr.opcode()) {
    case HloOpcode::kDot: {
      auto* dot = Cast<HloDotInstruction>(&instr);
      // Cases where lhs or rhs have no non-contracting dims are not handled.
      if (NoNonContractingDimension(*dot)) {
        return CodegenDecision::Forbid("No non-contracting dimensions.");
      }
      return CanTritonHandleGEMM(*dot, gpu_version);
    }
    case HloOpcode::kTuple: {
      if (instr.IsRoot()) {
        return CodegenDecision::Allow();
      }
      return CodegenDecision::Forbid("Only supports root tuples.");
    }
    case HloOpcode::kDynamicSlice: {
      return IsTritonSupportedDynamicSlice(
          *Cast<HloDynamicSliceInstruction>(&instr));
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kTranspose:
    case HloOpcode::kSlice:
    case HloOpcode::kReshape:
    case HloOpcode::kPad:
    case HloOpcode::kConcatenate:
    case HloOpcode::kParameter:
    case HloOpcode::kBroadcast:
      return CodegenDecision::Allow();
    default:
      break;
  }
  return CodegenDecision::Forbid("Unsupported opcode.");
}

}  // namespace legacy_triton
}  // namespace gpu
}  // namespace xla
