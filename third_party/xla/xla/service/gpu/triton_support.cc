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

#include "xla/service/gpu/triton_support.h"

#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
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

std::vector<HloOpcode> TritonSupportedUnaryElementwise(
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

std::vector<HloOpcode> TritonSupportedBinaryElementwise(
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
      element_type == PrimitiveType::F64) {
    ret.push_back(HloOpcode::kAtan2);
    ret.push_back(HloOpcode::kDivide);
    ret.push_back(HloOpcode::kPower);
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedTernaryElementwise(
    PrimitiveType element_type) {
  return {HloOpcode::kSelect, HloOpcode::kClamp};
}

bool IsTritonSupportedElementwise(HloOpcode opcode,
                                  PrimitiveType element_type) {
  return absl::c_linear_search(TritonSupportedUnaryElementwise(element_type),
                               opcode) ||
         absl::c_linear_search(TritonSupportedBinaryElementwise(element_type),
                               opcode) ||
         absl::c_linear_search(TritonSupportedTernaryElementwise(element_type),
                               opcode);
}

CodegenDecision CanTritonHandleElementwise(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version) {
  if (!IsTritonSupportedDataType(instr.shape().element_type(), gpu_version)) {
    return "Unsupported output data type.";
  }

  for (const HloInstruction* operand : instr.operands()) {
    if (!IsTritonSupportedDataType(operand->shape().element_type(),
                                   gpu_version)) {
      return "Unsupported input data type.";
    }
  }

  if (instr.opcode() == HloOpcode::kConstant) {
    return CodegenDecision{};
  } else if (!IsTritonSupportedElementwise(
                 instr.opcode(), instr.operand(0)->shape().element_type())) {
    return "Unsupported elementwise operation.";
  }
  return CodegenDecision{};
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
      if (cuda_compute_capability) {
        return true;
      }
      return false;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
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
    // TODO(b/311331155): Triton F32 is about 3x slower than Triton TF32 and is
    // slow to compile. Disable it for now.
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    default:
      return false;
  }
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
      return "Having non-default operand precisions or TensorFloat-32 disabled "
             "for Dot op with unset algorithm.";
    }
  } else {
    if (!IsDotAlgorithmSupportedByTriton(dot.precision_config().algorithm(),
                                         gpu_version)) {
      return "Unsupported algorithm on the current device(s).";
    }
  }

  // TODO(b/266862493): Support more output types.
  if (!IsTritonSupportedDotOutputType(dot.shape().element_type(),
                                      gpu_version)) {
    return "Unsupported output data type for Dot op.";
  }

  if (!IsTritonSupportedDataType(dot.operand(0)->shape().element_type(),
                                 gpu_version) ||
      !IsTritonSupportedDataType(dot.operand(1)->shape().element_type(),
                                 gpu_version)) {
    return "Unsupported input data type for Dot op.";
  }

  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  // TODO(b/269580541): support multiple batch dimensions.
  if (dim_numbers.lhs_batch_dimensions().size() > 1) {
    return "Multiple batch dimensions.";
  }

  return CodegenDecision{};
}

// Filters Reduces which can be handled using Triton.
CodegenDecision CanTritonHandleReduce(
    const HloReduceInstruction& reduce,
    const se::GpuComputeCapability& gpu_version) {
  if (!IsTritonSupportedDataType(reduce.shape().element_type(), gpu_version)) {
    return "Unsupported output data type for Reduce op.";
  }

  for (const HloInstruction* operand : reduce.operands()) {
    if (!IsTritonSupportedDataType(operand->shape().element_type(),
                                   gpu_version)) {
      return "Unsupported input data type for Reduce op.";
    }
  }

  bool is_triton_supported_reduction_computation = [&]() {
    return absl::c_all_of(
        reduce.to_apply()->instructions(), [&](const HloInstruction* instr) {
          return IsTritonSupportedInstruction(*instr, gpu_version);
        });
  }();
  if (!is_triton_supported_reduction_computation) {
    return "Unsupported reduction computation by Triton.";
  }

  if (reduce.dimensions().size() == 1 &&
      reduce.dimensions().front() == reduce.operand(0)->shape().rank() - 1 &&
      reduce.operand_count() == 2) {
    const HloInstruction* operand = reduce.operand(1);
    // We assume that the reduction init value was input as a constant, or in
    // the case of a data type affected by float normalization, a convert of a
    // constant.
    if (operand->opcode() == HloOpcode::kConvert) {
      if (operand->operand(0)->opcode() == HloOpcode::kConstant &&
          operand->operand(0)->shape().element_type() == BF16 &&
          operand->shape().element_type() == F32) {
        return CodegenDecision{};
      }
    } else if (operand->opcode() == HloOpcode::kConstant) {
      return CodegenDecision{};
    }
    return "Reduction init value should be a constant or a convert of a "
           "constant.";
  }
  return "Reduction is not a row-reduction of a single operand.";
}

bool NoNonContractingDimension(const HloDotInstruction& dot) {
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();
  if (dim_numbers.lhs_batch_dimensions().size() +
              dim_numbers.lhs_contracting_dimensions().size() ==
          dot.operand(0)->shape().rank() ||
      dim_numbers.rhs_batch_dimensions().size() +
              dim_numbers.rhs_contracting_dimensions().size() ==
          dot.operand(1)->shape().rank()) {
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
        return CodegenDecision(
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
      return CodegenDecision(
          "Unsupported dynamic slice on non-major-most dimension.");
    }
  }

  // TODO(b/343143854): Check the subtleties of which dynamic slices are
  // supported, for example that a fragmented dimension cannot be sliced.
  return CodegenDecision{};
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
        return "No non-contracting dimensions.";
      }
      return CanTritonHandleGEMM(*dot, gpu_version);
    }
    case HloOpcode::kReduce: {
      return CanTritonHandleReduce(*Cast<HloReduceInstruction>(&instr),
                                   gpu_version);
    }
    case HloOpcode::kTuple: {
      if (instr.IsRoot()) {
        return CodegenDecision{};
      }
      return "Only supports root tuples.";
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
      return CodegenDecision{};
    default:
      break;
  }
  return "Unsupported opcode.";
}

}  // namespace legacy_triton

CodegenDecision IsTritonSupportedInstruction(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version) {
  bool output_type_is_supported = legacy_triton::IsTritonSupportedDataType(
      instr.shape().element_type(), gpu_version);

  if (!output_type_is_supported) {
    return "Unsupported output data type.";
  }

  bool input_types_are_supported =
      absl::c_all_of(instr.operands(), [&](const HloInstruction* operand) {
        return legacy_triton::IsTritonSupportedDataType(
            operand->shape().element_type(), gpu_version);
      });

  if (!input_types_are_supported) {
    return "Unsupported input data type.";
  }

  if (instr.IsElementwise()) {
    return legacy_triton::CanTritonHandleElementwise(instr, gpu_version);
  }

  // TODO(bchetioui): support kDot, kPad, and kDynamicSlice.
  switch (instr.opcode()) {
    case HloOpcode::kReduce: {
      return legacy_triton::CanTritonHandleReduce(
          *Cast<HloReduceInstruction>(&instr), gpu_version);
    }
    case HloOpcode::kTranspose:
    case HloOpcode::kSlice:
    case HloOpcode::kParameter:
    case HloOpcode::kBroadcast:
      return CodegenDecision{};
    default:
      VLOG(1) << "Unsupported instruction: " << instr.ToString();
      break;
  }
  return "Unsupported opcode.";
}

}  // namespace gpu
}  // namespace xla
