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

#include "xla/service/gpu/fusions/triton/triton_support.h"

#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
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
      element_type == PrimitiveType::F64) {
    ret.push_back(HloOpcode::kAtan2);
    ret.push_back(HloOpcode::kDivide);
    ret.push_back(HloOpcode::kPower);
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
  // TODO(b/358580281): remove DebugOptions from this function after enabling
  // int4 in Triton GEMM.
  const auto debug_options = instr.GetModule()->config().debug_options();
  LOG(ERROR) << "S4: " << instr.opcode() << " "
             << instr.operand(0)->shape().element_type();
  if (instr.opcode() == HloOpcode::kConvert &&
      instr.operand(0)->shape().element_type() == S4) {
    if (debug_options.xla_gpu_enable_triton_gemm_int4()) {
      LOG(ERROR) << "return supported for S4: " << instr.opcode() << " "
                 << instr.operand(0)->shape().element_type();
      return CodegenDecision{};
    }
    LOG(ERROR) << "return not supported for S4: " << instr.opcode() << " "
               << instr.operand(0)->shape().element_type();
    return "xla_gpu_enable_triton_gemm_int4 is not enabled.";
  }
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
  } else if (!IsTritonSupportedElementwiseUpToFloatNormalization(
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

  if (reduce.dimensions().size() == 1 && reduce.operand_count() == 2) {
    return CodegenDecision{};
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

namespace {

bool IsTritonSupportedDataType(PrimitiveType type,
                               const se::GpuComputeCapability& gpu_version) {
  switch (type) {
    case PRED:
    case S8:
    case S16:
    case S32:
    case S64:
    case F16:
    case F32:
    case F64:
      return true;
    case F8E5M2:
    case F8E4M3FN:
      return std::holds_alternative<se::CudaComputeCapability>(gpu_version);
    case BF16:
      return std::holds_alternative<se::CudaComputeCapability>(gpu_version) ||
             (std::holds_alternative<se::RocmComputeCapability>(gpu_version) &&
              std::get<se::RocmComputeCapability>(gpu_version)
                  .has_bf16_dtype_support());
    default:
      return false;
  }
}

// Set of unary elementwise ops that are genuinely supported by Triton.
// TODO(b/345763510): make sure that this is accurate. At the moment, this is
// mostly a fork of the same code in legacy_triton::.
absl::flat_hash_set<HloOpcode> TritonSupportedUnaryElementwiseOps(
    PrimitiveType element_type) {
  if (element_type == PrimitiveType::PRED) {
    return {HloOpcode::kNot};
  }

  if (element_type == PrimitiveType::U16) {
    return {HloOpcode::kAbs};
  }

  absl::flat_hash_set<HloOpcode> ret{HloOpcode::kAbs, HloOpcode::kConvert};

  if (element_type != PrimitiveType::F8E5M2 &&
      element_type != PrimitiveType::F8E4M3FN) {
    ret.insert(HloOpcode::kNegate);
  }

  if (primitive_util::IsIntegralType(element_type)) {
    ret.insert(HloOpcode::kNot);
  }

  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::F64) {
    absl::flat_hash_set<HloOpcode> additional_opcodes{
        HloOpcode::kCos,   HloOpcode::kExp,   HloOpcode::kExpm1,
        HloOpcode::kFloor, HloOpcode::kCeil,  HloOpcode::kLog,
        HloOpcode::kLog1p, HloOpcode::kRsqrt, HloOpcode::kSin,
        HloOpcode::kSqrt,  HloOpcode::kCbrt,  HloOpcode::kTan,
        HloOpcode::kTanh,  HloOpcode::kErf};
    ret.insert(additional_opcodes.begin(), additional_opcodes.end());
  }

  if (element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F16) {
    absl::flat_hash_set<HloOpcode> additional_opcodes{HloOpcode::kFloor,
                                                      HloOpcode::kCeil};
    ret.insert(additional_opcodes.begin(), additional_opcodes.end());
  }
  return ret;
}

// Set of binary elementwise ops that are genuinely supported by Triton.
// TODO(b/345763510): make sure that this is accurate. At the moment, this is
// mostly a fork of the same code in legacy_triton::.
absl::flat_hash_set<HloOpcode> TritonSupportedBinaryElementwiseOps(
    PrimitiveType element_type, const se::GpuComputeCapability& gpu_version) {
  if (element_type == PrimitiveType::F8E5M2 ||
      element_type == PrimitiveType::F8E4M3FN) {
    return {};
  }

  if (element_type == PrimitiveType::PRED) {
    return {HloOpcode::kAnd,     HloOpcode::kOr,     HloOpcode::kXor,
            HloOpcode::kCompare, HloOpcode::kAdd,    HloOpcode::kMultiply,
            HloOpcode::kMaximum, HloOpcode::kMinimum};
  }

  absl::flat_hash_set<HloOpcode> ret{HloOpcode::kCompare};

  if (element_type != PrimitiveType::U16) {
    ret.insert(HloOpcode::kAdd);
    ret.insert(HloOpcode::kSubtract);
    ret.insert(HloOpcode::kMaximum);
    ret.insert(HloOpcode::kMinimum);
    ret.insert(HloOpcode::kMultiply);

    if (primitive_util::IsIntegralType(element_type)) {
      ret.insert(HloOpcode::kDivide);
      ret.insert(HloOpcode::kAnd);
      ret.insert(HloOpcode::kOr);
      ret.insert(HloOpcode::kXor);
    }
  }

  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::F64) {
    ret.insert(HloOpcode::kAtan2);
    ret.insert(HloOpcode::kDivide);
    ret.insert(HloOpcode::kRemainder);
    ret.insert(HloOpcode::kPower);
  }
  return ret;
}

// Set of ternary elementwise ops that are genuinely supported by Triton.
// TODO(b/345763510): make sure that this is accurate. At the moment, this is
// mostly a fork of the same code in legacy_triton::.
absl::flat_hash_set<HloOpcode> TritonSupportedTernaryElementwiseOps(
    PrimitiveType element_type, const se::GpuComputeCapability& gpu_version) {
  if (element_type == PrimitiveType::U16) {
    return {};
  }

  if (element_type == PrimitiveType::F8E5M2 ||
      element_type == PrimitiveType::F8E4M3FN) {
    return {HloOpcode::kSelect};
  }

  return {HloOpcode::kSelect, HloOpcode::kClamp};
}

// Returns `true` if the given opcode and element type correspond to a n-ary
// elementwise op that is genuinely supported by Triton. The caller is
// responsible for ensuring that the relevant data type is supported on the
// device of interest.
bool IsTritonSupportedElementwise(HloOpcode opcode, PrimitiveType element_type,
                                  const se::GpuComputeCapability& gpu_version) {
  return TritonSupportedUnaryElementwiseOps(element_type).contains(opcode) ||
         TritonSupportedBinaryElementwiseOps(element_type, gpu_version)
             .contains(opcode) ||
         TritonSupportedTernaryElementwiseOps(element_type, gpu_version)
             .contains(opcode);
}

CodegenDecision IsTritonSupportedInstructionImpl(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version) {
  // Special handling for the kCompare instruction, which codegens correctly
  // with a U16 data type despite the fact that it is not supported by Triton
  // itself.
  if (instr.opcode() == HloOpcode::kCompare &&
      instr.operand(0)->shape().element_type() == PrimitiveType::U16) {
    return CodegenDecision{};
  }

  auto type = instr.shape().element_type();
  bool output_type_is_supported = IsTritonSupportedDataType(type, gpu_version);

  if (!output_type_is_supported) {
    return "Unsupported output data type.";
  }

  bool input_types_are_supported =
      absl::c_all_of(instr.operands(), [&](const HloInstruction* operand) {
        return IsTritonSupportedDataType(operand->shape().element_type(),
                                         gpu_version);
      });

  if (!input_types_are_supported) {
    return "Unsupported input data type.";
  }

  if (instr.IsElementwise()) {
    if (!IsTritonSupportedElementwise(
            instr.opcode(),
            // Use the last operand below in order to support both `compare`
            // and `select` which have a fixed PRED type in the output and first
            // operand.
            instr.operand(instr.operand_count() - 1)->shape().element_type(),
            gpu_version)) {
      return "Unsupported elementwise operation.";
    }
    return CodegenDecision{};
  }

  // TODO(bchetioui): support kDot, kPad, and kDynamicSlice.
  switch (instr.opcode()) {
    case HloOpcode::kReduce: {
      // TODO(bchetioui): upgrade `CanTritonHandleReduce` to correspond to
      // the new implementation.
      return legacy_triton::CanTritonHandleReduce(
          *Cast<HloReduceInstruction>(&instr), gpu_version);
    }
    case HloOpcode::kTranspose:
    case HloOpcode::kSlice:
    case HloOpcode::kParameter:
    case HloOpcode::kBroadcast:
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
      return CodegenDecision{};
    default:
      VLOG(2) << "Unsupported instruction: " << instr.ToString();
      break;
  }
  return "Unsupported opcode.";
}

}  // namespace

absl::Status EnsureTritonSupportsComputeCapability(
    const se::GpuComputeCapability& gpu_compute_capability) {
  auto cuda_compute_capability =
      std::get_if<se::CudaComputeCapability>(&gpu_compute_capability);
  auto rocm_compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_compute_capability);
  if (!cuda_compute_capability && !rocm_compute_capability) {
    return absl::FailedPreconditionError(
        "Triton support is only enabled for CUDA and ROCm GPUs.");
  }

  if (cuda_compute_capability && !cuda_compute_capability->IsAtLeastAmpere()) {
    return absl::FailedPreconditionError(
        absl::StrCat("CUDA Triton support is only enabled for Ampere GPUs ",
                     "(compute capability 8.0) and up, but got compute ",
                     "capability ", cuda_compute_capability->major, ".",
                     cuda_compute_capability->minor, "."));
  }

  return absl::OkStatus();
}

CodegenDecision IsTritonSupportedInstruction(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version) {
  CodegenDecision decision =
      IsTritonSupportedInstructionImpl(instr, gpu_version);
  VLOG(2) << "IsTritonSupportedInstruction: " << instr.ToString() << " "
          << bool(decision);
  return decision;
}

bool IsTritonFusedComputation(const HloComputation& computation) {
  HloFusionInstruction* fusion =
      static_cast<HloFusionInstruction*>(computation.FusionInstruction());
  return fusion != nullptr &&
         fusion->fusion_kind() == HloInstruction::FusionKind::kCustom &&
         fusion->backend_config<gpu::GpuBackendConfig>()
                 ->fusion_backend_config()
                 .kind() == kTritonGemmFusionKind;
}

}  // namespace gpu
}  // namespace xla
