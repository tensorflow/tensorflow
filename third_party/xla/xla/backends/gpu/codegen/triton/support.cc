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

#include "xla/backends/gpu/codegen/triton/support.h"

#include <string>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

bool IsTritonSupportedDataType(PrimitiveType type,
                               const se::GpuComputeCapability& gpu_version) {
  switch (type) {
    case PRED:
    case S4:
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
absl::flat_hash_set<HloOpcode> TritonSupportedUnaryElementwiseOps(
    PrimitiveType element_type) {
  if (element_type == PrimitiveType::PRED) {
    return {HloOpcode::kNot, HloOpcode::kCopy};
  }

  if (element_type == PrimitiveType::S4) {
    return {};
  }

  if (element_type == PrimitiveType::U16) {
    return {HloOpcode::kAbs};
  }

  absl::flat_hash_set<HloOpcode> ret{HloOpcode::kAbs, HloOpcode::kCopy};

  if (element_type != PrimitiveType::F8E5M2 &&
      element_type != PrimitiveType::F8E4M3FN) {
    ret.insert(HloOpcode::kNegate);
  }

  if (primitive_util::IsIntegralType(element_type)) {
    ret.insert(HloOpcode::kNot);
  }

  if (element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F16 ||
      element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::F64) {
    absl::flat_hash_set<HloOpcode> additional_opcodes{
        HloOpcode::kCos,   HloOpcode::kExp,   HloOpcode::kExpm1,
        HloOpcode::kFloor, HloOpcode::kCeil,  HloOpcode::kLog,
        HloOpcode::kLog1p, HloOpcode::kRsqrt, HloOpcode::kSin,
        HloOpcode::kSqrt,  HloOpcode::kCbrt,  HloOpcode::kTan,
        HloOpcode::kTanh,  HloOpcode::kErf};
    ret.insert(additional_opcodes.begin(), additional_opcodes.end());
  }

  if (primitive_util::IsFloatingPointType(element_type)) {
    ret.insert(HloOpcode::kReducePrecision);
  }

  return ret;
}

CodegenDecision IsTritonSupportedConversion(
    PrimitiveType output, PrimitiveType input,
    const se::GpuComputeCapability& gpu_version) {
  auto any_is = [=](PrimitiveType compare) {
    return input == compare || output == compare;
  };

  auto error_message = [&]() {
    return CodegenDecision::Forbid(
        absl::StrCat("Unsupported conversion in Triton: ",
                     primitive_util::LowercasePrimitiveTypeName(input), " to ",
                     primitive_util::LowercasePrimitiveTypeName(output)));
  };

  if (input != output && any_is(PrimitiveType::F8E4M3FN) &&
      std::holds_alternative<se::CudaComputeCapability>(gpu_version) &&
      !std::get<se::CudaComputeCapability>(gpu_version).IsAtLeastHopper()) {
    return error_message();
  }

  bool is_f8_conversion =
      any_is(PrimitiveType::F8E4M3FN) && any_is(PrimitiveType::F8E5M2);
  bool is_f8 = any_is(PrimitiveType::F8E4M3FN) || any_is(PrimitiveType::F8E5M2);
  bool is_f16_or_f32 = any_is(PrimitiveType::F16) ||
                       any_is(PrimitiveType::BF16) ||
                       any_is(PrimitiveType::F32);
  if (input != output && is_f8 && !is_f8_conversion && !is_f16_or_f32) {
    return error_message();
  }

  if (input == S4 && output != S8) {
    return error_message();
  }
  if (output == S4) {
    return error_message();
  }

  if (IsTritonSupportedDataType(input, gpu_version) &&
      IsTritonSupportedDataType(output, gpu_version)) {
    return CodegenDecision::Allow();
  }

  return error_message();
}

// Set of binary element-wise ops that are genuinely supported by Triton.
absl::flat_hash_set<HloOpcode> TritonSupportedBinaryElementwiseOps(
    PrimitiveType element_type, const se::GpuComputeCapability& gpu_version) {
  if (element_type == PrimitiveType::S4 || element_type == PrimitiveType::U16 ||
      element_type == PrimitiveType::F8E5M2 ||
      element_type == PrimitiveType::F8E4M3FN) {
    return {};
  }

  absl::flat_hash_set<HloOpcode> ret{HloOpcode::kAdd, HloOpcode::kCompare,
                                     HloOpcode::kMaximum, HloOpcode::kMinimum,
                                     HloOpcode::kMultiply};

  if (element_type == PrimitiveType::PRED) {
    ret.insert(HloOpcode::kAnd);
    ret.insert(HloOpcode::kOr);
    ret.insert(HloOpcode::kXor);
    return ret;
  }

  ret.insert(HloOpcode::kSubtract);

  if (primitive_util::IsIntegralType(element_type)) {
    ret.insert(HloOpcode::kDivide);
    ret.insert(HloOpcode::kAnd);
    ret.insert(HloOpcode::kOr);
    ret.insert(HloOpcode::kXor);
  }

  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::F64) {
    ret.insert(HloOpcode::kAtan2);
    ret.insert(HloOpcode::kDivide);
    ret.insert(HloOpcode::kRemainder);
    ret.insert(HloOpcode::kPower);
  }
  if (element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F16) {
    ret.insert(HloOpcode::kAtan2);
    ret.insert(HloOpcode::kPower);
    ret.insert(HloOpcode::kRemainder);
  }

  return ret;
}

// Set of ternary elementwise ops that are genuinely supported by Triton.
absl::flat_hash_set<HloOpcode> TritonSupportedTernaryElementwiseOps(
    PrimitiveType element_type, const se::GpuComputeCapability& gpu_version) {
  if (element_type == PrimitiveType::S4 || element_type == PrimitiveType::U16) {
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
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version);

// Filters Reduces which can be handled using Triton.
CodegenDecision CanTritonHandleReduce(
    const HloReduceInstruction& reduce,
    const se::GpuComputeCapability& gpu_version) {
  if (reduce.shape().element_type() == PrimitiveType::F8E4M3FN ||
      reduce.shape().element_type() == PrimitiveType::F8E5M2) {
    return CodegenDecision::Forbid(
        "F8E4M3FN and F8E5M2 are not supported for reductions.");
  }

  bool is_triton_supported_reduction_computation = absl::c_all_of(
      reduce.to_apply()->instructions(), [&](const HloInstruction* instr) {
        return IsTritonSupportedInstructionImpl(*instr, gpu_version).CanFuse();
      });
  if (!is_triton_supported_reduction_computation) {
    return CodegenDecision::Forbid(
        "Unsupported reduction computation by Triton.");
  }

  if (reduce.dimensions().size() == 1 && reduce.operand_count() == 2) {
    return CodegenDecision::Allow();
  }
  return CodegenDecision::Forbid(
      "Reduction is not a row-reduction of a single operand.");
}

bool IsInTritonNestedGemmFusion(const HloInstruction& hlo) {
  const HloComputation* computation = hlo.parent();
  if (!computation->IsFusionComputation()) {
    return false;
  }
  absl::StatusOr<GpuBackendConfig> backend_config =
      computation->FusionInstruction()->backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return false;
  }
  absl::string_view fusion_kind =
      backend_config.value().fusion_backend_config().kind();
  return fusion_kind == kTritonNestedGemmFusionKind;
}

absl::Status CheckSupportedCheckDotDimensions(const HloDotInstruction& dot) {
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();
  // Only checking one side of bach and contracting dimensions, since they must
  // be the same for left and right.
  if (dim_numbers.lhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Exactly one contracting dimension is supported, got ",
        absl::StrJoin(dim_numbers.lhs_contracting_dimensions(), ",")));
  }
  return absl::OkStatus();
}

bool IsSupportedDotAlgorithm(PrecisionConfig::Algorithm algorithm) {
  switch (algorithm) {
    case PrecisionConfig::ALG_UNSET:
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      return true;
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
    default:
      break;
  }

  return false;
}

CodegenDecision AreTypesSupportedByAlgUnsetDot(
    PrimitiveType input_type, PrimitiveType result_type,
    const se::GpuComputeCapability& gpu_version) {
  if (input_type == F64 && result_type != F64) {
    return CodegenDecision::Forbid(
        "Dot operation only supports F64 result type for F64 input type.");
  }

  if (input_type == F8E4M3FN || result_type == F8E4M3FN) {
    if (auto* cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version);
        cuda_cc && !cuda_cc->IsAtLeastHopper()) {
      return CodegenDecision::Forbid(
          "Dot operation for F8E4M3FN is not supported before Hopper.");
    }
  }

  auto supported_float_types = {BF16, F16, F32, F64, F8E5M2, F8E4M3FN};
  if (absl::c_linear_search(supported_float_types, input_type)) {
    return CodegenDecision::Allow();
  }

  if (input_type == S8 && result_type == S32) {
    return CodegenDecision::Allow();
  }

  auto partially_supported_signed_types = {S4, S8, S16, S32, S64};
  if (absl::c_linear_search(partially_supported_signed_types, input_type)) {
    if (absl::c_linear_search(partially_supported_signed_types, result_type)) {
      return CodegenDecision::Forbid(
          "Dot operation does not support these signed integer types.");
    }
    if (primitive_util::IsFloatingPointType(result_type)) {
      return CodegenDecision::Forbid(
          "Dot operation does not support floating point input and signed "
          "integer result types.");
    }
    return CodegenDecision::Allow();
  }

  return CodegenDecision::Forbid("Unsupported types.");
}

// Checks whether the conversions generated during the lowering of the relevant
// dot algorithm for the relevant input and output types are supported by
// Triton.
//
// When the algorithm is `ALG_UNSET`, nothing is checked.
CodegenDecision AreDotAlgorithmInputAndOutputConversionsSupported(
    PrecisionConfig::Algorithm algorithm, PrimitiveType lhs_type,
    PrimitiveType rhs_type, PrimitiveType result_type,
    const se::GpuComputeCapability& gpu_version) {
  if (algorithm == PrecisionConfig::ALG_UNSET) {
    return CodegenDecision::Allow();
  }

  auto forbid = [&algorithm](absl::string_view message) {
    return CodegenDecision::Forbid(
        absl::StrCat(message, " for dot algorithm ",
                     PrecisionConfig::Algorithm_Name(algorithm)));
  };

  absl::StatusOr<std::vector<PrimitiveType>> allowed_operands_types_or =
      algorithm_util::GetAllowedOperandsTypeForAlgorithm(algorithm);
  absl::StatusOr<PrimitiveType> expected_accumulator_type =
      algorithm_util::GetDotAccumulatorType(algorithm);
  if (!allowed_operands_types_or.ok() || !expected_accumulator_type.ok()) {
    return forbid("Failed to recover operands types or accumulator type");
  }
  CHECK(!allowed_operands_types_or->empty());

  if (result_type != *expected_accumulator_type) {
    if (!IsTritonSupportedConversion(*expected_accumulator_type, result_type,
                                     gpu_version) ||
        !IsTritonSupportedConversion(result_type, *expected_accumulator_type,
                                     gpu_version)) {
      return forbid("Unsupported result conversion");
    }
  }

  if (allowed_operands_types_or->size() != 1 &&
      (lhs_type != rhs_type ||
       !absl::c_linear_search(*allowed_operands_types_or, lhs_type))) {
    return forbid("Unsupported operand types");
  } else if (allowed_operands_types_or->size() == 1) {
    return CodegenDecision::Allow();
  }

  PrimitiveType expected_operands_type = allowed_operands_types_or->front();

  if (lhs_type != expected_operands_type &&
      !IsTritonSupportedConversion(expected_operands_type, lhs_type,
                                   gpu_version)) {
    return forbid("Unsupported lhs conversion");
  }
  if (rhs_type != expected_operands_type &&
      !IsTritonSupportedConversion(expected_operands_type, rhs_type,
                                   gpu_version)) {
    return forbid("Unsupported rhs conversion");
  }

  return CodegenDecision::Allow();
}

CodegenDecision IsTritonSupportedDot(
    const HloDotInstruction& dot, const se::GpuComputeCapability& gpu_version) {
  if (!IsInTritonNestedGemmFusion(dot)) {
    return CodegenDecision::Forbid(
        "Dot operation is only supported in nested GEMM fusions.");
  }
  PrimitiveType result_type = dot.shape().element_type();
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  PrimitiveType lhs_type = lhs_shape.element_type();
  PrimitiveType rhs_type = rhs_shape.element_type();

  if (dot.operand(0)->opcode() != HloOpcode::kFusion ||
      dot.operand(1)->opcode() != HloOpcode::kFusion) {
    return CodegenDecision::Forbid(
        "Only operands that are fusions are supported.");
  }

  // TODO(b/393299275): add support tests for mixed types.
  if (lhs_type != rhs_type) {
    return CodegenDecision::Forbid(
        "Dot operation only supports same types for lhs and rhs.");
  }

  if (result_type == PrimitiveType::S4) {
    return CodegenDecision::Forbid("S4 is not supported.");
  }

  absl::Status status = CheckSupportedCheckDotDimensions(dot);
  if (!status.ok()) {
    return CodegenDecision::Forbid(status.message());
  }

  const PrecisionConfig& precision_config = dot.precision_config();
  const PrecisionConfig::Algorithm algorithm = precision_config.algorithm();

  if (!IsSupportedDotAlgorithm(algorithm)) {
    return CodegenDecision::Forbid(
        absl::StrCat("Unsupported dot algorithm: ",
                     PrecisionConfig::Algorithm_Name(algorithm)));
  }

  if (algorithm == PrecisionConfig::ALG_UNSET) {
    if (CodegenDecision decision =
            AreTypesSupportedByAlgUnsetDot(lhs_type, result_type, gpu_version);
        !decision) {
      return decision;
    }
  }

  if (CodegenDecision conversion_decision =
          AreDotAlgorithmInputAndOutputConversionsSupported(
              algorithm, lhs_type, rhs_type, result_type, gpu_version);
      !conversion_decision) {
    return conversion_decision;
  }

  return CodegenDecision::Allow();
}

// Verifies that the nested fusion instruction conforms to the assumptions of
// the emitter. Currently, we expect nested fusions:
// - of kind `__triton_nested_gemm_fusion`;
// - to have a single user that is either a `dot` or a `concatenate`;
// - calls a supported computation.
CodegenDecision IsSupportedFusion(const HloFusionInstruction& hlo,
                                  const se::GpuComputeCapability& capability) {
  // TODO(b/393299275): test cases when there are multiple dot users of the
  // same fusion.
  if (hlo.user_count() != 1) {
    return CodegenDecision::Forbid(
        absl::StrCat("Expected only one user for fusion ", hlo.ToString(),
                     " but got ", hlo.user_count()));
  }
  absl::StatusOr<GpuBackendConfig> backend_config =
      hlo.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return CodegenDecision(backend_config.status());
  }
  if (const std::string& kind =
          backend_config.value().fusion_backend_config().kind();
      kind != kTritonNestedGemmFusionKind) {
    return CodegenDecision::Forbid(
        absl::StrCat("Expected ", hlo.ToString(), " with fusion backend kind ",
                     kTritonNestedGemmFusionKind, ", got ", kind));
  }
  const HloInstruction* user = hlo.users().front();
  switch (user->opcode()) {
    case HloOpcode::kDot:
    case HloOpcode::kConcatenate:
      break;
    default:
      return CodegenDecision::Forbid(absl::StrCat(
          "Unexpected user opcode ", user->opcode(), " of nested fusion"));
  }
  CodegenDecision decision =
      IsTritonSupportedComputation(*hlo.called_computation(), capability);
  if (decision.CanFuse()) {
    return CodegenDecision::Allow();
  }
  return CodegenDecision::Forbid(
      absl::StrCat("Computation called by fusion ", hlo.ToString(),
                   " is not supported: ", decision.Explain()));
}

CodegenDecision IsTritonSupportedConcatenate(const HloInstruction& hlo) {
  CHECK(hlo.opcode() == HloOpcode::kConcatenate);
  if (!IsInTritonNestedGemmFusion(hlo)) {
    return CodegenDecision::Forbid(
        "Only concatenates in nested GEMM fusions are supported.");
  }
  // TODO(b/393299275): remove this operand filter once migration is
  // complete and priority fusion can produce nests.
  if (absl::c_any_of(hlo.operands(), [](const HloInstruction* operand) {
        return operand->opcode() != HloOpcode::kFusion;
      })) {
    return CodegenDecision::Forbid(
        "Only support concatenates with nested GEMM fusions as a "
        "parameter.");
  }
  return CodegenDecision(hlo.shape().element_type() != S4,
                         "S4 is not supported.");
}

CodegenDecision IsTritonSupportedInstructionImpl(
    const HloInstruction& instr, const se::GpuComputeCapability& gpu_version) {
  if (internal::IsTritonUnsupportedOpcode(instr.opcode())) {
    return CodegenDecision::Forbid(
        absl::StrCat("Unsupported opcode ", HloOpcodeString(instr.opcode())));
  }

  // Special handling for the kConvert instruction, which has a non-standard
  // set of supported types.
  if (instr.opcode() == HloOpcode::kConvert) {
    return IsTritonSupportedConversion(instr.shape().element_type(),
                                       instr.operand(0)->shape().element_type(),
                                       gpu_version);
  }

  auto type = instr.shape().element_type();
  bool output_type_is_supported = IsTritonSupportedDataType(type, gpu_version);

  if (!output_type_is_supported) {
    return CodegenDecision::Forbid("Unsupported output data type.");
  }

  bool input_types_are_supported =
      absl::c_all_of(instr.operands(), [&](const HloInstruction* operand) {
        return IsTritonSupportedDataType(operand->shape().element_type(),
                                         gpu_version);
      });

  if (!input_types_are_supported) {
    return CodegenDecision::Forbid("Unsupported input data type.");
  }

  if (instr.opcode() == HloOpcode::kConcatenate) {
    return IsTritonSupportedConcatenate(instr);
  }

  // Const is technically an elementwise op, so this check must be before the
  // elementwise check.
  if (instr.opcode() == HloOpcode::kConstant) {
    if (type == PrimitiveType::S4) {
      return CodegenDecision::Forbid("S4 is not supported.");
    }
    return CodegenDecision(ShapeUtil::IsEffectiveScalar(instr.shape()),
                           "Only scalar constants are supported in Triton.");
  }

  if (instr.opcode() == HloOpcode::kIota) {
    PrimitiveType element_type = instr.shape().element_type();
    return CodegenDecision(
        element_type != PrimitiveType::F8E4M3FN &&
            element_type != PrimitiveType::F8E5M2 &&
            element_type != PrimitiveType::S4,
        "F8E4M3FN, F8E5M2 and S4 are not supported for iota.");
  }

  switch (instr.opcode()) {
    case HloOpcode::kReduce: {
      return CanTritonHandleReduce(*Cast<HloReduceInstruction>(&instr),
                                   gpu_version);
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kParameter:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return CodegenDecision(instr.shape().element_type() != S4,
                             "S4 is not supported.");
    case HloOpcode::kDot:
      return IsTritonSupportedDot(*Cast<HloDotInstruction>(&instr),
                                  gpu_version);
    case HloOpcode::kFusion:
      return IsSupportedFusion(*Cast<HloFusionInstruction>(&instr),
                               gpu_version);
    default:
      // Not all instructions have a special handling.
      break;
  }

  if (instr.IsElementwise()) {
    if (!IsTritonSupportedElementwise(
            instr.opcode(),
            // Use the last operand below in order to support both `compare`
            // and `select` which have a fixed PRED type in the output and first
            // operand.
            instr.operand(instr.operand_count() - 1)->shape().element_type(),
            gpu_version)) {
      return CodegenDecision::Forbid("Unsupported elementwise operation.");
    }
    return CodegenDecision::Allow();
  }
  return CodegenDecision::Forbid(absl::StrCat("Unsupported instruction opcode ",
                                              HloOpcodeString(instr.opcode())));
}

}  // namespace

namespace internal {
bool IsTritonUnsupportedOpcode(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kConvolution:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kSort:
      return true;
    default:
      return false;
  }
}

}  // namespace internal

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
  VLOG(2) << absl::StrCat("IsTritonSupportedInstruction: ", instr.ToString(),
                          " ",
                          (decision.CanFuse() ? "yes" : decision.Explain()));
  return decision;
}

CodegenDecision IsTritonSupportedComputation(
    const HloComputation& computation,
    const se::GpuComputeCapability& gpu_compute_capability) {
  VLOG(3) << "IsTritonSupportedComputation: " << computation.ToString();
  for (const auto* instruction : computation.instructions()) {
    if (CodegenDecision can_codegen =
            IsTritonSupportedInstruction(*instruction, gpu_compute_capability);
        !can_codegen) {
      return can_codegen;
    }
  }
  return CodegenDecision::Allow();
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
