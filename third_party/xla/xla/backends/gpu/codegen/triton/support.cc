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

#include <cstdint>
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
    return {HloOpcode::kConvert, HloOpcode::kNot, HloOpcode::kCopy};
  }

  if (element_type == PrimitiveType::U16) {
    return {HloOpcode::kAbs};
  }

  absl::flat_hash_set<HloOpcode> ret{HloOpcode::kAbs, HloOpcode::kConvert,
                                     HloOpcode::kCopy};

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

  if (IsTritonSupportedDataType(input, gpu_version) &&
      IsTritonSupportedDataType(output, gpu_version)) {
    return CodegenDecision::Allow();
  }

  return error_message();
}

// Set of binary element-wise ops that are genuinely supported by Triton.
absl::flat_hash_set<HloOpcode> TritonSupportedBinaryElementwiseOps(
    PrimitiveType element_type, const se::GpuComputeCapability& gpu_version) {
  if (element_type == PrimitiveType::U16 ||
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
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> lhs_non_contracting_dims,
      GetNonContractingDims(lhs_shape, dim_numbers.lhs_batch_dimensions(),
                            dim_numbers.lhs_contracting_dimensions()));
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_non_contracting_dims,
      GetNonContractingDims(rhs_shape, dim_numbers.rhs_batch_dimensions(),
                            dim_numbers.rhs_contracting_dimensions()));
  if (lhs_non_contracting_dims.size() > 1 ||
      rhs_non_contracting_dims.size() > 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Multiple non-contracting dimensions are not supported, got LHS: [",
        absl::StrJoin(lhs_non_contracting_dims, ","), "], RHS: [",
        absl::StrJoin(rhs_non_contracting_dims, ","), "]"));
  }
  // Only checking one side of bach and contracting dimensions, since they must
  // be the same for left and right.
  if (dim_numbers.lhs_batch_dimensions_size() > 0) {
    return absl::UnimplementedError(
        absl::StrCat("Batch dimensions are not supported yet, got ",
                     absl::StrJoin(dim_numbers.lhs_batch_dimensions(), ",")));
  }
  if (dim_numbers.lhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Exactly one contracting dimension is supported, got ",
        absl::StrJoin(dim_numbers.lhs_contracting_dimensions(), ",")));
  }
  if (dim_numbers.lhs_contracting_dimensions(0) != 1 ||
      dim_numbers.rhs_contracting_dimensions(0) != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Only lhs_contracting_dimensions=1 (got ",
        absl::StrJoin(dim_numbers.lhs_contracting_dimensions(), ","),
        ") and  rhs_contracting_dimensions=0 (got ",
        absl::StrJoin(dim_numbers.rhs_contracting_dimensions(), ","),
        ") are supported."));
  }
  return absl::OkStatus();
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

  if (result_type != lhs_type || result_type != rhs_type) {
    return CodegenDecision::Forbid(
        "Dot operation only supports same types for the result, lhs and rhs.");
  }
  if (absl::c_linear_search(
          std::vector<PrimitiveType>{PrimitiveType::F8E5M2, PrimitiveType::BF16,
                                     PrimitiveType::F8E4M3FN,
                                     PrimitiveType::S32, PrimitiveType::S64,
                                     PrimitiveType::S16, PrimitiveType::S8},
          result_type)) {
    return CodegenDecision::Forbid(
        absl::StrCat(PrimitiveType_Name(result_type), " is not supported"));
  }

  absl::Status status = CheckSupportedCheckDotDimensions(dot);
  if (!status.ok()) {
    return CodegenDecision::Forbid(status.message());
  }

  const PrecisionConfig& precision_config = dot.precision_config();
  if (precision_config.algorithm() != PrecisionConfig::ALG_UNSET ||
      absl::c_any_of(precision_config.operand_precision(),
                     [](const int precision) {
                       return precision != PrecisionConfig::DEFAULT;
                     })) {
    LOG(INFO) << "Unsupported precision config: "
              << precision_config.ShortDebugString();
    return CodegenDecision::Forbid(absl::StrCat(
        "Unsupported precision config: ", precision_config.ShortDebugString()));
  }

  return CodegenDecision::Allow();
}

CodegenDecision IsSupportedFusion(const HloFusionInstruction& fusion) {
  absl::StatusOr<GpuBackendConfig> backend_config =
      fusion.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return CodegenDecision(backend_config.status());
  }
  absl::string_view fusion_kind =
      backend_config.value().fusion_backend_config().kind();
  // Note: kTritonFusionKind is NOT expected to be set for nested fusions.
  if (fusion_kind != kTritonNestedGemmFusionKind) {
    return CodegenDecision::Forbid(
        absl::StrCat("Unsupported fusion kind: ", fusion_kind));
  }
  return CodegenDecision::Allow();
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
  return CodegenDecision::Allow();
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
    return ShapeUtil::IsEffectiveScalar(instr.shape())
               ? CodegenDecision::Allow()
               : CodegenDecision::Forbid(
                     "Only scalar constants are supported in Triton.");
  }

  if (instr.opcode() == HloOpcode::kIota) {
    PrimitiveType element_type = instr.shape().element_type();
    return element_type != PrimitiveType::F8E4M3FN &&
                   element_type != PrimitiveType::F8E5M2
               ? CodegenDecision::Allow()
               : CodegenDecision::Forbid(
                     "F8E4M3FN and F8E5M2 are not supported for iota.");
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
      return CodegenDecision::Allow();
    case HloOpcode::kDot:
      return IsTritonSupportedDot(*Cast<HloDotInstruction>(&instr),
                                  gpu_version);
    case HloOpcode::kFusion:
      return IsSupportedFusion(*Cast<HloFusionInstruction>(&instr));
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
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCholesky:
    case HloOpcode::kConvolution:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kFft:
    case HloOpcode::kGather:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kInfeed:
    case HloOpcode::kMap:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kOutfeed:
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
    case HloOpcode::kStochasticConvert:
    case HloOpcode::kTopK:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kTuple:
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
