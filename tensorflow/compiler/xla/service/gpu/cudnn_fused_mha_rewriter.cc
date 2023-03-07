/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_mha_rewriter.h"

#include <functional>
#include <string>

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {
namespace {
namespace m = match;

bool IsFMHACustomCall(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kCustomCall &&
         (instr->custom_call_target() == kCudnnfMHABmmBmmCallTarget ||
          instr->custom_call_target() ==
              kCudnnfMHAScaleBiasMaskSoftmaxCallTarget ||
          instr->custom_call_target() ==
              kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget ||
          instr->custom_call_target() == kCudnnfMHAScaleMaskSoftmaxCallTarget ||
          instr->custom_call_target() ==
              kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget ||
          instr->custom_call_target() == kCudnnfMHASoftmaxDropoutCallTarget);
}

StatusOr<bool> IsBatchedDot(const HloInstruction* gemm) {
  GemmBackendConfig config;
  TF_ASSIGN_OR_RETURN(config, gemm->backend_config<GemmBackendConfig>());
  const DotDimensionNumbers& dot_dims = config.dot_dimension_numbers();
  bool is_batch_dot = !dot_dims.lhs_batch_dimensions().empty() ||
                      !dot_dims.rhs_batch_dimensions().empty();
  return is_batch_dot;
}

bool IsBatchedMatmul(const HloInstruction* gemm) {
  return IsCublasGemm(*gemm) && IsBatchedDot(gemm).value();
}

// Give this instruction a more useful name than "custom-call.42".
Status SetName(HloModule* module, HloInstruction* fmha) {
  if (fmha->custom_call_target() == kCudnnfMHABmmBmmCallTarget) {
    module->SetAndUniquifyInstrName(fmha, "fmha-bmm-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() == kCudnnfMHASoftmaxDropoutCallTarget) {
    module->SetAndUniquifyInstrName(fmha, "fmha-bmm-softmax-dropout-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() == kCudnnfMHAScaleMaskSoftmaxCallTarget) {
    module->SetAndUniquifyInstrName(fmha, "fmha-bmm-scale-mask-softmax-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() ==
      kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget) {
    module->SetAndUniquifyInstrName(fmha,
                                    "fmha-bmm-scale-mask-softmax-dropout-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() == kCudnnfMHAScaleBiasMaskSoftmaxCallTarget) {
    module->SetAndUniquifyInstrName(fmha,
                                    "fmha-bmm-scale-bias-mask-softmax-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() ==
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget) {
    module->SetAndUniquifyInstrName(
        fmha, "fmha-bmm-scale-bias-mask-softmax-dropout-bmm");
    return OkStatus();
  }

  return InternalError(
      "Found invalid FMHA custom-call target while setting custom-call name");
}

bool IsComputeCapabilitySupported(stream_executor::CudaComputeCapability cc) {
  if (!(cc.IsAtLeast(
          se::CudaComputeCapability::AMPERE)  && cc.minor == 0 )) {
      VLOG(2) << "CudnnFusedMHARewriter did not run. Unsupported compute "
                 "capability. Only GA100 supported.";
    return false;
  }
  return true;
}

bool IsDimensionMostMinor(absl::Span<const int64_t> minor_to_major,
                          int64_t dim_num) {
  return minor_to_major[0] == dim_num;
}

bool IsSupportedPrimitiveType(const HloInstruction* bmm) {
  auto dtype = bmm->shape().element_type();
  return dtype == BF16 || dtype == F16;
}

bool IsContractingDimSupported(
    absl::Span<const int64_t> contracting_dims) {
  return absl::c_all_of(contracting_dims,
                        [](int64_t dim) { return dim == 64; });
}

bool IsNonContractingDimSupported(
    const std::vector<int64_t>& non_contracting_dims) {
  return absl::c_all_of(non_contracting_dims,
                        [](int64_t dim) { return dim <= 512; });
}

std::vector<int64_t> GetDimensionVector(absl::Span<const int64_t> dimensions,
                                        absl::Span<const int64_t> dim_nums) {
  std::vector<int64_t> vec(dim_nums.size());
  for (int i = 0; i < dim_nums.size(); i++) {
    vec[i] = dimensions.at(dim_nums.at(i));
  }
  return vec;
}

StatusOr<bool> IsSupportedBMM1(const HloInstruction* bmm_1) {
  GemmBackendConfig config_bmm1;
  TF_ASSIGN_OR_RETURN(config_bmm1, bmm_1->backend_config<GemmBackendConfig>());
  const DotDimensionNumbers& dot_dims_bmm1 =
      config_bmm1.dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> lhs_non_contracting_dim_nums_bmm1,
      GetNonContractingDims(bmm_1->operand(0)->shape(),
                            dot_dims_bmm1.lhs_batch_dimensions(),
                            dot_dims_bmm1.lhs_contracting_dimensions()));
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_non_contracting_dim_nums_bmm1,
      GetNonContractingDims(bmm_1->operand(1)->shape(),
                            dot_dims_bmm1.rhs_batch_dimensions(),
                            dot_dims_bmm1.rhs_contracting_dimensions()));
  std::vector<int64_t> lhs_non_contracting_dims_bmm1 =
      GetDimensionVector(bmm_1->operand(0)->shape().dimensions(),
                         lhs_non_contracting_dim_nums_bmm1);
  std::vector<int64_t> rhs_non_contracting_dims_bmm1 =
      GetDimensionVector(bmm_1->operand(1)->shape().dimensions(),
                         rhs_non_contracting_dim_nums_bmm1);
  // The non contracting dimensions for BMM1 need to be less than or equal to
  // 512.
  if (!IsNonContractingDimSupported(lhs_non_contracting_dims_bmm1) ||
      !IsNonContractingDimSupported(rhs_non_contracting_dims_bmm1)) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "BMM1 lhs_non_contracting_dims: "
              << absl::StrJoin(lhs_non_contracting_dims_bmm1, ",")
              << " BMM1 rhs_non_contracting_dims: "
              << absl::StrJoin(rhs_non_contracting_dims_bmm1, ",")
              << " are not supported. The non-contracting dims should be less "
                 "than 512. This is a criteria for current cuDNN 8.8 support.";
    }
    return false;
  }

  std::vector<int64_t> lhs_contracting_dims_bmm1 =
      GetDimensionVector(bmm_1->operand(0)->shape().dimensions(),
                         dot_dims_bmm1.lhs_contracting_dimensions());
  std::vector<int64_t> rhs_contracting_dims_bmm1 =
      GetDimensionVector(bmm_1->operand(1)->shape().dimensions(),
                         dot_dims_bmm1.rhs_contracting_dimensions());

  // The contracting dimensions for BMM1 (both lhs and rhs) need to be 64 and should be the fastest moving dimension.
  absl::Span<const int64_t> rhs_minor_to_major_bmm1 =
      bmm_1->operand(1)->shape().layout().minor_to_major();
  absl::Span<const int64_t> lhs_minor_to_major_bmm1 =
      bmm_1->operand(0)->shape().layout().minor_to_major();
  if (!IsContractingDimSupported(lhs_contracting_dims_bmm1) ||
      !IsDimensionMostMinor(lhs_minor_to_major_bmm1,
                            dot_dims_bmm1.lhs_contracting_dimensions()[0]) ||
      !IsContractingDimSupported(rhs_contracting_dims_bmm1) ||
      !IsDimensionMostMinor(rhs_minor_to_major_bmm1,
                            dot_dims_bmm1.rhs_contracting_dimensions()[0])) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "BMM1 lhs_contracting_dims: "
              << bmm_1->operand(0)->shape().ToString(/*print layout*/ true)
              << " BMM1 rhs_contracting_dims: "
              << bmm_1->operand(1)->shape().ToString(true)
              << " are not supported. Either contracting dim is not equal to "
                 "64 or they are not the fastest moving. This is a criteria "
                 "for current cuDNN 8.8 supports";
    }
    return false;
  }
  return true;
}

StatusOr<bool> IsSupportedBMM2(const HloInstruction* bmm_2) {
  GemmBackendConfig config_bmm2;
  TF_ASSIGN_OR_RETURN(config_bmm2, bmm_2->backend_config<GemmBackendConfig>());
  const DotDimensionNumbers& dot_dims_bmm2 =
      config_bmm2.dot_dimension_numbers();

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_non_contracting_dim_nums_bmm2,
      GetNonContractingDims(bmm_2->operand(1)->shape(),
                            dot_dims_bmm2.rhs_batch_dimensions(),
                            dot_dims_bmm2.rhs_contracting_dimensions()));

  std::vector<int64_t> rhs_non_contracting_dims_bmm2 =
      GetDimensionVector(bmm_2->operand(1)->shape().dimensions(),
                         rhs_non_contracting_dim_nums_bmm2);
  // The non contracting dimension for BMM2 needs to be 64 for the input matrix.
  // The input matrix is the second argument to BMM2 i.e, rhs. It should also be the fastest moving dimension.
  absl::Span<const int64_t> rhs_minor_to_major_bmm2 =
      bmm_2->operand(1)->shape().layout().minor_to_major();
  if (!absl::c_all_of(rhs_non_contracting_dims_bmm2,
                      [](int64_t dim) { return dim == 64; }) ||
      !IsDimensionMostMinor(rhs_minor_to_major_bmm2,
                            rhs_non_contracting_dim_nums_bmm2[0])) {
    if (VLOG_IS_ON(2)) {
      VLOG(2)
          << " BMM1 rhs_non_contracting_dims: "
          << bmm_2->operand(1)->shape().ToString(true)
          << " is not supported. Either non-contracting dim is not equal to "
             "64 or it is not the fastest moving. This is a criteria "
             "for current cuDNN 8.8 supports";
    }
    return false;
  }
  return true;
}

StatusOr<bool> IsMHABlockSupported(HloInstruction* bmm_1,
                                   HloInstruction* bmm_2) {
  // cuDNN 8.8 currently only supports BF16 and F16 data types.
  if (!IsSupportedPrimitiveType(bmm_1) || !IsSupportedPrimitiveType(bmm_2)) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "Unsupported primitive type for cuDNN MHA fusion:\n"
              << bmm_1->ToString() << "\nOR\n"
              << bmm_2->ToString() << "\n"
              << "BF16 and F16 are the supported Dtypes.";
    }
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool is_bmm1_supported, IsSupportedBMM1(bmm_1));
  if (!is_bmm1_supported) return false;
  TF_ASSIGN_OR_RETURN(bool is_bmm2_supported, IsSupportedBMM2(bmm_2));
  if (!is_bmm2_supported) return false;
  return true;
}

// The following pattern is matched here:
// Q    K
// |    |
// v    v
//  BMM1
//   |    V
//   |    |
//   v    v
//    BMM2
//     |
//     v
//     O

StatusOr<bool> FuseBatchedMatmuls(HloComputation* comp,
                                  stream_executor::CudaComputeCapability cc) {
  const DebugOptions& debug_options = comp->parent()->config().debug_options();
  if (!IsComputeCapabilitySupported(cc)) {
    return false;
  }

  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* bmm_1;
    HloInstruction* bmm_2;
    auto pattern =
        m::Op(&bmm_2)
            .WithPredicate(IsBatchedMatmul)
            .WithOperand(
                0, m::Op(&bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse());
    if (!Match(instr, pattern)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(bool is_mha_module_supported,
                        IsMHABlockSupported(bmm_1, bmm_2));
    if (!is_mha_module_supported) continue;

    TF_ASSIGN_OR_RETURN(auto config_bmm1,
                        bmm_1->backend_config<GemmBackendConfig>());
    TF_ASSIGN_OR_RETURN(auto config_bmm2,
                        bmm_2->backend_config<GemmBackendConfig>());

    FusedMHABackendConfig fmha_config;
    *fmha_config.mutable_bmm1_dot_dimension_numbers() =
        config_bmm1.dot_dimension_numbers();
    *fmha_config.mutable_bmm2_dot_dimension_numbers() =
        config_bmm2.dot_dimension_numbers();
    fmha_config.set_fmha_scale(1.0);
    fmha_config.set_dropout_rate(0.0);
    *fmha_config.mutable_intermediate_tensor_shape() = bmm_1->shape().ToProto();
    {
      auto* algorithm = fmha_config.mutable_algorithm();
      algorithm->set_algo_id(0);  // engine id
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
      std::vector<int64_t> knob_ids = /* {0, 1} */{17, 24};
      std::vector<int64_t> knob_vals = {1, 0};
      for (int i = 0; i < knob_ids.size(); ++i) {
        (*algorithm->mutable_tuning_knobs())[knob_ids[i]] = knob_vals[i];
      }
      algorithm->set_is_cudnn_frontend(true);
      algorithm->mutable_workspace_size()->set_value(0);
    }
    HloInstruction* lhs_bmm1 = bmm_1->mutable_operand(0);
    HloInstruction* rhs_bmm1 = bmm_1->mutable_operand(1);
    HloInstruction* rhs_bmm2 = bmm_2->mutable_operand(1);
    const Shape& output_shape = bmm_2->shape();
    Shape call_shape = ShapeUtil::MakeTupleShape(
        {output_shape, ShapeUtil::MakeShape(U8, {0})});
    HloInstruction* fmha_call =
        comp->AddInstruction(HloInstruction::CreateCustomCall(
            call_shape, {lhs_bmm1, rhs_bmm1, rhs_bmm2},
            absl::string_view(kCudnnfMHABmmBmmCallTarget)));
    TF_RETURN_IF_ERROR(fmha_call->set_backend_config(fmha_config));
    TF_RETURN_IF_ERROR(SetName(bmm_1->GetModule(), fmha_call));
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        instr,
        HloInstruction::CreateGetTupleElement(instr->shape(), fmha_call, 0)));
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "After CudnnFusedMHARewriter: \n"
              << comp->parent()->ToString();
    }
    changed = true;
  }
  return changed;
}
}  // namespace

StatusOr<bool> CudnnFusedMHARewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    bool changed = false;
    TF_ASSIGN_OR_RETURN(changed, FuseBatchedMatmuls(comp, compute_capability_));
    any_changed |= changed;
  }

  return any_changed;
}
}  // namespace gpu
}  // namespace xla

