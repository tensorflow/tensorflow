/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {
namespace m = match;

template <typename Pattern>
auto OptionalReshape(Pattern pattern) {
  auto shared = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(m::Reshape(shared), shared);
}

template <typename Pattern>
auto OptionalConvert(Pattern pattern) {
  auto shared = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(m::Convert(shared), shared);
}

template <typename Pattern>
auto OptionalBroadcast(Pattern pattern) {
  auto shared = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(m::Broadcast(shared), shared);
}

bool IsBatchedMatmul(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kDot) return false;
  const DotDimensionNumbers& dot_dims = instr->dot_dimension_numbers();
  bool is_batch_dot = !dot_dims.lhs_batch_dimensions().empty() ||
                      !dot_dims.rhs_batch_dimensions().empty();
  return is_batch_dot;
}

bool IsScalar(const HloInstruction* instr) {
  return ShapeUtil::IsEffectiveScalar(instr->shape());
}

bool IsReduceMax(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kReduce &&
         instr->to_apply()->root_instruction()->opcode() == HloOpcode::kMaximum;
}

bool IsReduceSum(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kReduce &&
         instr->to_apply()->root_instruction()->opcode() == HloOpcode::kAdd;
}

bool IsFusedSoftmaxCall(const HloInstruction* instr) { return false; }

bool IsScaledMaskedFusedSoftmaxCall(const HloInstruction* instr) {
  return IsFusedSoftmaxCall(instr) &&
         (instr->operands().size() == 3 || instr->operands().size() == 4) &&
         instr->operands()[0]->shape().element_type() == PRED;
}

// Set up subpatterns for re-use.
// Matches softmax sub-pattern ->
// divide(exp(Subtract(producer, reduce_max(producer))),
// broadcast(reduce_add(exp(Subtract(...))))). There might be reshape and
// convert nodes between reduce and Subtract.
// TODO TJ: Make this more general to any patterns that has this structure when
// cudnn runner supports generic cudnnOpGraphs. producer
// |   \
// |  reduce
// |     |
// |  broadcast
// |   /
// root
auto GetUnfusedReduceMaxSumSoftmaxPattern(
    HloInstruction** softmax_input = nullptr) {
  // The reduce-max part of the softmax
  auto unfused_softmax_max_subpattern = m::SharedSubpattern(m::Subtract(
      m::Op(),
      m::Broadcast(OptionalConvert(OptionalConvert(
          m::Op()
              .WithPredicate(IsReduceMax)
              .WithOperand(0, OptionalConvert(m::Op(softmax_input))))))));

  // The reduce-add part of the softmax
  auto unfused_softmax_sum_subpattern = m::SharedSubpattern(m::Divide(
      m::Exp(unfused_softmax_max_subpattern),
      m::Broadcast(OptionalConvert(OptionalConvert(
                       m::Op()
                           .WithOperand(0, OptionalConvert(m::Exp(
                                               unfused_softmax_max_subpattern)))
                           .WithPredicate(IsReduceSum)
                           .WithOneUse())))
          .WithOneUse()));
  return unfused_softmax_sum_subpattern;
}

std::optional<double> GetConstantValue(const HloInstruction* inst) {
  if (!IsScalar(inst)) {
    return std::nullopt;
  }
  switch (inst->shape().element_type()) {
    case F16:
      return static_cast<float>(inst->literal().GetFirstElement<half>());
    case BF16:
      return static_cast<float>(inst->literal().GetFirstElement<bfloat16>());
    case F32:
      return inst->literal().GetFirstElement<float>();
    case F64:
      return inst->literal().GetFirstElement<double>();
    default:
      return std::nullopt;
  }
}

double GetDropoutRateFromHlo(HloInstruction* dropout) {
  std::optional<double> dropout_rate_inv;
  dropout_rate_inv = GetConstantValue(dropout);
  if (!dropout_rate_inv.has_value()) {
    return 0.0;
  }
  // In dropout, inputs are divided by (1 - rate), we need to divide 1 by
  // the constant in dropout node and substract
  // from 1 here to get the actual dropout rate.
  return (1.0 - (1.0 / *dropout_rate_inv));
}

HloInstruction* GetBiasFromBmm(HloInstruction* bmm) {
  return bmm->mutable_operand(2);
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
  if (fmha->custom_call_target() == kCudnnfMHASoftmaxCallTarget) {
    module->SetAndUniquifyInstrName(fmha, "fmha-bmm-softmax-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() == kCudnnfMHAScaleBiasSoftmaxCallTarget) {
    module->SetAndUniquifyInstrName(fmha, "fmha-bmm-scale-bias-softmax-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() ==
      kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget) {
    module->SetAndUniquifyInstrName(fmha,
                                    "fmha-bmm-scale-bias-softmax-dropout-bmm");
    return OkStatus();
  }

  return InternalError(
      "Found invalid FMHA custom-call target while setting custom-call name");
}

bool IsComputeCapabilityAndCudnnSupported(
    stream_executor::CudaComputeCapability cc,
    stream_executor::dnn::VersionInfo cudnn_version,
    stream_executor::StreamExecutor* stream_exec) {
  // return true;
  se::dnn::VersionInfo real_cudnn_version;
  if (stream_exec) {
    stream_executor::dnn::DnnSupport* dnn = stream_exec->AsDnn();
    StatusOr<se::dnn::VersionInfo> se_cudnn_version = dnn->GetVersion();
    if (se_cudnn_version.ok()) {
      real_cudnn_version = (*se_cudnn_version);
    }
  } else {
    real_cudnn_version = cudnn_version;
  }

  if (!((cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0) &&
        (real_cudnn_version.major_version() == 8 &&
         real_cudnn_version.minor_version() >= 8))) {
    VLOG(2) << "CudnnFusedMHARewriter did not run. Unsupported compute "
               "capability(==8.0) or cudnn version(>=8.8)";
    return false;
  }
  return true;
}

bool IsSupportedPrimitiveType(const HloInstruction* bmm) {
  PrimitiveType dtype = bmm->shape().element_type();
  return dtype == BF16 || dtype == F16;
}

bool IsContractingDimSupported(absl::Span<const int64_t> contracting_dims) {
  return absl::c_all_of(contracting_dims,
                        [](int64_t dim) { return dim == 64; });
}

bool IsNonContractingDimSupported(
    const std::vector<int64_t>& non_contracting_dims) {
  return absl::c_all_of(non_contracting_dims,
                        [](int64_t dim) { return dim <= 512; });
}

bool IsRankSupported(const HloInstruction* bmm) {
  return bmm->operand(0)->shape().dimensions().size() == 4 &&
         bmm->operand(1)->shape().dimensions().size() == 4;
}

bool IsBatchDimSizeSupported(const DotDimensionNumbers& dot_dims) {
  return dot_dims.lhs_batch_dimensions().size() == 2 &&
         dot_dims.rhs_batch_dimensions().size() == 2;
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
  if (!IsRankSupported(bmm_1)) return false;
  const DotDimensionNumbers& dot_dims_bmm1 = bmm_1->dot_dimension_numbers();
  if (!IsBatchDimSizeSupported(dot_dims_bmm1)) return false;
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

  // The contracting dimensions for BMM1 need to be 64.
  if (!IsContractingDimSupported(lhs_contracting_dims_bmm1) ||
      !IsContractingDimSupported(rhs_contracting_dims_bmm1)) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "BMM1 lhs_contracting_dims: "
              << absl::StrJoin(lhs_contracting_dims_bmm1, ",")
              << " BMM1 rhs_contracting_dims: "
              << absl::StrJoin(rhs_contracting_dims_bmm1, ",")
              << " are not supported.";
    }
    return false;
  }
  return true;
}

StatusOr<bool> IsSupportedBMM2(const HloInstruction* bmm_2,
                               bool need_canonicalization) {
  if (!IsRankSupported(bmm_2)) return false;
  const DotDimensionNumbers& dot_dims_bmm2 = bmm_2->dot_dimension_numbers();
  if (!IsBatchDimSizeSupported(dot_dims_bmm2)) return false;
  // need swap lhs and rhs for bmm2 if canonicalization is needed
  int operand_index = need_canonicalization ? 0 : 1;
  auto batch_dim = need_canonicalization ? dot_dims_bmm2.lhs_batch_dimensions()
                                         : dot_dims_bmm2.rhs_batch_dimensions();
  auto contracting_dim = need_canonicalization
                             ? dot_dims_bmm2.lhs_contracting_dimensions()
                             : dot_dims_bmm2.rhs_contracting_dimensions();

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> non_contracting_dim_nums_bmm2,
      GetNonContractingDims(bmm_2->operand(operand_index)->shape(), batch_dim,
                            contracting_dim));

  std::vector<int64_t> non_contracting_dims_bmm2 =
      GetDimensionVector(bmm_2->operand(operand_index)->shape().dimensions(),
                         non_contracting_dim_nums_bmm2);
  // The non contracting dimension for BMM2 needs to be 64 for the input matrix.
  // The input matrix is the second argument to BMM2 i.e, rhs.
  if (!absl::c_all_of(non_contracting_dims_bmm2,
                      [](int64_t dim) { return dim == 64; })) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << " BMM2 rhs_non_contracting_dims: "
              << absl::StrJoin(non_contracting_dims_bmm2, ",")
              << " are not supported.";
    }
    return false;
  }
  return true;
}

bool MatchDefaultBmmBmm(int64_t bmm2_operand_position, HloInstruction* instr,
                        HloInstruction** bmm_1, HloInstruction** bmm_2,
                        std::string& custom_call_name) {
  // Try matching default bmm1-bmm2 pattern
  auto default_bmm_bmm_pattern =
      m::Op(bmm_2)
          .WithPredicate(IsBatchedMatmul)
          .WithOperand(
              bmm2_operand_position,
              m::Op(bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse());

  if (Match(instr, default_bmm_bmm_pattern)) {
    custom_call_name = kCudnnfMHABmmBmmCallTarget;
    return true;
  }
  return false;
}

bool MatchSoftmaxDropoutBmm(int64_t bmm2_operand_position,
                            HloInstruction* instr, HloInstruction** bmm_1,
                            HloInstruction** bmm_2,
                            HloInstruction** softmax_input,
                            double& dropout_rate, HloInstruction** dropout,
                            std::string& custom_call_name) {
  // Matches the softmax-dropout subpattern.
  // Softmax_output can be coming from either a divide or a custom_call
  auto dropout_softmax_pattern =
      m::Select(
          OptionalBroadcast(m::Op().WithOperand(
              0, m::Compare(m::Op(), m::Op())
                     .WithComparisonDirection(ComparisonDirection::kLt))),
          OptionalConvert(m::MultiplyAnyOrder(
              OptionalReshape(OptionalConvert(
                  GetUnfusedReduceMaxSumSoftmaxPattern(softmax_input))),
              m::Broadcast(m::Constant(dropout).WithPredicate(IsScalar)))),
          m::Op())
          .WithOneUse();

  // Try matching BMM1 - (Scale) - (Bias) - (Mask) - Softmax - (Dropout) -
  // BMM2 Dropout with non-zero drop rate has select(divide(softmax_output,
  // broadcast(1-dropout_rate)))
  auto softmax_dropout_bmm2_pattern =
      m::Op(bmm_2)
          .WithPredicate(IsBatchedMatmul)
          .WithOperand(
              bmm2_operand_position,
              m::AnyOf<HloInstruction>(
                  OptionalReshape(OptionalConvert(
                      GetUnfusedReduceMaxSumSoftmaxPattern(softmax_input))),
                  dropout_softmax_pattern));
  if (!Match(instr, softmax_dropout_bmm2_pattern) ||
      !IsSupportedPrimitiveType((*bmm_2))) {
    return false;
  }
  *bmm_1 = *softmax_input;
  if (*dropout != nullptr) {
    dropout_rate = GetDropoutRateFromHlo(*dropout);
    custom_call_name = kCudnnfMHASoftmaxDropoutCallTarget;
  } else {
    custom_call_name = kCudnnfMHASoftmaxCallTarget;
  }
  return true;
}

bool MatchBmm1UnfusedBiasSoftmaxBmm2(HloInstruction* softmax_input,
                                     HloInstruction** bmm_1,
                                     HloInstruction** bias,
                                     HloInstruction** scale, bool has_dropout,
                                     HloInstruction* dropout,
                                     double& dropout_rate,
                                     std::string& custom_call_name) {
  auto first_bmm_pattern = m::SharedSubpattern(
      m::Op(bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse());
  auto unfused_scaled_bmm_subpattern = m::MultiplyAnyOrder(
      OptionalConvert(first_bmm_pattern),
      OptionalConvert(
          m::Broadcast(m::Constant(scale).WithPredicate(IsScalar))));
  auto pattern =
      m::AddAnyOrder(OptionalConvert(m::AnyOf<HloInstruction>(
                         unfused_scaled_bmm_subpattern, first_bmm_pattern)),
                     m::Op(bias));

  if (Match(softmax_input, pattern)) {
    custom_call_name = has_dropout ? kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget
                                   : kCudnnfMHAScaleBiasSoftmaxCallTarget;
    if (has_dropout) {
      dropout_rate = GetDropoutRateFromHlo(dropout);
    }
    return true;
  }
  return false;
}

bool MatchBmm1ScaleBiasMaskSoftmaxDropoutBmm2(
    HloInstruction* softmax_input, HloInstruction** bmm_1,
    HloInstruction** bias, HloInstruction** scale, HloInstruction** mask,
    bool has_dropout, HloInstruction* dropout, double& dropout_rate,
    std::string& custom_call_name) {
  // This is the subpattern for unfused scaled gemm since cublas
  // doesn't always fuse the scale into alpha.
  auto unfused_scaled_bmm_subpattern = m::SharedSubpattern(m::MultiplyAnyOrder(
      OptionalConvert(m::Op(bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse()),
      m::Broadcast(m::Constant(scale).WithPredicate(IsScalar))));
  auto pattern = OptionalConvert(m::Select(
      m::Op(mask).WithPredicate([](const HloInstruction* instr) {
        return instr->shape().element_type() == PRED;
      }),
      // Match bmm1-scale-bias-mask
      m::AnyOf<HloInstruction>(
          // Scale and bias might or might not be fused with gemm
          m::Op(bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse(),
          OptionalConvert(m::AnyOf<HloInstruction>(
              // Try to match unfused bias
              m::AddAnyOrder(
                  m::Op(bias),
                  m::AnyOf<HloInstruction>(
                      OptionalConvert(m::Op(bmm_1)
                                          .WithPredicate(IsBatchedMatmul)
                                          .WithOneUse()),
                      unfused_scaled_bmm_subpattern)),
              unfused_scaled_bmm_subpattern))),
      m::Op()));

  if (Match(softmax_input, pattern)) {
    if (!IsSupportedPrimitiveType((*bmm_1))) {
      return false;
    }
    if ((*bmm_1)->operands().size() > 2) {
      *bias = GetBiasFromBmm((*bmm_1));
    }

    if (has_dropout) {
      // Found BMM1 - Scale - (bias) - Mask - Softmax - dropout - BMM2
      custom_call_name = (*bias) == nullptr
                             ? kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget
                             : kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget;
      dropout_rate = GetDropoutRateFromHlo(dropout);
    } else {
      // Found BMM1 - Scale - Mask - Softmax - BMM2
      custom_call_name = (*bias) == nullptr
                             ? kCudnnfMHAScaleMaskSoftmaxCallTarget
                             : kCudnnfMHAScaleBiasMaskSoftmaxCallTarget;
    }
    return true;
  }
  return false;
}

// We will try to match all the patterns below:
// BMM1 - Scale - Bias - Mask - Softmax - Dropout - BMM2
// BMM1 - Scale - Mask - Softmax - Dropout - BMM2
// BMM1 - Scale - Bias - Mask - Softmax - BMM2
// BMM1 - Scale - Mask - Softmax - BMM2
// BMM1 - Scale - bias - Softmax - BMM2
// BMM1 - Softmax - Dropout - BMM2
// BMM1 - Softmax - BMM2
// BMM1 - BMM2
// Softmax might have been fused into a custom call, we will capture both the
// fused and unfused cases.
bool MatchMHAPatternsForCanonicalization(
    HloInstruction* instr, HloInstruction** bmm_1, HloInstruction** bmm_2,
    HloInstruction** bias, HloInstruction** mask, HloInstruction** scale,
    double& dropout_rate, std::string& custom_call_name,
    bool& need_canonicalization) {
  // TJ - TODO add backward support

  // We need to match 2 general cases:
  // 1. bmm1 --> (intermediate nodes) --> bmm2 <-- V matrix
  // 2. V matrix --> bmm2 <-- (intermediate nodes) <-- bmm1
  // to determine if we need to canonicalize bmm2.
  // So we go through both of bmm2's operands and see which one matches our
  // desired patterns, if operand 1 consumes them, then we need to canonicalize.
  for (int bmm2_operand_pos : {0, 1}) {
    if (bmm2_operand_pos == 1) {
      need_canonicalization = true;
    }

    if (MatchDefaultBmmBmm(bmm2_operand_pos, instr, bmm_1, bmm_2,
                           custom_call_name)) {
      return true;
    }

    HloInstruction* softmax_input = nullptr;

    HloInstruction* dropout = nullptr;

    bool has_dropout = false;
    // We first check if bmm2 is connect to a softmax or dropout.
    // If so, we set softmax input and dropout nodes to their corresponding ops.
    if (!MatchSoftmaxDropoutBmm(bmm2_operand_pos, instr, bmm_1, bmm_2,
                                &softmax_input, dropout_rate, &dropout,
                                custom_call_name)) {
      continue;
    }
    has_dropout = dropout != nullptr;
    if (MatchBmm1UnfusedBiasSoftmaxBmm2(softmax_input, bmm_1, bias, scale,
                                        has_dropout, dropout, dropout_rate,
                                        custom_call_name)) {
      return true;
    }

    if (MatchBmm1ScaleBiasMaskSoftmaxDropoutBmm2(
            softmax_input, bmm_1, bias, scale, mask, has_dropout, dropout,
            dropout_rate, custom_call_name)) {
      return true;
    }

    // If the execution has reached this point, it means that BMMSoftmaxBMM or
    // BMMSoftmaxDropoutBMM pattern has been matched. Hence, returning true.
    return true;
    // continue;
  }
  // Didn't find any match
  need_canonicalization = false;
  return false;
}

StatusOr<bool> IsMHABlockSupported(HloInstruction* bmm_1, HloInstruction* bmm_2,
                                   bool need_canonicalization) {
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
  TF_ASSIGN_OR_RETURN(bool is_bmm2_supported,
                      IsSupportedBMM2(bmm_2, need_canonicalization));
  if (!is_bmm2_supported) return false;
  return true;
}

StatusOr<HloInstruction*> CanonicalizeBatchedGemmForcuDNNFMHA(
    HloInstruction* bmm_2, HloComputation* comp) {
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Before FMHA Dot Cannonicalization: \n"
            << comp->parent()->ToString();
  }

  HloInstruction* lhs_bmm2 = bmm_2->mutable_operand(0);
  HloInstruction* rhs_bmm2 = bmm_2->mutable_operand(1);
  const DotDimensionNumbers& dnums = bmm_2->dot_dimension_numbers();
  int64_t rank = bmm_2->shape().dimensions_size();
  std::vector<int64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  // Swap the non-contracting dims of BMM2 shape. By contract, the
  // non-contracting dims in the output are the last two dimensions.
  std::swap(perm[rank - 1], perm[rank - 2]);

  DotDimensionNumbers new_dnums = dnums;
  std::swap(*new_dnums.mutable_lhs_contracting_dimensions(),
            *new_dnums.mutable_rhs_contracting_dimensions());
  std::swap(*new_dnums.mutable_lhs_batch_dimensions(),
            *new_dnums.mutable_rhs_batch_dimensions());
  const Shape& original_bmm2_shape = bmm_2->shape();
  HloInstruction* new_dot = comp->AddInstruction(HloInstruction::CreateDot(
      ShapeUtil::MakeShape(original_bmm2_shape.element_type(),
                           Permute(original_bmm2_shape.dimensions(), perm)),
      /* lhs */ rhs_bmm2, /* rhs */ lhs_bmm2, new_dnums,
      bmm_2->precision_config()));
  new_dot->set_metadata(bmm_2->metadata());

  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_2,
      HloInstruction::CreateTranspose(original_bmm2_shape, new_dot, perm)));
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "After FMHA Dot Cannonicalization: \n"
            << comp->parent()->ToString();
  }
  return new_dot;
}

StatusOr<bool> FuseMultiHeadedAttentionBlock(
    HloComputation* comp, HloInstruction* bmm_1, HloInstruction* bmm_2,
    HloInstruction* bias, HloInstruction* mask, HloInstruction* scale,
    double dropout_rate, std::string& custom_call_name,
    stream_executor::CudaComputeCapability cc) {
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Before CudnnFusedMHARewriter: \n" << comp->parent()->ToString();
  }
  bool changed = false;
  double scale_value = 1.0;

  // Introduce a transpose to make sure rhs contracting dimension is the
  // fastest moving one.
  absl::Span<const int64_t> rhs_minor_to_major_bmm1 =
      bmm_1->operand(1)->shape().layout().minor_to_major();
  const DotDimensionNumbers& dot_dims_bmm1 = bmm_1->dot_dimension_numbers();
  absl::Span<const int64_t> rhs_contracting_dims_bmm1 =
      dot_dims_bmm1.rhs_contracting_dimensions();
  CHECK_EQ(rhs_contracting_dims_bmm1.size(), 1);
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_non_contracting_dim_nums_bmm1,
      GetNonContractingDims(bmm_1->operand(1)->shape(),
                            dot_dims_bmm1.rhs_batch_dimensions(),
                            rhs_contracting_dims_bmm1));
  CHECK_EQ(rhs_non_contracting_dim_nums_bmm1.size(), 1);
  DotDimensionNumbers new_dot_dims_bmm1 = dot_dims_bmm1;
  HloInstruction* rhs_bmm1 = bmm_1->mutable_operand(1);
  // If the contracting dimension of rhs is not the fastest moving
  // dimension, make it so.
  if (rhs_minor_to_major_bmm1[0] != rhs_contracting_dims_bmm1[0]) {
    std::vector<int64_t> perm(bmm_1->shape().dimensions_size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[rhs_contracting_dims_bmm1[0]],
              perm[rhs_minor_to_major_bmm1[0]]);

    new_dot_dims_bmm1.set_rhs_contracting_dimensions(
        0, rhs_non_contracting_dim_nums_bmm1[0]);

    rhs_bmm1 = comp->AddInstruction(
        HloInstruction::CreateTranspose(
            ShapeUtil::MakeShapeWithDenseLayout(
                bmm_1->shape().element_type(),
                Permute(rhs_bmm1->shape().dimensions(), perm),
                rhs_bmm1->shape().layout().minor_to_major()),
            rhs_bmm1, perm),
        &rhs_bmm1->metadata());
  }

  // Introduce a transpose to make sure lhs contracting dimension is the
  // fastest moving one.
  absl::Span<const int64_t> lhs_minor_to_major_bmm1 =
      bmm_1->operand(0)->shape().layout().minor_to_major();
  absl::Span<const int64_t> lhs_contracting_dims_bmm1 =
      dot_dims_bmm1.lhs_contracting_dimensions();
  CHECK_EQ(lhs_contracting_dims_bmm1.size(), 1);
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> lhs_non_contracting_dim_nums_bmm1,
      GetNonContractingDims(bmm_1->operand(0)->shape(),
                            dot_dims_bmm1.lhs_batch_dimensions(),
                            lhs_contracting_dims_bmm1));
  CHECK_EQ(lhs_non_contracting_dim_nums_bmm1.size(), 1);
  HloInstruction* lhs_bmm1 = bmm_1->mutable_operand(0);
  // If the contracting dimension of lhs is not the fastest moving
  // dimension, make it so.
  if (lhs_minor_to_major_bmm1[0] != lhs_contracting_dims_bmm1[0]) {
    std::vector<int64_t> perm(bmm_1->shape().dimensions_size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[lhs_contracting_dims_bmm1[0]],
              perm[lhs_minor_to_major_bmm1[0]]);

    new_dot_dims_bmm1.set_lhs_contracting_dimensions(
        0, lhs_non_contracting_dim_nums_bmm1[0]);

    lhs_bmm1 = comp->AddInstruction(
        HloInstruction::CreateTranspose(
            ShapeUtil::MakeShapeWithDenseLayout(
                bmm_1->shape().element_type(),
                Permute(lhs_bmm1->shape().dimensions(), perm),
                rhs_bmm1->shape().layout().minor_to_major()),
            lhs_bmm1, perm),
        &lhs_bmm1->metadata());
  }

  // Introduce a transpose to make sure rhs non-contracting dimension  for
  // BMM2 is the fastest moving one.
  absl::Span<const int64_t> rhs_minor_to_major_bmm2 =
      bmm_2->operand(1)->shape().layout().minor_to_major();
  const DotDimensionNumbers& dot_dims_bmm2 = bmm_2->dot_dimension_numbers();
  absl::Span<const int64_t> rhs_contracting_dims_bmm2 =
      dot_dims_bmm2.rhs_contracting_dimensions();
  CHECK_EQ(rhs_contracting_dims_bmm2.size(), 1);
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_non_contracting_dim_nums_bmm2,
      GetNonContractingDims(bmm_2->operand(1)->shape(),
                            dot_dims_bmm2.rhs_batch_dimensions(),
                            rhs_contracting_dims_bmm2));
  CHECK_EQ(rhs_non_contracting_dim_nums_bmm2.size(), 1);
  DotDimensionNumbers new_dot_dims_bmm2 = dot_dims_bmm2;
  HloInstruction* rhs_bmm2 = bmm_2->mutable_operand(1);
  // If the non-contracting dimension of rhs is not the fastest moving
  // dimension, make it so.
  if (rhs_minor_to_major_bmm2[0] != rhs_non_contracting_dim_nums_bmm2[0]) {
    int64_t new_rhs_contracting_dim = rhs_non_contracting_dim_nums_bmm2[0];
    std::vector<int64_t> perm(bmm_2->shape().dimensions_size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[rhs_non_contracting_dim_nums_bmm2[0]],
              perm[rhs_minor_to_major_bmm2[0]]);

    new_dot_dims_bmm2.set_rhs_contracting_dimensions(0,
                                                     new_rhs_contracting_dim);

    rhs_bmm2 = comp->AddInstruction(
        HloInstruction::CreateTranspose(
            ShapeUtil::MakeShapeWithDenseLayout(
                bmm_2->shape().element_type(),
                Permute(rhs_bmm2->shape().dimensions(), perm),
                rhs_bmm2->shape().layout().minor_to_major()),
            rhs_bmm2, perm),
        &rhs_bmm2->metadata());
  }

  CudnnfMHABackendConfig fmha_config;
  *fmha_config.mutable_bmm1_dot_dimension_numbers() = new_dot_dims_bmm1;
  *fmha_config.mutable_bmm2_dot_dimension_numbers() = new_dot_dims_bmm2;

  TF_RET_CHECK((dropout_rate >= 0.0 && dropout_rate <= 1.0));

  // If scale node is assigned, extract value from it.
  if (scale != nullptr) {
    std::optional<double> value;
    value = GetConstantValue(scale);
    TF_RET_CHECK(value.has_value());
    scale_value = (double)*value;
  }

  fmha_config.set_fmha_scale(scale_value);
  fmha_config.set_dropout_rate(dropout_rate);
  // Set to an arbitrary seed for now, seed is not exposed to XLA in HLO
  // graph.
  // TODO Find a way to compute original seed from dropout keys.
  fmha_config.set_seed(42);

  *fmha_config.mutable_intermediate_tensor_shape() = bmm_1->shape().ToProto();
  {
    auto* algorithm = fmha_config.mutable_algorithm();
    algorithm->set_algo_id(0);  // engine id
    algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
    std::vector<int64_t> knob_ids = /* {0, 1} */ {17, 24};
    std::vector<int64_t> knob_vals = {1, 0};
    for (int i = 0; i < knob_ids.size(); ++i) {
      (*algorithm->mutable_tuning_knobs())[knob_ids[i]] = knob_vals[i];
    }
    algorithm->set_is_cudnn_frontend(true);
    algorithm->mutable_workspace_size()->set_value(0);
  }
  const Shape& output_shape = bmm_2->shape();
  Shape call_shape =
      ShapeUtil::MakeTupleShape({output_shape, ShapeUtil::MakeShape(U8, {0})});

  std::vector<HloInstruction*> operands = {lhs_bmm1, rhs_bmm1, rhs_bmm2};
  if (mask != nullptr) {
    HloInstruction* converted_mask = comp->AddInstruction(
        HloInstruction::CreateConvert(bmm_1->shape(), mask));
    operands.push_back(converted_mask);
  }
  if (bias != nullptr) {
    HloInstruction* original_bias;
    HloInstruction* original_broadcast;
    // There will be cases where the bias is up-casted to wider float type,
    // we need to take the original bias node and broadcast it without
    // converting.
    if (Match(bias, m::Broadcast(
                        &original_broadcast,
                        m::Convert(
                            m::Op(&original_bias)
                                .WithPredicate([](const HloInstruction* instr) {
                                  return instr->shape().element_type() == F16 ||
                                         instr->shape().element_type() == BF16;
                                }))
                            .WithPredicate([](const HloInstruction* instr) {
                              return instr->shape().element_type() == F32 ||
                                     instr->shape().element_type() == F64;
                            })))) {
      Shape bcast_shape = bmm_1->shape();
      bias = comp->AddInstruction(HloInstruction::CreateBroadcast(
          bcast_shape, original_bias,
          (DynCast<HloBroadcastInstruction>(original_broadcast))
              ->dimensions()));
    }
    operands.push_back(bias);
  }
  HloInstruction* fmha_call =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, operands, absl::string_view(custom_call_name)));
  TF_RETURN_IF_ERROR(fmha_call->set_backend_config(fmha_config));
  TF_RETURN_IF_ERROR(SetName(bmm_1->GetModule(), fmha_call));
  fmha_call->set_metadata(bmm_1->metadata());

  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_2,
      HloInstruction::CreateGetTupleElement(bmm_2->shape(), fmha_call, 0)));
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "After CudnnFusedMHARewriter: \n" << comp->parent()->ToString();
  }
  changed = true;
  return changed;
}
}  // namespace

StatusOr<bool> CudnnFusedMHARewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    const DebugOptions& debug_options =
        comp->parent()->config().debug_options();
    if (!debug_options.xla_gpu_enable_cudnn_fmha() ||
        !IsComputeCapabilityAndCudnnSupported(
            compute_capability_, cudnn_version_, stream_executor_)) {
      return false;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      HloInstruction* bmm_1;
      HloInstruction* bmm_2;
      // All of the below instructions are optional
      HloInstruction* bias = nullptr;
      HloInstruction* mask = nullptr;
      HloInstruction* scale = nullptr;
      double dropout_rate = 0.0f;
      std::string custom_call_name;
      bool need_canonicalization = false;

      if (!MatchMHAPatternsForCanonicalization(
              instr, &bmm_1, &bmm_2, &bias, &mask, &scale, dropout_rate,
              custom_call_name, need_canonicalization)) {
        continue;
      }

      // move constraint check before canonicalization so we don't modify the
      // graph if mha fusion is not possible
      TF_ASSIGN_OR_RETURN(
          bool is_mha_module_supported,
          IsMHABlockSupported(bmm_1, bmm_2, need_canonicalization));
      if (!is_mha_module_supported) continue;

      // If we need to canonicalize the bmm, we will assign the newly
      // canonicalized bmm to bmm_2.
      if (need_canonicalization) {
        TF_ASSIGN_OR_RETURN(bmm_2,
                            CanonicalizeBatchedGemmForcuDNNFMHA(bmm_2, comp));
      }

      bool changed = false;
      // Fuse the bmms and intermediate nodes into fMHA call, the fused call
      // will replace bmm_2.
      TF_ASSIGN_OR_RETURN(
          changed, FuseMultiHeadedAttentionBlock(
                       comp, bmm_1, bmm_2, bias, mask, scale, dropout_rate,
                       custom_call_name, compute_capability_));
      any_changed |= changed;
    }
  }

  return any_changed;
}
}  // namespace gpu
}  // namespace xla
