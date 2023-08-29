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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {
namespace m = match;

// A struct that contains all the matched nodes
// and results from pattern matching forward graph
struct MatchFwdResult {
  HloInstruction* matched_bmm_1 = nullptr;
  HloInstruction* matched_bmm_2 = nullptr;
  HloInstruction* matched_bias = nullptr;
  HloInstruction* matched_mask = nullptr;
  HloInstruction* matched_scale = nullptr;
  HloInstruction* matched_softmax_input = nullptr;

  double matched_dropout_rate = 0.0;
  bool need_canonicalization = false;
  bool is_training = false;
  bool has_match = false;
  std::string matched_custom_call_name;
};

// A struct that contains all the matched nodes
// and results from pattern matching backward graph
struct MatchBwdResult {
  HloInstruction* matched_bmm_1_grad_1 = nullptr;
  HloInstruction* matched_bmm_1_grad_2 = nullptr;

  HloInstruction* matched_bmm_2_grad_1 = nullptr;
  HloInstruction* matched_bmm_2_grad_2 = nullptr;
  HloInstruction* matched_d_intermediate = nullptr;
  // We use this to keep track of all gradient bmms that need
  // canonicalization.
  bool bmm_1_grad_1_need_canonicalization = false;
  bool bmm_1_grad_2_need_canonicalization = false;
  bool bmm_2_grad_1_need_canonicalization = false;
  bool bmm_2_grad_2_need_canonicalization = false;

  bool has_match = false;
  std::string matched_custom_call_name;
};

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
auto OptionalBitcast(Pattern pattern) {
  auto shared = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(m::Bitcast(shared), shared);
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

// We need to check if current gemm is sharing a parent node with a forward
// fMHA call because when we match backward gemms, the only way that we can be
// sure this is a backward gemm is to see if it's sharing the same operand with
// forward mha call(i.e Q,K,V,activation tensors). We can also use this function
// to infer if a gemm is a forward fmha gemm or not. We check this by doing a
// BFS of all operands to see if there's any user that is a forward fMHA custom
// call. We continue the traversal for shape ops like bitcast, reshape and
// transpose until we see a forward fmha call or there's no shape ops in path
// which means that current node will never share the same operand with a
// forward fmha call.
bool IsSharingOperandWithFwdMha(HloInstruction* gemm) {
  for (int64_t i = 0; i < gemm->operands().size(); i++) {
    std::queue<HloInstruction*> visit_list;
    visit_list.push(gemm->mutable_operand(i));
    while (!visit_list.empty()) {
      HloInstruction* current_instr = visit_list.front();
      for (auto user : current_instr->users()) {
        switch (user->opcode()) {
          case HloOpcode::kBitcast:
          case HloOpcode::kReshape:
          case HloOpcode::kTranspose: {
            visit_list.push(user);
            break;
          }
          case HloOpcode::kCustomCall: {
            if (IsFwdCustomCallTofMHA(*user)) {
              return true;
            }
          } break;
          default:
            break;
        }
      }
      visit_list.pop();
    }
  }
  return false;
}
// When we reach a gemm instruction, it could be one of the 3 cases:
//   1. one of the 2 gemms in forward fmha call
//   2. one of the 4 gemms in backward fmha call
//   3. gemms of other un-related layers
// 3 can be easily ruled out by the pattern matcher.
// However, 1 and 2 have very similar bmm-bmm structures.
// We need to determine that we exactly match case 1 for forward gemms
// which have below properties:
//    - A batched matmul
//    - None of the operands is a forward fmha call, in which case would make it
//      a backward gemm.
//    - It's not directly or indirectly sharing an operand with any other fmha
//      call, in which case would make it a backward gemm
bool IsFirstFwdMatmul(HloInstruction* gemm) {
  return IsBatchedMatmul(gemm) && !IsFwdCustomCallTofMHA(*gemm->operand(0)) &&
         !IsFwdCustomCallTofMHA(*gemm->operand(1)) &&
         !IsSharingOperandWithFwdMha(gemm);
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
    HloInstruction** softmax_input = nullptr,
    HloInstruction** softmax_reduce_sum = nullptr,
    HloInstruction** softmax_reduce_sum_bcast = nullptr) {
  // The reduce-max part of the softmax
  auto unfused_softmax_max_subpattern = m::SharedSubpattern(m::Subtract(
      m::Op(), m::Broadcast(OptionalConvert(OptionalConvert(
                   m::Op()
                       .WithPredicate(IsReduceMax)
                       .WithOperand(0, OptionalBitcast(OptionalConvert(
                                           m::Op(softmax_input)))))))));
  // The reduce-add part of the softmax
  auto unfused_softmax_sum_subpattern = m::SharedSubpattern(m::Divide(
      OptionalBitcast(m::Exp(unfused_softmax_max_subpattern)),
      m::Broadcast(
          softmax_reduce_sum_bcast,
          OptionalConvert(OptionalConvert(
              m::Op(softmax_reduce_sum)
                  .WithOperand(0, OptionalBitcast(OptionalConvert(
                                      m::Exp(unfused_softmax_max_subpattern))))
                  .WithPredicate(IsReduceSum))))));
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

bool IsComputeCapabilityAndCudnnSupported(
    stream_executor::CudaComputeCapability cc,
    stream_executor::dnn::VersionInfo cudnn_version,
    stream_executor::StreamExecutor* stream_exec,
    stream_executor::dnn::VersionInfo supported_cudnn_version) {
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
        (real_cudnn_version >= supported_cudnn_version))) {
    VLOG(2) << absl::StrFormat(
        "CudnnFusedMHARewriter did not run. Unsupported compute "
        "capability(==8.0) or cudnn version(>=%d.%d.%d)",
        supported_cudnn_version.major_version(),
        supported_cudnn_version.minor_version(),
        supported_cudnn_version.patch());
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

std::vector<int64_t> GetDimensionVector(absl::Span<const int64_t> dimensions,
                                        absl::Span<const int64_t> dim_nums) {
  std::vector<int64_t> vec(dim_nums.size());
  for (int i = 0; i < dim_nums.size(); i++) {
    vec[i] = dimensions.at(dim_nums.at(i));
  }
  return vec;
}

StatusOr<bool> IsSupportedBMM1(const HloInstruction* bmm_1) {
  const DotDimensionNumbers& dot_dims_bmm1 = bmm_1->dot_dimension_numbers();
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
  const DotDimensionNumbers& dot_dims_bmm2 = bmm_2->dot_dimension_numbers();
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

MatchFwdResult MatchDefaultFwdBmmBmm(MatchFwdResult previous_result,
                                     int64_t bmm2_operand_position,
                                     HloInstruction* instr) {
  MatchFwdResult match_result = previous_result;
  // Try matching default bmm1-bmm2 pattern
  HloInstruction* bmm_1;
  HloInstruction* bmm_2;

  auto default_bmm_bmm_pattern =
      m::Op(&bmm_2)
          .WithPredicate(IsBatchedMatmul)
          .WithOperand(bmm2_operand_position,
                       m::Op(&bmm_1).WithPredicate(IsBatchedMatmul));

  // If any of bmm1's operands is coming from a forward fMHA call, then return
  // false
  if (Match(instr, default_bmm_bmm_pattern) && IsFirstFwdMatmul(bmm_1)) {
    match_result.matched_bmm_1 = bmm_1;
    match_result.matched_bmm_2 = bmm_2;
    // In training mode, the forward fmha call needs to output an activation
    // to backward graph. In the case of bmm-bmm pattern, if the first bmm
    // has 2 users namely:
    //    1. the second forward bmm
    //    2. one of the backward bmms(activation)
    // then we know this is a training graph, otherwise it's an inference graph.
    match_result.is_training = bmm_1->user_count() == 2;
    match_result.has_match = true;
    match_result.matched_custom_call_name = kCudnnfMHABmmBmmCallTarget;
  }
  return match_result;
}

MatchFwdResult MatchSoftmaxDropoutBmm(MatchFwdResult previous_result,
                                      int64_t bmm2_operand_position,
                                      HloInstruction* instr) {
  // Matches the dropout-softmax subpattern.
  // Softmax_output is a divide
  // Dropout can take multiple forms, we capture 2 forms here based on
  // heurustics Form 1 -> softmax - mul - select(dropout) - BMM2
  MatchFwdResult match_result = previous_result;
  HloInstruction* softmax_reduce_sum;
  HloInstruction* softmax_reduce_sum_bcast;
  HloInstruction* bmm_2;
  HloInstruction* softmax_input;
  HloInstruction* dropout = nullptr;
  auto dropout_softmax_pattern_form_1 = m::Select(
      m::Op(),
      OptionalConvert(m::MultiplyAnyOrder(
          OptionalBitcast(OptionalReshape(
              OptionalConvert(GetUnfusedReduceMaxSumSoftmaxPattern(
                  &softmax_input, &softmax_reduce_sum,
                  &softmax_reduce_sum_bcast)))),
          m::Broadcast(
              OptionalConvert(m::Constant(&dropout).WithPredicate(IsScalar))))),
      m::Op());

  // Form 2 -> softmax - mul - BMM2
  //                     /
  //                    /
  //                 select(dropout)
  auto dropout_softmax_pattern_form_2 =
      OptionalBitcast(OptionalBitcast(OptionalConvert(m::MultiplyAnyOrder(
          OptionalReshape(OptionalConvert(GetUnfusedReduceMaxSumSoftmaxPattern(
              &softmax_input, &softmax_reduce_sum, &softmax_reduce_sum_bcast))),
          m::Broadcast(
              OptionalConvert(OptionalBitcast(OptionalReshape(m::Select(
                  m::Op(),
                  m::Broadcast(m::Constant(&dropout).WithPredicate(IsScalar)),
                  m::Op())))))))));

  // Try matching BMM1 - (Scale) - (Bias) - (Mask) - Softmax - (Dropout) -
  // BMM2 Dropout with non-zero drop rate has select(divide(softmax_output,
  // broadcast(1-dropout_rate)))
  auto softmax_dropout_bmm2_pattern =
      m::Op(&bmm_2)
          .WithPredicate(IsBatchedMatmul)
          .WithOperand(bmm2_operand_position,
                       m::AnyOf<HloInstruction>(
                           OptionalBitcast(OptionalConvert(
                               GetUnfusedReduceMaxSumSoftmaxPattern(
                                   &softmax_input, &softmax_reduce_sum,
                                   &softmax_reduce_sum_bcast))),
                           dropout_softmax_pattern_form_1,
                           dropout_softmax_pattern_form_2));

  if (!Match(instr, softmax_dropout_bmm2_pattern) ||
      !IsSupportedPrimitiveType(bmm_2)) {
    match_result.has_match = false;
    return match_result;
  }
  if (softmax_reduce_sum->users()[0]->opcode() == HloOpcode::kConvert) {
    softmax_reduce_sum = softmax_reduce_sum->users()[0];
  }
  match_result.is_training = softmax_reduce_sum->user_count() == 2 &&
                             softmax_reduce_sum_bcast->user_count() == 2;
  match_result.matched_bmm_2 = bmm_2;
  if (dropout) {
    match_result.matched_dropout_rate = GetDropoutRateFromHlo(dropout);
  }
  match_result.matched_softmax_input = softmax_input;
  match_result.has_match = true;
  return match_result;
}

MatchFwdResult MatchBmm1UnfusedBiasSoftmaxBmm2(MatchFwdResult previous_result,
                                               HloInstruction* softmax_input,
                                               bool has_dropout) {
  MatchFwdResult match_result = previous_result;
  HloInstruction* bmm_1;
  HloInstruction* bias = nullptr;
  HloInstruction* scale = nullptr;

  auto first_bmm_pattern =
      m::SharedSubpattern(m::Op(&bmm_1).WithPredicate(IsBatchedMatmul));
  auto unfused_scaled_bmm_subpattern = m::MultiplyAnyOrder(
      OptionalConvert(first_bmm_pattern),
      OptionalConvert(
          m::Broadcast(m::Constant(&scale).WithPredicate(IsScalar))));

  if (Match(softmax_input,
            OptionalConvert(OptionalBitcast(first_bmm_pattern)))) {
    match_result.matched_bmm_1 = bmm_1;
    match_result.matched_custom_call_name =
        has_dropout ? kCudnnfMHASoftmaxDropoutCallTarget
                    : kCudnnfMHASoftmaxCallTarget;
    match_result.has_match = true;
  } else if (Match(softmax_input,
                   OptionalBitcast(m::AddAnyOrder(
                       OptionalConvert(OptionalBitcast(m::AnyOf<HloInstruction>(
                           unfused_scaled_bmm_subpattern, first_bmm_pattern))),
                       m::Op(&bias))))) {
    match_result.matched_bmm_1 = bmm_1;
    match_result.matched_scale = scale;
    match_result.matched_bias = bias;
    match_result.matched_custom_call_name =
        has_dropout ? kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget
                    : kCudnnfMHAScaleBiasSoftmaxCallTarget;
    match_result.has_match = true;
  } else {
    match_result.has_match = false;
  }
  return match_result;
}

MatchFwdResult MatchBmm1ScaleBiasMaskSoftmaxDropoutBmm2(
    MatchFwdResult previous_result, HloInstruction* softmax_input,
    bool has_dropout) {
  MatchFwdResult matched_result = previous_result;
  HloInstruction* bmm_1;
  HloInstruction* bias = nullptr;
  HloInstruction* scale = nullptr;
  HloInstruction* mask = nullptr;

  // This is the subpattern for unfused scaled gemm since cublas
  // doesn't always fuse the scale into alpha.
  auto unfused_scaled_bmm_subpattern = m::SharedSubpattern(m::MultiplyAnyOrder(
      OptionalConvert(
          m::Op(&bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse()),
      m::Broadcast(m::Constant(&scale).WithPredicate(IsScalar))));

  if (Match(
          softmax_input,
          OptionalConvert(m::Select(
              m::Op(&mask).WithPredicate([](const HloInstruction* instr) {
                return instr->shape().element_type() == PRED;
              }),
              // Match bmm1-scale-bias-mask
              m::AnyOf<HloInstruction>(
                  // Scale and bias might or might not be fused
                  // with gemm
                  m::Op(&bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse(),
                  OptionalConvert(m::AnyOf<HloInstruction>(
                      // Try to match unfused bias
                      m::AddAnyOrder(m::Op(&bias),
                                     m::AnyOf<HloInstruction>(
                                         OptionalConvert(
                                             m::Op(&bmm_1)
                                                 .WithPredicate(IsBatchedMatmul)
                                                 .WithOneUse()),
                                         unfused_scaled_bmm_subpattern)),
                      unfused_scaled_bmm_subpattern))),
              m::Op())))) {
    if (!IsSupportedPrimitiveType(bmm_1)) {
      matched_result.has_match = false;
      return matched_result;
    }

    if (has_dropout) {
      // Found BMM1 - Scale - (bias) - Mask - Softmax - dropout - BMM2
      matched_result.matched_custom_call_name =
          bias == nullptr ? kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget
                          : kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget;
    } else {
      // Found BMM1 - Scale - Mask - Softmax - BMM2
      matched_result.matched_custom_call_name =
          bias == nullptr ? kCudnnfMHAScaleMaskSoftmaxCallTarget
                          : kCudnnfMHAScaleBiasMaskSoftmaxCallTarget;
    }
    matched_result.matched_bmm_1 = bmm_1;
    matched_result.matched_scale = scale;
    matched_result.matched_mask = mask;
    matched_result.matched_bias = bias;
    matched_result.has_match = true;
  } else {
    matched_result.has_match = false;
  }
  return matched_result;
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
MatchFwdResult MatchFwdMHAPatternsForCanonicalization(HloInstruction* instr) {
  // We need to match 2 general cases:
  // 1. bmm1 --> (intermediate nodes) --> bmm2 <-- V matrix
  // 2. V matrix --> bmm2 <-- (intermediate nodes) <-- bmm1
  // to determine if we need to canonicalize bmm2.
  // So we go through both of bmm2's operands and see which one matches our
  // desired patterns, if operand 1 consumes them, then we need to canonicalize.
  MatchFwdResult match_result;
  for (auto bmm2_operand_pos : {0, 1}) {
    if (bmm2_operand_pos == 1) {
      match_result.need_canonicalization = true;
    }
    match_result = MatchDefaultFwdBmmBmm(match_result, bmm2_operand_pos, instr);
    if (match_result.has_match) {
      return match_result;
    }

    bool has_dropout = false;
    // We first check if bmm2 is connect to a softmax or dropout.
    // If so, we set softmax input and dropout rate to their corresponding
    // values.
    match_result =
        MatchSoftmaxDropoutBmm(match_result, bmm2_operand_pos, instr);
    if (!match_result.has_match) {
      continue;
    }
    has_dropout = match_result.matched_dropout_rate > 0.0;
    match_result = MatchBmm1UnfusedBiasSoftmaxBmm2(
        match_result, match_result.matched_softmax_input, has_dropout);
    if (match_result.has_match) {
      return match_result;
    }
    match_result = MatchBmm1ScaleBiasMaskSoftmaxDropoutBmm2(
        match_result, match_result.matched_softmax_input, has_dropout);
    if (match_result.has_match) {
      return match_result;
    }
  }
  // Didn't find any match
  match_result.need_canonicalization = false;
  return match_result;
}

bool IsBmm2GradGemm2(HloInstruction* instr) {
  // Check to see if input bmm is bmm2 gradient gemm2, it needs to be either:
  // 1. having 1 user in cases of dropout
  // 2. having 2 users in other cases.
  return (instr->user_count() == 1) || (instr->user_count() == 2);
}

MatchBwdResult MatchBmm1GradGemm1(MatchBwdResult previous_result,
                                  HloInstruction* fwd_fmha_call,
                                  HloInstruction* bmm_1) {
  MatchBwdResult match_result = previous_result;
  match_result.has_match = false;
  const HloInstruction* q_tensor = fwd_fmha_call->operand(0);
  for (int64_t i = 0; i < q_tensor->user_count(); i++) {
    HloInstruction* q_tensor_user_i = q_tensor->users()[i];
    if (IsBatchedMatmul(q_tensor_user_i) && q_tensor_user_i != bmm_1) {
      match_result.matched_bmm_1_grad_1 = q_tensor_user_i;
      // Check for canonicalization.
      if (match_result.matched_bmm_1_grad_1->operand_index(q_tensor) != 1) {
        match_result.bmm_1_grad_1_need_canonicalization = true;
      }
      match_result.has_match = true;
    }
  }
  return match_result;
}

MatchBwdResult MatchBmm1GradGemm2(MatchBwdResult previous_result,
                                  HloInstruction* fwd_fmha_call) {
  HloInstruction* bmm_1_grad_2 = nullptr;
  MatchBwdResult match_result = previous_result;
  match_result.has_match = false;
  // bmm1 gradient gemm2 shares the same input as bmm1 gradient gemm1.
  // Check to see if bmm1 grad gemm1 needs canonicalization or not, if not,
  // then the shared input is the first operand.
  int64_t parent_nodex_index =
      match_result.bmm_1_grad_1_need_canonicalization ? 1 : 0;
  HloInstruction* d_s_user_0 = match_result.matched_bmm_1_grad_1;

  HloInstruction* parent_node = d_s_user_0->mutable_operand(parent_nodex_index);
  if (parent_node->opcode() == HloOpcode::kBitcast &&
      parent_node->user_count() == 1) {
    d_s_user_0 = parent_node;
    parent_node = parent_node->mutable_operand(0);
  }

  auto bmm_1_grad_2_it =
      std::find_if(parent_node->users().begin(), parent_node->users().end(),
                   [&](HloInstruction* instr) {
                     return instr != match_result.matched_bmm_1_grad_1 &&
                            instr->opcode() != HloOpcode::kReduce;
                   });
  if (bmm_1_grad_2_it != parent_node->users().end()) {
    bmm_1_grad_2 = *bmm_1_grad_2_it;
  } else {
    return match_result;
  }
  if (bmm_1_grad_2->opcode() == HloOpcode::kBitcast &&
      bmm_1_grad_2->user_count() == 1) {
    parent_node = bmm_1_grad_2;
    bmm_1_grad_2 = bmm_1_grad_2->users()[0];
  }

  match_result.matched_bmm_1_grad_2 = bmm_1_grad_2;

  if (match_result.matched_bmm_1_grad_2->operand_index(parent_node) != 0) {
    match_result.bmm_1_grad_2_need_canonicalization = true;
  }
  match_result.has_match = true;
  return match_result;
}

MatchBwdResult MatchBmm2GradGemm1(HloInstruction* fwd_fmha_call) {
  HloInstruction* bmm_2_grad_1 = nullptr;
  MatchBwdResult matched_result;
  // The second GTE of the forward MHA call is the input of the bmm2's gradient
  // gemm 1, we check to see if the current gemm satisfies above condition.
  int64_t activation_out_gte_index = 1;
  if (fwd_fmha_call->user_count() < 2 ||
      fwd_fmha_call->users()[activation_out_gte_index]->opcode() !=
          HloOpcode::kGetTupleElement ||
      fwd_fmha_call->users()[activation_out_gte_index]->user_count() > 1 ||
      !IsBatchedMatmul(
          fwd_fmha_call->users()[activation_out_gte_index]->users()[0])) {
    matched_result.has_match = false;
    return matched_result;
  }
  // Found fmha->GTE->gemm, assign it to bmm_2_grad_1 and check to see if it
  // needs canonicalization.
  bmm_2_grad_1 = fwd_fmha_call->users()[activation_out_gte_index]->users()[0];
  matched_result.matched_bmm_2_grad_1 = bmm_2_grad_1;
  if (bmm_2_grad_1->operand_index(
          fwd_fmha_call->users()[activation_out_gte_index]) != 0) {
    matched_result.bmm_2_grad_1_need_canonicalization = true;
  }

  matched_result.has_match = true;
  return matched_result;
}

MatchBwdResult MatchBmm2GradGemm2(MatchBwdResult previous_result,
                                  HloInstruction* fwd_fmha_call,
                                  bool v_transposed) {
  MatchBwdResult match_result = previous_result;
  match_result.has_match = false;
  // If v tensor is transposed by forward fmha call, then we need to take fmha v
  // input's producer's producer.
  const HloInstruction* v_tensor = v_transposed
                                       ? fwd_fmha_call->operand(2)->operand(0)
                                       : fwd_fmha_call->operand(2);
  for (int64_t i = 0; i < v_tensor->user_count(); i++) {
    HloInstruction* v_tensor_user_i = v_tensor->users()[i];
    if (IsBatchedMatmul(v_tensor_user_i) && IsBmm2GradGemm2(v_tensor_user_i)) {
      match_result.matched_bmm_2_grad_2 = v_tensor_user_i;
      // Check for canonicalization.
      if (match_result.matched_bmm_2_grad_2->operand_index(v_tensor) != 1) {
        match_result.bmm_2_grad_2_need_canonicalization = true;
      }
      match_result.has_match = true;
    }
  }
  return match_result;
}

MatchBwdResult MatchBwdBmmSoftmaxDropoutBmm(MatchBwdResult previous_result,
                                            HloInstruction* fwd_fmha_call,
                                            HloInstruction* mask) {
  MatchBwdResult match_result = previous_result;
  bool is_bmm1_grad1_canonicalized =
      match_result.bmm_1_grad_1_need_canonicalization;
  match_result.has_match = false;
  bool has_dropout = false;
  bool has_mask = false;
  // Backward dropout pattern
  // select(mask, bmm2_grad2, broadcast())
  auto bwd_dropout_pattern_form_1 = m::SharedSubpattern(
      OptionalBitcast(OptionalReshape(OptionalConvert(m::Select(
          m::Op(), m::Op().WithPredicate([&](const HloInstruction* instr) {
            return instr == match_result.matched_bmm_2_grad_2;
          }),
          m::Broadcast(
              OptionalConvert(m::Constant().WithPredicate(IsScalar))))))));

  // multiply(bmm2_grad2, broadcast(select(mask, broadcast(), op())))
  auto bwd_dropout_pattern_form_2 =
      m::SharedSubpattern(OptionalBitcast(m::MultiplyAnyOrder(
          OptionalConvert(
              m::Op().WithPredicate([&](const HloInstruction* instr) {
                return instr == match_result.matched_bmm_2_grad_2;
              })),
          m::Broadcast(OptionalConvert(OptionalBitcast(OptionalReshape(
              m::Select(m::Op(),
                        m::Broadcast(OptionalConvert(
                            m::Constant().WithPredicate(IsScalar))),
                        m::Op()))))))));
  auto bwd_dropout_pattern = m::AnyOf<HloInstruction>(
      bwd_dropout_pattern_form_1, bwd_dropout_pattern_form_2);
  // Backward softmax pattern
  HloInstruction* bwd_softmax_input = nullptr;
  HloInstruction* exp_1;
  HloInstruction* exp_2;
  HloInstruction* d_softmax;

  auto bwd_softmax_pattern =
      OptionalBitcast(OptionalConvert(m::MultiplyAnyOrder(
          &d_softmax,
          m::AddAnyOrder(
              m::Divide(),
              m::Broadcast(OptionalBitcast(
                  OptionalConvert(OptionalConvert(m::Negate(OptionalBitcast(
                      m::Op()
                          .WithPredicate(IsReduceSum)
                          .WithOperand(0, OptionalBitcast(m::MultiplyAnyOrder(
                                              m::MultiplyAnyOrder(
                                                  m::Op(&bwd_softmax_input),
                                                  m::Broadcast()),
                                              m::Exp(&exp_2, m::Op()))))))))))),
          m::Exp(&exp_1, m::Op()))));

  // Backward mask input pattern
  // we already matched this in the fwd. Just make sure the same mask is used in
  // the bwd
  HloInstruction* bwd_mask_input = nullptr;
  HloInstruction* bwd_mask = nullptr;
  auto bwd_mask_pattern = OptionalConvert(
      m::Select(m::Op(&bwd_mask).WithPredicate([](const HloInstruction* instr) {
        return instr->shape().element_type() == PRED;
      }),
                m::Op(&bwd_mask_input), m::Op()));

  // Backward scale input pattern
  HloInstruction* bwd_scale_input = nullptr;

  auto bwd_scale_pattern =
      m::MultiplyAnyOrder(m::Op(&bwd_scale_input),
                          m::Broadcast(m::Constant().WithPredicate(IsScalar)));
  int intermediate_input_pos = is_bmm1_grad1_canonicalized ? 1 : 0;

  HloInstruction* intermediate_input =
      match_result.matched_bmm_1_grad_1->mutable_operand(
          intermediate_input_pos);

  if (Match(intermediate_input, bwd_scale_pattern)) {
    intermediate_input = bwd_scale_input;
  }

  has_mask = Match(intermediate_input, bwd_mask_pattern) && mask == bwd_mask;

  if (has_mask) {
    intermediate_input = bwd_mask_input;
  }
  if (!Match(intermediate_input, bwd_softmax_pattern) || exp_1 != exp_2) {
    return match_result;
  }
  has_dropout = Match(bwd_softmax_input, bwd_dropout_pattern);
  // If no dropout but softmax input is not coming from bmm2 gradient gemm 2,
  // then it's not the pattern that we care about.
  if (!has_dropout &&
      !Match(bwd_softmax_input,
             OptionalConvert((OptionalBitcast(
                 m::Op().WithPredicate([&](const HloInstruction* instr) {
                   return instr == match_result.matched_bmm_2_grad_2;
                 })))))) {
    return match_result;
  }

  if (has_mask && has_dropout) {
    // has bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget)
      match_result.matched_custom_call_name =
          kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget;
    // no bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget)
      match_result.matched_custom_call_name =
          kCudnnfMHAScaleMaskSoftmaxDropoutBackwardCallTarget;
  } else if (!has_mask && has_dropout) {
    // has bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget)
      match_result.matched_custom_call_name =
          kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
    // no bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHASoftmaxDropoutCallTarget)
      match_result.matched_custom_call_name =
          kCudnnfMHASoftmaxDropoutBackwardCallTarget;
  } else if (has_mask && !has_dropout) {
    // has bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleBiasMaskSoftmaxCallTarget)
      match_result.matched_custom_call_name =
          kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget;
    // no bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleMaskSoftmaxCallTarget)
      match_result.matched_custom_call_name =
          kCudnnfMHAScaleMaskSoftmaxBackwardCallTarget;
  } else {
    // has bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleBiasSoftmaxCallTarget)
      match_result.matched_custom_call_name =
          kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget;
    // no bias
    if (fwd_fmha_call->custom_call_target() == kCudnnfMHASoftmaxCallTarget)
      match_result.matched_custom_call_name =
          kCudnnfMHASoftmaxBackwardCallTarget;
  }

  // If d_softmax tensor has 3 consumers, then we need to output the
  // intermediate tensor.
  bool need_d_intermediate = d_softmax->user_count() == 3;
  if ((match_result.matched_custom_call_name ==
           kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget ||
       match_result.matched_custom_call_name ==
           kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget ||
       match_result.matched_custom_call_name ==
           kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget ||
       match_result.matched_custom_call_name ==
           kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget) &&
      need_d_intermediate) {
    match_result.matched_d_intermediate = d_softmax;
  }
  match_result.has_match = true;
  return match_result;
}
// First, we look for the bmm2 gradient gemm 1 which takes the activation
// output from a forward fmha call.
// Secondly, look for bmm2 gradient gemm 2 that takes the v tensor as an
// input. We take the v tensor from the third operand of the forward fmha
// call. If forward is canonicalized, then we skip the additional transpose in
// between.
// Then we look for bmm1 gradient gemm1 by searching for gemms that share q
// tensor with current fmha call.
MatchBwdResult MatchBackwardBmms(HloInstruction* fwd_fmha_call,
                                 HloInstruction* bmm_1, bool v_transposed) {
  MatchBwdResult matched_result = MatchBmm2GradGemm1(fwd_fmha_call);
  if (!matched_result.has_match) {
    return matched_result;
  }

  matched_result =
      MatchBmm2GradGemm2(matched_result, fwd_fmha_call, v_transposed);
  if (!matched_result.has_match) {
    return matched_result;
  }

  matched_result = MatchBmm1GradGemm1(matched_result, fwd_fmha_call, bmm_1);
  if (!matched_result.has_match) {
    return matched_result;
  }

  matched_result = MatchBmm1GradGemm2(matched_result, fwd_fmha_call);
  if (!matched_result.has_match) {
    return matched_result;
  }
  return matched_result;
}
// We will match the backward graphs for all forward patterns defined in
// MatchFwdMHAPatternsForCanonicalization
MatchBwdResult MatchBwdMHAPatternsForCanonicalization(
    HloInstruction* fwd_fmha_call, HloInstruction* bmm_1, HloInstruction* mask,
    bool v_transposed) {
  MatchBwdResult match_result =
      MatchBackwardBmms(fwd_fmha_call, bmm_1, v_transposed);
  if (!match_result.has_match) {
    return match_result;
  }

  // Found default bmm-bmm backward graph.
  if (match_result.matched_bmm_2_grad_2->users().size() == 2 &&
      (match_result.matched_bmm_1_grad_1->IsUserOf(
          match_result.matched_bmm_2_grad_2)) &&
      (match_result.matched_bmm_1_grad_2->IsUserOf(
          match_result.matched_bmm_2_grad_2))) {
    match_result.matched_custom_call_name = kCudnnfMHABmmBmmBackwardCallTarget;
    return match_result;
  }
  // TODO match all other patterns
  match_result =
      MatchBwdBmmSoftmaxDropoutBmm(match_result, fwd_fmha_call, mask);
  return match_result;
}

StatusOr<bool> IsMHABlockSupported(HloInstruction* bmm_1, HloInstruction* bmm_2,
                                   bool need_canonicalization, bool is_training,
                                   std::string& custom_call_name,
                                   const DebugOptions& debug_options) {
  if (MHACallHasDropout(custom_call_name) &&
      !debug_options.xla_gpu_fused_attention_use_cudnn_rng()) {
    VLOG(3) << "Using CUDNN RNG for fused attention dropout is not enabled.\n";
    return false;
  }

  if (is_training &&
      (custom_call_name != kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget &&
       custom_call_name != kCudnnfMHAScaleBiasSoftmaxCallTarget &&
       custom_call_name != kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget &&
       custom_call_name != kCudnnfMHAScaleBiasMaskSoftmaxCallTarget)) {
    VLOG(3) << "Unsupported fused MHA training pattern.\n";
    return false;
  }

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
    HloInstruction* bmm, HloComputation* comp) {
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Before FMHA Dot Cannonicalization: \n"
            << comp->parent()->ToString();
  }
  HloInstruction* lhs_bmm = bmm->mutable_operand(0);
  HloInstruction* rhs_bmm = bmm->mutable_operand(1);
  const DotDimensionNumbers& dnums = bmm->dot_dimension_numbers();

  int64_t rank = bmm->shape().dimensions_size();
  std::vector<int64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  // Swap the non-contracting dims of BMM shape. By contract, the
  // non-contracting dims in the output are the last two dimensions.
  std::swap(perm[rank - 1], perm[rank - 2]);

  DotDimensionNumbers new_dnums = dnums;
  std::swap(*new_dnums.mutable_lhs_contracting_dimensions(),
            *new_dnums.mutable_rhs_contracting_dimensions());
  std::swap(*new_dnums.mutable_lhs_batch_dimensions(),
            *new_dnums.mutable_rhs_batch_dimensions());
  auto original_bmm_shape = bmm->shape();
  HloInstruction* new_dot = comp->AddInstruction(HloInstruction::CreateDot(
      ShapeUtil::MakeShape(original_bmm_shape.element_type(),
                           Permute(original_bmm_shape.dimensions(), perm)),
      /* lhs */ rhs_bmm, /* rhs */ lhs_bmm, new_dnums,
      bmm->precision_config()));

  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm, HloInstruction::CreateTranspose(original_bmm_shape, new_dot, perm)));
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "After FMHA Dot Cannonicalization: \n"
            << comp->parent()->ToString();
  }
  return new_dot;
}

StatusOr<HloInstruction*> ChangeCheckedDimToFastest(
    HloComputation* comp, HloInstruction* bmm, bool is_lhs,
    bool should_contracting_be_fastest) {
  const DotDimensionNumbers& dot_dims_bmm = bmm->dot_dimension_numbers();
  DotDimensionNumbers new_dot_dims_bmm = dot_dims_bmm;
  int64_t bmm_operand = is_lhs ? 0 : 1;
  absl::Span<const int64_t> contracting_dims =
      is_lhs ? dot_dims_bmm.lhs_contracting_dimensions()
             : dot_dims_bmm.rhs_contracting_dimensions();
  absl::Span<const int64_t> batch_dims =
      is_lhs ? dot_dims_bmm.lhs_batch_dimensions()
             : dot_dims_bmm.rhs_batch_dimensions();
  absl::Span<const int64_t> lhs_minor_to_major_bmm =
      bmm->operand(0)->shape().layout().minor_to_major();
  absl::Span<const int64_t> rhs_minor_to_major_bmm =
      bmm->operand(1)->shape().layout().minor_to_major();

  absl::Span<const int64_t>& minor_to_major_to_check =
      is_lhs ? lhs_minor_to_major_bmm : rhs_minor_to_major_bmm;

  CHECK_EQ(contracting_dims.size(), 1);
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> non_contracting_dim_nums_bmm,
                      GetNonContractingDims(bmm->operand(bmm_operand)->shape(),
                                            batch_dims, contracting_dims));
  CHECK_EQ(non_contracting_dim_nums_bmm.size(), 1);
  HloInstruction* operand_bmm = bmm->mutable_operand(bmm_operand);
  std::vector<int64_t> contracting_dims_to_check{contracting_dims[0]};
  std::vector<int64_t> dims_to_set = should_contracting_be_fastest
                                         ? contracting_dims_to_check
                                         : non_contracting_dim_nums_bmm;
  // If the dimension being checked(contracting or non-contracting) of the
  // target operand is not the fastest moving dimension, make it so.
  if (minor_to_major_to_check[0] != dims_to_set[0]) {
    std::vector<int64_t> perm(bmm->shape().dimensions_size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dims_to_set[0]], perm[minor_to_major_to_check[0]]);

    if (is_lhs) {
      new_dot_dims_bmm.set_lhs_contracting_dimensions(
          0, non_contracting_dim_nums_bmm[0]);
    } else {
      new_dot_dims_bmm.set_rhs_contracting_dimensions(
          0, non_contracting_dim_nums_bmm[0]);
    }

    operand_bmm = comp->AddInstruction(
        HloInstruction::CreateTranspose(
            ShapeUtil::MakeShapeWithDenseLayout(
                bmm->shape().element_type(),
                Permute(operand_bmm->shape().dimensions(), perm),
                rhs_minor_to_major_bmm),
            operand_bmm, perm),
        &operand_bmm->metadata());
    *((DynCast<HloDotInstruction>(bmm))->mutable_dot_dimension_numbers()) =
        new_dot_dims_bmm;
  }
  return operand_bmm;
}

StatusOr<HloInstruction*> FuseFwdMultiHeadedAttentionBlock(
    HloComputation* comp, HloInstruction* bmm_1, HloInstruction* bmm_2,
    HloInstruction* bias, HloInstruction* mask, HloInstruction* scale,
    double dropout_rate, std::string& custom_call_name,
    stream_executor::CudaComputeCapability cc, bool is_training, bool& changed,
    bool& v_transposed) {
  double scale_value = 1.0;
  HloInstruction* lhs_bmm1;
  HloInstruction* rhs_bmm1;
  HloInstruction* rhs_bmm2;
  TF_ASSIGN_OR_RETURN(rhs_bmm1, ChangeCheckedDimToFastest(
                                    comp, bmm_1, false /*is_lhs*/,
                                    true /*should_contracting_be_fastest*/));
  TF_ASSIGN_OR_RETURN(lhs_bmm1, ChangeCheckedDimToFastest(
                                    comp, bmm_1, true /*is_lhs*/,
                                    true /*should_contracting_be_fastest*/));

  TF_ASSIGN_OR_RETURN(rhs_bmm2, ChangeCheckedDimToFastest(
                                    comp, bmm_2, false /*is_lhs*/,
                                    false /*should_contracting_be_fastest*/));

  if (rhs_bmm2 != bmm_2->mutable_operand(1)) {
    v_transposed = true;
  }

  CudnnfMHABackendConfig fmha_config;
  *fmha_config.mutable_bmm1_dot_dimension_numbers() =
      bmm_1->dot_dimension_numbers();
  *fmha_config.mutable_bmm2_dot_dimension_numbers() =
      bmm_2->dot_dimension_numbers();

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

  Shape call_shape;
  // Activation output is used by backward gemm.
  HloInstruction* activation_output = nullptr;

  std::vector<Shape> output_shapes = {output_shape,
                                      ShapeUtil::MakeShape(U8, {0})};
  if (is_training) {
    // TODO Flush attention will have a different shape in training.
    activation_output = bmm_2->mutable_operand(0);
    // Sometimes activation output is bitcast, the actual activation is the
    // second user of the producer of bmm_2's first operand.
    if (activation_output->user_count() < 2 &&
        activation_output->opcode() == HloOpcode::kBitcast) {
      HloInstruction* producer = activation_output->mutable_operand(0);
      TF_RET_CHECK(producer->user_count() == 2);
      activation_output = producer->UserId(activation_output) == 0
                              ? producer->users()[1]
                              : producer->users()[0];
    }
    output_shapes.push_back(activation_output->shape());
  }
  call_shape = ShapeUtil::MakeTupleShape(output_shapes);

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
      absl::Span<const int64_t> original_bcast_dims =
          (DynCast<HloBroadcastInstruction>(original_broadcast))->dimensions();
      // This is to deal with cases like paxml where an extra dimension of 1 is
      // added to the left of the tensor.
      // TODO Make this logic more generic
      absl::Span<const int64_t> original_broadcast_shape_dims =
          original_broadcast->shape().dimensions();
      int64_t starting_index = original_broadcast_shape_dims.size() == 5 &&
                                       original_broadcast_shape_dims[0] == 1
                                   ? 1
                                   : 0;
      std::vector<int64_t> bcast_dimensions;
      for (auto& dim : original_bcast_dims) {
        bcast_dimensions.push_back(dim - starting_index);
      }

      Shape bcast_shape = bmm_1->shape();
      bias = comp->AddInstruction(HloInstruction::CreateBroadcast(
          bcast_shape, original_bias, bcast_dimensions));
    }
    operands.push_back(bias);
  }

  HloInstruction* fmha_call =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, operands, absl::string_view(custom_call_name)));
  TF_RETURN_IF_ERROR(fmha_call->set_backend_config(fmha_config));
  TF_RETURN_IF_ERROR(SetFMHAInstructionName(bmm_1->GetModule(), fmha_call));

  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_2,
      HloInstruction::CreateGetTupleElement(bmm_2->shape(), fmha_call, 0)));

  if (activation_output) {
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        activation_output, HloInstruction::CreateGetTupleElement(
                               activation_output->shape(), fmha_call, 2)));
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "After CudnnFusedMHARewriter: \n" << comp->parent()->ToString();
  }
  changed = true;
  return fmha_call;
}

bool IsDbiasOnlyUserBesidesGradGemm(HloInstruction* d_intermediate,
                                    HloInstruction* bmm_1_grad_1,
                                    HloInstruction* bmm_1_grad_2,
                                    HloInstruction** dbias) {
  auto user_count = d_intermediate->user_count();
  HloInstruction* dbias_user = nullptr;
  for (auto user : d_intermediate->users()) {
    if (user == bmm_1_grad_1) {
      user_count -= 1;
    } else if (user == bmm_1_grad_2) {
      user_count -= 1;
    } else {
      dbias_user = user;
    }
  }
  HloInstruction* reduce;
  auto ConsumeExtraConvert = [](HloInstruction** instr) {
    Match((*instr)->users()[0], m::Convert(instr, m::Op()).WithOneUse());
    return true;
  };
  // user_count == 1 && (reduce-> {convert} ->bitcast)
  return user_count == 1 &&
         Match(dbias_user, m::Reduce(&reduce, m::Op(), m::Op()).WithOneUse()) &&
         ConsumeExtraConvert(&reduce) &&
         Match(reduce->users()[0],
               m::AnyOf<HloInstruction>(m::Reshape(dbias, m::Op()),
                                        m::Bitcast(dbias, m::Op()))
                   .WithOneUse());
}

StatusOr<bool> FuseBwdMultiHeadedAttentionBlock(
    HloComputation* comp, HloInstruction* bmm_1_grad_1,
    HloInstruction* bmm_1_grad_2, HloInstruction* bmm_2_grad_1,
    HloInstruction* bmm_2_grad_2, HloInstruction* fwd_fmha_call,
    HloInstruction* d_intermediate, HloInstruction* mask,
    std::string& bwd_custom_call_name, bool fwd_bmm_2_canonicalized,
    bool is_bmm2_grad1_canonicalized) {
  HloInstruction* rhs_bmm1_grad_gemm1;
  HloInstruction* lhs_bmm1_grad_gemm2;
  HloInstruction* lhs_bmm2_grad_gemm1;
  HloInstruction* rhs_bmm2_grad_gemm2;
  HloInstruction* d_output_grad;

  // Q tensor
  TF_ASSIGN_OR_RETURN(
      rhs_bmm1_grad_gemm1,
      ChangeCheckedDimToFastest(comp, bmm_1_grad_1, false /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
  // K tensor
  TF_ASSIGN_OR_RETURN(
      lhs_bmm1_grad_gemm2,
      ChangeCheckedDimToFastest(comp, bmm_1_grad_2, false /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
  // Forward activation
  TF_ASSIGN_OR_RETURN(
      lhs_bmm2_grad_gemm1,
      ChangeCheckedDimToFastest(comp, bmm_2_grad_1, true /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
  // V tensor
  TF_ASSIGN_OR_RETURN(
      rhs_bmm2_grad_gemm2,
      ChangeCheckedDimToFastest(comp, bmm_2_grad_2, false /*is_lhs*/,
                                true /*should_contracting_be_fastest*/));
  // d output
  // Since d_o is the input of 2 bmms, we set the dim number using the
  // constraint
  // -> the contracting dimension of the lhs of bmm_2_grad_2 needs to be the
  // fastest moving dimension.
  TF_ASSIGN_OR_RETURN(d_output_grad, ChangeCheckedDimToFastest(
                                         comp, bmm_2_grad_2, true /*is_lhs*/,
                                         true /*check_contracting_dim*/));
  // Operand order {Q, K, V, Fwd act, d_o, mask*}
  std::vector<HloInstruction*> operands = {
      rhs_bmm1_grad_gemm1, lhs_bmm1_grad_gemm2, rhs_bmm2_grad_gemm2,
      lhs_bmm2_grad_gemm1, d_output_grad};
  if (mask) {
    HloInstruction* converted_mask = comp->AddInstruction(
        HloInstruction::CreateConvert(bmm_2_grad_2->shape(), mask));
    operands.push_back(converted_mask);
  }
  TF_ASSIGN_OR_RETURN(CudnnfMHABackendConfig fwd_config,
                      fwd_fmha_call->backend_config<CudnnfMHABackendConfig>());
  CudnnfMHABackendConfig bwd_fmha_config;

  // If forward bmm_2 is canonicalized, the contracting dimension of lhs
  // of bmm_2_grad_1 needs to be changed to the non-contracting dimension.

  if (fwd_bmm_2_canonicalized) {
    TF_ASSIGN_OR_RETURN(
        std::vector<int64_t> bmm_2_grad_1_lhs_non_contracting_dims,
        GetNonContractingDims(
            bmm_2_grad_1->shape(),
            bmm_2_grad_1->dot_dimension_numbers().lhs_batch_dimensions(),
            bmm_2_grad_1->dot_dimension_numbers()
                .lhs_contracting_dimensions()));
    CHECK_EQ(bmm_2_grad_1_lhs_non_contracting_dims.size(), 1);
    (DynCast<HloDotInstruction>(bmm_2_grad_1))
        ->mutable_dot_dimension_numbers()
        ->set_lhs_contracting_dimensions(
            0, bmm_2_grad_1_lhs_non_contracting_dims[0]);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> bmm_2_grad_1_new_contracting_dims,
      GetNonContractingDims(
          bmm_2_grad_1->shape(),
          bmm_2_grad_1->dot_dimension_numbers().rhs_batch_dimensions(),
          bmm_2_grad_1->dot_dimension_numbers().rhs_contracting_dimensions()));

  if (is_bmm2_grad1_canonicalized) {
    (DynCast<HloDotInstruction>(bmm_2_grad_1))
        ->mutable_dot_dimension_numbers()
        ->set_rhs_contracting_dimensions(0,
                                         bmm_2_grad_1_new_contracting_dims[0]);
  }

  *bwd_fmha_config.mutable_bmm1_grad_gemm1_dot_dimension_numbers() =
      bmm_1_grad_1->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm1_grad_gemm2_dot_dimension_numbers() =
      bmm_1_grad_2->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm2_grad_gemm1_dot_dimension_numbers() =
      bmm_2_grad_1->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm2_grad_gemm2_dot_dimension_numbers() =
      bmm_2_grad_2->dot_dimension_numbers();

  bwd_fmha_config.set_fmha_scale(fwd_config.fmha_scale());
  bwd_fmha_config.set_dropout_rate(fwd_config.dropout_rate());
  // Set to an arbitrary seed for now, seed is not exposed to XLA in HLO
  // graph.
  // TODO Find a way to compute original seed from dropout keys.
  bwd_fmha_config.set_seed(fwd_config.seed());

  *bwd_fmha_config.mutable_intermediate_tensor_shape() =
      fwd_config.intermediate_tensor_shape();
  {
    auto* algorithm = bwd_fmha_config.mutable_algorithm();
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

  // Output order:
  // dQ(bmm_1_grad_2), dK(bmm_1_grad_1), dV(bmm_2_grad_1),
  // d_intermediate_tensor, d_bias_tensor
  std::vector<Shape> output_shapes = {
      bmm_1_grad_2->shape(), bmm_1_grad_1->shape(), bmm_2_grad_1->shape()};
  // d_intermediate is required to be output
  output_shapes.push_back(lhs_bmm2_grad_gemm1->shape());

  // Reserved placeholder for workspace
  output_shapes.push_back(ShapeUtil::MakeShape(U8, {0}));

  HloInstruction* dbias = nullptr;
  if (d_intermediate &&
      IsDbiasOnlyUserBesidesGradGemm(d_intermediate, bmm_1_grad_1, bmm_1_grad_2,
                                     &dbias)) {
    output_shapes.push_back(dbias->shape());
  }

  Shape call_shape = ShapeUtil::MakeTupleShape(output_shapes);
  HloInstruction* fmha_bwd_call =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, operands, absl::string_view(bwd_custom_call_name)));
  TF_RETURN_IF_ERROR(fmha_bwd_call->set_backend_config(bwd_fmha_config));
  TF_RETURN_IF_ERROR(
      SetFMHAInstructionName(bmm_1_grad_1->GetModule(), fmha_bwd_call));

  // Q gradient
  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_1_grad_2, HloInstruction::CreateGetTupleElement(bmm_1_grad_2->shape(),
                                                          fmha_bwd_call, 0)));
  // K gradient
  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_1_grad_1, HloInstruction::CreateGetTupleElement(bmm_1_grad_1->shape(),
                                                          fmha_bwd_call, 1)));
  // V gradient
  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_2_grad_1, HloInstruction::CreateGetTupleElement(bmm_2_grad_1->shape(),
                                                          fmha_bwd_call, 2)));
  // d_intermediate tensor
  if (dbias) {
    // does not really need d_intermediate
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        dbias, HloInstruction::CreateGetTupleElement(dbias->shape(),
                                                     fmha_bwd_call, 5)));
  }
  return true;
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
            compute_capability_, cudnn_version_, stream_executor_,
            stream_executor::dnn::VersionInfo(8, 8, 0))) {
      return false;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      bool v_transposed = false;
      MatchFwdResult matched_result =
          MatchFwdMHAPatternsForCanonicalization(instr);
      if (!matched_result.has_match) {
        continue;
      }
      // We check the validity of bmms here before canonicalization so we don't
      // modify the graph if mha fusion is not possible
      TF_ASSIGN_OR_RETURN(
          bool is_mha_module_supported,
          IsMHABlockSupported(
              matched_result.matched_bmm_1, matched_result.matched_bmm_2,
              matched_result.need_canonicalization, matched_result.is_training,
              matched_result.matched_custom_call_name, debug_options));
      if (!is_mha_module_supported) continue;
      // If we need to canonicalize the bmm, we will assign the newly
      // canonicalized bmm to bmm_2.
      if (matched_result.need_canonicalization) {
        TF_ASSIGN_OR_RETURN(matched_result.matched_bmm_2,
                            CanonicalizeBatchedGemmForcuDNNFMHA(
                                matched_result.matched_bmm_2, comp));
      }
      bool changed = false;
      // Fuse the bmms and intermediate nodes into fMHA call, the fused call
      // will replace bmm_2.
      TF_ASSIGN_OR_RETURN(
          HloInstruction * fwd_fmha_call,
          FuseFwdMultiHeadedAttentionBlock(
              comp, matched_result.matched_bmm_1, matched_result.matched_bmm_2,
              matched_result.matched_bias, matched_result.matched_mask,
              matched_result.matched_scale, matched_result.matched_dropout_rate,
              matched_result.matched_custom_call_name, compute_capability_,
              matched_result.is_training, changed, v_transposed));
      any_changed |= changed;

      if (matched_result.is_training) {
        // if fwd uses mask input, then bwd needs cudnn 8.9.1 to take in a mask
        // input if cudnn version < 8.9.1 we won't lower the bwd pass
        if (matched_result.matched_mask != nullptr &&
            !IsComputeCapabilityAndCudnnSupported(
                compute_capability_, cudnn_version_, stream_executor_,
                stream_executor::dnn::VersionInfo(8, 9, 1))) {
          continue;
        }
        MatchBwdResult matched_bwd_result =
            MatchBwdMHAPatternsForCanonicalization(
                fwd_fmha_call, matched_result.matched_bmm_1,
                matched_result.matched_mask, v_transposed);
        if (!matched_bwd_result.has_match) {
          continue;
        }
        // check if dbias is the only user of d_intermediate besides
        // bmm_1_grad_1 and bmm_1_grad_2 and the cudnn version is > 8.9.1. We
        // won't lower bwd if this condition is not met as we won't deal with
        // unswizzling now
        HloInstruction* dbias = nullptr;
        if (matched_bwd_result.matched_d_intermediate &&
            !IsDbiasOnlyUserBesidesGradGemm(
                matched_bwd_result.matched_d_intermediate,
                matched_bwd_result.matched_bmm_1_grad_1,
                matched_bwd_result.matched_bmm_1_grad_2, &dbias) &&
            !IsComputeCapabilityAndCudnnSupported(
                compute_capability_, cudnn_version_, stream_executor_,
                stream_executor::dnn::VersionInfo(8, 9, 1))) {
          continue;
        }
        // Canonicalize gemms
        if (matched_bwd_result.bmm_1_grad_1_need_canonicalization) {
          TF_ASSIGN_OR_RETURN(
              matched_bwd_result.matched_bmm_1_grad_1,
              CanonicalizeBatchedGemmForcuDNNFMHA(
                  matched_bwd_result.matched_bmm_1_grad_1, comp));
        }
        if (matched_bwd_result.bmm_1_grad_2_need_canonicalization) {
          TF_ASSIGN_OR_RETURN(
              matched_bwd_result.matched_bmm_1_grad_2,
              CanonicalizeBatchedGemmForcuDNNFMHA(
                  matched_bwd_result.matched_bmm_1_grad_2, comp));
        }
        if (matched_bwd_result.bmm_2_grad_1_need_canonicalization) {
          TF_ASSIGN_OR_RETURN(
              matched_bwd_result.matched_bmm_2_grad_1,
              CanonicalizeBatchedGemmForcuDNNFMHA(
                  matched_bwd_result.matched_bmm_2_grad_1, comp));
        }
        if (matched_bwd_result.bmm_2_grad_2_need_canonicalization) {
          TF_ASSIGN_OR_RETURN(
              matched_bwd_result.matched_bmm_2_grad_2,
              CanonicalizeBatchedGemmForcuDNNFMHA(
                  matched_bwd_result.matched_bmm_2_grad_2, comp));
        }

        // Fuse the corresponding gradient graph to an fMHA fused call.s
        TF_ASSIGN_OR_RETURN(
            changed,
            FuseBwdMultiHeadedAttentionBlock(
                comp, matched_bwd_result.matched_bmm_1_grad_1,
                matched_bwd_result.matched_bmm_1_grad_2,
                matched_bwd_result.matched_bmm_2_grad_1,
                matched_bwd_result.matched_bmm_2_grad_2, fwd_fmha_call,
                matched_bwd_result.matched_d_intermediate,
                matched_result.matched_mask,
                matched_bwd_result.matched_custom_call_name,
                matched_result.need_canonicalization,
                matched_bwd_result.bmm_2_grad_1_need_canonicalization));
        any_changed |= changed;
      }
    }
  }

  return any_changed;
}
}  // namespace gpu
}  // namespace xla
