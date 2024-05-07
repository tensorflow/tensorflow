/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/cudnn_fused_mha_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

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
  HloInstruction* matched_scale = nullptr;
  HloInstruction* matched_softmax_input = nullptr;
  HloInstruction* matched_reduce_sum = nullptr;

  double matched_dropout_rate = 0.0;
  bool need_canonicalization = false;
  bool is_training = false;
  // We use this to keep track of whether the bias is being
  // applied to the bmm1 is a causal mask, cuDNN can generate causal mask inside
  // the attention kernel to save I/O.
  bool is_causal_mask = false;
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
  HloInstruction* matched_dbias = nullptr;
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
  // reduce_max and subtract will always have exactly 1 user
  // in both training and inference
  // softmax_input should always have exactly 2 users
  auto unfused_softmax_max_subpattern = m::SharedSubpattern(
      m::Subtract(
          m::Op(),
          m::Broadcast(OptionalConvert(
              m::Op()
                  .WithPredicate(IsReduceMax)
                  .WithOneUse()
                  .WithOperand(0, OptionalBitcast(OptionalConvert(
                                      m::Op(softmax_input).WithNumUser(2)))))))
          .WithOneUse());
  // The reduce-add part of the softmax
  // reduce_sum and reduce_sum_broadcast should have 2 users in training
  // and 1 user in inference
  auto unfused_softmax_sum_subpattern = m::SharedSubpattern(m::Divide(
      OptionalBitcast(m::Exp(unfused_softmax_max_subpattern)),
      m::Broadcast(
          softmax_reduce_sum_bcast,
          OptionalConvert(
              m::Op(softmax_reduce_sum)
                  .WithOperand(0, OptionalBitcast(OptionalConvert(
                                      m::Exp(unfused_softmax_max_subpattern))))
                  .WithPredicate(IsReduceSum)
                  .WithAtMostNumUser(2)))
          .WithAtMostNumUser(2)));
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
    stream_executor::dnn::VersionInfo supported_cudnn_version) {
  // Enforce capability minor == 0 because hardware with a non-zero minor
  // number typically has insufficient shared memory for cuDNN FMHA.
  if (cc.IsAtLeastAmpere() && cc.minor == 0 &&
      cudnn_version >= supported_cudnn_version) {
    return true;
  }
  VLOG(2) << absl::StrFormat(
      "CudnnFusedMHARewriter did not run. Unsupported compute "
      "capability(%s; major should be >= 8, minor should be 0) or cudnn version"
      "(%s; should be >= %s)",
      cc.ToString(), cudnn_version.ToString(),
      supported_cudnn_version.ToString());
  return false;
}

bool IsSupportedPrimitiveType(const HloInstruction* bmm) {
  PrimitiveType dtype = bmm->shape().element_type();
  return dtype == BF16 || dtype == F16;
}

std::vector<int64_t> GetDimensionVector(absl::Span<const int64_t> dimensions,
                                        absl::Span<const int64_t> dim_nums) {
  std::vector<int64_t> vec(dim_nums.size());
  for (int i = 0; i < dim_nums.size(); i++) {
    vec[i] = dimensions.at(dim_nums.at(i));
  }
  return vec;
}

struct QKVLayout {
  int64_t batch;
  int64_t num_heads;
  int64_t seqlen_q;
  int64_t seqlen_kv;
  int64_t hidden_dim;
};

absl::StatusOr<std::optional<QKVLayout>> GetQKVLayout(
    HloInstruction* bmm_1, HloInstruction* bmm_2, bool need_canonicalization) {
  // get layout from bmm1
  const DotDimensionNumbers& bmm1_dnums = bmm_1->dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> bmm1_s_q_dims,
      GetNonContractingDims(bmm_1->operand(0)->shape(),
                            bmm1_dnums.lhs_batch_dimensions(),
                            bmm1_dnums.lhs_contracting_dimensions()));

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> bmm1_s_kv_dims,
      GetNonContractingDims(bmm_1->operand(1)->shape(),
                            bmm1_dnums.rhs_batch_dimensions(),
                            bmm1_dnums.rhs_contracting_dimensions()));

  std::vector<int64_t> bmm1_bh =
      GetDimensionVector(bmm_1->operand(0)->shape().dimensions(),
                         bmm1_dnums.lhs_batch_dimensions());

  std::vector<int64_t> bmm1_s_q = GetDimensionVector(
      bmm_1->operand(0)->shape().dimensions(), bmm1_s_q_dims);

  std::vector<int64_t> bmm1_s_kv = GetDimensionVector(
      bmm_1->operand(1)->shape().dimensions(), bmm1_s_kv_dims);

  std::vector<int64_t> bmm1_d =
      GetDimensionVector(bmm_1->operand(0)->shape().dimensions(),
                         bmm1_dnums.lhs_contracting_dimensions());

  TF_RET_CHECK(bmm1_bh.size() == 2);
  TF_RET_CHECK(bmm1_s_q.size() == 1);
  TF_RET_CHECK(bmm1_s_kv.size() == 1);
  TF_RET_CHECK(bmm1_d.size() == 1);

  // get layout from bmm2
  const DotDimensionNumbers& bmm2_dnums = bmm_2->dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> bmm2_lhs_non_contracting_dims,
      GetNonContractingDims(bmm_2->operand(0)->shape(),
                            bmm2_dnums.lhs_batch_dimensions(),
                            bmm2_dnums.lhs_contracting_dimensions()));

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> bmm2_rhs_non_contracting_dims,
      GetNonContractingDims(bmm_2->operand(1)->shape(),
                            bmm2_dnums.rhs_batch_dimensions(),
                            bmm2_dnums.rhs_contracting_dimensions()));

  std::vector<int64_t> bmm2_bh =
      GetDimensionVector(bmm_2->operand(0)->shape().dimensions(),
                         bmm2_dnums.lhs_batch_dimensions());

  std::vector<int64_t> bmm2_s_kv =
      GetDimensionVector(bmm_2->operand(0)->shape().dimensions(),
                         bmm2_dnums.lhs_contracting_dimensions());

  std::vector<int64_t> bmm2_s_q =
      need_canonicalization
          ? GetDimensionVector(bmm_2->operand(1)->shape().dimensions(),
                               bmm2_rhs_non_contracting_dims)
          : GetDimensionVector(bmm_2->operand(0)->shape().dimensions(),
                               bmm2_lhs_non_contracting_dims);

  std::vector<int64_t> bmm2_d =
      need_canonicalization
          ? GetDimensionVector(bmm_2->operand(0)->shape().dimensions(),
                               bmm2_lhs_non_contracting_dims)
          : GetDimensionVector(bmm_2->operand(1)->shape().dimensions(),
                               bmm2_rhs_non_contracting_dims);

  TF_RET_CHECK(bmm2_bh.size() == 2);
  TF_RET_CHECK(bmm2_s_q.size() == 1);
  TF_RET_CHECK(bmm2_s_kv.size() == 1);
  TF_RET_CHECK(bmm2_d.size() == 1);

  // check if bhsd is correct between bmm1 and bmm2
  if (bmm1_bh[0] != bmm2_bh[0] || bmm1_bh[1] != bmm2_bh[1] ||
      bmm1_s_q[0] != bmm2_s_q[0] || bmm1_s_kv[0] != bmm2_s_kv[0] ||
      bmm1_d[0] != bmm2_d[0]) {
    return std::nullopt;
  }

  QKVLayout qkv_layout;
  qkv_layout.batch = bmm1_bh[0];
  qkv_layout.num_heads = bmm1_bh[1];
  qkv_layout.seqlen_q = bmm1_s_q[0];
  qkv_layout.seqlen_kv = bmm1_s_kv[0];
  qkv_layout.hidden_dim = bmm1_d[0];
  return qkv_layout;
}

absl::StatusOr<bool> IsFlashAttention(
    QKVLayout qkv_layout, bool is_training,
    stream_executor::CudaComputeCapability cc,
    stream_executor::dnn::VersionInfo cudnn_version) {
  int64_t s_q = qkv_layout.seqlen_q;
  int64_t s_kv = qkv_layout.seqlen_kv;
  int64_t hidden_dim = qkv_layout.hidden_dim;
  // start with most relaxed constraint
  bool is_seqlen_supported = (!is_training || (s_q % 2 == 0 && s_kv % 2 == 0));
  bool is_hidden_dim_supported = hidden_dim <= 128 && hidden_dim % 8 == 0;
  bool is_flash_attention = is_seqlen_supported && is_hidden_dim_supported;
  if (!is_flash_attention) return false;

  // going backwards to check compatibility
  if ((is_training && (s_q < 64 || s_kv < 64)) &&
      !IsComputeCapabilityAndCudnnSupported(
          cc, cudnn_version, stream_executor::dnn::VersionInfo(9, 0, 0))) {
    VLOG(2) << "Flash attention training with seq < 64 not supported cuDNN < "
               "9.0.0.";
    return false;
  }

  if ((hidden_dim != 64 && hidden_dim != 128) &&
      !IsComputeCapabilityAndCudnnSupported(
          cc, cudnn_version, stream_executor::dnn::VersionInfo(8, 9, 6))) {
    VLOG(2) << "Flash attention head dim != 64 or 128 not supported with cuDNN "
               "< 8.9.6.";
    return false;
  }

  if ((is_training && s_kv % 64 != 0) &&
      !IsComputeCapabilityAndCudnnSupported(
          cc, cudnn_version, stream_executor::dnn::VersionInfo(8, 9, 5))) {
    VLOG(2) << "Flash attention training with seq kv % 64 != 0 not supported "
               "with cuDNN < 8.9.5.";
    return false;
  }

  if (!IsComputeCapabilityAndCudnnSupported(
          cc, cudnn_version, stream_executor::dnn::VersionInfo(8, 9, 4))) {
    VLOG(2) << "Require cuDNN 8.9.4 to run flash attention.";
    return false;
  }
  return is_flash_attention;
}

bool IsCausalMaskPattern(HloInstruction* mask) {
  auto causal_mask =
      m::Select(m::Compare(m::Iota(), m::Iota()), m::Broadcast(m::Constant()),
                m::Broadcast(m::Constant()));
  auto causal_mask_pattern_fwd_remat =
      m::Broadcast(OptionalBitcast(causal_mask));
  auto causal_mask_pattern_bwd = m::Broadcast(m::Convert(OptionalBitcast(
      m::Minimum(m::Op(), m::Broadcast(OptionalBitcast(causal_mask))))));
  HloInstruction* param = nullptr;
  HloInstruction* gte = nullptr;
  auto causal_mask_pattern_fwd = m::Broadcast(
      OptionalBitcast(m::GetTupleElement(&gte, m::Parameter(&param))));
  auto causal_mask_pattern = m::AnyOf<HloInstruction>(
      causal_mask_pattern_fwd_remat, causal_mask_pattern_fwd,
      causal_mask_pattern_bwd);
  if (Match(mask, causal_mask_pattern)) {
    if (param != nullptr && param->parent()->IsWhileBodyComputation()) {
      // need to track to outside of the while loop body to find the real mask.
      auto while_instr = param->parent()->WhileCallInstruction();
      auto mask_index = gte->tuple_index();
      auto actual_mask =
          while_instr->mutable_operand(0)->mutable_operand(mask_index);
      auto causal_mask_pattern_fwd =
          OptionalBitcast(m::Convert(m::MinimumAnyOrder(
              m::Op(),
              OptionalBitcast(m::MinimumAnyOrder(
                  m::Op(), m::Broadcast(OptionalBitcast(causal_mask)))))));
      return Match(actual_mask, causal_mask_pattern_fwd);
    }
    return true;
  }
  return false;
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

  // Form3 -> softmax - mul(dropout) - mul(scale) - BMM2
  auto dropout_softmax_pattern_form_3 = m::MultiplyAnyOrder(
      m::MultiplyAnyOrder(
          OptionalConvert(GetUnfusedReduceMaxSumSoftmaxPattern(
              &softmax_input, &softmax_reduce_sum, &softmax_reduce_sum_bcast)),
          m::Op()),
      m::Broadcast(m::Constant(&dropout).WithPredicate(IsScalar)));

  // Try matching BMM1 - (Scale) - (Bias) - Softmax - (Dropout) -
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
                           dropout_softmax_pattern_form_2,
                           dropout_softmax_pattern_form_3));

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
  match_result.matched_reduce_sum = softmax_reduce_sum;
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
  // bmm1/scale/bias add should have 2 users if being connected to softmax
  // otherwise should have exactly 1 user
  auto first_bmm_pattern =
      m::SharedSubpattern(m::Op(&bmm_1).WithPredicate(IsBatchedMatmul));
  auto unfused_scaled_bmm_subpattern = m::MultiplyAnyOrder(
      OptionalConvert(first_bmm_pattern.WithOneUse()),
      OptionalConvert(
          m::Broadcast(m::Constant(&scale).WithPredicate(IsScalar))));
  if (Match(softmax_input,
            OptionalConvert(OptionalBitcast(m::AnyOf<HloInstruction>(
                first_bmm_pattern, unfused_scaled_bmm_subpattern))))) {
    // bmm1 - (scale) - softmax
    match_result.matched_bmm_1 = bmm_1;
    match_result.matched_scale = scale;
    match_result.matched_custom_call_name =
        has_dropout ? kCudnnfMHASoftmaxDropoutCallTarget
                    : kCudnnfMHASoftmaxCallTarget;
    match_result.has_match = true;
  } else if (Match(softmax_input,
                   OptionalBitcast(m::AddAnyOrder(
                       OptionalConvert(OptionalBitcast(m::AnyOf<HloInstruction>(
                           unfused_scaled_bmm_subpattern.WithOneUse(),
                           first_bmm_pattern.WithOneUse()))),
                       m::Op(&bias))))) {
    // bmm1 - (scale) - bias - softmax
    match_result.matched_bmm_1 = bmm_1;
    match_result.matched_scale = scale;
    match_result.matched_custom_call_name =
        has_dropout ? kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget
                    : kCudnnfMHAScaleBiasSoftmaxCallTarget;
    match_result.is_causal_mask |= IsCausalMaskPattern(bias);
    if (!match_result.is_causal_mask &&
        bias->opcode() == HloOpcode::kBroadcast) {
      // we can take the bias before broadcast
      auto dims = Cast<HloBroadcastInstruction>(bias)->dimensions();
      if (dims == std::vector<int64_t>{2, 3} ||
          dims == std::vector<int64_t>{0, 2, 3} ||
          dims == std::vector<int64_t>{1, 2, 3}) {
        // shapes [1, 1, s, s], [b, 1, s, s], [1, h, s, s] are supported
        HloInstruction* bias_bc = bias->mutable_operand(0);
        // bitcast bias_before_broadcast to be 4D
        std::vector<int64_t> bitcast_dims(bias->shape().rank(), 1);
        for (int dim : dims) {
          bitcast_dims[dim] = bias->shape().dimensions()[dim];
        }
        bias = bias_bc->AddInstruction(HloInstruction::CreateBitcast(
            ShapeUtil::MakeShape(bias->shape().element_type(), bitcast_dims),
            bias_bc));
      }
    }
    match_result.matched_bias = bias;
    match_result.has_match = true;
  } else {
    match_result.has_match = false;
  }
  return match_result;
}

// We will try to match all the patterns below:
// BMM1 - Scale - bias - Softmax - Dropout - BMM2
// BMM1 - Scale - bias - Softmax - BMM2
// BMM1 - Softmax - Dropout - BMM2
// BMM1 - Softmax - BMM2
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
                                  HloInstruction* bmm_1) {
  MatchBwdResult match_result = previous_result;
  match_result.has_match = false;
  const HloInstruction* q_tensor = bmm_1->operand(0);
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
  // bmm1 gradient gemm2 shares the same input d_s as bmm1 gradient gemm1.
  // Check to see if bmm1 grad gemm1 needs canonicalization or not, if not,
  // then the shared input is the first operand.
  int64_t d_s_index = match_result.bmm_1_grad_1_need_canonicalization ? 1 : 0;
  HloInstruction* d_s_user_0 = match_result.matched_bmm_1_grad_1;

  HloInstruction* d_s = d_s_user_0->mutable_operand(d_s_index);
  if (d_s->opcode() == HloOpcode::kBitcast && d_s->user_count() == 1) {
    d_s = d_s->mutable_operand(0);
  }

  auto bmm_1_grad_2_it = std::find_if(
      d_s->users().begin(), d_s->users().end(), [&](HloInstruction* instr) {
        return instr != match_result.matched_bmm_1_grad_1 &&
               instr->opcode() == HloOpcode::kDot;
      });
  if (bmm_1_grad_2_it != d_s->users().end()) {
    bmm_1_grad_2 = *bmm_1_grad_2_it;
  } else {
    return match_result;
  }

  match_result.matched_bmm_1_grad_2 = bmm_1_grad_2;

  if (match_result.matched_bmm_1_grad_2->operand_index(d_s) != 0) {
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

MatchBwdResult MatchDbias(MatchBwdResult previous_result,
                          HloInstruction* d_intermediate,
                          const absl::flat_hash_set<HloInstruction*> users) {
  MatchBwdResult match_result = previous_result;
  auto user_count = d_intermediate->user_count();
  HloInstruction* dbias_user = nullptr;
  HloInstruction* dbias = nullptr;
  for (auto user : d_intermediate->users()) {
    if (users.contains(user)) {
      user_count -= 1;
    } else {
      dbias_user = user;
    }
  }
  auto ConsumeExtraConvert = [](HloInstruction* instr) {
    Match(instr->users()[0], m::Convert(&instr, m::Op()).WithOneUse());
    return true;
  };
  // user_count == 1 && (reduce-> {convert} ->bitcast)
  match_result.has_match =
      user_count == 1 &&
      Match(dbias_user, m::Reduce(&dbias, m::Op(), m::Op()).WithOneUse()) &&
      dbias->shape().rank() == 3 && ConsumeExtraConvert(dbias);
  if (match_result.has_match) {
    // cuDNN only supports dbias for [1, h, s, s]
    // make sure reduce dimension is on batch dim
    auto reduce_dim = dbias->dimensions();
    if (reduce_dim.size() == 1 && reduce_dim[0] == 0) {
      match_result.matched_dbias = dbias;
    } else {
      match_result.has_match = false;
    }
  }
  return match_result;
}

MatchBwdResult MatchBwdBmmSoftmaxDropoutBmm(MatchBwdResult previous_result,
                                            HloInstruction* fwd_fmha_call) {
  MatchBwdResult match_result = previous_result;
  bool is_bmm1_grad1_canonicalized =
      match_result.bmm_1_grad_1_need_canonicalization;
  match_result.has_match = false;
  bool has_scale = false;
  bool has_dropout = false;
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
  auto bwd_dropout_pattern_form_3 = OptionalConvert(m::MultiplyAnyOrder(
      m::MultiplyAnyOrder(
          m::Op().WithPredicate([&](const HloInstruction* instr) {
            return instr == match_result.matched_bmm_2_grad_2;
          }),
          m::Broadcast(m::Constant().WithPredicate(IsScalar))),
      m::Op()));
  auto bwd_dropout_pattern = m::AnyOf<HloInstruction>(
      bwd_dropout_pattern_form_1, bwd_dropout_pattern_form_2,
      bwd_dropout_pattern_form_3);
  // Backward softmax pattern
  HloInstruction* bwd_softmax_input = nullptr;
  HloInstruction* exp_1;
  HloInstruction* exp_2;
  HloInstruction* d_softmax;

  // d_softmax = exp * (dy / s_b - sum(dy * exp * 1 / s^2))
  // there could be at most 3 users of d_softmax: bmm1grad1 bmm1grad2 and dbias
  auto bwd_softmax_pattern = OptionalBitcast(OptionalConvert(
      m::MultiplyAnyOrder(
          &d_softmax,
          m::AddAnyOrder(
              m::Divide().WithOneUse(),
              m::Broadcast(OptionalBitcast(OptionalConvert(
                  m::Negate(
                      OptionalBitcast(
                          m::Op()
                              .WithPredicate(IsReduceSum)
                              .WithOneUse()
                              .WithOperand(
                                  0, OptionalBitcast(
                                         m::MultiplyAnyOrder(
                                             m::MultiplyAnyOrder(
                                                 m::Op(&bwd_softmax_input),
                                                 m::Broadcast())
                                                 .WithOneUse(),
                                             m::Exp(&exp_2, m::Op()))
                                             .WithOneUse()))))
                      .WithOneUse())))),
          m::Exp(&exp_1, m::Op()))
          .WithAtMostNumUser(3)));

  // Backward scale input pattern
  HloInstruction* bwd_scale_input = nullptr;
  HloInstruction* bwd_scale = nullptr;

  auto bwd_scale_pattern =
      m::MultiplyAnyOrder(&bwd_scale, m::Op(&bwd_scale_input),
                          m::Broadcast(m::Constant().WithPredicate(IsScalar)))
          .WithNumUser(2);
  int intermediate_input_pos = is_bmm1_grad1_canonicalized ? 1 : 0;

  HloInstruction* intermediate_input =
      match_result.matched_bmm_1_grad_1->mutable_operand(
          intermediate_input_pos);

  has_scale = Match(intermediate_input, bwd_scale_pattern);

  if (has_scale) {
    intermediate_input = bwd_scale_input;
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

  if (has_dropout) {
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
  // try to pattern match dbias
  HloInstruction* dS = d_softmax;
  if (dS->users()[0]->opcode() == HloOpcode::kConvert) {
    dS = dS->users()[0];
  }
  if (has_scale) {
    // bmm1-(scale)-(bias)-softmax pattern users could be dbias or scale bwd
    if (dS->user_count() == 1) {
      // no dbias
      match_result.has_match = true;
    } else if (dS->user_count() == 2) {
      match_result = MatchDbias(match_result, dS, {bwd_scale});
    } else {
      match_result.has_match = false;
    }
  } else {
    // bmm1-(bias)-softmax pattern
    // users could be dbias besides bmm1grad1 bmm1grad2
    if (dS->user_count() == 2) {
      match_result.has_match = true;
    } else if (dS->user_count() == 3) {
      match_result = MatchDbias(match_result, dS,
                                {match_result.matched_bmm_1_grad_1,
                                 match_result.matched_bmm_1_grad_2});
    } else {
      match_result.has_match = false;
    }
  }
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

  matched_result = MatchBmm1GradGemm1(matched_result, bmm_1);
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
    HloInstruction* fwd_fmha_call, HloInstruction* bmm_1, bool v_transposed) {
  MatchBwdResult match_result =
      MatchBackwardBmms(fwd_fmha_call, bmm_1, v_transposed);
  if (!match_result.has_match) {
    return match_result;
  }
  match_result = MatchBwdBmmSoftmaxDropoutBmm(match_result, fwd_fmha_call);
  return match_result;
}

absl::StatusOr<bool> IsMHABlockSupported(
    HloInstruction* bmm_1, HloInstruction* bmm_2, bool need_canonicalization,
    bool is_training, bool is_causal_mask, std::string& custom_call_name,
    const DebugOptions& debug_options,
    stream_executor::CudaComputeCapability cc,
    stream_executor::dnn::VersionInfo cudnn_version) {
  if (MHACallHasDropout(custom_call_name) &&
      !debug_options.xla_gpu_fused_attention_use_cudnn_rng()) {
    VLOG(3) << "Using CUDNN RNG for fused attention dropout is not enabled.\n";
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

  if (bmm_1->shape().rank() != 4 || bmm_2->shape().rank() != 4) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "Unsupported bmm rank for cuDNN MHA fusion:\n"
              << bmm_1->ToString() << "\nOR\n"
              << bmm_2->ToString() << "\n"
              << "Only bmm with rank 4 is supported.";
    }
    return false;
  }

  // get batch/num heads/sequence length/hidden dim from bmm1 and bmm2
  // also make sure they are the same between bmm1 and bmm2
  TF_ASSIGN_OR_RETURN(std::optional<QKVLayout> qkv_layout,
                      GetQKVLayout(bmm_1, bmm_2, need_canonicalization));
  if (!qkv_layout.has_value()) {
    VLOG(2) << "bmm1 and bmm2 have different qkv layout.";
    return false;
  }

  // check if matched attention block is supported by cuDNN flash attention.
  TF_ASSIGN_OR_RETURN(
      bool is_flash_attention,
      IsFlashAttention(qkv_layout.value(), is_training, cc, cudnn_version));
  if (is_flash_attention) {
    if (is_causal_mask) {
      // if bias is causal mask, needs to remove bias from name
      if (custom_call_name == kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget) {
        custom_call_name = kCudnnfMHASoftmaxDropoutCallTarget;
      } else if (custom_call_name == kCudnnfMHAScaleBiasSoftmaxCallTarget) {
        custom_call_name = kCudnnfMHASoftmaxCallTarget;
      }
    }
  }
  return is_flash_attention;
}

absl::StatusOr<HloInstruction*> CanonicalizeBatchedGemmForcuDNNFMHA(
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

absl::StatusOr<HloInstruction*> ChangeCheckedDimToFastest(
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
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> non_contracting_dims,
                      GetNonContractingDims(bmm->operand(bmm_operand)->shape(),
                                            batch_dims, contracting_dims));
  CHECK_EQ(non_contracting_dims.size(), 1);
  HloInstruction* operand_bmm = bmm->mutable_operand(bmm_operand);
  int64_t hidden_dim = should_contracting_be_fastest ? contracting_dims[0]
                                                     : non_contracting_dims[0];
  int64_t minor_dim = minor_to_major_to_check[0];
  // If the hidden dim of the target operand is not the fastest moving
  // dimension, make it so.
  if (minor_dim != hidden_dim) {
    std::vector<int64_t> perm(bmm->shape().dimensions_size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[hidden_dim], perm[minor_dim]);

    if (is_lhs) {
      new_dot_dims_bmm.set_lhs_contracting_dimensions(0,
                                                      non_contracting_dims[0]);
    } else {
      new_dot_dims_bmm.set_rhs_contracting_dimensions(0,
                                                      non_contracting_dims[0]);
    }

    operand_bmm = comp->AddInstruction(
        HloInstruction::CreateTranspose(
            ShapeUtil::MakeShapeWithDenseLayout(
                bmm->shape().element_type(),
                Permute(operand_bmm->shape().dimensions(), perm),
                minor_to_major_to_check),
            operand_bmm, perm),
        &operand_bmm->metadata());
    *((DynCast<HloDotInstruction>(bmm))->mutable_dot_dimension_numbers()) =
        new_dot_dims_bmm;
  }
  return operand_bmm;
}

absl::StatusOr<HloInstruction*> FuseFwdMultiHeadedAttentionBlock(
    HloComputation* comp, HloInstruction* bmm_1, HloInstruction* bmm_2,
    HloInstruction* bias, HloInstruction* scale, HloInstruction* reduce_sum,
    HloInstruction* softmax_input, double dropout_rate,
    std::string& custom_call_name, stream_executor::CudaComputeCapability cc,
    bool is_training, bool& changed, bool& v_transposed, bool is_causal_mask) {
  double scale_value = 1.0;
  HloInstruction* lhs_bmm1;
  HloInstruction* rhs_bmm1;
  HloInstruction* rhs_bmm2;
  DotDimensionNumbers orig_bmm1_dot_dim = bmm_1->dot_dimension_numbers();
  DotDimensionNumbers orig_bmm2_dot_dim = bmm_2->dot_dimension_numbers();

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

  GpuBackendConfig gpu_config;
  CudnnfMHABackendConfig& fmha_config =
      *gpu_config.mutable_cudnn_fmha_backend_config();

  *fmha_config.mutable_bmm1_dot_dimension_numbers() =
      bmm_1->dot_dimension_numbers();
  *fmha_config.mutable_bmm2_dot_dimension_numbers() =
      bmm_2->dot_dimension_numbers();

  TF_RET_CHECK((dropout_rate >= 0.0 && dropout_rate <= 1.0));
  // Restore original DotDimensionNumbers.
  *((DynCast<HloDotInstruction>(bmm_1))->mutable_dot_dimension_numbers()) =
      orig_bmm1_dot_dim;
  *((DynCast<HloDotInstruction>(bmm_2))->mutable_dot_dimension_numbers()) =
      orig_bmm2_dot_dim;

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
  // set is_causal_mask here
  // choose to generate causal mask inside cuDNN attention or not
  fmha_config.set_mask_type(is_causal_mask ? CudnnfMHABackendConfig::CAUSAL
                                           : CudnnfMHABackendConfig::NO_MASK);

  // Output Order: {O, scratch, Fwd act*}
  const Shape& output_shape = bmm_2->shape();

  Shape call_shape;
  // Activation output is used by backward gemm.
  HloInstruction* activation_output = nullptr;

  std::vector<Shape> output_shapes = {output_shape,
                                      ShapeUtil::MakeShape(U8, {0})};
  if (is_training) {
    activation_output = bmm_2->mutable_operand(0);
    // Sometimes activation output is bitcast, the actual activation is the
    // other user of the producer of bmm_2's first operand.
    if (activation_output->user_count() < 2 &&
        activation_output->opcode() == HloOpcode::kBitcast) {
      HloInstruction* producer = activation_output->mutable_operand(0);
      TF_RET_CHECK(producer->user_count() == 2);
      HloInstruction* bmm2_grad2_user =
          producer->users()[0] == activation_output ? producer->users()[1]
                                                    : producer->users()[0];
      // might be (transpose) - bmm2_grad2
      if (IsBatchedMatmul(bmm2_grad2_user)) {
        activation_output = producer;
      } else if (bmm2_grad2_user->opcode() == HloOpcode::kTranspose) {
        activation_output = bmm2_grad2_user;
      } else {
        return Internal("Unexpected activation patterns");
      }
    }
    // if it is flash attention, should output softmax stats to the bwd
    TF_RET_CHECK(reduce_sum != nullptr);
    output_shapes.push_back(
        ShapeUtil::MakeShape(F32, reduce_sum->shape().dimensions()));
  }
  call_shape = ShapeUtil::MakeTupleShape(output_shapes);

  // Input Order: {Q, K, V, bias*}
  std::vector<HloInstruction*> operands = {lhs_bmm1, rhs_bmm1, rhs_bmm2};
  if (!is_causal_mask && bias != nullptr) {
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
  TF_RETURN_IF_ERROR(fmha_call->set_backend_config(gpu_config));
  TF_RETURN_IF_ERROR(SetFMHAInstructionName(bmm_1->GetModule(), fmha_call));

  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_2,
      HloInstruction::CreateGetTupleElement(bmm_2->shape(), fmha_call, 0)));

  if (activation_output) {
    HloInstruction* activation_gte =
        comp->AddInstruction(HloInstruction::CreateGetTupleElement(
            activation_output->shape(), fmha_call, 2));
    TF_RETURN_IF_ERROR(comp->ReplaceInstructionWithDifferentShape(
                               activation_output, activation_gte,
                               /*preserve_sharding=*/false,
                               /*relay_control_dependency=*/false,
                               /*remove_unused_operands=*/false)
                           .status());
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "After CudnnFusedMHARewriter: \n" << comp->parent()->ToString();
  }
  changed = true;
  return fmha_call;
}

absl::StatusOr<bool> FuseBwdMultiHeadedAttentionBlock(
    HloComputation* comp, HloInstruction* bmm_1_grad_1,
    HloInstruction* bmm_1_grad_2, HloInstruction* bmm_2_grad_1,
    HloInstruction* bmm_2_grad_2, HloInstruction* fwd_fmha_call,
    HloInstruction* dbias, HloInstruction* bias,
    std::string& bwd_custom_call_name) {
  HloInstruction* rhs_bmm1_grad_gemm1;
  HloInstruction* lhs_bmm1_grad_gemm2;
  HloInstruction* rhs_bmm2_grad_gemm2;
  HloInstruction* d_output_grad;

  DotDimensionNumbers orig_bmm1_grad1_config =
      bmm_1_grad_1->dot_dimension_numbers();
  DotDimensionNumbers orig_bmm1_grad2_config =
      bmm_1_grad_2->dot_dimension_numbers();
  DotDimensionNumbers orig_bmm2_grad1_config =
      bmm_2_grad_1->dot_dimension_numbers();
  DotDimensionNumbers orig_bmm2_grad2_config =
      bmm_2_grad_2->dot_dimension_numbers();

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      fwd_fmha_call->backend_config<GpuBackendConfig>());
  CudnnfMHABackendConfig fwd_config = gpu_config.cudnn_fmha_backend_config();
  bool is_causal_mask =
      fwd_config.mask_type() == CudnnfMHABackendConfig::CAUSAL;
  CudnnfMHABackendConfig bwd_fmha_config;
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
  // softmax_stats
  HloInstruction* fwd_act;
  int64_t fwd_act_index = 2;
  fwd_act = comp->AddInstruction(HloInstruction::CreateGetTupleElement(
      fwd_fmha_call->shape().tuple_shapes(fwd_act_index), fwd_fmha_call,
      fwd_act_index));

  // V tensor
  TF_ASSIGN_OR_RETURN(
      rhs_bmm2_grad_gemm2,
      ChangeCheckedDimToFastest(comp, bmm_2_grad_2, false /*is_lhs*/,
                                true /*should_contracting_be_fastest*/));
  // d output to bmm2_grad2
  // Since d_o is the input of 2 bmms, we set the dim number using the
  // constraint
  // -> the contracting dimension of the lhs of bmm_2_grad_2 needs to be the
  // fastest moving dimension.
  TF_ASSIGN_OR_RETURN(
      d_output_grad,
      ChangeCheckedDimToFastest(comp, bmm_2_grad_2, true /*is_lhs*/,
                                true /*should_contracting_be_fastest*/));
  // d output to bmm2_grad1
  // we don't use this value but we call this to make sure dot number is being
  // set correctly
  TF_ASSIGN_OR_RETURN(
      HloInstruction * bmm_2_grad_1_rhs,
      ChangeCheckedDimToFastest(comp, bmm_2_grad_1, false /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
  (void)bmm_2_grad_1_rhs;
  // Operand order: {Q, K, V, Fwd act, d_o, bias*, O*}
  std::vector<HloInstruction*> operands = {
      rhs_bmm1_grad_gemm1, lhs_bmm1_grad_gemm2, rhs_bmm2_grad_gemm2, fwd_act,
      d_output_grad};

  // For flash attention, add fwd output to input list
  if (!is_causal_mask && bias) {
    operands.push_back(bias);
  }
  HloInstruction* fwd_output;
  for (auto user : fwd_fmha_call->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == 0) {
      fwd_output = user;
    }
  }
  // should be able to find the instruction
  TF_RET_CHECK(fwd_output != nullptr);
  // check dO and O have the same layout as it is required by cuDNN
  TF_RET_CHECK(fwd_output->shape() == d_output_grad->shape());
  operands.push_back(fwd_output);

  *bwd_fmha_config.mutable_bmm1_grad_gemm1_dot_dimension_numbers() =
      bmm_1_grad_1->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm1_grad_gemm2_dot_dimension_numbers() =
      bmm_1_grad_2->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm2_grad_gemm1_dot_dimension_numbers() =
      bmm_2_grad_1->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm2_grad_gemm2_dot_dimension_numbers() =
      bmm_2_grad_2->dot_dimension_numbers();

  // Restore original DotDimensionNumbers
  *((DynCast<HloDotInstruction>(bmm_1_grad_1))
        ->mutable_dot_dimension_numbers()) = orig_bmm1_grad1_config;
  *((DynCast<HloDotInstruction>(bmm_1_grad_2))
        ->mutable_dot_dimension_numbers()) = orig_bmm1_grad2_config;
  *((DynCast<HloDotInstruction>(bmm_2_grad_1))
        ->mutable_dot_dimension_numbers()) = orig_bmm2_grad1_config;
  *((DynCast<HloDotInstruction>(bmm_2_grad_2))
        ->mutable_dot_dimension_numbers()) = orig_bmm2_grad2_config;

  bwd_fmha_config.set_fmha_scale(fwd_config.fmha_scale());
  bwd_fmha_config.set_dropout_rate(fwd_config.dropout_rate());
  // Set to an arbitrary seed for now, seed is not exposed to XLA in HLO
  // graph.
  // TODO Find a way to compute original seed from dropout keys.
  bwd_fmha_config.set_seed(fwd_config.seed());
  bwd_fmha_config.set_mask_type(is_causal_mask
                                    ? CudnnfMHABackendConfig::CAUSAL
                                    : CudnnfMHABackendConfig::NO_MASK);

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
  // {dQ(bmm_1_grad_2), dK(bmm_1_grad_1), dV(bmm_2_grad_1),
  // scratch, dbias*}
  std::vector<Shape> output_shapes = {
      bmm_1_grad_2->shape(), bmm_1_grad_1->shape(), bmm_2_grad_1->shape()};
  // Reserved placeholder for workspace
  output_shapes.push_back(ShapeUtil::MakeShape(U8, {0}));

  if (dbias) {
    // Cudnn kernel only outputs dbias in this shape [1, num_heads, seq, seq],
    // so we add a dimension of 1 to existing dbias' shape.
    std::vector<int64_t> dbias_shape_vector =
        SpanToVector(dbias->shape().dimensions());
    dbias_shape_vector.insert(dbias_shape_vector.begin(), 1);
    Shape cudnn_dbias_shape =
        ShapeUtil::MakeShape(dbias->shape().element_type(), dbias_shape_vector);
    output_shapes.push_back(cudnn_dbias_shape);
  }
  Shape call_shape = ShapeUtil::MakeTupleShape(output_shapes);
  HloInstruction* fmha_bwd_call =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, operands, absl::string_view(bwd_custom_call_name)));
  GpuBackendConfig bwd_gpu_config;
  *bwd_gpu_config.mutable_cudnn_fmha_backend_config() = bwd_fmha_config;
  TF_RETURN_IF_ERROR(fmha_bwd_call->set_backend_config(bwd_gpu_config));
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

  if (dbias) {
    // Reshape fmha dbias output to original user's input shape.
    // If the reshape doesn't involve physical data movement,
    // algebraic simplifer can change it to a no-op bitcast.
    Shape original_shape = dbias->shape();
    HloInstruction* dbias_user = dbias->users()[0];
    HloInstruction* cudnn_dbias_output =
        comp->AddInstruction(HloInstruction::CreateGetTupleElement(
            output_shapes.back(), fmha_bwd_call, 4));
    HloInstruction* reshape_dbias = comp->AddInstruction(
        HloInstruction::CreateReshape(original_shape, cudnn_dbias_output));
    TF_RETURN_IF_ERROR(dbias_user->ReplaceOperandWith(
        dbias_user->operand_index(dbias), reshape_dbias));

    TF_RETURN_IF_ERROR(
        comp->ReplaceInstructionWithDifferentShape(dbias, cudnn_dbias_output));
  }
  return true;
}

Status RestoreFwdGraph(
    HloComputation* comp, HloInstruction* fwd_fmha_call, HloInstruction* bmm2,
    HloInstruction* activation, HloInstruction* original_bmm2_producer0,
    HloInstruction* original_bmm2_producer1,
    std::vector<HloInstruction*>& original_activation_producers,
    bool bmm_2_need_canonicalization) {
  // If backward pattern is not matched, we need to restore the
  // original graph structure.
  // Replacing new GTEs added by forward FMHA call with cloned old
  // activations and bmm2.
  HloInstruction* output_gte = fwd_fmha_call->users()[0];
  HloInstruction* activation_gte = fwd_fmha_call->users()[1];
  std::string suffix = "fmha_no_match_clone";
  HloInstruction* cloned_activation =
      comp->AddInstruction(activation->CloneWithNewOperands(
          activation->shape(), original_activation_producers, suffix));

  // Since old activation is detached by forward FMHA rewrite, we need
  // to use the newly cloned activation.
  HloInstruction* lhs = activation == original_bmm2_producer0
                            ? cloned_activation
                            : original_bmm2_producer0;
  HloInstruction* rhs = activation == original_bmm2_producer0
                            ? original_bmm2_producer1
                            : cloned_activation;
  HloInstruction* cloned_bmm2 = comp->AddInstruction(
      bmm2->CloneWithNewOperands(bmm2->shape(), {lhs, rhs}, suffix));
  if (bmm_2_need_canonicalization) {
    TF_RET_CHECK(output_gte->users()[0]->opcode() == HloOpcode::kTranspose);
    TF_RETURN_IF_ERROR(
        comp->ReplaceInstruction(output_gte->users()[0], cloned_bmm2));
  } else {
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(output_gte, cloned_bmm2));
  }
  TF_RETURN_IF_ERROR(
      comp->ReplaceInstruction(activation_gte, cloned_activation));
  return OkStatus();
}
}  // namespace

absl::StatusOr<bool> CudnnFusedMHARewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  // we use this set to keep track of all already matched attention block
  absl::flat_hash_set<HloInstruction*> matched_bmm1;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    const DebugOptions& debug_options =
        comp->parent()->config().debug_options();
    const se::dnn::VersionInfo cudnn_version =
        GetDnnVersionInfo(stream_executor_, cudnn_version_);
#if !defined(GOOGLE_CUDA) || CUDA_VERSION < 12000
    // CUDA needs to be >= 12.0 for cuDNN to work with all supported hardware.
    // Some cuDNN versions work with CUDA 11, but it is impractical for us to
    // test those combinations so just disable them.
    return false;
#endif
    if (!debug_options.xla_gpu_enable_cudnn_fmha() ||
        !IsComputeCapabilityAndCudnnSupported(
            compute_capability_, cudnn_version,
            stream_executor::dnn::VersionInfo(8, 8, 0))) {
      return false;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      bool v_transposed = false;
      bool changed = false;
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
              matched_result.is_causal_mask,
              matched_result.matched_custom_call_name, debug_options,
              compute_capability_, cudnn_version));

      if (!is_mha_module_supported) continue;
      // If we have an activation with more than 1 users in non-training mode,
      // we cannot rewrite the graph. So skip processing the rest.
      HloInstruction* activation =
          matched_result.need_canonicalization
              ? matched_result.matched_bmm_2->mutable_operand(1)
              : matched_result.matched_bmm_2->mutable_operand(0);
      if (!matched_result.is_training && activation->user_count() > 1) {
        VLOG(2)
            << "Activation: " << activation->ToString()
            << " cannot have more than 1 users in non-training mode. Skipping.";
        continue;
      }
      HloInstruction* original_bmm2_producer0 =
          matched_result.matched_bmm_2->mutable_operand(0);
      HloInstruction* original_bmm2_producer1 =
          matched_result.matched_bmm_2->mutable_operand(1);

      HloInstruction* original_bmm2 = matched_result.matched_bmm_2;
      std::vector<HloInstruction*> original_activation_producers;
      for (HloInstruction* operand : activation->mutable_operands()) {
        original_activation_producers.push_back(operand);
      }
      // We make sure no attention block is matched and replaced twice here
      if (!matched_bmm1.insert(matched_result.matched_bmm_1).second) {
        continue;
      }
      // If we need to canonicalize the bmm, we will assign the newly
      // canonicalized bmm to bmm_2.
      if (matched_result.need_canonicalization) {
        TF_ASSIGN_OR_RETURN(matched_result.matched_bmm_2,
                            CanonicalizeBatchedGemmForcuDNNFMHA(
                                matched_result.matched_bmm_2, comp));
      }

      // Fuse the bmms and intermediate nodes into fMHA call, the fused call
      // will replace bmm_2.
      TF_ASSIGN_OR_RETURN(
          HloInstruction * fwd_fmha_call,
          FuseFwdMultiHeadedAttentionBlock(
              comp, matched_result.matched_bmm_1, matched_result.matched_bmm_2,
              matched_result.matched_bias, matched_result.matched_scale,
              matched_result.matched_reduce_sum,
              matched_result.matched_softmax_input,
              matched_result.matched_dropout_rate,
              matched_result.matched_custom_call_name, compute_capability_,
              matched_result.is_training, changed, v_transposed,
              matched_result.is_causal_mask));
      any_changed |= changed;
      if (matched_result.is_training) {
        MatchBwdResult matched_bwd_result =
            MatchBwdMHAPatternsForCanonicalization(
                fwd_fmha_call, matched_result.matched_bmm_1, v_transposed);
        if (!matched_bwd_result.has_match) {
          VLOG(2) << "Backward pattern not matching, skipping.";
          // restore fwd graph if bwd pattern match failed
          TF_RETURN_IF_ERROR(
              RestoreFwdGraph(comp, fwd_fmha_call, original_bmm2, activation,
                              original_bmm2_producer0, original_bmm2_producer1,
                              original_activation_producers,
                              matched_result.need_canonicalization));
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
                matched_bwd_result.matched_dbias, matched_result.matched_bias,
                matched_bwd_result.matched_custom_call_name));
        any_changed |= changed;
      }
    }
  }

  return any_changed;
}
}  // namespace gpu
}  // namespace xla
