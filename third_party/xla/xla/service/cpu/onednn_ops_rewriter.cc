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
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/onednn_ops_rewriter.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_pattern_utils.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"

namespace xla {
namespace cpu {

namespace {
namespace m = match;
namespace pu = ::xla::cpu::onednn_pattern_utils_internal;

inline auto OneDnnConvertibleInstr(HloInstruction** instr) {
  return m::AnyOf<HloInstruction>(m::CustomCall(instr, {"__onednn$layernorm"}),
                                  m::CustomCall(instr, {"__onednn$softmax"}));
}

HloInstruction* FindLayerNormScale(HloInstruction* instr) {
  HloInstruction* scale = nullptr;
  auto scalePattern = m::Multiply().WithBinaryOperandsAnyOrder(
      m::Broadcast(m::Op(&scale).WithOpcode(HloOpcode::kReshape)),
      m::Broadcast(m::Reshape(m::Broadcast(m::Rsqrt()))).WithOneUser());
  Match(instr, scalePattern);
  return scale;
}

HloInstruction* FindLayerNormShift(HloInstruction* instr) {
  HloInstruction* shift = nullptr;
  Match(instr,
        m::Add().WithBinaryOperandsAnyOrder(
            m::Multiply()
                .WithBinaryOperandsAnyOrder(
                    m::Op(), m::Subtract(m::Op(), m::Broadcast().WithOneUser())
                                 .WithOneUser())
                .WithOneUser(),
            m::Broadcast(m::Op(&shift))));
  return shift;
}

std::optional<HloInstruction*> MatchSoftmax(HloInstruction* instr) {
  //
  // producer
  // |   \
  // |  reduce_max
  // |     |
  // |  reshape
  // |     |
  // |  broadcast
  // |     |
  // |  reshape
  // |     |
  // |  broadcast
  // |   /
  // subtract
  // |
  // exponential
  // |   \
  // |  reduce_sum
  // |     |
  // |  reshape
  // |     |
  // |  broadcast
  // |     |
  // |  reshape
  // |     |
  // |  broadcast
  // |   /
  // divide  // (instr parameter)
  //
  // where both reductions occur only on the last axis.
  HloInstruction* left_exponential;
  HloInstruction* right_exponential;
  HloInstruction* left_producer;
  HloInstruction* right_producer;

  // Lower diamond
  if (!Match(instr,
             m::Divide(
                 m::Exp(&left_exponential, m::Op()),
                 m::Broadcast(m::Reshape(m::Broadcast(
                     pu::OptionalConvert(m::Reshape(pu::OptionalConvert(
                         m::Reduce(pu::OptionalConvert(
                                       m::Exp(&right_exponential, m::Op())),
                                   m::Op())
                             .WithPredicate([](const HloInstruction* reduce) {
                               HloComputation* reducer = reduce->to_apply();
                               return (reducer->root_instruction()->opcode() ==
                                           HloOpcode::kAdd &&
                                       reduce->dimensions().size() == 1 &&
                                       reduce->dimensions()[0] !=
                                           reduce->shape().rank() - 1);
                             })
                             .WithOneUse()))))))))) {
    return std::nullopt;
  }

  if (left_exponential != right_exponential ||
      left_exponential->user_count() != 2) {
    return std::nullopt;
  }

  // Upper diamond
  if (!Match(left_exponential->mutable_operand(0),
             m::Subtract(
                 m::Op(&left_producer),
                 m::Broadcast(
                     m::Reshape(m::Broadcast(m::Reshape(
                         m::Reduce(m::Op(&right_producer), m::Op())
                             .WithPredicate([](const HloInstruction* reduce) {
                               HloComputation* reducer = reduce->to_apply();
                               return (reducer->root_instruction()->opcode() ==
                                           HloOpcode::kMaximum &&
                                       reduce->dimensions().size() == 1 &&
                                       reduce->dimensions()[0] !=
                                           reduce->shape().rank() - 1);
                             })
                             .WithOneUse()))))
                     .WithOneUse())
                 .WithOneUse())) {
    return std::nullopt;
  }

  if (left_producer != right_producer || left_producer->user_count() != 2) {
    return std::nullopt;
  }

  return left_producer;
}

auto MeanPattern(HloInstruction** input) {
  return m::Reshape(
      m::Convert(m::Divide(m::Reduce(m::Convert(m::Op(input)), m::Op()),
                           m::Broadcast(m::Convert()))));
}

template <typename Pattern>
auto Square(Pattern pattern) {
  return m::Multiply()
      .WithBinaryOperandsAnyOrder(pattern, pattern)
      .WithPredicate([](const HloInstruction* instr) {
        return instr->unique_operands().size() == 1;
      });
}

std::optional<bool> MatchTFKerasLayerNorm(HloInstruction* instr,
                                          HloInstruction** src,
                                          HloInstruction** scale,
                                          HloInstruction** bias, float* eps) {
  // variance = Mean((X - Mean(x))^2)
  // Z = scale / sqrt(variance + eps)
  // LN(X) = X*Z + Bias - Mean(X)*Z

  HloInstruction *src_a, *src_b, *src_c;
  HloInstruction *bias_node, *scaled_norm_a, *scaled_norm_b, *mean0_a, *epsilon,
      *sqrd_diff_mean, *scale_node, *sqrd_diff;

  // First Match X*Z + Bias - Mean(X)*Z
  if (!Match(
          instr,
          m::Add().WithBinaryOperandsAnyOrder(
              m::Multiply()
                  .WithBinaryOperandsAnyOrder(m::Op(src), m::Op(&scaled_norm_a))
                  .WithOneUser(),
              m::Subtract(m::Op(&bias_node),
                          m::Multiply().WithBinaryOperandsAnyOrder(
                              m::Broadcast(m::Reshape(m::Op(&mean0_a))),
                              m::Op(&scaled_norm_b)))
                  .WithOneUser()))) {
    return std::nullopt;
  }

  if (scaled_norm_a != scaled_norm_b) return std::nullopt;

  const Shape& src_shape = (*src)->shape();
  if (!IsSupportedType(src_shape.element_type())) return std::nullopt;

  // Get bias
  if (!Match(bias_node, m::Broadcast(m::Op(bias)))) return std::nullopt;

  // Match Z = scale / sqrt(variance + eps)
  if (!Match(scaled_norm_a,
             m::Multiply().WithBinaryOperandsAnyOrder(
                 m::Op(&scale_node),
                 m::Broadcast(
                     m::Reshape(m::Rsqrt(m::Add().WithBinaryOperandsAnyOrder(
                         m::Broadcast(m::ConstantScalar(&epsilon)),
                         m::Op(&sqrd_diff_mean)))))))) {
    return std::nullopt;
  }

  // get epsilon
  *eps = static_cast<float>(epsilon->literal().GetAsDouble({}).value());
  // get scale
  if (!Match(scale_node, m::Broadcast(m::Op(scale)))) return std::nullopt;

  // match variance
  if (!Match(sqrd_diff_mean, MeanPattern(&sqrd_diff))) return std::nullopt;

  if (!Match(sqrd_diff, Square(m::Subtract(
                            m::Op(&src_a),
                            m::Broadcast(m::Reshape(MeanPattern(&src_b))))))) {
    return std::nullopt;
  }

  if (src_a != src_b && src_a != *src) return std::nullopt;

  // Match mean from Bias - Mean(X)*Z
  if (!Match(mean0_a, MeanPattern(&src_c))) return std::nullopt;

  if (src_c != *src) return std::nullopt;

  return true;
}

bool MatchFlaxLayerNorm(HloInstruction* instr, HloInstruction** src,
                        HloInstruction** scale, HloInstruction** bias,
                        float* eps, bool* is_bf16orfp16_convert,
                        bool* is_producer_bf16orfp16,
                        HloInstruction** convert_instr) {
  HloInstruction *prod_s, *hinge;
  HloInstruction *div0, *div1, *div_red;
  HloInstruction *mul_in0, *mul_in1, *main_pipe_mul_in0;
  HloInstruction *reduce_in0, *epsilon;
  HloInstruction *broadcast0, *broadcast1;

  bool scaleFound = false;
  bool shiftFound = false;

  auto spine = m::Add().WithBinaryOperandsAnyOrder(
      m::Broadcast(),
      m::Multiply()
          .WithBinaryOperandsAnyOrder(
              m::Op(&hinge).WithOneUser(),
              m::Subtract(
                  pu::OptionalConvert(m::Op(&prod_s)),
                  m::Broadcast(
                      m::Reshape(
                          m::Broadcast(m::Reshape(m::Op(&div_red).WithOpcode(
                                                      HloOpcode::kDivide))
                                           .WithOneUser())
                              .WithOneUser())
                          .WithOneUser())
                      .WithOneUser())
                  .WithOneUser())
          .WithOneUser());

  if (!Match(instr, spine)) return false;

  const Shape& prod_shape = prod_s->shape();
  if (!IsSupportedType(prod_shape.element_type())) return false;

  HloInstruction* shift = FindLayerNormShift(instr);
  shiftFound = (shift != nullptr);

  HloInstruction* scale_gamma = FindLayerNormScale(hinge);
  scaleFound = (scale_gamma != nullptr);

  // Currently patterns without scale and shift are not supported.
  // OneDNN only supports 2 <= rank <= 5
  if (!(prod_shape.rank() >= 2 && prod_shape.rank() <= 5) || !shiftFound ||
      !scaleFound) {
    return false;
  }

  // NOLINTBEGIN
  auto main_pipeline = m::Multiply().WithBinaryOperandsAnyOrder(
      m::Op(),
      m::Broadcast(
          m::Reshape(
              m::Broadcast(
                  m::Rsqrt(
                      m::Add()
                          .WithBinaryOperandsAnyOrder(
                              m::Broadcast(m::ConstantScalar(&epsilon)),
                              m::Reshape(
                                  m::Maximum()
                                      .WithBinaryOperandsAnyOrder(
                                          m::Broadcast(),
                                          m::Subtract(
                                              m::Op(&div0).WithOpcode(
                                                  HloOpcode::kDivide),
                                              m::Multiply()
                                                  .WithBinaryOperandsAnyOrder(
                                                      m::Op(&main_pipe_mul_in0),
                                                      m::Op(&div1).WithOpcode(
                                                          HloOpcode::kDivide))
                                                  .WithOneUser())
                                              .WithOneUser())
                                      .WithOneUser())
                                  .WithOneUser())
                          .WithOneUser())
                      .WithOneUser())
                  .WithOneUser())
              .WithOneUser())
          .WithOneUser());
  // NOLINTEND

  if (!Match(hinge, main_pipeline)) return false;

  if ((div_red != div1) || (main_pipe_mul_in0 != div1)) return false;

  auto div_red_mul_src =
      m::Divide()
          .WithOperand(0, m::Reduce(m::Multiply().WithBinaryOperandsAnyOrder(
                                        pu::OptionalConvert(m::Op(&mul_in0)),
                                        pu::OptionalConvert(m::Op(&mul_in1))),
                                    m::Constant())
                              .WithPredicate([](const HloInstruction* reduce) {
                                HloComputation* reducer = reduce->to_apply();
                                return (reducer->root_instruction()->opcode() ==
                                            HloOpcode::kAdd &&
                                        reduce->dimensions().size() == 1 &&
                                        reduce->dimensions()[0] ==
                                            reduce->shape().rank());
                              }))
          .WithOperand(1, m::Op(&broadcast0).WithOpcode(HloOpcode::kBroadcast))
          .WithOneUser();

  if (!Match(div0, div_red_mul_src)) return false;

  if (mul_in0 != mul_in1) return false;

  auto div_red_subgraph =
      m::Divide()
          .WithOperand(
              0,
              m::Reduce(pu::OptionalConvert(m::Op(&reduce_in0)), m::Constant())
                  .WithPredicate([](const HloInstruction* reduce) {
                    HloComputation* reducer = reduce->to_apply();
                    return (reducer->root_instruction()->opcode() ==
                                HloOpcode::kAdd &&
                            reduce->dimensions().size() == 1 &&
                            reduce->dimensions()[0] == reduce->shape().rank());
                  }))
          .WithOperand(1, m::Op(&broadcast1).WithOpcode(HloOpcode::kBroadcast));

  if (!Match(div1, div_red_subgraph)) return false;

  if (broadcast1 != broadcast0 || reduce_in0 != mul_in0 || mul_in0 != prod_s) {
    return false;
  }

  *is_producer_bf16orfp16 =
      (prod_s->shape().element_type() == PrimitiveType::F16) ||
      (prod_s->shape().element_type() == PrimitiveType::BF16);
  if (instr->user_count() == 1 &&
      instr->users().at(0)->opcode() == HloOpcode::kConvert) {
    *convert_instr = instr->users().at(0);
    *is_bf16orfp16_convert =
        ((*convert_instr)->shape().element_type() == PrimitiveType::F16 ||
         (*convert_instr)->shape().element_type() == PrimitiveType::BF16);
  }

  *src = prod_s;
  *scale = scale_gamma;
  *bias = shift;
  // get epsilon
  *eps = static_cast<float>(epsilon->literal().GetAsDouble({}).value());

  return true;
}

}  // namespace

class OneDnnOpsRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleAdd(HloInstruction* instr) override {
    HloInstruction *src, *scale, *bias;
    float eps;
    bool is_bf16orfp16_convert = false;
    bool is_producer_bf16orfp16 = false;
    HloInstruction* convert_instr;

    bool found_ln =
        MatchTFKerasLayerNorm(instr, &src, &scale, &bias, &eps).value_or(false);

    if (!found_ln) {
      found_ln = MatchFlaxLayerNorm(instr, &src, &scale, &bias, &eps,
                                    &is_bf16orfp16_convert,
                                    &is_producer_bf16orfp16, &convert_instr);
    }

    if (!found_ln) return absl::OkStatus();

    const Shape& src_shape = src->shape();
    auto scale_type = scale->shape().element_type();
    auto bias_type = bias->shape().element_type();
    HloInstruction* scale_operand = scale;
    HloInstruction* bias_operand = bias;

    // oneDNN requires scale and shift float32
    if ((scale_type == PrimitiveType::BF16) ||
        (scale_type == PrimitiveType::F16)) {
      scale_operand = instr->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(scale->shape(), PrimitiveType::F32),
          scale));
    }

    if ((bias_type == PrimitiveType::BF16) ||
        (bias_type == PrimitiveType::F16)) {
      bias_operand = instr->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(bias->shape(), PrimitiveType::F32),
          bias));
    }

    HloInstruction* ln_call =
        instr->AddInstruction(HloInstruction::CreateCustomCall(
            src_shape, {src, scale_operand, bias_operand},
            "__onednn$layernorm"));
    BackendConfig backend_config;
    OneDnnLayerNormConfig* ln_config =
        backend_config.mutable_onednn_layer_norm_config();
    ln_config->set_fused_ops(OneDnnLayerNormConfig::SCALE_AND_SHIFT);
    ln_config->set_epsilon_typecast(*(reinterpret_cast<int32_t*>(&eps)));
    TF_RETURN_IF_ERROR(ln_call->set_backend_config(backend_config));

    if (convert_instr != nullptr && is_bf16orfp16_convert &&
        is_producer_bf16orfp16) {
      TF_RETURN_IF_ERROR(ReplaceInstruction(convert_instr, ln_call));
    } else {
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, ln_call));
    }

    return absl::OkStatus();
  }

  absl::Status HandleConvert(HloInstruction* instr) override {
    HloInstruction* custom_call;
    HloInstruction* convert_instr;
    auto pattern =
        m::Op(&convert_instr)
            .WithOpcode(HloOpcode::kConvert)
            .WithOperand(0, OneDnnConvertibleInstr(&custom_call)
                                .WithOneUser()
                                .WithElementType(PrimitiveType::F32));

    if (!IsSupportedType(instr->shape().element_type()))
      return absl::OkStatus();
    if (Match(instr, pattern)) {
      bool is_bf16orfp16_convert =
          (convert_instr->shape().element_type() == PrimitiveType::BF16) ||
          (convert_instr->shape().element_type() == PrimitiveType::F16);
      if (!is_bf16orfp16_convert) return absl::OkStatus();
      HloInstruction* producer = instr->mutable_operand(0)->mutable_operand(0);
      HloInstruction* newinp =
          producer->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(producer->shape(),
                                           instr->shape().element_type()),
              producer));
      absl::InlinedVector<HloInstruction*, 2> newoperands =
          custom_call->mutable_operands();
      newoperands.at(0) = newinp;
      HloInstruction* updated_call = instr->AddInstruction(
          custom_call->CloneWithNewOperands(instr->shape(), newoperands));
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, updated_call));
    }

    return absl::OkStatus();
  }

  absl::Status HandleDivide(HloInstruction* divide_instr) override {
    if (divide_instr->HasControlDependencies()) return absl::OkStatus();
    if (!IsSupportedType(divide_instr->shape().element_type()))
      return absl::OkStatus();
    std::optional<HloInstruction*> producer = MatchSoftmax(divide_instr);
    if (producer == std::nullopt) return absl::OkStatus();

    const Shape& output_shape = divide_instr->shape();
    HloInstruction* softmax_call =
        divide_instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {producer.value()}, "__onednn$softmax"));
    TF_RETURN_IF_ERROR(ReplaceInstruction(divide_instr, softmax_call));

    return absl::OkStatus();
  }
};

absl::StatusOr<bool> OneDnnOpsRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OneDnnOpsRewriterVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
