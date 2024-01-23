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
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"

namespace xla {
namespace cpu {

namespace {
namespace m = match;

auto ConvertPattern(HloInstruction** instr) {
  return m::Convert(m::Op(instr).WithElementType(PrimitiveType::BF16))
      .WithElementType(PrimitiveType::F32);
}

template <typename Pattern>
auto OptionalConvert(Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Convert(pattern), std::move(pattern));
}

HloInstruction* FindLayerNormScale(HloInstruction* instr) {
  HloInstruction* scale = nullptr;
  auto scalePattern = m::Multiply().WithBinaryOperandsAnyOrder(
      m::Broadcast(m::Op(&scale).WithOpcode(HloOpcode::kReshape)),
      m::Broadcast(m::Reshape(m::Broadcast(m::Rsqrt()))).WithOneUser());
  auto m = Match(instr, scalePattern);
  return scale;
}

HloInstruction* FindLayerNormShift(HloInstruction* instr) {
  HloInstruction* shift = nullptr;
  auto m = Match(
      instr,
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
  if (!Match(
          instr,
          m::Divide(
              m::Exp(&left_exponential, m::Op()),
              m::Broadcast(m::Reshape(m::Broadcast(OptionalConvert(m::Reshape(
                  m::Reduce(
                      OptionalConvert(m::Exp(&right_exponential, m::Op())),
                      m::Op())
                      .WithPredicate([](const HloInstruction* reduce) {
                        HloComputation* reducer = reduce->to_apply();
                        return (reducer->root_instruction()->opcode() ==
                                    HloOpcode::kAdd &&
                                reduce->dimensions().size() == 1 &&
                                reduce->dimensions()[0] !=
                                    reduce->shape().rank() - 1);
                      })
                      .WithOneUse())))))))) {
    return std::nullopt;
  }

  if (left_exponential != right_exponential ||
      left_exponential->user_count() != 2)
    return std::nullopt;

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

  if (left_producer != right_producer || left_producer->user_count() != 2)
    return std::nullopt;

  return left_producer;
}

}  // namespace

class OneDnnOpsRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleAdd(HloInstruction* instr) override {
    HloInstruction *slicemu1, *slicemu2;
    HloInstruction *slicesource1, *slicesource2;
    HloInstruction *musquare1, *musquare2;
    HloInstruction *prod_c, *prod_l, *prod_s, *prod_r;
    HloInstruction *slicevar, *hinge;

    bool scaleFound = false;
    bool shiftFound = false;

    auto spine = m::Add().WithBinaryOperandsAnyOrder(
        m::Broadcast(),
        m::Multiply()
            .WithBinaryOperandsAnyOrder(
                m::Op(&hinge).WithOneUser(),
                m::Subtract(
                    m::Op(&prod_s),
                    m::Broadcast(
                        m::Reshape(
                            m::Broadcast(
                                m::Reshape(
                                    m::Op(&slicemu1)
                                        .WithOpcode(HloOpcode::kSlice)
                                        .WithOperand(
                                            0, m::Op(&slicesource1)
                                                   .WithOpcode(
                                                       HloOpcode::kDivide)))
                                    .WithOneUser())
                                .WithOneUser())
                            .WithOneUser())
                        .WithOneUser())
                    .WithOneUser())
            .WithOneUser());

    if (!Match(instr, spine)) {
      return OkStatus();
    }

    const Shape& prod_shape = prod_s->shape();
    if (!IsSupportedType(prod_shape.element_type())) return OkStatus();

    HloInstruction* shift = FindLayerNormShift(instr);
    shiftFound = (shift != nullptr);

    HloInstruction* scale = FindLayerNormScale(hinge);
    scaleFound = (scale != nullptr);

    // Currently patterns without scale and shift are
    // not supported.
    // OneDNN only supports 2 <= rank <= 5
    if (!(prod_shape.rank() >= 2 && prod_shape.rank() <= 5) || !shiftFound ||
        !scaleFound) {
      return OkStatus();
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
                                m::Broadcast(m::Constant()),
                                m::Reshape(
                                    m::Maximum()
                                        .WithBinaryOperandsAnyOrder(
                                            m::Broadcast(),
                                            m::Subtract(
                                                m::Reshape(
                                                    m::Op(&slicevar)
                                                        .WithOpcode(
                                                            HloOpcode::kSlice)
                                                        .WithOperand(
                                                            0,
                                                            m::Op(&slicesource2)
                                                                .WithOpcode(
                                                                    HloOpcode::
                                                                        kDivide)))
                                                    .WithOneUser(),
                                                m::Multiply(
                                                    m::Op(&musquare1),
                                                    m::Op(&musquare2)
                                                        .WithOperand(
                                                            0,
                                                            m::Op(&slicemu2)
                                                                .WithOpcode(
                                                                    HloOpcode::
                                                                        kSlice)))
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

    if (!Match(hinge, main_pipeline) || slicemu1 != slicemu2 ||
        musquare1 != musquare2 || slicesource1 != slicesource2) {
      return OkStatus();
    }

    // Check if the slices are compatible
    if (!(absl::c_all_of(slicemu1->slice_starts(),
                         [](int64_t i) { return i == 0; }) &&
          absl::c_equal(slicemu1->slice_limits(),
                        slicemu1->shape().dimensions())) &&
        !(absl::c_all_of(slicevar->slice_starts(),
                         [](int64_t i) { return i == 0; }) &&
          absl::c_equal(slicevar->slice_limits(),
                        slicevar->shape().dimensions()))) {
      return OkStatus();
    }

    auto empirical_expectations = m::Divide(
        m::Reduce(m::Concatenate()
                      .WithBinaryOperandsAnyOrder(
                          m::Reshape(m::Multiply(m::Op(&prod_l), m::Op(&prod_c))
                                         .WithOneUser())
                              .WithOneUser(),
                          m::Reshape(m::Op(&prod_r)).WithOneUser())
                      .WithPredicate([](const HloInstruction* comb) {
                        return (comb->dimensions().size() == 1 &&
                                comb->dimensions()[0] == 0 &&
                                comb->shape().dimensions(0) == 2);
                      })
                      .WithOneUser(),
                  m::Constant())
            .WithPredicate([](const HloInstruction* reduce) {
              HloComputation* reducer = reduce->to_apply();
              return (reducer->root_instruction()->opcode() ==
                          HloOpcode::kAdd &&
                      reduce->dimensions().size() == 1 &&
                      reduce->dimensions()[0] == reduce->shape().rank());
            })
            .WithOneUser(),
        m::Broadcast(m::ConstantScalar().WithPredicate(
            [orig = prod_s](const HloInstruction* divisor) {
              std::optional<double> actual =
                  static_cast<const HloConstantInstruction*>(divisor)
                      ->literal()
                      .GetAsDouble({});
              return (actual.has_value() &&
                      orig->shape().dimensions(orig->shape().rank() - 1) ==
                          *actual);
            })));

    HloInstruction *src1, *src2;
    if (Match(slicesource2, empirical_expectations) &&
        // Float32 pattern check
        ((prod_l == prod_c && prod_c == prod_r && prod_l == prod_s) ||
         // Bfloat16 pattern check
         (prod_l == prod_c && prod_c == prod_r &&
          Match(prod_l, ConvertPattern(&src1)) &&
          Match(prod_s, ConvertPattern(&src2)) && src1 == src2))) {
      HloInstruction* ln_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              prod_shape, {prod_r, scale, shift}, "__onednn$layernorm"));
      BackendConfig backend_config;
      OneDnnLayerNormConfig* ln_config =
          backend_config.mutable_onednn_layer_norm_config();
      ln_config->set_fused_ops(OneDnnLayerNormConfig::SCALE_AND_SHIFT);
      TF_RETURN_IF_ERROR(ln_call->set_backend_config(backend_config));
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, ln_call));
    }

    return OkStatus();
  }

  Status HandleConvert(HloInstruction* instr) override {
    HloInstruction* ln_instr;
    auto pattern = m::Convert(m::Op(&ln_instr)
                                  .WithOneUser()
                                  .WithOpcode(HloOpcode::kCustomCall)
                                  .WithCustomCallTarget({"__onednn$layernorm"})
                                  .WithElementType(PrimitiveType::F32))
                       .WithElementType(PrimitiveType::BF16);

    if (!IsSupportedType(instr->shape().element_type())) return OkStatus();
    if (Match(instr, pattern)) {
      HloInstruction* producer = instr->mutable_operand(0)->mutable_operand(0);
      HloInstruction* newinp =
          producer->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(producer->shape(),
                                           instr->shape().element_type()),
              {producer}));
      absl::InlinedVector<HloInstruction*, 2> newoperands =
          ln_instr->mutable_operands();
      newoperands.at(0) = newinp;
      HloInstruction* ln_call = instr->AddInstruction(
          ln_instr->CloneWithNewOperands(instr->shape(), newoperands));
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, ln_call));
    }

    return OkStatus();
  }

  Status HandleDivide(HloInstruction* divide_instr) override {
    if (divide_instr->HasControlDependencies()) return OkStatus();
    if (!IsSupportedType(divide_instr->shape().element_type()))
      return OkStatus();
    std::optional<HloInstruction*> producer;
    bool found_pattern = false;
    if (producer = MatchSoftmax(divide_instr)) {
      found_pattern = true;
    }

    if (!found_pattern) return OkStatus();

    const Shape& output_shape = divide_instr->shape();
    HloInstruction* softmax_call =
        divide_instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {producer.value()}, "__onednn$softmax"));
    TF_RETURN_IF_ERROR(ReplaceInstruction(divide_instr, softmax_call));

    return OkStatus();
  }
};

StatusOr<bool> OneDnnOpsRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OneDnnOpsRewriterVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
