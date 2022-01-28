/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_conv_rewriter.h"

#include <functional>
#include <string>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/dnn.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

// If VLOG is on and `instr` matches `filter_pattern`, prints out why it doesn't
// match `log_pattern`.  You can use this to explain "near-hits".
template <typename FilterPattern, typename LogPattern>
void VlogIfFailureToMatch(HloInstruction* instr, const LogPattern& log_pattern,
                          absl::string_view desc,
                          const FilterPattern& filter_pattern) {
  if (!VLOG_IS_ON(3) || !Match(instr, filter_pattern)) {
    return;
  }
  std::stringstream os;
  if (!Match(instr, log_pattern, {/*capture=*/false, /*explain_os=*/&os})) {
    VLOG(3) << "Failed to match " << desc << ":\n" << os.str();
  }
}

bool IsConvCustomCall(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kCustomCall &&
         (instr->custom_call_target() == kCudnnConvForwardCallTarget ||
          instr->custom_call_target() ==
              kCudnnConvBiasActivationForwardCallTarget);
}

// Can instr be converted to type `dst_ty` without losing any precision?  For
// our purposes, this is true if:
//
//  - instr already has type dst_ty, or
//  - instr is convert<wider type>(op_with_dst_ty), or
//  - instr is a constant which we can convert orig_ty -> dst_ty -> orig_ty and
//    get back exactly the original value, or
//  - instr is a broadcast, reshape, or transpose of one of the above.
bool IsLosslesslyConvertibleTo(const HloInstruction* instr,
                               PrimitiveType dst_ty) {
  if (instr->shape().element_type() == dst_ty) {
    return true;
  }

  if (Match(instr, m::Convert(m::Op().WithElementType(dst_ty)))) {
    // Check that the convert from dst_ty to instr->element_type() doesn't lose
    // precision.  Otherwise, this convert is not lossless.
    return primitive_util::CastPreservesValues(dst_ty,
                                               instr->shape().element_type());
  }

  if (instr->opcode() == HloOpcode::kConstant) {
    if (!instr->shape().IsArray()) {
      return false;
    }
    // Check if instr's literal roundtrips to ty and back to its original type
    // without modification.
    PrimitiveType orig_ty = instr->shape().element_type();

    // The only reason Convert() should fail is if we don't support converting
    // from x to y, which indeed means it's not losslessly-convertible.
    StatusOr<Literal> converted1 = instr->literal().Convert(dst_ty);
    if (!converted1.ok()) {
      return false;
    }
    StatusOr<Literal> converted2 = converted1->Convert(orig_ty);
    if (!converted2.ok()) {
      return false;
    }

    return instr->literal() == *converted2;
  }

  if (instr->opcode() == HloOpcode::kBroadcast ||
      instr->opcode() == HloOpcode::kReshape ||
      instr->opcode() == HloOpcode::kTranspose) {
    return IsLosslesslyConvertibleTo(instr->operand(0), dst_ty);
  }

  return false;
}

// Helpers suitable for use in m::Op().WithPredicate(...).
bool IsLosslesslyConvertibleToS8(const HloInstruction* instr) {
  return IsLosslesslyConvertibleTo(instr, S8);
}
bool IsLosslesslyConvertibleToF16(const HloInstruction* instr) {
  return IsLosslesslyConvertibleTo(instr, F16);
}

// If `conv` is a vanilla forward conv, transforms it into a
// conv-bias-activation.  If it's already a conv-bias-activation, does nothing.
//
// If `conv` is anything else, returns an error.
StatusOr<HloInstruction*> EnsureIsConvBiasActivation(HloInstruction* conv) {
  CHECK_EQ(conv->opcode(), HloOpcode::kCustomCall);

  if (conv->custom_call_target() == kCudnnConvBiasActivationForwardCallTarget) {
    return conv;
  }

  if (conv->custom_call_target() == kCudnnConvForwardCallTarget) {
    HloComputation* comp = conv->parent();

    const Shape& shape = conv->shape().tuple_shapes(0);
    int64_t num_output_features = shape.dimensions(
        conv->convolution_dimension_numbers().output_feature_dimension());

    // bias for integer convs is always f32, see
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    PrimitiveType bias_ty;
    if (primitive_util::IsIntegralType(shape.element_type())) {
      bias_ty = F32;
    } else {
      bias_ty = shape.element_type();
    }
    auto bias = BroadcastZeros(comp, bias_ty, {num_output_features});

    absl::InlinedVector<HloInstruction*, 3> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands.push_back(bias);

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, new_conv));
    new_conv->set_custom_call_target(kCudnnConvBiasActivationForwardCallTarget);
    comp->parent()->SetAndUniquifyInstrName(new_conv,
                                            "cudnn-conv-bias-activation");
    return new_conv;
  }

  return FailedPrecondition("Unsupported conv: %s", conv->ToString());
}

// convert<float>(gte(custom-call<int32>(int8_x, int8_w))) ->
// gte(custom-call<float>(int8_x, int8_w))
StatusOr<bool> FuseConvertToFloat(HloComputation* comp) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    auto pattern =
        m::Convert(
            m::GetTupleElement(m::Op(&conv).WithPredicate(IsConvCustomCall), 0)
                .WithElementType(S32))
            .WithElementType(F32);
    if (!Match(instr, pattern)) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertToFloat: ", conv->ToString());
        })) {
      continue;
    }

    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(F32);
    HloInstruction* new_conv =
        comp->AddInstruction(conv->CloneWithNewShape(new_shape));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_gte,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_gte));

    changed = true;
  }

  return changed;
}

// alpha * gte(custom-call(...)) ->
// gte(custom-call(..., backend_config={alpha})).
StatusOr<bool> FuseConvAlpha(HloComputation* comp) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    HloInstruction* gte = nullptr;
    HloInstruction* alpha = nullptr;
    auto pattern = m::MultiplyAnyOrder(
        m::GetTupleElement(&gte, m::Op(&conv).WithPredicate(IsConvCustomCall),
                           0)
            .WithOneUse(),
        m::Broadcast(m::ConstantEffectiveScalar(&alpha)));
    if (!Match(instr, pattern)) {
      continue;
    }

    // alpha is f32 except for f64 convs, where it's f64.  See
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    PrimitiveType alpha_ty = gte->shape().element_type() == F64 ? F64 : F32;
    if (!IsLosslesslyConvertibleTo(alpha, alpha_ty)) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(auto config,
                        conv->backend_config<CudnnConvBackendConfig>());
    if (config.conv_result_scale() != 1) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvAlpha: ", conv->ToString());
        })) {
      continue;
    }

    // StreamExecutor doesn't support the alpha parameter on non-bias-activation
    // convs, so we have to upgrade `conv`.
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));

    TF_ASSIGN_OR_RETURN(Literal alpha_f64, alpha->literal().Convert(F64));
    config.set_conv_result_scale(alpha_f64.GetFirstElement<double>());

    TF_RETURN_IF_ERROR(conv->set_backend_config(config));
    TF_RETURN_IF_ERROR(conv->parent()->ReplaceInstruction(instr, gte));

    changed = true;
  }
  return changed;
}

StatusOr<bool> FuseBiasOrSideInput(HloComputation* comp) {
  bool changed = false;
  for (auto instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv = nullptr;
    HloInstruction* gte = nullptr;
    HloInstruction* addend = nullptr;
    auto pattern = m::AddAnyOrder(
        m::GetTupleElement(
            &gte, m::Op(&conv).WithPredicate(IsConvCustomCall).WithOneUse(), 0)
            .WithOneUse(),
        m::Op(&addend));
    if (!Match(instr, pattern)) {
      continue;
    }

    // If it's a vanilla forward conv, upgrade it to a bias-activation conv.  We
    // only want to do this if the fusion will succeed, but we're guaranteed
    // that it will, because the only reason we'll bail at this point is if
    // !can_accept_bias && !can_accept_side_input, and our shiny new
    // bias-activation conv will be able to accept both.
    if (conv->custom_call_target() == kCudnnConvForwardCallTarget) {
      TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    }

    // Can't fuse bias or side-input if the conv already has a relu (or other
    // activation), because bias and side-input are added before the activation
    // is applied.
    TF_ASSIGN_OR_RETURN(auto config,
                        conv->backend_config<CudnnConvBackendConfig>());
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }

    // Does `conv` already have a (nonzero) bias?  Does it already have a
    // side_input?
    bool can_accept_bias =
        Match(conv->operand(2), m::Broadcast(m::ConstantEffectiveScalar(0)));
    bool can_accept_side_input = conv->operand_count() < 4;

    // The addend can be fused as a bias if
    //  - it is 1D broadcasted in the output feature dimension, and
    //  - it is losslessly-convertible to the correct type (f32 for s8/f32/u32
    //    convs, and conv_ty for floating-point convs)
    PrimitiveType conv_ty = gte->shape().element_type();
    PrimitiveType bias_ty =
        primitive_util::IsFloatingPointType(conv_ty) ? conv_ty : F32;
    bool addend_may_be_rank1_bias =
        addend->opcode() == HloOpcode::kBroadcast &&
        addend->dimensions().size() == 1 &&
        addend->dimensions(0) ==
            conv->convolution_dimension_numbers().output_feature_dimension() &&
        IsLosslesslyConvertibleTo(addend, bias_ty);

    bool addend_may_be_rank0_bias = addend->opcode() == HloOpcode::kBroadcast &&
                                    addend->dimensions().empty() &&
                                    IsLosslesslyConvertibleTo(addend, bias_ty);

    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    if (can_accept_bias && addend_may_be_rank1_bias) {
      new_operands[2] = MakeConvertToHlo(addend->mutable_operand(0), bias_ty);
    } else if (can_accept_bias && addend_may_be_rank0_bias) {
      new_operands[2] = MakeBroadcastHlo(
          MakeConvertToHlo(addend->mutable_operand(0), bias_ty),
          /*broadcast_dimensions=*/{},
          /*result_shape_bounds=*/
          {gte->shape().dimensions(conv->convolution_dimension_numbers()
                                       .output_feature_dimension())});
    } else if (can_accept_side_input) {
      CHECK_EQ(new_operands.size(), 3);
      new_operands.push_back(addend);
      config.set_side_input_scale(1);
    } else {
      // Can't fuse; this op already has a bias and a side-input.
      continue;
    }

    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseBiasOrSideInput: ", conv->ToString());
        })) {
      continue;
    }

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_RETURN_IF_ERROR(new_conv->set_backend_config(config));
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}

// custom-call(..., alpha * side_input) ->
// custom-call(..., side_input, backend_config={alpha}).
//
// We also have to support the more complicated case of
//
//   custom-call(..., reshape(side_input * alpha)) -->
//   custom-call(..., reshape(side_input), backend_config={alpha}),
//
// where `reshape` can be an arbitrary chain of reshapes+transposes.  This idiom
// is created by the ReshapeMover pass.
StatusOr<bool> FuseSideInputAlpha(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* conv;
    HloInstruction* side_input;
    auto pattern = m::Op(&conv)
                       .WithPredicate(IsConvCustomCall)
                       .WithOperand(3, m::Op(&side_input));
    if (!Match(instr, pattern)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(auto config,
                        conv->backend_config<CudnnConvBackendConfig>());
    if (config.side_input_scale() != 1) {
      continue;
    }

    // Given side_input, pattern match the following (working from bottom up).
    //
    // before_reshape = multiply(base, broadcast(alpha))
    // side_input = chain_of_reshapes_and_transposes(before_reshape)
    //
    // where alpha is a scalar constant.
    //
    // alpha is f32 except for f64 convs, where it's f64.  See
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    HloInstruction* before_reshape = side_input;
    while (before_reshape->opcode() == HloOpcode::kReshape ||
           before_reshape->opcode() == HloOpcode::kTranspose) {
      before_reshape = before_reshape->mutable_operand(0);
    }

    PrimitiveType conv_ty = conv->shape().tuple_shapes(0).element_type();
    PrimitiveType alpha_ty = conv_ty == F64 ? F64 : F32;
    HloInstruction* base;
    HloInstruction* alpha;
    if (!Match(
            before_reshape,
            m::MultiplyAnyOrder(
                m::Op(&base),
                m::Broadcast(m::ConstantEffectiveScalar(&alpha).WithPredicate(
                    [&](const HloInstruction* instr) {
                      return IsLosslesslyConvertibleTo(instr, alpha_ty);
                    }))))) {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseSideInputAlpha: ", conv->ToString());
        })) {
      continue;
    }

    // Rewrite conv's operand 3 to
    //
    //   chain_of_reshapes_and_transposes(before_reshape).
    //
    // and store alpha in the conv's backend config.
    //
    // We're going to do something bad here: We aren't going to check that the
    // chain of reshapes/transposes has one use, so we're potentially
    // duplicating all these instructions (once with alpha and once without).
    //
    // This is justified because
    //
    //  - duplicating reshapes/transposes shouldn't be "that bad" -- these
    //    instructions can usually be fused, and
    //
    //  - *not* fusing alpha can be catastrophic.  For s8->s8 convolutions, the
    //    side-input must be s8.  But the product side_input * alpha is f32, so
    //    we can only see that side-input is s8 if we fuse alpha. IOW not fusing
    //    alpha means we'll run this s8->s8 conv as s8->f32, which is *much*
    //    slower than some extra transposes.

    // Recursively clone chain_of_reshapes_and_transposes until we get to
    // `before_reshape`, at which point we skip the multiply(base, alpha) and
    // just return base.
    std::function<HloInstruction*(const HloInstruction*)> clone =
        [&](const HloInstruction* instr) {
          if (instr == before_reshape) {
            return base;
          }
          CHECK(instr->opcode() == HloOpcode::kReshape ||
                instr->opcode() == HloOpcode::kTranspose)
              << "Must be reshape or transpose: " << instr->ToString();
          return comp->AddInstruction(instr->CloneWithNewOperands(
              instr->shape(), {clone(instr->operand(0))}));
        };
    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands[3] = clone(side_input);

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(conv->shape(), new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());

    TF_ASSIGN_OR_RETURN(Literal alpha_f64, alpha->literal().Convert(F64));
    config.set_side_input_scale(alpha_f64.GetFirstElement<double>());
    TF_RETURN_IF_ERROR(new_conv->set_backend_config(config));

    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, new_conv));
    changed = true;
  }
  return changed;
}

StatusOr<bool> FuseRelu(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte;
    HloInstruction* conv;
    if (!Match(
            instr,
            m::MaximumAnyOrder(
                m::Broadcast(m::ConstantEffectiveScalar(0)),
                m::GetTupleElement(
                    &gte,
                    m::Op(&conv).WithPredicate(IsConvCustomCall).WithOneUse())
                    .WithOneUse()))) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig config,
                        conv->backend_config<CudnnConvBackendConfig>());
    if (config.activation_mode() != se::dnn::kNone) {
      continue;
    }

    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseRelu: ", conv->ToString());
        })) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(conv, EnsureIsConvBiasActivation(conv));
    config.set_activation_mode(se::dnn::kRelu);
    TF_RETURN_IF_ERROR(conv->set_backend_config(config));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, gte));
    changed = true;
  }
  return changed;
}

StatusOr<bool> FuseConvertToF16(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte = nullptr;
    HloInstruction* conv = nullptr;

    auto f32_convertible_to_f16_pattern =
        m::Op().WithElementType(F32).WithPredicate(
            IsLosslesslyConvertibleToF16);
    auto pattern =
        m::Convert(
            m::GetTupleElement(
                &gte,
                m::Op(&conv)
                    .WithPredicate(IsConvCustomCall)
                    .WithOperand(0, f32_convertible_to_f16_pattern)
                    .WithOperand(1, f32_convertible_to_f16_pattern)
                    .WithOperandIfPresent(2, f32_convertible_to_f16_pattern)
                    .WithOperandIfPresent(3, f32_convertible_to_f16_pattern),
                0)
                .WithOneUse())
            .WithElementType(F16);
    if (!Match(instr, pattern)) {
      VlogIfFailureToMatch(
          instr, pattern, "fp16 conv",
          m::Op().WithOperand(
              0, m::GetTupleElement(m::Op().WithPredicate(IsConvCustomCall))));
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertToF16: ", conv->ToString());
        })) {
      continue;
    }

    VLOG(2) << "Matched fp16 conv: " << conv->ToString();

    // In fp16 convs, all operands, including `bias`, must be fp16.  This is
    // different from int8 convs, where the bias is fp32.  See table of
    // supported datatypes at
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    absl::InlinedVector<HloInstruction*, 4> new_operands;
    for (HloInstruction* operand : conv->operands()) {
      new_operands.push_back(MakeConvertToHlo(operand, F16));
    }

    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(F16);

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(new_shape, new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}

StatusOr<bool> FuseConvertToS8(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* gte = nullptr;
    HloInstruction* conv = nullptr;

    auto conv_pattern =
        m::Op(&conv)
            .WithPredicate(IsConvCustomCall)
            .WithOperand(0, m::Op().WithPredicate(IsLosslesslyConvertibleToS8))
            .WithOperand(1, m::Op().WithPredicate(IsLosslesslyConvertibleToS8));

    // int8 -> int8 conv
    auto s8_pattern =
        m::Convert(
            m::Clamp(
                m::Broadcast(m::ConstantEffectiveScalar(-128)),
                m::GetTupleElement(
                    &gte,
                    conv_pattern.WithOperandIfPresent(
                        3, m::Op().WithPredicate(IsLosslesslyConvertibleToS8)),
                    0)
                    .WithOneUse(),
                m::Broadcast(m::ConstantEffectiveScalar(127))))
            .WithElementType(S8);

    // int8 -> fp32 conv
    auto f32_pattern = m::GetTupleElement(&gte,
                                          conv_pattern.WithOperandIfPresent(
                                              3, m::Op().WithElementType(F32)),
                                          0)
                           .WithElementType(F32);

    VlogIfFailureToMatch(
        instr, s8_pattern, "s8->s8 conv",
        m::Convert(m::Clamp(m::Op(),  //
                            m::GetTupleElement(
                                m::Op().WithPredicate(IsConvCustomCall)),  //
                            m::Op()))
            .WithElementType(S8));

    VlogIfFailureToMatch(
        instr, f32_pattern, "s8->f32 conv",
        m::GetTupleElement(m::Op().WithPredicate(IsConvCustomCall))
            .WithElementType(F32));

    PrimitiveType conv_output_ty;
    if (Match(instr, s8_pattern)) {
      conv_output_ty = S8;
    } else if (Match(instr, f32_pattern)) {
      conv_output_ty = F32;
    } else {
      continue;
    }
    if (!ConsumeFuel("cudnn-fused-convolution-rewriter", [&] {
          return absl::StrCat("FuseConvertToS8: ", conv->ToString());
        })) {
      continue;
    }

    absl::InlinedVector<HloInstruction*, 4> new_operands(
        conv->operands().begin(), conv->operands().end());
    new_operands[0] = MakeConvertToHlo(new_operands[0], S8);
    new_operands[1] = MakeConvertToHlo(new_operands[1], S8);
    // Don't convert bias (operand 2); it's always f32 for s8 ops in cudnn.  See
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    if (new_operands.size() >= 4) {
      // side-input always matches conv output type.  We checked in the patterns
      // above that it's losslessly-convertible to this type.
      new_operands[3] = MakeConvertToHlo(new_operands[3], conv_output_ty);
    }

    Shape new_shape = conv->shape();
    new_shape.mutable_tuple_shapes(0)->set_element_type(conv_output_ty);

    HloInstruction* new_conv = comp->AddInstruction(
        conv->CloneWithNewOperands(new_shape, new_operands));
    comp->parent()->SetAndUniquifyInstrName(new_conv, conv->name());
    TF_ASSIGN_OR_RETURN(HloInstruction * new_instr,
                        MakeGetTupleElementHlo(new_conv, 0));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_instr));
    changed = true;
  }
  return changed;
}

Status CheckNoIllegalIntegerConvs(HloComputation* comp) {
  auto is_integral_not_s8 = [](const Shape& s) {
    return primitive_util::IsIntegralType(s.element_type()) &&
           s.element_type() != S8;
  };

  std::vector<HloInstruction*> bad_convs;
  for (HloInstruction* instr : comp->instructions()) {
    if (!IsConvCustomCall(instr)) {
      continue;
    }
    if (is_integral_not_s8(instr->shape().tuple_shapes(0)) ||
        is_integral_not_s8(instr->operand(0)->shape()) ||
        is_integral_not_s8(instr->operand(1)->shape()) ||
        (instr->operand_count() >= 4 &&
         is_integral_not_s8(instr->operand(3)->shape()))) {
      bad_convs.push_back(instr);
    }
  }

  if (bad_convs.empty()) {
    return Status::OK();
  }

  return Unimplemented(
      R"(
Can't lower one or more integer convolutions to idioms supported by CuDNN.

CuDNN integer convolutions must have:

  - s8 input and filter,
  - f32 bias (if present),
  - s8 or f32 output, and
  - s8 side_input (if present) if output is s8.

For each of the unsupported convs below, we weren't able to lower one of the
operands or the output to the appropriate type.

See specific HLO idioms in cudnn_fused_conv_rewriter.h, and see cudnn semantics:

https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward and
https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters

Unsupported convs:
%s

******* Full HLO module *******
%s
)",
      absl::StrJoin(bad_convs, "\n",
                    [](std::string* out, HloInstruction* instr) {
                      absl::StrAppend(out, " - ", instr->ToString());
                    }),
      comp->parent()->ToString());
}

void VlogStats(HloModule* module) {
  if (!VLOG_IS_ON(1)) {
    return;
  }

  VLOG(1) << "Results of CudnnFusedConvRewriter for " << module->name();
  absl::flat_hash_map<std::string, int> stats;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!Match(instr, m::Op().WithPredicate(IsConvCustomCall))) {
        continue;
      }

      VLOG(3) << instr->ToString();

      if (instr->custom_call_target() == kCudnnConvForwardCallTarget) {
        stats["01 non-fused forward convs"]++;
      } else if (instr->custom_call_target() ==
                 kCudnnConvBiasActivationForwardCallTarget) {
        stats["02 fused forward convs"]++;
      }

      PrimitiveType conv_in_ty = instr->operand(0)->shape().element_type();
      PrimitiveType conv_out_ty = instr->shape().tuple_shapes(0).element_type();
      if (conv_in_ty == F32) {
        stats["10 f32 convs"]++;
      } else if (conv_in_ty == F16) {
        stats["11 f16 convs"]++;
      } else if (conv_in_ty == S8) {
        if (conv_out_ty == S8) {
          stats["12 s8->s8 convs"]++;
        } else if (conv_out_ty == F32) {
          stats["13 s8->f32 convs"]++;
        } else {
          LOG(ERROR) << "Unexpected conv: " << instr->ToString();
        }
      }

      if (instr->operand_count() > 2) {
        stats["20 convs with bias"]++;
        if (Match(instr->operand(2),
                  m::Broadcast(m::ConstantEffectiveScalar(0)))) {
          stats["21 convs with 0 bias"]++;
        }
      }
      if (instr->operand_count() > 3) {
        stats["22 convs with side-input"]++;
      }

      auto config = instr->backend_config<CudnnConvBackendConfig>();
      if (!config.ok()) {
        LOG(ERROR) << "Couldn't parse backend config for " << instr->ToString();
        continue;
      }

      if (config->conv_result_scale() != 1) {
        stats["30 convs with result scale"]++;
      }
      if (config->side_input_scale() != 0 && config->side_input_scale() != 1) {
        stats["31 convs with side-input scale"]++;
      }
      stats[absl::StrCat(
          "32 convs with activation mode ",
          se::dnn::ActivationMode_Name(config->activation_mode()))]++;
    }
  }

  std::vector<std::pair<std::string, int>> stats_sorted(stats.begin(),
                                                        stats.end());
  absl::c_sort(stats_sorted);
  for (const auto& kv : stats_sorted) {
    VLOG(1) << absl::StreamFormat("%4d %s", kv.second,
                                  absl::string_view(kv.first).substr(3));
  }
}

}  // namespace

StatusOr<bool> CudnnFusedConvRewriter::Run(HloModule* module) {
  bool any_changed = false;

  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    // Fuse "inside out" starting with the operations closest to the conv.
    bool changed = false;

    TF_ASSIGN_OR_RETURN(changed, FuseConvertToFloat(comp));
    any_changed |= changed;

    TF_ASSIGN_OR_RETURN(changed, FuseConvAlpha(comp));
    any_changed |= changed;

    // s8 convs' bias and side-input appear before conversion to s8.
    //
    // Run FuseBiasOrSideInput twice, so we get both the bias and the side
    // input, if both are present.
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseSideInputAlpha(comp));
    any_changed |= changed;

    // Relu might appear before or after convert-to-f16/s8, so we check in both
    // cases.
    TF_ASSIGN_OR_RETURN(changed, FuseRelu(comp));
    any_changed |= changed;

    TF_ASSIGN_OR_RETURN(changed, FuseConvertToF16(comp));
    any_changed |= changed;

    TF_ASSIGN_OR_RETURN(changed, FuseConvertToS8(comp));
    any_changed |= changed;

    // f16 convs' bias+side-input can appear before or after conversion to f16.
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseBiasOrSideInput(comp));
    any_changed |= changed;
    TF_ASSIGN_OR_RETURN(changed, FuseSideInputAlpha(comp));
    any_changed |= changed;

    TF_ASSIGN_OR_RETURN(changed, FuseRelu(comp));
    any_changed |= changed;

    // Check that we don't have any convs outputing integer types other than s8.
    // cudnn does not support these.  They should have been transformed to
    // int8->int8 or int8->float above.
    TF_RETURN_IF_ERROR(CheckNoIllegalIntegerConvs(comp));
  }

  VlogStats(module);

  return any_changed;
}
}  // namespace gpu
}  // namespace xla
