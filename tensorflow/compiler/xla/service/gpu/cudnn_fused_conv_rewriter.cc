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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {
namespace {

// Describes matched patterns:
//   max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias));
//   for floating point types or
//   max(0, alpha1 * conv<float>(int8_x, int8_w) + alpha2 *
//   * side_input + broadcast(bias));
//   for int8.
// Where side_input has the shape of output buffer, and bias is a 1D array with
// the dimension of number of output features.
struct ConvWithRelu {
  HloInstruction* maximum;
  HloCustomCallInstruction* conv;
  HloInstruction* bias;
  HloInstruction* side_input;
  HloConstantInstruction* alpha_conv;
  HloConstantInstruction* alpha_side_input;
};

// The pattern we want to match:
//   max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias));
//   or
//   max(0, alpha1 * conv<float>(int8_x, int8_w) + alpha2 *
//   * side_input + broadcast(bias));
// With its variants involving commute/reassociation of adds, multiplies, and
// max, and omission of alpha1, side_input, alpha2, or bias.
absl::optional<ConvWithRelu> FindConvWithRelu(HloInstruction* instr) {
  using match::Add;
  using match::AddAnyOrder;
  using match::AnyOf;
  using match::Broadcast;
  using match::ConstantScalar;
  using match::GetTupleElement;
  using match::Maximum;
  using match::MultiplyAnyOrder;
  using match::Op;

  HloInstruction* relu_input;

  // Match max(0, relu_input).
  auto zero_pattern = Broadcast(ConstantScalar(0));
  if (!Match(instr, Maximum(zero_pattern, Op(&relu_input))) &&
      !Match(instr, Maximum(Op(&relu_input), zero_pattern))) {
    return absl::nullopt;
  }
  HloInstruction* conv_instr = nullptr;
  HloInstruction* alpha_conv_instr = nullptr;
  HloInstruction* alpha_side_input_instr = nullptr;
  HloInstruction* bias_broadcast_instr = nullptr;
  HloInstruction* bias = nullptr;
  HloInstruction* side_input = nullptr;

  // These nodes will not be in the returned value, but we need to check them
  // for single use.
  HloInstruction *gte = nullptr, *add1 = nullptr, *add2 = nullptr,
                 *mul1 = nullptr, *mul2 = nullptr;

  const auto bias_pattern = Broadcast(&bias_broadcast_instr, Op(&bias));
  const auto conv_pattern = [&] {
    auto alpha_pattern = Broadcast(ConstantScalar(&alpha_conv_instr));
    auto conv_pattern = GetTupleElement(
        &gte, Op(&conv_instr).WithOpcode(HloOpcode::kCustomCall), 0);
    return AnyOf<HloInstruction>(
        MultiplyAnyOrder(&mul1, alpha_pattern, conv_pattern), conv_pattern);
  }();
  const auto side_input_pattern = [&] {
    auto alpha_pattern = Broadcast(ConstantScalar(&alpha_side_input_instr));
    // If bias is already matched, match arbitrary additional input as side
    // input. Note this may force a cheap operation (e.g. broadcast) to be
    // materialized into a large buffer, as large as the output buffer.
    //
    // TODO(timshen): If in practice there are significant false positives, we
    // should fix it.
    auto side_input_pattern = Op(&side_input);
    return AnyOf<HloInstruction>(
        MultiplyAnyOrder(&mul2, alpha_pattern, side_input_pattern),
        side_input_pattern);
  }();

  {
    // Try to match any of the following form of add, in any association:
    //   addends[0]
    //   addends[0] + addends[1]
    //   addends[0] + addends[1] + addends[2]
    //
    // Then try to match each addend with one of the three patterns: bias, conv,
    // or side_input. Notice that side_input matching must go last, as it
    // also matches a conv or a bias.
    HloInstruction* addends[3] = {nullptr, nullptr, nullptr};
    auto add3_pattern = [&] {
      auto add2_pattern = Add(&add1, Op(&addends[0]), Op(&addends[1]));
      return AnyOf<HloInstruction>(
          AddAnyOrder(&add2, add2_pattern, Op(&addends[2])), add2_pattern,
          Op(&addends[0]));
    }();
    CHECK(Match(relu_input, add3_pattern));
    for (auto addend : addends) {
      if (addend) {
        if (bias == nullptr && Match(addend, bias_pattern)) {
          CHECK(bias);
        } else if (conv_instr == nullptr && Match(addend, conv_pattern)) {
          CHECK(conv_instr);
        } else if (side_input == nullptr && Match(addend, side_input_pattern)) {
          CHECK(side_input);
        } else {
          return absl::nullopt;
        }
      }
    }
  }

  if (conv_instr == nullptr) {
    return absl::nullopt;
  }

  for (HloInstruction* instr :
       {conv_instr, bias_broadcast_instr, gte, add1, add2, mul1, mul2}) {
    if (instr && instr->user_count() > 1) {
      return absl::nullopt;
    }
  }

  auto conv = Cast<HloCustomCallInstruction>(conv_instr);
  auto bias_broadcast =
      CastOrNull<HloBroadcastInstruction>(bias_broadcast_instr);

  if (conv->custom_call_target() != kCudnnConvForwardCallTarget) {
    return absl::nullopt;
  }

  // In order to map to cudnnConvolutionBiasActivationForward for int8, the
  // convolution output is float, i.e. conv<float>(int8_x, int8_w)
  if (conv->operand(0)->shape().element_type() == xla::S8) {
    if (conv->shape().tuple_shapes(0).element_type() != xla::F32) {
      return absl::nullopt;
    }
  }

  if (bias_broadcast) {
    // TODO(timshen): handle bias_broadcast_instr->dimensions() == {}.
    if (bias_broadcast_instr->dimensions().size() != 1) {
      return absl::nullopt;
    }
    if (bias_broadcast_instr->dimensions(0) !=
        conv->convolution_dimension_numbers().output_feature_dimension()) {
      return absl::nullopt;
    }
  }

  return ConvWithRelu{
      instr,
      conv,
      bias,
      side_input,
      CastOrNull<HloConstantInstruction>(alpha_conv_instr),
      CastOrNull<HloConstantInstruction>(alpha_side_input_instr)};
}

StatusOr<std::unique_ptr<HloInstruction>> TryRewriteToCudnnForwardRelu(
    ConvWithRelu match) {
  auto conv = match.conv;

  HloComputation* computation = conv->parent();

  const auto get_alpha_value =
      [](HloConstantInstruction* instr) -> StatusOr<double> {
    TF_ASSIGN_OR_RETURN(
        auto alpha,
        Cast<HloConstantInstruction>(instr)->literal().Convert(F64));
    return alpha.GetFirstElement<double>();
  };

  double alpha_conv = 1;
  if (match.alpha_conv) {
    TF_ASSIGN_OR_RETURN(alpha_conv, get_alpha_value(match.alpha_conv));
  }

  double alpha_side_input;
  if (match.side_input) {
    if (match.alpha_side_input) {
      TF_ASSIGN_OR_RETURN(alpha_side_input,
                          get_alpha_value(match.alpha_side_input));
    } else {
      alpha_side_input = 1;
    }
  } else {
    CHECK(match.alpha_side_input == nullptr);
    alpha_side_input = 0;
  }

  auto bias = match.bias;
  if (!bias) {
    PrimitiveType conv_output_type =
        conv->shape().tuple_shapes(0).element_type();
    auto zero = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(conv_output_type)));

    int64_t num_output_feature = conv->shape().tuple_shapes(0).dimensions(
        conv->convolution_dimension_numbers().output_feature_dimension());
    bias = computation->AddInstruction(HloInstruction::CreateBroadcast(
        ShapeUtil::MakeShapeWithDescendingLayout(conv_output_type,
                                                 {num_output_feature}),
        zero, {}));
  }

  CHECK(bias);
  std::vector<HloInstruction*> args = {conv->mutable_operand(0),
                                       conv->mutable_operand(1), bias};
  if (match.side_input) {
    args.push_back(match.side_input);
  }
  auto new_conv = computation->AddInstruction(HloInstruction::CreateCustomCall(
      conv->shape(), args, kCudnnConvBiasActivationForwardCallTarget));
  new_conv->set_feature_group_count(conv->feature_group_count());
  new_conv->set_window(conv->window());
  new_conv->set_convolution_dimension_numbers(
      conv->convolution_dimension_numbers());
  new_conv->set_metadata(conv->metadata());
  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig config,
                      conv->backend_config<CudnnConvBackendConfig>());
  config.set_activation_mode(
      static_cast<int64>(se::dnn::ActivationMode::kRelu));
  config.set_conv_result_scale(alpha_conv);
  config.set_side_input_scale(alpha_side_input);
  TF_RETURN_IF_ERROR(new_conv->set_backend_config(config));

  VLOG(1) << "Replacing convolution " << conv->ToString() << " with "
          << new_conv->ToString();
  return HloInstruction::CreateGetTupleElement(conv->shape().tuple_shapes(0),
                                               new_conv, 0);
}

// Fuse bias/scaling/ReLU with convolution custom call with floating point
// output
StatusOr<bool> RunFuseBiasSideActivation(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    std::vector<ConvWithRelu> matches;
    int num_forward_convs = 0;
    for (auto instr : computation->instructions()) {
      auto match = FindConvWithRelu(instr);
      if (match.has_value()) {
        matches.push_back(*match);
      }
      if (auto call = DynCast<HloCustomCallInstruction>(instr)) {
        if (call->custom_call_target() == kCudnnConvForwardCallTarget) {
          num_forward_convs++;
        }
      }
    }
    VLOG(1) << "Identified cuDNN forward conv + relu: " << matches.size()
            << " out of " << num_forward_convs << " forward convs.";
    std::vector<std::pair<HloInstruction*, std::unique_ptr<HloInstruction>>>
        replacements;
    for (const ConvWithRelu& match : matches) {
      TF_ASSIGN_OR_RETURN(auto new_instr, TryRewriteToCudnnForwardRelu(match));
      replacements.push_back({match.maximum, std::move(new_instr)});
      changed = true;
    }
    for (auto& replacement : replacements) {
      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          replacement.first, std::move(replacement.second)));
    }
  }
  return changed;
}

// Describes a matched pattern:
// convert_or_clamp(get_tuple_element(custom_call(x,w, ...)));
// where the custom_call targets CuDNN convolution (either pure convolution or
// fused convolution).
struct ConvWithConvertOrClamp {
  HloInstruction* convert_or_clamp;
  HloInstruction* gte;
  HloCustomCallInstruction* conv;
};

// The pattern we want to match:
//   convert<int8>(clamp(broadcast(-128), (get_tuple_element(custom_call(int8_x,
//   int8_w, ...)), broadcast(127));
absl::optional<ConvWithConvertOrClamp> FindConvWithClampAndConvertToInt8(
    HloInstruction* instr) {
  using match::Broadcast;
  using match::Clamp;
  using match::Convert;
  using match::GetTupleElement;
  using match::Op;

  HloInstruction* gte = nullptr;
  HloInstruction* conv_instr = nullptr;
  auto lower_pattern = Broadcast(match::ConstantScalar(-128));
  auto upper_pattern = Broadcast(match::ConstantScalar(127));
  auto pattern = Convert(
      Clamp(lower_pattern,
            GetTupleElement(
                &gte, Op(&conv_instr).WithOpcode(HloOpcode::kCustomCall), 0),
            upper_pattern));

  if (Match(instr, pattern)) {
    if (conv_instr->operand(0)->shape().element_type() == xla::S8 &&
        instr->shape().element_type() == xla::S8) {
      HloCustomCallInstruction* conv =
          CastOrNull<HloCustomCallInstruction>(conv_instr);
      return ConvWithConvertOrClamp{instr, gte, conv};
    }
  }
  return absl::nullopt;
}

// A help function to rewrite convert_or_clamp_or_other<new_type>(gte(conv()))
// to gte<new_type>(conv<new_type>()).  It bypasses convert_or_clamp_or_other
// and set the output data type on gte and conv.
Status RewriteForConvertOrClampImpl(ConvWithConvertOrClamp match) {
  auto conv = match.conv;
  auto gte = match.gte;
  auto convert_or_clamp = match.convert_or_clamp;

  // Change type on conv and gte
  auto convert_out_type = convert_or_clamp->shape().element_type();
  conv->mutable_shape()->mutable_tuple_shapes(0)->set_element_type(
      convert_out_type);
  gte->mutable_shape()->set_element_type(convert_out_type);

  // Remove clamp/convert and so on and just keep
  // get_tuple_element(custom_call(x,w, ...))
  TF_RETURN_IF_ERROR(convert_or_clamp->ReplaceAllUsesWithDifferentShape(gte));
  TF_RETURN_IF_ERROR(
      conv->parent()->RemoveInstructionAndUnusedOperands(convert_or_clamp));
  return Status::OK();
}

Status RewriteForFinalOutput(ConvWithConvertOrClamp match) {
  // When the matched clamp has a single user, which is convert<int8>, we
  // will absorb it, if
  // 1. the side_input matches a convert<float>(int8_side_input), or
  // 2. there is no side input
  const auto is_one_to_one_X_to_Y_cast = [](const HloInstruction* instr,
                                            PrimitiveType X,
                                            PrimitiveType Y) -> bool {
    return (instr->opcode() == HloOpcode::kConvert &&
            instr->shape().element_type() == Y && instr->operand_count() == 1 &&
            instr->operand(0)->user_count() == 1 &&
            instr->operand(0)->shape().element_type() == X);
  };

  if (match.conv->operand_count() < 4) {
    // Conv input #3 (zero based) is side_input, after x, w, and bias.
    // Side input doesn't exist in this case.
    TF_RETURN_IF_ERROR(RewriteForConvertOrClampImpl(match));
  } else if (is_one_to_one_X_to_Y_cast(match.conv->operand(3), S8, F32)) {
    // If side_input has a convert_float_to_int8, absorb it as well.
    auto side_converter = match.conv->mutable_operand(3);
    TF_RETURN_IF_ERROR(side_converter->ReplaceAllUsesWithDifferentShape(
        side_converter->mutable_operand(0)));
    TF_RETURN_IF_ERROR(
        side_converter->parent()->RemoveInstructionAndUnusedOperands(
            side_converter));

    TF_RETURN_IF_ERROR(RewriteForConvertOrClampImpl(match));
  }
  return Status::OK();
}

// Fuse the clamp/convert pattern with the int8 convolution custom call
// (either pure or fused) for int8 output
StatusOr<bool> RunFuseClamp(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    std::vector<ConvWithConvertOrClamp> matches;
    for (auto instr : computation->instructions()) {
      auto match = FindConvWithClampAndConvertToInt8(instr);
      if (match.has_value()) {
        matches.push_back(*match);
      }
    }
    for (const ConvWithConvertOrClamp& match : matches) {
      TF_RETURN_IF_ERROR(RewriteForFinalOutput(match));
      changed = true;
    }

    // Report error for any convolution still having int32 output.
    // Although int32 output convolution will trigger other sanity check errors
    // later, we want to give specific error message here.
    for (auto instr : computation->instructions()) {
      if (auto call = DynCast<HloCustomCallInstruction>(instr)) {
        if ((call->custom_call_target() == kCudnnConvForwardCallTarget ||
             call->custom_call_target() ==
                 kCudnnConvBiasActivationForwardCallTarget) &&
            call->shape().tuple_shapes(0).element_type() == xla::S32) {
          return Unimplemented(
              "Integer convolutions for CuDNN must have float or int8 output.  "
              "Use convert to cast output to float or the following pattern to "
              "int8: "
              "clamp(broadcast(-128), conv(int8_x, int8_w, ...), "
              "broadcast(127)).");
        }
      }
    }
  }
  return changed;
}

// The pattern we want to match:
//   convert<float>(get_tuple_element<int32>(custom_call()));
absl::optional<ConvWithConvertOrClamp> FindConvWithConvertToFloat(
    HloInstruction* instr) {
  using match::Convert;
  using match::GetTupleElement;
  using match::Op;

  HloInstruction* gte = nullptr;
  HloInstruction* conv_instr = nullptr;
  auto pattern =
      Convert(GetTupleElement(
                  &gte,
                  Op(&conv_instr)
                      .WithOpcode(HloOpcode::kCustomCall)
                      .WithCustomCallTarget(kCudnnConvForwardCallTarget),
                  0)
                  .WithShape(match::Shape().WithElementType(xla::S32)))
          .WithShape(match::Shape().WithElementType(xla::F32));
  if (Match(instr, pattern)) {
    HloCustomCallInstruction* conv =
        CastOrNull<HloCustomCallInstruction>(conv_instr);
    return ConvWithConvertOrClamp{instr, gte, conv};
  }
  return absl::nullopt;
}

// Transform
// convert<float>(GetTupleElement<int32>(custom_call<int32>(int8_x, int8_w)))
// to
// GetTupleElement<float>(custom_call<int32>(int8_x, int8_w))
StatusOr<bool> RunFuseConvertToFloat(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    std::vector<ConvWithConvertOrClamp> matches;
    for (auto instr : computation->instructions()) {
      auto match = FindConvWithConvertToFloat(instr);
      if (match.has_value()) {
        matches.push_back(*match);
      }
    }

    for (const ConvWithConvertOrClamp& match : matches) {
      TF_RETURN_IF_ERROR(RewriteForConvertOrClampImpl(match));
      changed = true;
    }
  }
  return changed;
}
}  // namespace

StatusOr<bool> CudnnFusedConvRewriter::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(bool fused_for_convert_to_float,
                      RunFuseConvertToFloat(module));

  TF_ASSIGN_OR_RETURN(bool fused_for_bias, RunFuseBiasSideActivation(module));

  TF_ASSIGN_OR_RETURN(bool fused_for_clamp, RunFuseClamp(module));

  return fused_for_convert_to_float || fused_for_bias || fused_for_clamp;
}
}  // namespace gpu
}  // namespace xla
