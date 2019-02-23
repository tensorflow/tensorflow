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

// Describes a matched pattern:
//   max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias));
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

absl::optional<ConvWithRelu> FindConvWithRelu(HloInstruction* instr) {
  using match::Add;
  using match::AddAnyOrder;
  using match::AnyOf;
  using match::Broadcast;
  using match::Constant;
  using match::GetTupleElement;
  using match::Maximum;
  using match::MultiplyAnyOrder;
  using match::Op;

  // The pattern we want to match:
  //   max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias));
  //
  // With its variants involving commute/reassociation of adds, multiplies, and
  // max, and omission of alpha1, side_input, alpha2, or bias.

  HloInstruction* relu_input;

  // Match max(0, relu_input).
  auto zero_pattern = Broadcast(match::ConstantScalar(0));
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
    auto alpha_pattern = Broadcast(Constant(&alpha_conv_instr));
    auto conv_pattern = GetTupleElement(
        &gte, Op(&conv_instr).WithOpcode(HloOpcode::kCustomCall), 0);
    return AnyOf<HloInstruction>(
        MultiplyAnyOrder(&mul1, alpha_pattern, conv_pattern), conv_pattern);
  }();
  const auto side_input_pattern = [&] {
    auto alpha_pattern = Broadcast(Constant(&alpha_side_input_instr));
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
  PrimitiveType element_type = conv->operand(0)->shape().element_type();

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
    auto zero = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(element_type)));

    int64 num_output_feature = conv->shape().tuple_shapes(0).dimensions(
        conv->convolution_dimension_numbers().output_feature_dimension());
    bias = computation->AddInstruction(HloInstruction::CreateBroadcast(
        ShapeUtil::MakeShapeWithDescendingLayout(element_type,
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

}  // namespace

StatusOr<bool> CudnnFusedConvRewriter::Run(HloModule* module) {
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

}  // namespace gpu
}  // namespace xla
