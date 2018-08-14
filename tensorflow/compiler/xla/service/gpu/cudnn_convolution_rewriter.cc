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

#include "tensorflow/compiler/xla/service/gpu/cudnn_convolution_rewriter.h"

#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

bool CanImplementAsCudnnForwardConv(HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  if (dnums.input_spatial_dimensions_size() > 3) {
    return false;
  }

  // CuDNN does not accept zero-element arguments
  if (ShapeUtil::IsZeroElementArray(conv->operand(0)->shape()) ||
      ShapeUtil::IsZeroElementArray(conv->operand(1)->shape())) {
    return false;
  }

  if (window_util::HasWindowReversal(conv->window())) {
    return false;
  }
  return true;
}

// Try to match a backward filter pattern that contains "conv".
// Precondition: "conv" is a kConvolution.
std::tuple<bool, Window, ConvolutionDimensionNumbers> MatchBackwardFilter(
    HloInstruction* conv) {
  const auto no_match_result =
      std::make_tuple(false, Window(), ConvolutionDimensionNumbers());
  // Step 1: match the instruction pattern without considering the paddings and
  // dimension numbers just yet. We may need some generic pattern matcher
  // similar to third_party/llvm/llvm/include/llvm/IR/PatternMatch.h
  //
  // Backward filter convolution is implemented in XLA as the forward
  // convolution of padded activations and dilated gradients. Padding on
  // activations and dilation on gradients are specified in the "window" field
  // of the forward convolution.
  //
  //        activations  gradients
  //              \         /
  //               v       v
  //              Convolution
  //                 conv
  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());

  // Step 2: match paddings and dimension numbers of the forward convolution.
  const ConvolutionDimensionNumbers& conv_dnums =
      conv->convolution_dimension_numbers();
  auto input_batch_dim = conv_dnums.input_batch_dimension();
  auto input_feature_dim = conv_dnums.input_feature_dimension();
  auto input_spatial_dims = conv_dnums.input_spatial_dimensions();
  auto kernel_input_feature_dim = conv_dnums.kernel_input_feature_dimension();
  auto kernel_output_feature_dim = conv_dnums.kernel_output_feature_dimension();
  auto kernel_spatial_dims = conv_dnums.kernel_spatial_dimensions();
  auto output_batch_dim = conv_dnums.output_batch_dimension();
  auto output_feature_dim = conv_dnums.output_feature_dimension();
  auto output_spatial_dims = conv_dnums.output_spatial_dimensions();

  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have stride of 1.";
      return no_match_result;
    }
    if (window_dim.base_dilation() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have no base (LHS) dilation.";
      return no_match_result;
    }
    if (window_dim.padding_low() < 0) {
      VLOG(1) << "Padding low should be non-negative.";
      return no_match_result;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return no_match_result;
    }
    // Padding high will be checked in Step 3.
  }
  if (input_batch_dim == output_batch_dim &&
      !window_util::HasWindowDilation(conv->window())) {
    VLOG(1) << conv->ToString()
            << " is a regular forward convolution. No need "
               "to fold it to a backward filter convolution.";
    return no_match_result;
  }

  // Step 3: fuse the matched HLOs into a backward convolution instruction.
  //
  // Compute the window of the backward convolution.
  Window backward_conv_window;
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    WindowDimension* dim = backward_conv_window.add_dimensions();
    // The window size of the backward convolution equals the output size of the
    // forward convolution.
    int64 filter_size = conv->shape().dimensions(output_spatial_dims[i]);
    dim->set_size(filter_size);
    // The window stride equals the window dilation of the forward convolution.
    dim->set_stride(conv->window().dimensions(i).window_dilation());
    // The window's low padding is the same as the low padding of the
    // activations.
    dim->set_padding_low(conv->window().dimensions(i).padding_low());

    int64 input_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    int64 output_size = conv->window().dimensions(i).size();
    // Compute the range of the amount of valid high padding. We first compute
    // min_padding_high, the amount of padding on the right/bottom to ensure the
    // last patch ends at the border, i.e.,
    //
    //   input_size + dim->padding_low() + min_padding_high
    //     = (output_size - 1) * stride + filter_size
    //
    // Because convolution ignores trailing incomplete windows, any amount of
    // padding high from min_padding_high to min_padding_high+stride-1
    // (max_padding_high) has the same effect.
    int64 padded_input_size = filter_size + (output_size - 1) * dim->stride();
    int64 min_padding_high =
        padded_input_size - input_size - dim->padding_low();
    int64 max_padding_high = min_padding_high + dim->stride() - 1;
    CHECK_GE(dim->padding_low(), 0);
    // In practice, since cuDNN convolution only supports even padding, we make
    // the amount of high padding the same as the amount of low padding as long
    // as it is between min_padding_high and max_padding_high. If it is not in
    // that range, we pick the one that's closest to dim->padding_low() and let
    // PadInsertion canonicalize the resultant backward convolution later.
    // Picking the closest one minimizes the cost of the kPad instruction to be
    // inserted by PadInsertion.
    if (dim->padding_low() >= min_padding_high &&
        dim->padding_low() <= max_padding_high) {
      dim->set_padding_high(dim->padding_low());
    } else {
      if (dim->padding_low() < min_padding_high) {
        dim->set_padding_high(min_padding_high);
      } else {
        dim->set_padding_high(max_padding_high);
      }
    }
    if (dim->padding_high() < 0) {
      LOG(ERROR)
          << "Fusing this pattern to backward filter convolution would cause "
             "negative padding ("
          << dim->padding_high()
          << ") on right/bottom of the weight gradients, which is not "
             "supported by PadInsertion (b/32744257). Falling back to "
             "unfused convolution for instruction: "
          << conv->ToString();
      return no_match_result;
    }
  }

  // Restore the dimension numbers of the backward convolution from the forward
  // convolution. The two activation dimensions are reversed (batch and
  // feature).
  ConvolutionDimensionNumbers backward_conv_dnums;
  backward_conv_dnums.set_input_batch_dimension(input_feature_dim);
  backward_conv_dnums.set_input_feature_dimension(input_batch_dim);
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_input_spatial_dimensions(input_spatial_dims[i]);
  }
  backward_conv_dnums.set_output_batch_dimension(kernel_input_feature_dim);
  backward_conv_dnums.set_output_feature_dimension(kernel_output_feature_dim);
  for (int i = 0; i < kernel_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_output_spatial_dimensions(kernel_spatial_dims[i]);
  }
  // The dimension numbering of the output of the forward convolution (before
  // transposition) is the same as that of the activations (according to the
  // semantics of kConvolution). The batch dimension of the activations should
  // be treated as the input feature dimension, and the feature dimension should
  // be treated as the output feature.
  backward_conv_dnums.set_kernel_input_feature_dimension(output_batch_dim);
  backward_conv_dnums.set_kernel_output_feature_dimension(output_feature_dim);
  for (int i = 0; i < output_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_kernel_spatial_dimensions(output_spatial_dims[i]);
  }

  return std::make_tuple(true, backward_conv_window, backward_conv_dnums);
}

// Try to match a backward input pattern that contains "conv".
// Precondition: "conv" is a kConvolution.
std::tuple<bool, Window, ConvolutionDimensionNumbers> MatchBackwardInput(
    HloInstruction* conv) {
  const auto no_match_result =
      std::make_tuple(false, Window(), ConvolutionDimensionNumbers());

  // Match instruction pattern.
  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());
  HloInstruction* reverse_filter = conv->mutable_operand(1);

  // Match the reverse of the filter.
  ConvolutionDimensionNumbers dnums = conv->convolution_dimension_numbers();
  const auto& kernel_spatial_dims = dnums.kernel_spatial_dimensions();
  if (reverse_filter->opcode() == HloOpcode::kReverse) {
    if (kernel_spatial_dims.size() != reverse_filter->dimensions().size() ||
        !std::is_permutation(kernel_spatial_dims.begin(),
                             kernel_spatial_dims.end(),
                             reverse_filter->dimensions().begin())) {
      VLOG(1)
          << "Backward input convolution should reverse all kernel dimensions.";
      return no_match_result;
    }
  } else {
    // Possibly 1x1 filter.
    for (int64 i = 0; i < kernel_spatial_dims.size(); ++i) {
      if (conv->window().dimensions(i).size() != 1) {
        VLOG(1) << "The reverse filter is neither a kReverse nor a 1x1 filter: "
                << reverse_filter->ToString();
        return no_match_result;
      }
    }
    if (!window_util::HasBaseDilation(conv->window())) {
      VLOG(1) << conv->ToString()
              << " is a regular forward convolution. No need "
                 "to fold it to a backward input convolution.";
      return no_match_result;
    }
  }

  // Match padding and dilation of the forward convolution.
  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have stride of 1.";
      return no_match_result;
    }
    if (window_dim.window_dilation() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have no window dilation.";
      return no_match_result;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return no_match_result;
    }
  }

  const auto& input_spatial_dims = dnums.input_spatial_dimensions();
  const auto& output_spatial_dims = dnums.output_spatial_dimensions();
  CHECK_EQ(conv->window().dimensions().size(), input_spatial_dims.size());
  CHECK_EQ(output_spatial_dims.size(), input_spatial_dims.size());

  const Window& old_window = conv->window();
  Window new_window = old_window;
  for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
    // Restore backward convolution's padding config from the matched pattern.
    // See the comment in tensorflow/core/kernels/conv_grad_tuple_ops.cc
    // for how we convert backward input convolution to a variant of forward
    // convolution.
    //
    // The stride of the backward convolution
    // = the base dilation factor of the forward convolution
    auto dim = new_window.mutable_dimensions(i);
    dim->set_stride(old_window.dimensions(i).base_dilation());

    // The low padding = kernel_size - 1 - low padding on the gradients
    // Make sure the low padding is not negative.
    auto kernel_size = old_window.dimensions(i).size();
    auto backward_padding_low =
        kernel_size - 1 - old_window.dimensions(i).padding_low();
    if (backward_padding_low < 0) {
      LOG(ERROR)
          << "The low padding of the backward convolution would be negative ("
          << backward_padding_low
          << "), which isn't supported by PadInsertion for now (b/32744257).";
      return no_match_result;
    }
    dim->set_padding_low(backward_padding_low);

    // Compute the range of the amount of padding on the right/bottom of the
    // activations. XLA's convolution requires all patches to be within the
    // padded base. This gives us flexiblity to choose the amount of high
    // padding from a set of values without changing the result of the backward
    // convolution. The minimum amount (min_padding_high) makes the last patch
    // end at the border. The maximum amount (max_padding_high) equals
    // min_padding_high+stride-1 -- max_padding_high+1 would cause the output
    // size to change.
    auto unpadded_input_size = conv->shape().dimensions(output_spatial_dims[i]);
    auto output_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    auto padded_input_size = kernel_size + dim->stride() * (output_size - 1);
    auto total_pad_size = padded_input_size - unpadded_input_size;
    auto min_padding_high = total_pad_size - backward_padding_low;
    auto max_padding_high = min_padding_high + dim->stride() - 1;

    if (backward_padding_low >= min_padding_high &&
        backward_padding_low <= max_padding_high) {
      // In the best case (most likely), if backward_padding_low is in the range
      // of the amounts of valid high padding, we choose backward_padding_low
      // because cuDNN supports even padding only.
      dim->set_padding_high(backward_padding_low);
    } else {
      // Otherwise, we choose the amount that's closest to backward_padding_low,
      // and PadInsertion will later insert kSlice instructions to enforce even
      // padding.
      //
      // For example, consider the backward convolution pattern
      //
      //   ab     xy
      //   | pad  | reverse
      //  .a.b    yx
      //     \   /
      //      ABC
      //
      // The amount of low padding on activations (in backward convolution) is
      //   backward_padding_low = kernel_size - 1 - forward_padding_low
      //                        = 2 - 1 - 1 = 0
      //
      // The amount of padding high must be between 1 and 2, in order to make
      // Conv(ABC, xy, stride=2) produce exactly 2 elements (ab). 0 is not in
      // the range of [1,2], so we pick the closest valid amount of padding
      // high, which is 1 in this case. Therefore, we fuse the above pattern to
      //
      //   ABC = BackwardInputConv(ab, xy, stride=2, padding_high=1)
      if (backward_padding_low < min_padding_high) {
        dim->set_padding_high(min_padding_high);
      } else {
        dim->set_padding_high(max_padding_high);
      }
    }
    // PadInsertion doesn't handle backward input convolution with negative
    // padding for now. So fall back to unfused convolution in case of negative
    // padding. For example,
    //   ABCD = Conv(abc, reverse(xy), padding_high=2)
    // could be fused to
    //   ABCD = BackwardInputConv(abc, xy, padding_low=1, padding_high=-1)
    // with positive padding low but negative padding high.
    if (dim->padding_high() < 0) {
      LOG(ERROR) << "Fusing this pattern to backward convolution would cause "
                    "negative padding ("
                 << dim->padding_high()
                 << ") on right/bottom of the activations, which is not "
                    "supported by PadInsertion (b/32744257). Falling back to "
                    "unfused convolution for instruction: "
                 << conv->ToString();
      return no_match_result;
    }
  }

  // Fuse the matched HLOs into a backward convolution instruction.
  //
  // If the reverse is omitted (for 1x1 filters) in the original pattern, we add
  // it back in the fusion instruction so that later passes (such as
  // PadInsertion) can handle such fusion instructions easily.
  if (reverse_filter->opcode() != HloOpcode::kReverse) {
    reverse_filter = reverse_filter->parent()->AddInstruction(
        HloInstruction::CreateReverse(reverse_filter->shape(), reverse_filter,
                                      AsInt64Slice(kernel_spatial_dims)));
    TF_CHECK_OK(conv->ReplaceOperandWith(/*operand_no=*/1, reverse_filter));
  }
  dnums.set_kernel_input_feature_dimension(
      conv->convolution_dimension_numbers().kernel_output_feature_dimension());
  dnums.set_kernel_output_feature_dimension(
      conv->convolution_dimension_numbers().kernel_input_feature_dimension());

  return std::make_tuple(true, new_window, dnums);
}

// Tries to rewrite a single convolution into a call to cudnn.
StatusOr<bool> RunOnInstruction(HloInstruction* conv) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);

  HloInstruction* custom_call = [&]() -> HloInstruction* {
    bool match;
    Window window;
    ConvolutionDimensionNumbers dnums;

    std::tie(match, window, dnums) = MatchBackwardFilter(conv);
    if (match) {
      return CreateCudnnConvBackwardFilter(
          conv->shape(), conv->mutable_operand(0), conv->mutable_operand(1),
          window, dnums);
    }

    std::tie(match, window, dnums) = MatchBackwardInput(conv);
    if (match) {
      // Backward input conv subsumes the conv plus the reverse in operand 1.
      HloInstruction* reverse = conv->mutable_operand(1);
      CHECK_EQ(reverse->opcode(), HloOpcode::kReverse);
      HloInstruction* rhs = reverse->mutable_operand(0);

      return CreateCudnnConvBackwardInput(
          conv->shape(), conv->mutable_operand(0), rhs, window, dnums);
    }

    // If all else fails, try a forward convolution.
    if (CanImplementAsCudnnForwardConv(conv)) {
      return CreateCudnnConvForward(conv->shape(), conv->mutable_operand(0),
                                    conv->mutable_operand(1), conv->window(),
                                    conv->convolution_dimension_numbers());
    }

    return nullptr;
  }();

  if (custom_call == nullptr) {
    return false;
  }

  // The CustomCall returns a tuple (conv_result, scratch_memory).  Extract out
  // the conv result and replace `conv` with it.
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceWithNewInstruction(
      conv,
      HloInstruction::CreateGetTupleElement(conv->shape(), custom_call, 0)));
  return true;
}

// Rewrites the convolutions in the given computation into calls to cudnn.
// Returns true if it made any changes.
StatusOr<bool> RunOnComputation(HloComputation* computation) {
  std::vector<HloInstruction*> convs;
  for (auto* hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kConvolution) {
      convs.push_back(hlo);
    }
  }

  bool changed = false;
  for (HloInstruction* conv : convs) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(conv));
    changed |= result;
  }
  return changed;
}
}  // namespace

StatusOr<bool> CudnnConvolutionRewriter::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
