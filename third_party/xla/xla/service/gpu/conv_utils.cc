/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/conv_utils.h"

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla {
namespace gpu {

std::optional<Window> RestoreWindowFromBackwardFilter(
    const HloConvolutionInstruction* conv) {
  Window backward_conv_window;
  const ConvolutionDimensionNumbers& conv_dnums =
      conv->convolution_dimension_numbers();
  auto input_spatial_dims = conv_dnums.input_spatial_dimensions();
  auto output_spatial_dims = conv_dnums.output_spatial_dimensions();

  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    WindowDimension* dim = backward_conv_window.add_dimensions();
    // The window size of the backward convolution equals the output size of the
    // forward convolution.
    int64_t filter_size = conv->shape().dimensions(output_spatial_dims[i]);
    dim->set_size(filter_size);
    // The window stride equals the window dilation of the forward convolution.
    dim->set_stride(conv->window().dimensions(i).window_dilation());
    // The window's low padding is the same as the low padding of the
    // activations.
    dim->set_padding_low(conv->window().dimensions(i).padding_low());
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);

    int64_t input_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    int64_t output_size = conv->window().dimensions(i).size();
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
    int64_t padded_input_size = filter_size + (output_size - 1) * dim->stride();
    int64_t min_padding_high =
        padded_input_size - input_size - dim->padding_low();
    int64_t max_padding_high = min_padding_high + dim->stride() - 1;
    CHECK_GE(dim->padding_low(), 0);
    // In practice, since cuDNN convolution only supports even padding, we make
    // the amount of high padding the same as the amount of low padding as long
    // as it is between min_padding_high and max_padding_high. If it is not in
    // that range, we pick the one that's closest to dim->padding_low() and let
    // GpuConvPaddingLegalization canonicalize the resultant backward
    // convolution later. Picking the closest one minimizes the cost of the kPad
    // instruction to be inserted by GpuConvPaddingLegalization.
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
      LOG(WARNING)
          << "Fusing this pattern to backward filter convolution would cause "
             "negative padding ("
          << dim->padding_high()
          << ") on right/bottom of the weight gradients, which is not "
             "supported by GpuConvPaddingLegalization (b/32744257). "
             "Falling back to "
             "unfused convolution for instruction: "
          << conv->ToString();
      return std::nullopt;
    }
  }
  return backward_conv_window;
}

std::optional<Window> RestoreWindowFromBackwardInput(
    const HloConvolutionInstruction* conv) {
  ConvolutionDimensionNumbers dnums = conv->convolution_dimension_numbers();
  const auto& input_spatial_dims = dnums.input_spatial_dimensions();
  const auto& output_spatial_dims = dnums.output_spatial_dimensions();
  CHECK_EQ(conv->window().dimensions().size(), input_spatial_dims.size());
  CHECK_EQ(output_spatial_dims.size(), input_spatial_dims.size());

  const Window& old_window = conv->window();
  Window new_window = old_window;
  for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
    // Restore backward convolution's padding config from the matched pattern.
    // See the comment in tensorflow/core/kernels/conv_grad_ops.h for how we
    // convert backward input convolution to a variant of forward convolution.
    //
    // The stride of the backward convolution
    // = the base dilation factor of the forward convolution
    auto dim = new_window.mutable_dimensions(i);
    dim->set_stride(old_window.dimensions(i).base_dilation());
    dim->set_base_dilation(1);

    // The low padding = kernel_size - 1 - low padding on the gradients
    // Make sure the low padding is not negative.
    auto kernel_size = old_window.dimensions(i).size();
    auto backward_padding_low =
        kernel_size - 1 - old_window.dimensions(i).padding_low();
    if (backward_padding_low < 0) {
      LOG(WARNING)
          << "The low padding of the backward convolution would be negative ("
          << backward_padding_low
          << "), which isn't supported by GpuConvPaddingLegalization "
             "for now (b/32744257).";
      return std::nullopt;
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
      // and GpuConvPaddingLegalization will later insert kSlice
      // instructions to enforce even padding.
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
    // GpuConvPaddingLegalization doesn't handle backward input
    // convolution with negative padding for now. So fall back to unfused
    // convolution in case of negative padding. For example,
    //   ABCD = Conv(abc, reverse(xy), padding_high=2)
    // could be fused to
    //   ABCD = BackwardInputConv(abc, xy, padding_low=1, padding_high=-1)
    // with positive padding low but negative padding high.
    if (dim->padding_high() < 0) {
      LOG(WARNING) << "Fusing this pattern to backward convolution would cause "
                      "negative padding ("
                   << dim->padding_high()
                   << ") on right/bottom of the activations, which is not "
                      "supported by GpuConvPaddingLegalization (b/32744257). "
                      "Falling back to unfused convolution for instruction: "
                   << conv->ToString();
      return std::nullopt;
    }
  }
  return new_window;
}

using ConvKind = HloConvolutionInstruction::ConvKind;

std::optional<Window> RestoreWindow(const HloConvolutionInstruction* conv) {
  ConvKind conv_kind = conv->conv_kind();
  if (conv_kind == ConvKind::WGRAD) {
    return RestoreWindowFromBackwardFilter(conv);
  } else if (conv_kind == ConvKind::DGRAD) {
    return RestoreWindowFromBackwardInput(conv);
  }
  return conv->window();
}

ConvolutionDimensionNumbers RestoreDimNumberFromBackwardInput(
    const HloConvolutionInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ConvolutionDimensionNumbers dnums_for_layout = dnums;

  dnums_for_layout.set_kernel_input_feature_dimension(
      dnums.kernel_output_feature_dimension());
  dnums_for_layout.set_kernel_output_feature_dimension(
      dnums.kernel_input_feature_dimension());
  return dnums_for_layout;
}

ConvolutionDimensionNumbers RestoreDimNumberFromBackwardFilter(
    const HloConvolutionInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ConvolutionDimensionNumbers dnums_for_layout = dnums;

  dnums_for_layout.set_input_batch_dimension(dnums.input_feature_dimension());
  dnums_for_layout.set_input_feature_dimension(dnums.input_batch_dimension());
  dnums_for_layout.set_output_batch_dimension(dnums.output_feature_dimension());
  dnums_for_layout.set_output_feature_dimension(dnums.output_batch_dimension());
  dnums_for_layout.set_kernel_input_feature_dimension(
      dnums.kernel_output_feature_dimension());
  dnums_for_layout.set_kernel_output_feature_dimension(
      dnums.kernel_input_feature_dimension());
  return dnums_for_layout;
}

ConvolutionDimensionNumbers RestoreDimNumber(
    const HloConvolutionInstruction* conv) {
  ConvKind conv_kind = conv->conv_kind();
  if (conv_kind == ConvKind::WGRAD) {
    return RestoreDimNumberFromBackwardFilter(conv);
  } else if (conv_kind == ConvKind::DGRAD) {
    return RestoreDimNumberFromBackwardInput(conv);
  }
  return conv->convolution_dimension_numbers();
}

}  // end namespace gpu
}  // end namespace xla
