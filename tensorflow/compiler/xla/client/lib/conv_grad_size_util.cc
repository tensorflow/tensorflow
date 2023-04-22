/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/conv_grad_size_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

StatusOr<SpatialDimensionOutputSizeAndPadding> GetWindowedOutputSize(
    int64 input_size, int64 filter_size, int64 dilation_rate, int64 stride,
    Padding padding_type) {
  if (stride <= 0) {
    return tensorflow::errors::InvalidArgument("Stride must be > 0, but got ",
                                               stride);
  }
  if (dilation_rate < 1) {
    return tensorflow::errors::InvalidArgument(
        "Dilation rate must be >= 1, but got ", dilation_rate);
  }

  int64 effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  SpatialDimensionOutputSizeAndPadding dim;
  switch (padding_type) {
    case Padding::kValid:
      dim.output_size = (input_size - effective_filter_size + stride) / stride;
      dim.pad_before = dim.pad_after = 0;
      break;
    case Padding::kSame:
      dim.output_size = (input_size + stride - 1) / stride;
      const int64 padding_needed =
          std::max(int64{0}, (dim.output_size - 1) * stride +
                                 effective_filter_size - input_size);
      // For odd values of total padding, add more padding on the "after" side
      // of the given dimension.
      dim.pad_before = padding_needed / 2;
      dim.pad_after = padding_needed - dim.pad_before;
      break;
  }
  if (dim.output_size < 0) {
    return tensorflow::errors::InvalidArgument(
        "Computed output size would be negative: ", dim.output_size,
        " [input_size: ", input_size,
        ", effective_filter_size: ", effective_filter_size,
        ", stride: ", stride, "]");
  }
  return dim;
}

}  // namespace

StatusOr<SpatialDimensionOutputSizeAndPadding>
ConvGradExtractAndVerifyDimension(int64 input_size, int64 filter_size,
                                  int64 output_size, int64 dilation,
                                  int64 stride, Padding padding) {
  TF_ASSIGN_OR_RETURN(SpatialDimensionOutputSizeAndPadding output_dim,
                      GetWindowedOutputSize(input_size, filter_size, dilation,
                                            stride, padding));
  if (output_size != output_dim.output_size) {
    return tensorflow::errors::InvalidArgument(
        "Size of out_backprop doesn't match computed: ", "actual = ",
        output_size, ", computed = ", output_dim.output_size,
        " input: ", input_size, " filter: ", filter_size,
        " output: ", output_size, " stride: ", stride, " dilation: ", dilation);
  }

  SpatialDimensionOutputSizeAndPadding dim;
  int64 effective_filter_size = (filter_size - 1) * dilation + 1;
  dim.output_size = (output_dim.output_size - 1) * stride + 1;
  const auto padded_out_size = input_size + effective_filter_size - 1;
  dim.pad_before = effective_filter_size - 1 - output_dim.pad_before;
  dim.pad_after = padded_out_size - dim.output_size - dim.pad_before;
  VLOG(2) << "expanded_out = " << dim.output_size
          << ", effective_filter_size = " << effective_filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim.pad_before
          << ", pad_after = " << dim.pad_after << ", dilation = " << dilation
          << ", strides = " << stride;
  return dim;
}

}  // namespace xla
