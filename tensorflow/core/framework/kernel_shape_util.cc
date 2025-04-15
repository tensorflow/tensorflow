/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/kernel_shape_util.h"

#include <algorithm>
#include <array>

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
absl::Status GetWindowedOutputSizeVerbose(
    int64_t input_size, int64_t filter_size, int64_t dilation_rate,
    int64_t stride, Padding padding_type, int64_t* output_size,
    int64_t* padding_before, int64_t* padding_after) {
  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }
  if (dilation_rate < 1) {
    return errors::InvalidArgument("Dilation rate must be >= 1, but got ",
                                   dilation_rate);
  }

  // See also the parallel implementation in GetWindowedOutputSizeFromDimsV2.
  int64_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding_type) {
    case Padding::VALID:
      *output_size = (input_size - effective_filter_size + stride) / stride;
      *padding_before = *padding_after = 0;
      break;
    case Padding::EXPLICIT:
      *output_size = (input_size + *padding_before + *padding_after -
                      effective_filter_size + stride) /
                     stride;
      break;
    case Padding::SAME:
      *output_size = (input_size + stride - 1) / stride;
      const int64_t padding_needed =
          std::max(int64_t{0}, (*output_size - 1) * stride +
                                   effective_filter_size - input_size);
      // For odd values of total padding, add more padding at the 'right'
      // side of the given dimension.
      *padding_before = padding_needed / 2;
      *padding_after = padding_needed - *padding_before;
      break;
  }
  if (*output_size < 0) {
    return errors::InvalidArgument(
        "Computed output size would be negative: ", *output_size,
        " [input_size: ", input_size,
        ", effective_filter_size: ", effective_filter_size,
        ", stride: ", stride, "]");
  }
  return absl::OkStatus();
}

absl::Status GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                                   int dilation_rate, int64_t stride,
                                   Padding padding_type, int64_t* output_size,
                                   int64_t* padding_size) {
  if (padding_type == Padding::EXPLICIT) {
    return errors::Internal(
        "GetWindowedOutputSize does not handle EXPLICIT padding; call "
        "GetWindowedOutputSizeVerbose instead");
  }
  int64_t padding_after_unused;
  return GetWindowedOutputSizeVerbose(input_size, filter_size, dilation_rate,
                                      stride, padding_type, output_size,
                                      padding_size, &padding_after_unused);
}

absl::Status Get3dOutputSizeV2(const std::array<int64_t, 3>& input,
                               const std::array<int64_t, 3>& window,
                               const std::array<int64_t, 3>& dilations,
                               const std::array<int64_t, 3>& strides,
                               Padding padding_type,
                               std::array<int64_t, 3>* output_ptr,
                               std::array<int64_t, 3>* padding_ptr) {
  for (size_t i = 0; i < input.size(); ++i) {
    TF_RETURN_IF_ERROR(GetWindowedOutputSize(
        input[i], window[i], dilations[i], strides[i], padding_type,
        &(*output_ptr)[i], &(*padding_ptr)[i]));
  }
  return absl::OkStatus();
}
}  // namespace tensorflow
