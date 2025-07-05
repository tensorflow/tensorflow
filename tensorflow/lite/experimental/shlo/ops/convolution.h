/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_H_

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/dot_general.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

class ConvolutionOp {
 public:
  struct Attributes {
    absl::Span<int64_t> window_strides;
    Tensor padding;
    absl::Span<int64_t> lhs_dilation;
    absl::Span<int64_t> rhs_dilation;
    int64_t input_batch_dimension;
    int64_t input_feature_dimension;
    absl::Span<int64_t> input_spatial_dimensions;
    int64_t kernel_input_feature_dimension;
    int64_t kernel_output_feature_dimension;
    absl::Span<int64_t> kernel_spatial_dimensions;
    int64_t output_batch_dimension;
    int64_t output_feature_dimension;
    absl::Span<int64_t> output_spatial_dimensions;
    int64_t feature_group_count;
    int64_t batch_group_count;
    std::array<PrecisionTypes, 2> precision_configs;
  };
  Attributes attributes;
  Tensor lhs_dot_general;
  Tensor rhs_dot_general;
  Tensor output_dot_general;
  std::vector<std::byte> lhs_dot_general_data;
  std::vector<std::byte> rhs_dot_general_data;
  std::vector<std::byte> output_dot_general_data;
  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_contracting_dimensions;
  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_contracting_dimensions;
  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_result_dims;
  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_result_dims;
};

ConvolutionOp Create(const ConvolutionOp::Attributes& attributes);
absl::Status Prepare(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output);
absl::Status Evaluate(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_CONVOLUTION_H_