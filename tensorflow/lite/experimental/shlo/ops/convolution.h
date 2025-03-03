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
  ~ConvolutionOp() {
    free(lhs_permutation_data);
    free(lhs_transposed_data);
    free(rhs_permutation_data);
    free(rhs_transposed_data);
    free(output_permutation_data);
    free(output_transposed_data);
    free(lhs_padded_data);
    free(lhs_dot_general_data);
    free(rhs_dot_general_data);
    free(output_dot_general_data);
    free(lhs_contracting_dimensions_data);
    free(rhs_contracting_dimensions_data);
    if (lhs_dequantized_data != NULL) {
      free(lhs_dequantized_data);
      free(rhs_dequantized_data);
      free(output_dequantized_data);
    }
    for (int64_t i = 0;
         i < attributes.batch_group_count * attributes.feature_group_count;
         ++i) {
      free(lhs_splits_data[i]);
      free(rhs_splits_data[i]);
    }
  }
  struct Attributes {
    Tensor window_strides;
    Tensor padding;
    Tensor lhs_dilation;
    Tensor rhs_dilation;
    Tensor window_reversal;
    int64_t input_batch_dimension;
    int64_t input_feature_dimension;
    Tensor input_spacial_dimensions;
    int64_t kernel_input_feature_dimension;
    int64_t kernel_output_feature_dimension;
    Tensor kernel_spacial_dimensions;
    int64_t output_batch_dimension;
    int64_t output_feature_dimension;
    Tensor output_spacial_dimensions;
    int64_t feature_group_count;
    int64_t batch_group_count;
    absl::InlinedVector<PrecisionTypes, 2> precision_configs;
  };
  Attributes attributes;
  DotGeneralOp dot_general_op;
  Tensor lhs_transposed;
  Tensor lhs_permutations;
  Tensor rhs_transposed;
  Tensor rhs_permutations;
  Tensor output_transposed;
  Tensor output_permutations;
  Tensor lhs_padded;
  Tensor lhs_dot_general;
  Tensor rhs_dot_general;
  Tensor output_dot_general;
  std::vector<Tensor> lhs_splits;
  std::vector<Tensor> rhs_splits;
  Tensor lhs_dequantized;
  Tensor rhs_dequantized;
  Tensor output_dequantized;
  void* lhs_permutation_data;
  void* lhs_transposed_data;
  void* rhs_permutation_data;
  void* rhs_transposed_data;
  void* output_permutation_data;
  void* output_transposed_data;
  void* lhs_padded_data;
  void* lhs_dot_general_data;
  void* rhs_dot_general_data;
  void* output_dot_general_data;
  void* lhs_contracting_dimensions_data;
  void* rhs_contracting_dimensions_data;
  void* lhs_dequantized_data;
  void* rhs_dequantized_data;
  void* output_dequantized_data;
  std::vector<void*> lhs_splits_data;
  std::vector<void*> rhs_splits_data;
};

ConvolutionOp Create(const ConvolutionOp::Attributes& attributes);
absl::Status Prepare(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output);
absl::Status Evaluate(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_ABS_H_