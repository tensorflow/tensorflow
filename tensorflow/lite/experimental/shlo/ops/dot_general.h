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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_DOT_GENERAL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_DOT_GENERAL_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

enum class PrecisionTypes {
  DEFAULT,
  HIGH,
  HIGHEST,
};

class DotGeneralOp {
 public:
  struct Attributes {
    absl::Span<const Axis> lhs_batching_dimensions;
    absl::Span<const Axis> rhs_batching_dimensions;
    absl::Span<const Axis> lhs_contracting_dimensions;
    absl::Span<const Axis> rhs_contracting_dimensions;
    std::array<PrecisionTypes, 2> precision_configs;
  };
  Attributes attributes;
  Tensor lhs_dequantized;
  Tensor rhs_dequantized;
  Tensor output_dequantized;
  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_result_dims;
  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_result_dims;
  std::vector<std::byte> lhs_dequantized_data;
  std::vector<std::byte> rhs_dequantized_data;
  std::vector<std::byte> output_dequantized_data;
};

DotGeneralOp Create(DotGeneralOp::Attributes attributes);
absl::Status Prepare(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output);
absl::Status Evaluate(DotGeneralOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_DOT_GENERAL_H_
