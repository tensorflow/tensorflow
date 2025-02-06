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

#include "tensorflow/lite/experimental/shlo/ops/floor.h"

#include <cmath>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Floor {
  template <class T>
  T operator()(T v) const {
    return std::floor(v);
  }
};

template <>
F16 Floor::operator()<F16>(F16 val) const {
  return F16(operator()(static_cast<float>(val)));
}

template <>
BF16 Floor::operator()<BF16>(BF16 val) const {
  return BF16(operator()(static_cast<float>(val)));
}

FloorOp Create(FloorOp::Attributes) { return {}; }

absl::Status Prepare(FloorOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("floor"), input, IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("floor"), input, output));
  return absl::OkStatus();
}

absl::Status Evaluate(FloorOp& op, const Tensor& input, Tensor& output) {
  Floor floor;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), floor, input,
        output)
  } else if (IsFloatTensor(input)) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   floor, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.floor: Unsupported tensor type.");
}

};  // namespace shlo_ref
