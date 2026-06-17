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

#include "tensorflow/lite/experimental/shlo/ops/subtract.h"

#include <functional>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Subtract : std::minus<void> {};

SubtractOp Create(SubtractOp::Attributes) { return {}; }

absl::Status Prepare(SubtractOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(CheckCtx("subtract"), lhs,
                                               IsIntTensor, IsFloatTensor,
                                               IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("subtract"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("subtract"), rhs, output));
  return absl::OkStatus();
}

absl::Status Evaluate(SubtractOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  Subtract subtract;
  if (IsIntTensor(lhs) || IsFloatTensor(lhs)) {
    // Note: all the arithmetic types share the same implementation.
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       lhs.tensor_element_type(), subtract, lhs, rhs, output);
  } else if (IsQuantizedPerTensorTensor(lhs)) {
    DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerTensor,
                       lhs.quantized_per_tensor_element_type().StorageType(),
                       lhs.quantized_per_tensor_element_type().ExpressedType(),
                       subtract, lhs, rhs, output)
  }
  return absl::FailedPreconditionError(
      "stablehlo.subtract: Unsupported tensor type.");
}

}  // namespace shlo_ref
