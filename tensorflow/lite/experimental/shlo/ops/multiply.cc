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

#include "tensorflow/lite/experimental/shlo/ops/multiply.h"

#include <functional>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

template <DataType expressed_type>
struct Multiply : std::multiplies<void> {};

template <>
struct Multiply<DataType::kI1> {
  template <class T>
  T operator()(const T& lhs, const T& rhs) const {
    return static_cast<T>(lhs && rhs);
  }
};

MultiplyOp Create(MultiplyOp::Attributes) { return {}; }

absl::Status Prepare(MultiplyOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("multiply"), lhs, IsBoolTensor, IsIntTensor,
                          IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("multiply"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("multiply"), rhs, output));
  return absl::OkStatus();
}

absl::Status Evaluate(MultiplyOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (IsBoolTensor(lhs)) {
    detail::EvaluateNoQuantization<DataType::kI1>(Multiply<DataType::kI1>(),
                                                  lhs, rhs, output);
    return absl::OkStatus();
  } else if (IsIntTensor(lhs) || IsFloatTensor(lhs)) {
    // Note: all the arithmetic types share the same implementation.
    Multiply<DataType::kF32> multiply;
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       lhs.tensor_element_type(), multiply, lhs, rhs, output);
  } else if (IsQuantizedPerTensorTensor(lhs)) {
    Multiply<DataType::kF32> multiply;
    DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerTensor,
                       lhs.quantized_per_tensor_element_type().StorageType(),
                       lhs.quantized_per_tensor_element_type().ExpressedType(),
                       multiply, lhs, rhs, output)
  }
  return absl::FailedPreconditionError(
      "stablehlo.multiply: Unsupported tensor type.");
}

}  // namespace shlo_ref
