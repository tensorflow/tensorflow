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

#include "tensorflow/lite/experimental/shlo/ops/and.h"

#include <functional>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

template <DataType>
struct And : std::bit_and<void> {};

template <>
struct And<DataType::kI1> : std::logical_and<void> {};

AndOp Create(AndOp::Attributes) { return {}; }

absl::Status Prepare(AndOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("and"), lhs, IsBoolTensor, IsIntTensor));
  SHLO_REF_RETURN_ON_ERROR(CheckSameBaselineType(CheckCtx("and"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(CheckSameBaselineType(CheckCtx("and"), rhs, output));
  return absl::OkStatus();
}

absl::Status Evaluate(AndOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (IsIntTensor(lhs)) {
    // Note: all the integer types share the same implementation.
    And<DataType::kSI32> and_func;
    DISPATCH_INT(detail::EvaluateNoQuantization, lhs.tensor_element_type(),
                 and_func, lhs, rhs, output);
  } else if (IsBoolTensor(lhs)) {
    And<DataType::kI1> and_func;
    detail::EvaluateNoQuantization<DataType::kI1>(and_func, lhs, rhs, output);
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      "stablehlo.and: Unsupported tensor type in Evaluate.");
}

}  // namespace shlo_ref
