/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions xor
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/experimental/shlo/ops/xor.h"

#include <functional>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

template <DataType>
struct Xor : std::bit_xor<void> {};

template <>
struct Xor<DataType::kI1> {
  template <class T>
  bool operator()(T lhs, T rhs) const {
    return static_cast<bool>(lhs) != static_cast<bool>(rhs);
  }
};

XorOp Create(XorOp::Attributes) { return {}; }

absl::Status Prepare(XorOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("xor"), lhs, IsBoolTensor, IsIntTensor));
  SHLO_REF_RETURN_ON_ERROR(CheckSameBaselineType(CheckCtx("xor"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(CheckSameBaselineType(CheckCtx("xor"), rhs, output));
  return absl::OkStatus();
}

absl::Status Evaluate(XorOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (IsIntTensor(lhs)) {
    // Note: all the integer types share the same implementation.
    Xor<DataType::kSI32> xor_func;
    DISPATCH_INT(detail::EvaluateNoQuantization, lhs.tensor_element_type(),
                 xor_func, lhs, rhs, output);
  } else if (IsBoolTensor(lhs)) {
    Xor<DataType::kI1> xor_func;
    detail::EvaluateNoQuantization<DataType::kI1>(xor_func, lhs, rhs, output);
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      "stablehlo.xor: Unsupported tensor type in Evaluate.");
}

}  // namespace shlo_ref
