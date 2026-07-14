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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UTIL_H_

#include <string>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

#define SHLO_REF_RETURN_ON_ERROR(EXPR)    \
  if (absl::Status s = (EXPR); !s.ok()) { \
    return s;                             \
  }

// Propagates the input shape to the output shape.
//
// If the output shape is already populated, checks that is it compatible with
// the input.
absl::Status Propagate(const Shape& input_shape, Shape& output_shape);

// Propagates the input shapes to the output shape.
//
// If the output shape is already populated, checks that is it compatible with
// the inputs.
absl::Status Propagate(const Shape& lhs_shape, const Shape& rhs_shape,
                       Shape& output_shape);

// Provides context information for the `Check*` functions error messages.
struct CheckCtx {
  explicit CheckCtx(std::string name) : op_name(name) {}
  // The operation that requested the check.
  std::string op_name;
};

// Checks that the `tensor` element type is supported by one the the `checks`
// functions.
//
// Returns a failed precondition error when no check succeeds.
//
// The check functions should have the following signature.
//
// ```
// bool Check(const Tensor& tensor);
// ```
template <class... CheckFuncs>
absl::Status CheckSupportedTypes(CheckCtx ctx, const Tensor& tensor,
                                 CheckFuncs&&... checks) {
  if ((static_cast<CheckFuncs&&>(checks)(tensor) || ...)) {
    return absl::OkStatus();
  }
  std::string tensor_type_repr = std::visit(
      [](auto v) -> std::string { return ToString(v); }, tensor.element_type());
  return absl::FailedPreconditionError("stablehlo." + ctx.op_name +
                                       ": Unsupported tensor type (" +
                                       tensor_type_repr + ").");
}

// Returns true if the tensor's storage type is boolean.
bool IsBoolTensor(const Tensor& tensor);

// Returns true if the tensor's storage type is a signed integer type.
bool IsSignedIntTensor(const Tensor& tensor);

// Returns true if the tensor's storage type is an unsigned integer type.
bool IsUnsignedIntTensor(const Tensor& tensor);

// Returns true if the tensor's storage type is an integer type.
bool IsIntTensor(const Tensor& tensor);

// Returns true if the tensor's storage type is an floating point type.
bool IsFloatTensor(const Tensor& tensor);

// Returns true if the tensor's storage type is quantized per tensor.
bool IsQuantizedPerTensorTensor(const Tensor& tensor);

// Returns true if the tensor's storage type is quantized per axis.
bool IsQuantizedPerAxisTensor(const Tensor& tensor);

// Checks that both tensors have the same baseline element type.
absl::Status CheckSameBaselineType(CheckCtx ctx, const Tensor& tensor1,
                                   const Tensor& tensor2);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UTIL_H_
