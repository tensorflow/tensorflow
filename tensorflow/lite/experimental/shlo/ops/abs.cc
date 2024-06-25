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
#include "tensorflow/lite/experimental/shlo/ops/abs.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Abs {
  template <class T>
  T operator()(const T& val) {
    return val < static_cast<T>(0) ? static_cast<T>(-val) : val;
  }
};

AbsOp Create(typename AbsOp::Attributes) { return AbsOp{}; }

absl::Status Prepare(AbsOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(CheckCtx("abs"), input,
                                               IsSignedIntTensor, IsFloatTensor,
                                               IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("abs"), input, output));
  return absl::OkStatus();
}

absl::Status Evaluate(AbsOp& op, const Tensor& input, Tensor& output) {
  Abs abs;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), abs, input,
        output)
  } else if (IsSignedIntTensor(input) || IsFloatTensor(input)) {
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       input.tensor_element_type(), abs, input, output);
  }
  return absl::FailedPreconditionError("Unsupported tensor type.");
}

}  // namespace shlo_ref
