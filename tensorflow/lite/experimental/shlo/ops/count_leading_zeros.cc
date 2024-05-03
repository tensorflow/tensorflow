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

#include "tensorflow/lite/experimental/shlo/ops/count_leading_zeros.h"

#include <cstdint>
#include <type_traits>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/i4.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct CountLeadingZeros {
  template <class T>
  T operator()(T v) const {
    if constexpr (std::is_same_v<I4, T>) {
      return I4(absl::countl_zero(static_cast<uint8_t>(v << 4 | 0xf)));
    } else {
      return absl::countl_zero(static_cast<std::make_unsigned_t<T>>(v));
    }
  }
};

CountLeadingZerosOp Create(CountLeadingZerosOp::Attributes) { return {}; }

absl::Status Prepare(CountLeadingZerosOp& op, const Tensor& input,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("count_leading_zeros"), input, IsIntTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("count_leading_zeros"), input, output));
  return absl::OkStatus();
}

absl::Status Evaluate(CountLeadingZerosOp& op, const Tensor& input,
                      Tensor& output) {
  CountLeadingZeros count_leading_zeros;
  if (IsIntTensor(input)) {
    DISPATCH_INT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                 count_leading_zeros, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.count_leading_zeros: Unsupported tensor type.");
}

};  // namespace shlo_ref
