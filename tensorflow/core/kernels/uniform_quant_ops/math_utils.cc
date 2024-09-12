/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/uniform_quant_ops/math_utils.h"

#include <algorithm>
#include <cmath>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

using errors::InvalidArgument;

// Reference:
// https://github.com/tensorflow/tensorflow/blob/57946ceb4b6119d6d0f49abbb2e3d1636a3b83a0/tensorflow/lite/kernels/internal/quantization_util.cc#L53
// Where double_multiplier >= 0 and TFLITE_EMULATE_FLOAT is not defined.
Status QuantizeMultiplier(double double_multiplier,
                          int32_t& quantized_multiplier, int32_t& shift) {
  if (!isfinite(double_multiplier) || double_multiplier <= 0) {
    return InvalidArgument(
        "double_multiplier must be a poisitive finite number. Given ",
        double_multiplier);
  }
  const double q = std::frexp(double_multiplier, &shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));
  if (q_fixed == (1LL << 31)) {
    q_fixed /= 2;
    ++shift;
  }
  if (shift < -31) {
    shift = 0;
    q_fixed = 0;
  }
  if (shift > 30) {
    shift = 30;
    q_fixed = (1LL << 31) - 1;
  }
  quantized_multiplier = static_cast<int32_t>(q_fixed);
  return absl::OkStatus();
}

}  // namespace tensorflow
