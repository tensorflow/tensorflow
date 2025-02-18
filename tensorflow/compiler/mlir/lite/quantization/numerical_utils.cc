/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/quantization/numerical_utils.h"

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "absl/types/optional.h"

namespace mlir {
namespace quant {

// Converts a double-precision floating-point multiplier to a quantized
// multiplier.
//
// Args:
//   double_multiplier: The double-precision floating-point multiplier.
//
// Returns:
//   A quantized multiplier, represented as a pair of integers: the quantized
//   multiplier and the shift amount. The shift amount is the number of bits
//   that the quantized multiplier should be shifted to the right before being
//   used.
QuantizedMultiplier QuantizeMultiplier(double double_multiplier) {
  if (double_multiplier < 1e-6) {
    return {0, 0};
  }

  int32_t shift;
  const double q = frexp(double_multiplier, &shift);
  int64_t quantized_multiplier = round(q * (1LL << 31));
  assert(quantized_multiplier <= (1LL << 31));
  if (quantized_multiplier == (1LL << 31)) {
    quantized_multiplier /= 2;
    ++shift;
  }
  assert(quantized_multiplier <= std::numeric_limits<int32_t>::max());

  // Check that the shift amount is not greater than 31 or less than -31.
  if (shift > 31 || shift < -31) {
    return {0, 0};
  }

  return {static_cast<int32_t>(quantized_multiplier), shift};
}

// Calculates the quantized range for a given scale, zero point, minimum and
// maximum values, and quantization range.
//
// Args:
//   scale: The scale factor for the quantized values.
//   zero_point: The zero point for the quantized values.
//   rmin: The minimum value of the quantized values.
//   rmax: The maximum value of the quantized values.
//   qmin: The minimum value of the quantization range.
//   qmax: The maximum value of the quantization range.
//
// Returns:
//   A quantized range, represented as a pair of integers: the minimum and
//   maximum quantized values.
QuantizedRange CalculateQuantizedRange(double scale, int32_t zero_point,
                                       std::optional<double> rmin,
                                       std::optional<double> rmax, int32_t qmin,
                                       int32_t qmax) {
  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(std::round(f / scale));
  };

  if (rmin.has_value() && rmax.has_value()) {
    return {std::max(qmin, quantize(rmin.value())),
            std::min(qmax, quantize(rmax.value()))};
  } else if (rmin.has_value()) {
    return {std::max(qmin, quantize(rmin.value())), qmax};
  } else if (rmax.has_value()) {
    return {qmin, std::min(qmax, quantize(rmax.value()))};
  } else {
    return {qmin, qmax};
  }
}

}  // namespace quant
}  // namespace mlir
