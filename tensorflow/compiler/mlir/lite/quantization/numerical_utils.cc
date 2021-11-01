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
#include <limits>

#include "absl/types/optional.h"

namespace mlir {
namespace quant {

// This method is adopted from TFLite:
// ["tensorflow/lite/kernels/internal/quantization_util.cc"]
QuantizedMultiplier QuantizeMultiplier(double double_multiplier) {
  if (double_multiplier < 1e-6) {
    return {0, 0};
  }

  int32_t shift;
  const double q = frexp(double_multiplier, &shift);
  auto q_fixed = static_cast<int64_t>(round(q * (1LL << 31)));
  assert(q_fixed <= (1LL << 31));
  if (q_fixed == (1LL << 31)) {
    q_fixed /= 2;
    ++shift;
  }
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (shift < -31) {
    shift = 0;
    q_fixed = 0;
  }
  return {static_cast<int32_t>(q_fixed), shift};
}

QuantizedRange CalculateQuantizedRange(double scale, int32_t zero_point,
                                       absl::optional<double> rmin,
                                       absl::optional<double> rmax,
                                       int32_t qmin, int32_t qmax) {
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
