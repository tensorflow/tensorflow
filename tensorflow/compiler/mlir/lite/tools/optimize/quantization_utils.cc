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
#include "tensorflow/compiler/mlir/lite/tools/optimize/quantization_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/log/check.h"

namespace mlir {
namespace TFL {

namespace {

const int8_t kMinQuantizedValue8bit = -127;
const int8_t kMaxQuantizedValue8bit = 127;

// const int8_t kMinQuantizedValue4bit = -7;
// const int8_t kMaxQuantizedValue4bit = 7;

// The maximum number of dimensions supported in per-channel quantization.
constexpr int kPerChannelMaxDim = 4;

// Forked from tensorflow/compiler/mlir/tools/safe_cast.h.
// Converts a floating-point number to an integer. For all inputs x where
// static_cast<IntOut>(x) is legal according to the C++ standard, the result
// is identical to that cast (i.e. the result is x with its fractional part
// truncated whenever that is representable as IntOut).
//
// static_cast would cause undefined behavior for the following cases, which
// have well-defined behavior for this function:
//
//  1. If x is NaN, the result is zero.
//
//  2. If the truncated form of x is above the representable range of IntOut,
//     the result is std::numeric_limits<IntOut>::max().
//
//  3. If the truncated form of x is below the representable range of IntOut,
//     the result is std::numeric_limits<IntOut>::min().
//
// Note that cases #2 and #3 cover infinities as well as finite numbers.
//
// The range of FloatIn must include the range of IntOut, otherwise
// the results are undefined.
// TODO(sfeuz): Replace by absl::SafeCast once available.
template <class IntOut, class FloatIn>
IntOut SafeCast(FloatIn x) {
  static_assert(!std::numeric_limits<FloatIn>::is_integer,
                "FloatIn is integer");
  static_assert(std::numeric_limits<IntOut>::is_integer,
                "IntOut is not integer");
  static_assert(std::numeric_limits<IntOut>::radix == 2, "IntOut is base 2");

  // Special case NaN, for which the logic below doesn't work.
  if (std::isnan(x)) {
    return 0;
  }

  // Negative values all clip to zero for unsigned results.
  if (!std::numeric_limits<IntOut>::is_signed && x < 0) {
    return 0;
  }

  // Handle infinities.
  if (std::isinf(x)) {
    return x < 0 ? std::numeric_limits<IntOut>::min()
                 : std::numeric_limits<IntOut>::max();
  }

  // Set exp such that x == f * 2^exp for some f with |f| in [0.5, 1.0),
  // unless x is zero in which case exp == 0. Note that this implies that the
  // magnitude of x is strictly less than 2^exp.
  int exp = 0;
  std::frexp(x, &exp);

  // Let N be the number of non-sign bits in the representation of IntOut. If
  // the magnitude of x is strictly less than 2^N, the truncated version of x
  // is representable as IntOut. The only representable integer for which this
  // is not the case is kMin for signed types (i.e. -2^N), but that is covered
  // by the fall-through below.
  if (exp <= std::numeric_limits<IntOut>::digits) {
    return x;
  }

  // Handle numbers with magnitude >= 2^N.
  return x < 0 ? std::numeric_limits<IntOut>::min()
               : std::numeric_limits<IntOut>::max();
}
}  // namespace

template <class BiasType>
std::vector<BiasType> SymmetricBiasQuantize(const float* data,
                                            uint64_t num_elements,
                                            const std::vector<float>& scales) {
  CHECK(!scales.empty()) << "Scales must not be empty.";      // Crash OK
  CHECK(scales.size() == 1 || scales.size() >= num_elements)  // Crash OK
      << "Expected 1 or at least " << num_elements << " scales, got "
      << scales.size() << ".";
  std::vector<BiasType> buffer(num_elements);
  const BiasType kScale = std::numeric_limits<BiasType>::max();
  float scaling_factor_inv_per_layer = (scales[0] == 0) ? 0 : 1.0 / scales[0];

  for (uint64_t idx = 0; idx < num_elements; idx++) {
    float scaling_factor_inv =
        scales.size() == 1 ? scaling_factor_inv_per_layer
                           : ((scales[idx] == 0) ? 0 : 1.0 / scales[idx]);
    const BiasType quantized_value =
        SafeCast<BiasType>(std::round(data[idx] * scaling_factor_inv));
    buffer[idx] = std::min(kScale, std::max(-kScale, quantized_value));
  }
  return buffer;
}

template std::vector<std::int32_t> SymmetricBiasQuantize<std::int32_t>(
    const float* data, uint64_t num_elements, const std::vector<float>& scales);

template std::vector<std::int64_t> SymmetricBiasQuantize<std::int64_t>(
    const float* data, uint64_t num_elements, const std::vector<float>& scales);

std::vector<int16_t> SymmetricQuantizeFloatsToInt16(const float* data,
                                                    uint64_t num_elements,
                                                    float scaling_factor) {
  // Compute the inverse of scale.
  const float scaling_factor_inv =
      (scaling_factor == 0) ? 0 : 1.0 / scaling_factor;
  std::vector<int16_t> buffer(num_elements);
  const int32_t kScale = std::numeric_limits<int16_t>::max();

  for (size_t i = 0; i < num_elements; i++) {
    const int32_t quantized_value =
        static_cast<int32_t>(std::round(data[i] * scaling_factor_inv));
    buffer[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
  return buffer;
}

void SymmetricPerChannelQuantizeValues(const float* const input,
                                       const std::vector<float>& scales_inv,
                                       const std::vector<int32_t>& dimension,
                                       int32_t channel_dim_index,
                                       std::vector<int8_t>* output_value) {
  // Quantize the values.
  int indices[kPerChannelMaxDim];
  // Left-pad the shape with 1s up to `kPerChannelMaxDim` dimensions.
  const int num_dims = static_cast<int>(dimension.size());
  CHECK_LE(num_dims, kPerChannelMaxDim)  // Crash OK
      << "At most " << kPerChannelMaxDim << " dimensions are supported.";
  CHECK_GE(channel_dim_index, 0) << "Invalid channel dimension.";  // Crash OK
  CHECK_LT(channel_dim_index, num_dims)                            // Crash OK
      << "Invalid channel dimension.";
  std::vector<int64_t> shape;
  shape.reserve(std::max(num_dims, kPerChannelMaxDim));
  for (int i = 0; i < kPerChannelMaxDim - num_dims; ++i) {
    shape.push_back(1);
  }
  for (const int32_t dim : dimension) {
    CHECK_GE(dim, 0) << "Dimension must be non-negative.";  // Crash OK
    shape.push_back(dim);
  }
  CHECK_GE(scales_inv.size(),  // Crash OK
           static_cast<size_t>(dimension[channel_dim_index]))
      << "Expected one scale per channel.";
  channel_dim_index += kPerChannelMaxDim - num_dims;

  for (indices[0] = 0; indices[0] < shape.at(0); indices[0]++) {
    for (indices[1] = 0; indices[1] < shape.at(1); indices[1]++) {
      for (indices[2] = 0; indices[2] < shape.at(2); indices[2]++) {
        for (indices[3] = 0; indices[3] < shape.at(3); indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          int index = 0;
          int current_stride = 1;
          for (int i = kPerChannelMaxDim - 1; i >= 0; --i) {
            index += indices[i] * current_stride;
            current_stride *= shape.at(i);
          }
          const float val = input[index];
          const int32_t quantized_value =
              static_cast<int32_t>(std::round(val * scales_inv[channel_idx]));
          output_value->at(index) = std::min<int8_t>(
              kMaxQuantizedValue8bit,
              std::max<int8_t>(kMinQuantizedValue8bit, quantized_value));
        }
      }
    }
  }
}

}  // namespace TFL
}  // namespace mlir
