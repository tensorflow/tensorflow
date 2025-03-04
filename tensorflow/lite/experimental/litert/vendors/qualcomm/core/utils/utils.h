// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_UTILS_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_UTILS_UTILS_H_

#include <cmath>

namespace qnn {

template <typename...>
inline constexpr bool always_false = false;

template <typename T>
T Quantize(const float val, const float scale, const int32_t zero_point) {
  static_assert(std::is_integral<T>::value,
                "Integral required in Quantize function.");
  return std::round(val / scale) + zero_point;
}

template <typename T>
float Dequantize(const T val, const float scale, const int32_t zero_point) {
  static_assert(std::is_integral<T>::value,
                "Integral required in Dequantize function.");
  return scale * (val - zero_point);
}

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_UTILS_UTILS_H_
