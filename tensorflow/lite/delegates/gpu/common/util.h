/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_UTIL_H_

#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

// @param n must be non negative
// @param divisor must be greater than zero
template <typename T, typename N>
T IntegralDivideRoundUp(T n, N divisor) {
  const T div = static_cast<T>(divisor);
  const T q = n / div;
  return n % div == 0 ? q : q + 1;
}

template <>
inline uint3 IntegralDivideRoundUp(uint3 n, uint3 divisor) {
  return uint3(IntegralDivideRoundUp(n.x, divisor.x),
               IntegralDivideRoundUp(n.y, divisor.y),
               IntegralDivideRoundUp(n.z, divisor.z));
}

// @param number or its components must be greater than zero
// @param n must be greater than zero
template <typename T, typename N>
T AlignByN(T number, N n) {
  return IntegralDivideRoundUp(number, n) * n;
}

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_UTIL_H_
