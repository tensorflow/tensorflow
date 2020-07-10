/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_
#define TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_

#include <cmath>
#include <complex>
#include <iostream>
#include <limits>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/byte_order.h"

namespace Eigen {
struct bfloat16;
struct half;
}  // namespace Eigen

namespace tensorflow {
typedef Eigen::bfloat16 bfloat16;
}  // end namespace tensorflow

namespace std {

using tensorflow::bfloat16;
inline bool isinf(const bfloat16& a) { return std::isinf(float(a)); }
inline bool isnan(const bfloat16& a) { return std::isnan(float(a)); }
inline bool isfinite(const bfloat16& a) { return std::isfinite(float(a)); }
inline bfloat16 abs(const bfloat16& a) { return bfloat16(std::abs(float(a))); }
inline bfloat16 exp(const bfloat16& a) { return bfloat16(std::exp(float(a))); }
inline bfloat16 expm1(const bfloat16& a) {
  return bfloat16(std::expm1(float(a)));
}
inline bfloat16 log(const bfloat16& a) { return bfloat16(std::log(float(a))); }
inline bfloat16 log1p(const bfloat16& a) {
  return bfloat16(std::log1p(float(a)));
}
inline bfloat16 log10(const bfloat16& a) {
  return bfloat16(std::log10(float(a)));
}
inline bfloat16 sqrt(const bfloat16& a) {
  return bfloat16(std::sqrt(float(a)));
}
inline bfloat16 pow(const bfloat16& a, const bfloat16& b) {
  return bfloat16(std::pow(float(a), float(b)));
}
inline bfloat16 sin(const bfloat16& a) { return bfloat16(std::sin(float(a))); }
inline bfloat16 cos(const bfloat16& a) { return bfloat16(std::cos(float(a))); }
inline bfloat16 tan(const bfloat16& a) { return bfloat16(std::tan(float(a))); }
inline bfloat16 tanh(const bfloat16& a) {
  return bfloat16(std::tanh(float(a)));
}
inline bfloat16 floor(const bfloat16& a) {
  return bfloat16(std::floor(float(a)));
}
inline bfloat16 ceil(const bfloat16& a) {
  return bfloat16(std::ceil(float(a)));
}
}  // namespace std

#endif  // TENSORFLOW_CORE_LIB_BFLOAT16_BFLOAT16_H_
