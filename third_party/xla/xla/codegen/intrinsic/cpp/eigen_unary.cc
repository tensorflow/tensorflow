/* Copyright 2025 The OpenXLA Authors.

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

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#include "xla/codegen/intrinsic/cpp/eigen_unary.h"

#include <cmath>

#include "Eigen/Core"  // NOLINT(misc-include-cleaner)
#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

//===--------------------------------------------------------------------===//
// Generic conversion and operation
//===--------------------------------------------------------------------===//

template <typename VecType>
inline VecType VectorTanh(const VecType x) {
  using ArrayType = typename ArrayMap<VecType>::type;
  ArrayType x_array = *reinterpret_cast<const ArrayType*>(&x);
  ArrayType result = x_array.tanh();
  return *reinterpret_cast<const VecType*>(&result);
}

template <typename VecType>
inline VecType VectorAtan(const VecType x) {
  using ArrayType = typename ArrayMap<VecType>::type;
  ArrayType x_array = *reinterpret_cast<const ArrayType*>(&x);
  ArrayType result = x_array.atan();
  return *reinterpret_cast<const VecType*>(&result);
}

//===--------------------------------------------------------------------===//
// XLA entrypoints, renamed with asm in header file.
//===--------------------------------------------------------------------===//

// Single precision
float tanh_f32(float x) {
  return Eigen::internal::ptanh_float(x);  // NOLINT(misc-include-cleaner)
}
Vec4f tanh_v4f32(Vec4f x) { return VectorTanh(x); }
Vec8f tanh_v8f32(Vec8f x) { return VectorTanh(x); }
Vec16f tanh_v16f32(Vec16f x) { return VectorTanh(x); }

// Double precision
double tanh_f64(double x) {
  return Eigen::internal::ptanh_double(x);  // NOLINT(misc-include-cleaner)
}
Vec4d tanh_v4f64(Vec4d x) { return VectorTanh(x); }
Vec8d tanh_v8f64(Vec8d x) { return VectorTanh(x); }

// Single precision.
// This uses the same polynomial approximation as Eigen's vectorized version
// (generic_atan) for numerical consistency. Written manually to avoid
// inefficient generic scalar bitwise operations in Eigen.
float atan_f32(float x_in) {
  constexpr float kPiOverTwo = 1.5707963267948966f;

  float abs_x = std::abs(x_in);
  // For tiny inputs (|x| < 1e-3), atan(x) is indistinguishable from x up to
  // single precision epsilon. Bypassing Remez polynomial approximation avoids
  // intermediate squaring underflows and reciprocal division traps when invoked
  // via scalar loop expansion on ARM NEON.
  if (abs_x < 1e-3f) {
    return x_in;
  }
  const bool large_x = abs_x > 1.0f;
  // For |x| > 1, use atan(x) = sign(x)*pi/2 - atan(1/x). Use direct
  // approximation otherwise.
  float x = large_x ? (1.0f / abs_x) : abs_x;

  constexpr float kAlpha[] = {1.12026982009410858154296875e-01f,
                              7.296695709228515625e-01f,
                              8.109951019287109375e-01f};

  constexpr float kBeta[] = {1.00917108356952667236328125e-02f,
                             2.8318560123443603515625e-01f, 1.0f,
                             8.109951019287109375e-01f};

  float x2 = x * x;
  float p = (kAlpha[0] * x2 + kAlpha[1]) * x2 + kAlpha[2];
  float q = ((kBeta[0] * x2 + kBeta[1]) * x2 + kBeta[2]) * x2 + kBeta[3];
  float r = x * (p / q);

  float result = large_x ? (kPiOverTwo - r) : r;
  return std::copysign(result, x_in);
}
Vec4f atan_v4f32(Vec4f x) { return VectorAtan(x); }
Vec8f atan_v8f32(Vec8f x) { return VectorAtan(x); }
Vec16f atan_v16f32(Vec16f x) { return VectorAtan(x); }

// Double precision.
// This uses the same polynomial approximation as Eigen's vectorized version
// (generic_atan) for numerical consistency. Written manually to avoid
// inefficient generic scalar bitwise operations in Eigen.
double atan_f64(double x_in) {
  constexpr double kPiOverTwo = 1.57079632679489661923;

  double abs_x = std::abs(x_in);
  // For tiny inputs (|x| < 1e-9), atan(x) is indistinguishable from x.
  // Short-circuit bypass avoids intermediate Remez approximation underflows and
  // reciprocal division anomalies.
  if (abs_x < 1e-9) {
    return x_in;
  }
  const bool large_x = abs_x > 1.0;
  // For |x| > 1, use atan(x) = sign(x)*pi/2 - atan(1/x). Use direct
  // approximation otherwise.
  double x = large_x ? (1.0 / abs_x) : abs_x;

  constexpr double kAlpha[] = {2.6667153866462208e-05, 3.0917513112462781e-03,
                               5.2574296781008604e-02, 3.0409318473444424e-01,
                               7.5365702534987022e-01, 8.2704055405494614e-01,
                               3.3004361289279920e-01};

  constexpr double kBeta[] = {2.7311202462436667e-04,
                              1.0899150928962708e-02,
                              1.1548932646420353e-01,
                              4.9716458728465573e-01,
                              1.0,
                              9.3705509168587852e-01,
                              3.3004361289279920e-01};

  double x2 = x * x;
  double p =
      (((((kAlpha[0] * x2 + kAlpha[1]) * x2 + kAlpha[2]) * x2 + kAlpha[3]) *
            x2 +
        kAlpha[4]) *
           x2 +
       kAlpha[5]) *
          x2 +
      kAlpha[6];

  double q =
      (((((kBeta[0] * x2 + kBeta[1]) * x2 + kBeta[2]) * x2 + kBeta[3]) * x2 +
        kBeta[4]) *
           x2 +
       kBeta[5]) *
          x2 +
      kBeta[6];

  double r = x * (p / q);
  double result = large_x ? (kPiOverTwo - r) : r;
  return std::copysign(result, x_in);
}
Vec4d atan_v4f64(Vec4d x) { return VectorAtan(x); }
Vec8d atan_v8f64(Vec8d x) { return VectorAtan(x); }

}  // namespace xla::codegen
#endif  // defined(__has_attribute) && __has_attribute(vector_size) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
