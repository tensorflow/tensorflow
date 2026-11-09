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

#include "xla/codegen/intrinsic/cpp/atan.h"

#include <type_traits>

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

namespace {

// Coefficients match Eigen's generic_atan; written directly so every width,
// including scalar, lowers without libm calls.
template <typename T>
inline T Atan(const T x_in) {
  using Scalar = ElementType<T>;
  const T abs_x = BitwiseAbs(x_in);
  const auto large_x = abs_x > Splat<T>(Scalar{1});
  const T x = large_x ? (Splat<T>(Scalar{1}) / abs_x) : abs_x;
  const T x2 = x * x;

  if constexpr (std::is_same_v<Scalar, float>) {
    constexpr float kPiOverTwo = 1.5707963267948966f;
    const T p =
        HornerPoly(x2, 1.12026982009410858154296875e-01f,
                   7.296695709228515625e-01f, 8.109951019287109375e-01f);
    const T q = HornerPoly(x2, 1.00917108356952667236328125e-02f,
                           2.8318560123443603515625e-01f, 1.0f,
                           8.109951019287109375e-01f);
    const T r = x * (p / q);
    const T result = large_x ? (Splat<T>(kPiOverTwo) - r) : r;
    return BitwiseCopysign(result, x_in);
  } else {
    static_assert(std::is_same_v<Scalar, double>);
    constexpr double kPiOverTwo = 1.57079632679489661923;
    const T p = HornerPoly(x2, 2.6667153866462208e-05, 3.0917513112462781e-03,
                           5.2574296781008604e-02, 3.0409318473444424e-01,
                           7.5365702534987022e-01, 8.2704055405494614e-01,
                           3.3004361289279920e-01);
    const T q = HornerPoly(x2, 2.7311202462436667e-04, 1.0899150928962708e-02,
                           1.1548932646420353e-01, 4.9716458728465573e-01, 1.0,
                           9.3705509168587852e-01, 3.3004361289279920e-01);
    const T r = x * (p / q);
    const T result = large_x ? (Splat<T>(kPiOverTwo) - r) : r;
    return BitwiseCopysign(result, x_in);
  }
}

}  // namespace

//===--------------------------------------------------------------------===//
// XLA entrypoints, renamed with asm in header file.
//===--------------------------------------------------------------------===//

// Single precision
float atan_f32(float x) { return Atan(x); }
Vec4f atan_v4f32(Vec4f x) { return Atan(x); }
Vec8f atan_v8f32(Vec8f x) { return Atan(x); }
Vec16f atan_v16f32(Vec16f x) { return Atan(x); }

// Double precision
double atan_f64(double x) { return Atan(x); }
Vec2d atan_v2f64(Vec2d x) { return Atan(x); }
Vec4d atan_v4f64(Vec4d x) { return Atan(x); }
Vec8d atan_v8f64(Vec8d x) { return Atan(x); }

}  // namespace xla::codegen
#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
